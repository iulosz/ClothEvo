//! ClothEvo — Fase 0: protótipo do simulador de tecido XPBD (egui + wgpu).
//!
//! Modo de captura (para validar o render sem interação), via variáveis de ambiente:
//! `CLOTHEVO_SCENE=1|2|3`, `CLOTHEVO_FRAMES=<n passos simulados>` e
//! `CLOTHEVO_SCREENSHOT=<arquivo.ppm>` — salva o viewport e fecha o app.

use cloth_render::{
    primitives, ClothTopology, OrbitCamera, SceneRenderer, StaticMeshData, Vertex,
    ViewStyle, ViewportCallback,
};
use cloth_sim::{
    build_scene, ClothBuilder, Collider, FabricPreset, PanelMesh, Placement, SceneKind,
    Simulation, SimParams,
};
use eframe::{egui, egui_wgpu};
use glam::{Quat, Vec2, Vec3};
use pattern_core::{
    BoundaryCurve, Curve2D, CurveSegment, EdgeRef, FoldKind, FoldLine, GrainLine, PatternPiece,
    PatternProject, Pleat, Point2 as PatternPoint2, SeamDefinition,
};
use pattern_geom::{
    cut_closed_outline, offset_polygon, point_in_polygon, rectangle, regular_polygon, sample_curve,
    sample_curve_with_ranges, validate_simple_polygon, CutAndSewResult, SourceSegmentFragment,
};
use pattern_mesh::triangulate_outline;

const FIXED_DT: f32 = 1.0 / 60.0;
const DEFAULT_RES: u32 = 40;
const PATTERN_PIXELS_PER_METER: f32 = 360.0;
const PATTERN_SIM_BODY_RADIUS: f32 = 0.14;
const PATTERN_SIM_PERTURBATION: f32 = 0.001;
const PATTERN_SIM_LAYER_SPACING: f32 = 0.003;

/// Mantém os moldes do lado externo do colisor corporal, inclusive após o ruído inicial.
fn pattern_sim_layer_z(piece_index: usize, thickness: f32) -> f32 {
    PATTERN_SIM_BODY_RADIUS + thickness + PATTERN_SIM_PERTURBATION
        + piece_index as f32 * PATTERN_SIM_LAYER_SPACING
}

fn main() -> eframe::Result {
    let options = eframe::NativeOptions {
        viewport: egui::ViewportBuilder::default()
            .with_title("ClothEvo — Pattern Studio + XPBD")
            .with_inner_size([1360.0, 840.0])
            .with_min_inner_size([900.0, 600.0]),
        renderer: eframe::Renderer::Wgpu,
        // O viewport 3D precisa de depth buffer no render pass do egui.
        depth_buffer: 32,
        ..Default::default()
    };
    eframe::run_native(
        "ClothEvo",
        options,
        Box::new(|cc| Ok(Box::new(ClothEvoApp::new(cc)))),
    )
}

/// Parâmetros do modo de captura.
struct Capture {
    path: String,
    frames: u32,
    requested: bool,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum WorkspaceMode {
    Simulation,
    Pattern,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum PatternTool {
    Draw,
    Select,
}

#[derive(Clone, Copy, Debug, PartialEq)]
enum PatternEditTarget {
    Piece { piece: usize },
    Anchor { piece: usize, index: usize },
    Control1 { piece: usize, segment: usize },
    Control2 { piece: usize, segment: usize },
    Edge { piece: usize, segment: usize, t: f64 },
}

#[derive(Clone, Debug, PartialEq)]
struct PatternEditorSnapshot {
    project: PatternProject,
    draft_points: Vec<PatternPoint2>,
    selected_piece: usize,
    selected_target: Option<PatternEditTarget>,
}

impl PatternEditTarget {
    fn piece(self) -> usize {
        match self {
            Self::Piece { piece } | Self::Anchor { piece, .. } | Self::Control1 { piece, .. }
            | Self::Control2 { piece, .. } | Self::Edge { piece, .. } => piece,
        }
    }

    fn is_movable(self) -> bool {
        !matches!(self, Self::Edge { .. })
    }
}

struct ClothEvoApp {
    sim: Simulation,
    kind: SceneKind,
    res: u32,
    preset: FabricPreset,
    playing: bool,
    accumulator: f32,
    camera: OrbitCamera,
    style: ViewStyle,
    seam_pairs: Vec<(u32, u32)>,
    normals: Vec<Vec3>,
    render_state: Option<egui_wgpu::RenderState>,
    diverged: bool,
    last_step_ms: f32,
    capture: Option<Capture>,
    workspace: WorkspaceMode,
    pattern_project: PatternProject,
    draft_points: Vec<PatternPoint2>,
    pattern_spacing: f64,
    pattern_polygon_sides: usize,
    pattern_show_mesh: bool,
    selected_piece: usize,
    seam_a_piece: usize,
    seam_b_piece: usize,
    seam_a_segment: usize,
    seam_b_segment: usize,
    seam_a_reversed: bool,
    seam_b_reversed: bool,
    seam_ease_m: f64,
    pattern_message: String,
    using_pattern_sim: bool,
    pattern_tool: PatternTool,
    pattern_zoom: f32,
    pattern_pan: egui::Vec2,
    pattern_snap: bool,
    selected_target: Option<PatternEditTarget>,
    dragging_target: Option<PatternEditTarget>,
    dragging_before: Option<PatternEditorSnapshot>,
    drag_last_world: Option<PatternPoint2>,
    pattern_undo: Vec<PatternEditorSnapshot>,
    pattern_redo: Vec<PatternEditorSnapshot>,
    pattern_saved_path: Option<std::path::PathBuf>,
    pattern_dirty: bool,
    pattern_fit_pending: bool,
}

fn statics_from(colliders: &[Collider]) -> Vec<StaticMeshData> {
    let mut out = Vec::new();
    for c in colliders {
        match *c {
            Collider::Plane { normal, offset } => {
                if normal.dot(Vec3::Y) > 0.99 {
                    out.push(primitives::ground_quad(offset, 8.0, [0.62, 0.64, 0.66]));
                }
            }
            Collider::Sphere { center, radius } => {
                out.push(primitives::uv_sphere(center, radius, 48, 24, [0.78, 0.78, 0.80]));
            }
            Collider::Cylinder { a, b, radius } => {
                out.push(primitives::cylinder(a, b, radius, 48, [0.70, 0.76, 0.70]));
            }
        }
    }
    out
}

impl ClothEvoApp {
    fn new(cc: &eframe::CreationContext<'_>) -> Self {
        let render_state = cc.wgpu_render_state.clone();
        if let Some(rs) = &render_state {
            let depth = egui_wgpu::depth_format_from_bits(32, 0)
                .expect("formato de profundidade Depth32Float");
            let renderer = SceneRenderer::new(&rs.device, rs.target_format, depth);
            rs.renderer.write().callback_resources.insert(renderer);
        }

        // Modo de captura (opcional)
        let capture = std::env::var("CLOTHEVO_SCREENSHOT").ok().map(|path| Capture {
            path,
            frames: std::env::var("CLOTHEVO_FRAMES").ok().and_then(|s| s.parse().ok()).unwrap_or(120),
            requested: false,
        });
        let kind = std::env::var("CLOTHEVO_SCENE")
            .ok()
            .and_then(|s| s.parse::<usize>().ok())
            .and_then(|i| SceneKind::ALL.get(i.wrapping_sub(1)).copied())
            .unwrap_or(SceneKind::HangingCloth);

        let params = SimParams::default();
        let sim = build_scene(kind, DEFAULT_RES, params);
        let mut app = Self {
            seam_pairs: sim.seam_pairs(),
            sim,
            kind,
            res: DEFAULT_RES,
            preset: FabricPreset::Cotton,
            playing: true,
            accumulator: 0.0,
            camera: OrbitCamera::default(),
            style: ViewStyle::default(),
            normals: Vec::new(),
            render_state,
            diverged: false,
            last_step_ms: 0.0,
            capture,
            workspace: WorkspaceMode::Simulation,
            pattern_project: PatternProject::new("Novo projeto"),
            draft_points: Vec::new(),
            pattern_spacing: 0.08,
            pattern_polygon_sides: 6,
            pattern_show_mesh: true,
            selected_piece: 0,
            seam_a_piece: 0,
            seam_b_piece: 1,
            seam_a_segment: 0,
            seam_b_segment: 0,
            seam_a_reversed: false,
            seam_b_reversed: false,
            seam_ease_m: 0.0,
            pattern_message: String::new(),
            using_pattern_sim: false,
            pattern_tool: PatternTool::Draw,
            pattern_zoom: PATTERN_PIXELS_PER_METER,
            pattern_pan: egui::Vec2::ZERO,
            pattern_snap: false,
            selected_target: None,
            dragging_target: None,
            dragging_before: None,
            drag_last_world: None,
            pattern_undo: Vec::new(),
            pattern_redo: Vec::new(),
            pattern_saved_path: None,
            pattern_dirty: false,
            pattern_fit_pending: false,
        };
        app.upload_scene();
        app
    }

    /// Reconstrói a cena preservando os parâmetros atuais.
    fn load_scene(&mut self, kind: SceneKind, res: u32) {
        let params = self.sim.params.clone();
        self.kind = kind;
        self.using_pattern_sim = false;
        self.res = res;
        self.sim = build_scene(kind, res, params);
        self.seam_pairs = self.sim.seam_pairs();
        self.diverged = false;
        self.accumulator = 0.0;
        self.upload_scene();
    }

    /// Envia topologia do tecido e colisores para a GPU.
    fn upload_scene(&mut self) {
        let Some(rs) = &self.render_state else {
            return;
        };
        let edges = self.sim.mesh_edges();
        let statics = statics_from(&self.sim.colliders);
        let mut renderer = rs.renderer.write();
        if let Some(r) = renderer.callback_resources.get_mut::<SceneRenderer>() {
            r.set_statics(&rs.device, &statics);
            r.set_cloth_topology(
                &rs.device,
                &ClothTopology {
                    vertex_count: self.sim.particle_count(),
                    triangles: &self.sim.triangles,
                    wire_edges: &edges,
                    seam_pairs: &self.seam_pairs,
                },
            );
        }
    }

    fn advance(&mut self, frame_dt: f32) {
        if !self.playing {
            return;
        }
        // Passo fixo; se a simulação não acompanha o tempo real, ela simplesmente desacelera.
        self.accumulator = (self.accumulator + frame_dt).min(2.0 * FIXED_DT);
        while self.accumulator >= FIXED_DT {
            self.sim.step(FIXED_DT);
            self.accumulator -= FIXED_DT;
        }
        if !self.sim.is_finite() {
            self.playing = false;
            self.diverged = true;
        }
    }

    fn seam_gap_max(&self) -> f32 {
        self.seam_pairs
            .iter()
            .map(|&(a, b)| (self.sim.pos[a as usize] - self.sim.pos[b as usize]).length())
            .fold(0.0, f32::max)
    }

    fn controls(&mut self, ui: &mut egui::Ui) {
        let mut new_scene: Option<(SceneKind, u32)> = None;

        ui.heading("ClothEvo");
        ui.horizontal(|ui| {
            if ui.selectable_label(self.workspace == WorkspaceMode::Simulation, "Simulação 3D").clicked() {
                self.workspace = WorkspaceMode::Simulation;
            }
            if ui.selectable_label(self.workspace == WorkspaceMode::Pattern, "Moldes 2D").clicked() {
                self.workspace = WorkspaceMode::Pattern;
            }
        });
        ui.separator();
        if self.workspace == WorkspaceMode::Pattern {
            self.pattern_controls(ui);
            return;
        }
        if self.using_pattern_sim {
            ui.strong("Cena atual: moldes personalizados");
            ui.label("A malha 2D foi convertida para o solver XPBD.");
            if ui.button("Restaurar cena de teste").clicked() {
                self.load_scene(self.kind, self.res);
            }
            ui.separator();
        } else {
            ui.label("Simulação XPBD · cenas de validação");
            ui.separator();
        }

        // ----- Cena
        ui.strong("Cena");
        egui::ComboBox::from_id_salt("scene")
            .selected_text(self.kind.name())
            .width(240.0)
            .show_ui(ui, |ui| {
                for k in SceneKind::ALL {
                    if ui.selectable_label(self.kind == k, k.name()).clicked() && self.kind != k {
                        new_scene = Some((k, self.res));
                    }
                }
            });
        let mut res = self.res;
        let slider = ui.add(egui::Slider::new(&mut res, 16..=141).text("células/m"));
        if slider.drag_stopped() || (slider.changed() && !slider.dragged()) {
            new_scene = Some((self.kind, res));
        } else if slider.changed() {
            self.res = res; // só atualiza o valor; a malha é refeita ao soltar
        }
        ui.horizontal(|ui| {
            let label = if self.playing { "⏸ Pausar" } else { "▶ Continuar" };
            if ui.button(label).clicked() {
                self.playing = !self.playing;
                self.diverged = false;
            }
            if ui.button("⏭ Passo").clicked() {
                self.sim.step(FIXED_DT);
            }
            if ui.button("↺ Reiniciar").clicked() {
                self.sim.reset();
                self.diverged = false;
                self.accumulator = 0.0;
            }
        });
        if self.diverged {
            ui.colored_label(egui::Color32::LIGHT_RED, "A simulação divergiu (NaN). Reinicie e aumente os substeps.");
        }

        // ----- Tecido
        ui.separator();
        ui.strong("Tecido");
        egui::ComboBox::from_id_salt("preset")
            .selected_text(self.preset.name())
            .width(240.0)
            .show_ui(ui, |ui| {
                for p in FabricPreset::ALL {
                    if ui.selectable_label(self.preset == p, p.name()).clicked() {
                        self.preset = p;
                        p.apply(&mut self.sim.params);
                    }
                }
            });
        let p = &mut self.sim.params;
        ui.add(egui::Slider::new(&mut p.density, 0.02..=1.0).logarithmic(true).text("densidade kg/m²"));
        ui.add(egui::Slider::new(&mut p.stretch_warp, 50.0..=100_000.0).logarithmic(true).text("stretch urdume N/m"));
        ui.add(egui::Slider::new(&mut p.stretch_weft, 50.0..=100_000.0).logarithmic(true).text("stretch trama N/m"));
        ui.add(egui::Slider::new(&mut p.shear, 5.0..=20_000.0).logarithmic(true).text("cisalhamento N/m"));
        ui.add(egui::Slider::new(&mut p.bend, 1e-7..=1e-2).logarithmic(true).text("flexão N·m"));
        ui.add(egui::Slider::new(&mut p.seam, 100.0..=100_000.0).logarithmic(true).text("costura N/m"));
        ui.add(egui::Slider::new(&mut p.damping, 0.0..=5.0).text("amortecimento 1/s"));
        ui.add(egui::Slider::new(&mut p.friction, 0.0..=1.5).text("atrito"));
        ui.add(egui::Slider::new(&mut p.thickness, 0.0..=0.03).text("espessura m"));

        // ----- Solver
        ui.separator();
        ui.strong("Solver");
        ui.add(egui::Slider::new(&mut p.substeps, 1..=60).text("substeps"));
        ui.add(egui::Slider::new(&mut p.iterations, 1..=8).text("iterações"));
        ui.checkbox(&mut p.parallel, "paralelo (rayon)");

        // ----- Visualização
        ui.separator();
        ui.strong("Visualização");
        ui.checkbox(&mut self.style.wireframe, "malha (wireframe)");
        ui.checkbox(&mut self.style.seams, "costuras");

        // ----- Métricas
        ui.separator();
        ui.strong("Métricas");
        let sim = &self.sim;
        ui.label(format!("partículas: {}", sim.particle_count()));
        ui.label(format!(
            "restrições: {} dist. · {} flexão",
            sim.distance_constraint_count(),
            sim.bend_constraint_count()
        ));
        let budget = 1000.0 / 60.0;
        let ms = self.last_step_ms;
        let color = if ms <= budget { egui::Color32::LIGHT_GREEN } else { egui::Color32::YELLOW };
        ui.colored_label(color, format!("passo: {ms:.2} ms (orçamento {budget:.1} ms a 60 Hz)"));
        ui.label(format!("deformação máx.: {:.1} %", sim.max_strain() * 100.0));
        ui.label(format!("energia cinética: {:.4} J", sim.kinetic_energy()));
        if !self.seam_pairs.is_empty() {
            ui.label(format!("abertura máx. das costuras: {:.1} mm", self.seam_gap_max() * 1000.0));
        }

        ui.separator();
        ui.weak("Arraste: orbitar · botão direito/meio: mover · roda: zoom");

        if let Some((k, r)) = new_scene {
            self.load_scene(k, r);
        }
    }

    fn editor_snapshot(&self) -> PatternEditorSnapshot {
        PatternEditorSnapshot {
            project: self.pattern_project.clone(),
            draft_points: self.draft_points.clone(),
            selected_piece: self.selected_piece,
            selected_target: self.selected_target,
        }
    }

    fn remember_pattern_change(&mut self, before: PatternEditorSnapshot) {
        if before == self.editor_snapshot() { return; }
        self.pattern_undo.push(before);
        if self.pattern_undo.len() > 100 {
            self.pattern_undo.remove(0);
        }
        self.pattern_redo.clear();
        self.pattern_dirty = true;
    }

    fn restore_editor_snapshot(&mut self, snapshot: PatternEditorSnapshot) {
        self.pattern_project = snapshot.project;
        self.draft_points = snapshot.draft_points;
        self.selected_piece = snapshot.selected_piece.min(
            self.pattern_project.pieces.len().saturating_sub(1)
        );
        self.selected_target = snapshot.selected_target.filter(|target| {
            target.piece() < self.pattern_project.pieces.len()
        });
        self.dragging_target = None;
        self.dragging_before = None;
        self.drag_last_world = None;
        self.pattern_dirty = true;
    }

    fn undo_pattern_edit(&mut self) {
        let Some(previous) = self.pattern_undo.pop() else {
            self.pattern_message = "Nada para desfazer.".into();
            return;
        };
        let current = self.editor_snapshot();
        self.pattern_redo.push(current);
        self.restore_editor_snapshot(previous);
        self.pattern_message = "Última edição desfeita.".into();
    }

    fn redo_pattern_edit(&mut self) {
        let Some(next) = self.pattern_redo.pop() else {
            self.pattern_message = "Nada para refazer.".into();
            return;
        };
        let current = self.editor_snapshot();
        self.pattern_undo.push(current);
        self.restore_editor_snapshot(next);
        self.pattern_message = "Edição refeita.".into();
    }

    fn confirm_discard_unsaved_changes(&self) -> bool {
        if !self.pattern_dirty { return true; }
        rfd::MessageDialog::new()
            .set_level(rfd::MessageLevel::Warning)
            .set_title("Alterações não salvas")
            .set_description("O projeto atual tem alterações não salvas. Descartar essas alterações?")
            .set_buttons(rfd::MessageButtons::YesNo)
            .show() == rfd::MessageDialogResult::Yes
    }

    fn new_pattern_project(&mut self) {
        if !self.confirm_discard_unsaved_changes() { return; }
        self.pattern_project = PatternProject::new("Novo projeto");
        self.draft_points.clear();
        self.selected_piece = 0;
        self.selected_target = None;
        self.pattern_saved_path = None;
        self.pattern_undo.clear();
        self.pattern_redo.clear();
        self.dragging_target = None;
        self.dragging_before = None;
        self.drag_last_world = None;
        self.pattern_dirty = false;
        self.pattern_message = "Novo projeto criado.".into();
    }

    fn open_pattern_project_dialog(&mut self) {
        let mut dialog = rfd::FileDialog::new()
            .add_filter("Projeto ClothEvo / JSON", &["clothevo", "json"]);
        if let Some(parent) = self.pattern_saved_path.as_deref().and_then(std::path::Path::parent) {
            dialog = dialog.set_directory(parent);
        }
        let Some(path) = dialog.pick_file() else { return; };
        let loaded = std::fs::read_to_string(&path)
            .map_err(|e| format!("Falha ao ler {}: {e}", path.display()))
            .and_then(|json| PatternProject::from_json(&json));
        match loaded {
            Ok(project) => {
                if !self.confirm_discard_unsaved_changes() { return; }
                self.pattern_project = project;
                self.draft_points.clear();
                self.selected_piece = 0;
                self.selected_target = None;
                self.pattern_saved_path = Some(path.clone());
                self.pattern_undo.clear();
                self.pattern_redo.clear();
                self.dragging_target = None;
                self.dragging_before = None;
                self.drag_last_world = None;
                self.pattern_dirty = false;
                self.pattern_message = format!("Projeto carregado: {}", path.display());
            }
            Err(error) => self.pattern_message = error,
        }
    }

    fn save_pattern_project(&mut self) {
        if let Some(path) = self.pattern_saved_path.clone() {
            self.write_pattern_project(path);
        } else {
            self.save_pattern_project_as();
        }
    }

    fn save_pattern_project_as(&mut self) {
        let mut dialog = rfd::FileDialog::new()
            .add_filter("Projeto ClothEvo", &["clothevo", "json"])
            .set_file_name("projeto.clothevo");
        if let Some(parent) = self.pattern_saved_path.as_deref().and_then(std::path::Path::parent) {
            dialog = dialog.set_directory(parent);
        }
        if let Some(path) = dialog.save_file() {
            self.write_pattern_project(path);
        }
    }

    fn write_pattern_project(&mut self, mut path: std::path::PathBuf) {
        if path.extension().is_none() {
            path.set_extension("clothevo");
        }
        let result = self.pattern_project.to_json()
            .and_then(|json| std::fs::write(&path, json)
                .map_err(|e| format!("Falha ao salvar {}: {e}", path.display())));
        match result {
            Ok(()) => {
                self.pattern_saved_path = Some(path.clone());
                self.pattern_dirty = false;
                self.pattern_message = format!("Projeto salvo: {}", path.display());
            }
            Err(error) => self.pattern_message = error,
        }
    }

    fn fit_pattern_view(&mut self, rect: egui::Rect) {
        let mut min_x = f64::INFINITY;
        let mut min_y = f64::INFINITY;
        let mut max_x = f64::NEG_INFINITY;
        let mut max_y = f64::NEG_INFINITY;
        for piece in &self.pattern_project.pieces {
            let points = sample_curve(&piece.outline, 16);
            for p in points {
                min_x = min_x.min(p.x);
                min_y = min_y.min(p.y);
                max_x = max_x.max(p.x);
                max_y = max_y.max(p.y);
            }
        }
        if !min_x.is_finite() {
            self.pattern_zoom = PATTERN_PIXELS_PER_METER;
            self.pattern_pan = egui::Vec2::ZERO;
            return;
        }
        let width = (max_x - min_x).max(0.1) as f32;
        let height = (max_y - min_y).max(0.1) as f32;
        let available_width = (rect.width() - 120.0).max(120.0);
        let available_height = (rect.height() - 120.0).max(120.0);
        self.pattern_zoom = (available_width / width).min(available_height / height).clamp(40.0, 1800.0);
        let mid_x = ((min_x + max_x) * 0.5) as f32;
        let mid_y = ((min_y + max_y) * 0.5) as f32;
        self.pattern_pan = egui::vec2(-mid_x * self.pattern_zoom, mid_y * self.pattern_zoom);
    }

    fn duplicate_selected_piece(&mut self) {
        if self.pattern_project.pieces.is_empty() { return; }
        let index = self.selected_piece.min(self.pattern_project.pieces.len() - 1);
        let before = self.editor_snapshot();
        let Some(mut duplicate) = self.pattern_project.pieces.get(index).cloned() else { return; };
        let new_id = self.pattern_project.pieces.iter().map(|p| p.id).max().unwrap_or(0) + 1;
        duplicate.id = new_id;
        duplicate.name = format!("{} (cópia)", duplicate.name);
        translate_piece(&mut duplicate, 0.20, 0.20);
        self.pattern_project.pieces.push(duplicate);
        self.selected_piece = self.pattern_project.pieces.len() - 1;
        self.selected_target = Some(PatternEditTarget::Piece { piece: self.selected_piece });
        self.pattern_message = "Peça duplicada e deslocada em 20 cm; as costuras não foram duplicadas.".into();
        self.remember_pattern_change(before);
    }

    fn pattern_controls(&mut self, ui: &mut egui::Ui) {
        ui.heading("Pattern Studio 0.4 · Geometria vetorial");
        ui.label("Editor 2D · medidas internas em metros");
        ui.horizontal(|ui| {
            ui.selectable_value(&mut self.pattern_tool, PatternTool::Draw, "Desenhar");
            ui.selectable_value(&mut self.pattern_tool, PatternTool::Select, "Selecionar / editar");
        });
        ui.horizontal(|ui| {
            if ui.add_enabled(!self.pattern_undo.is_empty(), egui::Button::new("↶ Desfazer")).clicked() {
                self.undo_pattern_edit();
            }
            if ui.add_enabled(!self.pattern_redo.is_empty(), egui::Button::new("↷ Refazer")).clicked() {
                self.redo_pattern_edit();
            }
        });
        ui.horizontal(|ui| {
            if ui.button("Novo projeto").clicked() { self.new_pattern_project(); }
            if ui.button("Abrir…").clicked() { self.open_pattern_project_dialog(); }
        });
        ui.horizontal(|ui| {
            if ui.button("Salvar").clicked() { self.save_pattern_project(); }
            if ui.button("Salvar como…").clicked() { self.save_pattern_project_as(); }
        });
        if self.pattern_dirty {
            ui.colored_label(egui::Color32::from_rgb(240, 180, 75), "● Modificações não salvas");
        } else if let Some(path) = &self.pattern_saved_path {
            ui.weak(format!("Arquivo: {}", path.file_name().unwrap_or_default().to_string_lossy()));
        }
        ui.horizontal(|ui| {
            if ui.button("Desfazer ponto").clicked() {
                let before = self.editor_snapshot();
                self.draft_points.pop();
                self.remember_pattern_change(before);
            }
            if ui.button("Limpar rascunho").clicked() {
                let before = self.editor_snapshot();
                self.draft_points.clear();
                self.remember_pattern_change(before);
            }
        });
        ui.checkbox(&mut self.pattern_snap, "Ajustar à grade (1 cm)");
        ui.horizontal(|ui| {
            if ui.button("Enquadrar todos os moldes").clicked() {
                self.pattern_fit_pending = true;
            }
            if ui.button("Vista padrão").clicked() {
                self.pattern_zoom = PATTERN_PIXELS_PER_METER;
                self.pattern_pan = egui::Vec2::ZERO;
            }
        });
        ui.separator();
        ui.strong("Geometria vetorial");
        ui.horizontal(|ui| {
            if ui.button("Adicionar retângulo").clicked() {
                self.create_rectangle_primitive();
            }
            if ui.button("Adicionar polígono regular").clicked() {
                self.create_regular_polygon_primitive();
            }
        });
        ui.add(egui::Slider::new(&mut self.pattern_polygon_sides, 3..=12).text("lados do polígono"));
        ui.horizontal(|ui| {
            if ui.button("Fechar peça a partir dos pontos").clicked() {
                self.create_piece_from_draft();
            }
            if ui.button("Adicionar linha interna").clicked() {
                self.create_internal_line_from_draft();
            }
            if ui.button("Cut & Sew").clicked() {
                self.cut_and_sew_from_draft();
            }
        });
        ui.weak("Cut & Sew: desenhe um caminho dentro do molde, começando e terminando no contorno.");
        ui.horizontal(|ui| {
            if ui.button("Criar dobra (vale)").clicked() {
                self.create_fold_line_from_draft();
            }
            if ui.button("Criar prega").clicked() {
                self.create_pleat_from_draft();
            }
            if ui.button("Definir linha de fio").clicked() {
                self.create_grainline_from_draft();
            }
        });
        ui.label(format!("Pontos no rascunho: {}", self.draft_points.len()));
        if let Some(target) = self.selected_target {
            ui.label(format!("Selecionado: {}", match target {
                PatternEditTarget::Piece { .. } => "peça inteira".to_string(),
                PatternEditTarget::Anchor { index, .. } => format!("vértice {}", index + 1),
                PatternEditTarget::Control1 { segment, .. } => format!("alça 1 · segmento {}", segment + 1),
                PatternEditTarget::Control2 { segment, .. } => format!("alça 2 · segmento {}", segment + 1),
                PatternEditTarget::Edge { segment, .. } => format!("borda · segmento {}", segment + 1),
            }));
            ui.horizontal(|ui| {
                if matches!(target, PatternEditTarget::Edge { .. })
                    && ui.button("Inserir vértice").clicked() {
                    self.insert_selected_vertex();
                }
                if matches!(target, PatternEditTarget::Edge { .. })
                    && ui.button("Converter em Bézier").clicked() {
                    self.convert_selected_edge_to_bezier();
                }
                if matches!(target, PatternEditTarget::Anchor { .. })
                    && ui.button("Excluir vértice").clicked() {
                    self.delete_selected_vertex();
                }
            });
        }
        if !self.pattern_project.pieces.is_empty() {
            ui.horizontal(|ui| {
                if ui.button("Duplicar peça selecionada").clicked() {
                    self.duplicate_selected_piece();
                }
                if ui.button("Excluir peça").clicked() {
                    self.delete_selected_piece();
                }
            });
        }

        ui.separator();
        ui.strong("Malha 2D");
        ui.add(egui::Slider::new(&mut self.pattern_spacing, 0.025..=0.20).text("espaçamento (m)"));
        ui.checkbox(&mut self.pattern_show_mesh, "Mostrar triangulação");
        ui.label("Os projetos são armazenados em JSON no formato nativo .clothevo.");

        ui.separator();
        ui.strong(format!("Peças ({})", self.pattern_project.pieces.len()));
        for piece in &self.pattern_project.pieces {
            ui.label(format!("{} · {} segmentos", piece.name, piece.outline.segment_count()));
        }
        if !self.pattern_project.pieces.is_empty() {
            let options: Vec<(u64, String)> = self.pattern_project.pieces.iter()
                .map(|piece| (piece.id, piece.name.clone())).collect();
            self.selected_piece = self.selected_piece.min(options.len() - 1);
            egui::ComboBox::from_id_salt("allowance_piece")
                .selected_text(format!("Margem: {}", options[self.selected_piece].1))
                .show_ui(ui, |ui| {
                    for (index, (_, name)) in options.iter().enumerate() {
                        ui.selectable_value(&mut self.selected_piece, index, name);
                    }
                });
            let allowance_before = self.editor_snapshot();
            let allowance_response = ui.add(egui::Slider::new(
                &mut self.pattern_project.pieces[self.selected_piece].seam_allowance_meters,
                0.0..=0.05,
            ).text("margem (m)"));
            if allowance_response.changed() {
                self.remember_pattern_change(allowance_before);
            }
        }

        if self.pattern_project.pieces.len() >= 2 {
            ui.separator();
            ui.strong("Costura entre bordas");
            let options: Vec<(u64, String)> = self.pattern_project.pieces.iter()
                .map(|piece| (piece.id, piece.name.clone())).collect();
            self.seam_a_piece = self.seam_a_piece.min(options.len() - 1);
            self.seam_b_piece = self.seam_b_piece.min(options.len() - 1);

            egui::ComboBox::from_id_salt("seam_piece_a")
                .selected_text(format!("A: {}", options[self.seam_a_piece].1))
                .show_ui(ui, |ui| {
                    for (index, (_, name)) in options.iter().enumerate() {
                        ui.selectable_value(&mut self.seam_a_piece, index, name);
                    }
                });
            egui::ComboBox::from_id_salt("seam_piece_b")
                .selected_text(format!("B: {}", options[self.seam_b_piece].1))
                .show_ui(ui, |ui| {
                    for (index, (_, name)) in options.iter().enumerate() {
                        ui.selectable_value(&mut self.seam_b_piece, index, name);
                    }
                });

            let a_count = self.pattern_project.pieces[self.seam_a_piece].outline.segment_count().max(1);
            let b_count = self.pattern_project.pieces[self.seam_b_piece].outline.segment_count().max(1);
            self.seam_a_segment = self.seam_a_segment.min(a_count - 1);
            self.seam_b_segment = self.seam_b_segment.min(b_count - 1);
            egui::ComboBox::from_id_salt("seam_edge_a")
                .selected_text(format!("Borda A: {}", self.seam_a_segment + 1))
                .show_ui(ui, |ui| {
                    for index in 0..a_count {
                        ui.selectable_value(&mut self.seam_a_segment, index, format!("Segmento {}", index + 1));
                    }
                });
            egui::ComboBox::from_id_salt("seam_edge_b")
                .selected_text(format!("Borda B: {}", self.seam_b_segment + 1))
                .show_ui(ui, |ui| {
                    for index in 0..b_count {
                        ui.selectable_value(&mut self.seam_b_segment, index, format!("Segmento {}", index + 1));
                    }
                });
            ui.horizontal(|ui| {
                if ui.button("Usar borda selecionada como A").clicked() {
                    if let Some(PatternEditTarget::Edge { piece, segment, .. }) = self.selected_target {
                        self.seam_a_piece = piece.min(options.len() - 1);
                        self.seam_a_segment = segment;
                    } else {
                        self.pattern_message = "Selecione uma borda no canvas antes de defini-la como A.".into();
                    }
                }
                if ui.button("Usar borda selecionada como B").clicked() {
                    if let Some(PatternEditTarget::Edge { piece, segment, .. }) = self.selected_target {
                        self.seam_b_piece = piece.min(options.len() - 1);
                        self.seam_b_segment = segment;
                    } else {
                        self.pattern_message = "Selecione uma borda no canvas antes de defini-la como B.".into();
                    }
                }
            });
            ui.label(format!("A: {} · borda {}", options[self.seam_a_piece].1, self.seam_a_segment + 1));
            ui.label(format!("B: {} · borda {}", options[self.seam_b_piece].1, self.seam_b_segment + 1));
            ui.add(egui::DragValue::new(&mut self.seam_ease_m).speed(0.001).prefix("Folga (m) "));
            ui.checkbox(&mut self.seam_a_reversed, "Inverter orientação da borda A");
            ui.checkbox(&mut self.seam_b_reversed, "Inverter orientação da borda B");
            ui.weak("A folga é salva no projeto; o solver ainda não a aplica como ease diferencial.");
            if ui.button("Adicionar costura").clicked() { self.add_seam_from_controls(); }
            ui.label(format!("Costuras no projeto: {}", self.pattern_project.seams.len()));
        } else {
            ui.weak("Crie ao menos duas peças para configurar uma costura.");
        }

        ui.separator();
        if ui.button("Simular moldes na viewport 3D").clicked() {
            match self.start_pattern_simulation() {
                Ok(()) => {
                    self.pattern_message = if self.pattern_project.seams.iter().any(|seam| seam.ease_meters.abs() > 1.0e-6) {
                        "Moldes simulando em 3D. A folga foi preservada no projeto, mas ainda não é aplicada fisicamente.".into()
                    } else {
                        "Moldes enviados à simulação 3D.".into()
                    };
                },
                Err(e) => self.pattern_message = e,
            }
        }
        if !self.pattern_message.is_empty() {
            ui.separator();
            ui.label(&self.pattern_message);
        }
        ui.separator();
        ui.weak("Canvas: clique para adicionar pontos; as peças fechadas aparecem com contorno azul.");
    }

    fn hit_test_pattern(&self, pointer: egui::Pos2, center: egui::Pos2) -> Option<PatternEditTarget> {
        let origin = center + self.pattern_pan;
        let scale = self.pattern_zoom;
        let to_screen = |p: PatternPoint2| {
            egui::pos2(origin.x + p.x as f32 * scale, origin.y - p.y as f32 * scale)
        };

        let mut best_point: Option<(f32, PatternEditTarget)> = None;
        for (piece_index, piece) in self.pattern_project.pieces.iter().enumerate() {
            for (index, point) in outline_anchors(&piece.outline).iter().enumerate() {
                let d = to_screen(*point).distance(pointer);
                if d <= 12.0 && best_point.map(|best| d < best.0).unwrap_or(true) {
                    best_point = Some((d, PatternEditTarget::Anchor { piece: piece_index, index }));
                }
            }
            for (segment_index, segment) in piece.outline.segments.iter().enumerate() {
                if let CurveSegment::CubicBezier { control1, control2, .. } = segment {
                    for (point, target) in [
                        (*control1, PatternEditTarget::Control1 { piece: piece_index, segment: segment_index }),
                        (*control2, PatternEditTarget::Control2 { piece: piece_index, segment: segment_index }),
                    ] {
                        let d = to_screen(point).distance(pointer);
                        if d <= 12.0 && best_point.map(|best| d < best.0).unwrap_or(true) {
                            best_point = Some((d, target));
                        }
                    }
                }
            }
        }
        if let Some((_, target)) = best_point { return Some(target); }

        // Hit-test das bordas usando amostras de Bézier; o parâmetro local permite subdividir
        // o segmento selecionado sem alterar a forma original da curva.
        let mut best_edge: Option<(f32, PatternEditTarget)> = None;
        for (piece_index, piece) in self.pattern_project.pieces.iter().enumerate() {
            let sampled = sample_curve_with_ranges(&piece.outline, 16);
            let count = sampled.points.len();
            for (segment_index, &(start, end)) in sampled.segment_vertex_ranges.iter().enumerate() {
                if count < 2 { continue; }
                let sample_count = if end >= start { end - start } else { count - start };
                if sample_count == 0 { continue; }
                for step in 0..sample_count {
                    let ia = (start + step) % count;
                    let ib = (ia + 1) % count;
                    let a = to_screen(sampled.points[ia]);
                    let b = to_screen(sampled.points[ib]);
                    let (distance, segment_t) = closest_segment_parameter(pointer, a, b);
                    if distance <= 8.0 && best_edge.map(|best| distance < best.0).unwrap_or(true) {
                        let t = ((step as f64 + segment_t as f64) / sample_count as f64).clamp(0.02, 0.98);
                        best_edge = Some((distance, PatternEditTarget::Edge {
                            piece: piece_index,
                            segment: segment_index,
                            t,
                        }));
                    }
                }
            }
        }
        if let Some((_, target)) = best_edge { return Some(target); }

        let world = PatternPoint2::new(
            (pointer.x - origin.x) as f64 / scale as f64,
            (origin.y - pointer.y) as f64 / scale as f64,
        );
        for (piece_index, piece) in self.pattern_project.pieces.iter().enumerate().rev() {
            let polygon = sample_curve(&piece.outline, 16);
            if point_in_polygon(world, &polygon) {
                return Some(PatternEditTarget::Piece { piece: piece_index });
            }
        }
        None
    }

    fn move_pattern_target(&mut self, target: PatternEditTarget, point: PatternPoint2) {
        let Some(piece) = self.pattern_project.pieces.get_mut(target.piece()) else { return; };
        match target {
            PatternEditTarget::Anchor { index, .. } => {
                set_curve_anchor(&mut piece.outline, index, point);
            }
            PatternEditTarget::Control1 { segment, .. } => {
                if let Some(CurveSegment::CubicBezier { control1, .. }) = piece.outline.segments.get_mut(segment) {
                    *control1 = point;
                }
            }
            PatternEditTarget::Control2 { segment, .. } => {
                if let Some(CurveSegment::CubicBezier { control2, .. }) = piece.outline.segments.get_mut(segment) {
                    *control2 = point;
                }
            }
            PatternEditTarget::Piece { .. } | PatternEditTarget::Edge { .. } => {}
        }
    }

    fn insert_selected_vertex(&mut self) {
        let Some(PatternEditTarget::Edge { piece, segment, t }) = self.selected_target else {
            self.pattern_message = "Selecione uma borda no modo Editar antes de inserir um vértice.".into();
            return;
        };
        let Some(piece_data) = self.pattern_project.pieces.get(piece) else { return; };
        let id = piece_data.id;
        let before = self.editor_snapshot();
        match split_project_segment(&mut self.pattern_project, id, segment, t) {
            Ok(new_anchor) => {
                self.selected_target = Some(PatternEditTarget::Anchor { piece, index: new_anchor });
                self.pattern_message = "Vértice inserido; a curva mantém sua forma original.".into();
                self.remember_pattern_change(before);
            }
            Err(e) => self.pattern_message = e,
        }
    }

    fn convert_selected_edge_to_bezier(&mut self) {
        let Some(PatternEditTarget::Edge { piece, segment, .. }) = self.selected_target else {
            self.pattern_message = "Selecione um segmento para convertê-lo em Bézier.".into();
            return;
        };
        let Some(piece_data) = self.pattern_project.pieces.get(piece) else { return; };
        let id = piece_data.id;
        let before = self.editor_snapshot();
        match convert_project_segment_to_bezier(&mut self.pattern_project, id, segment) {
            Ok(()) => {
                self.selected_target = Some(PatternEditTarget::Control1 { piece, segment });
                self.pattern_message = "Segmento convertido. Arraste as alças roxas para editar a curva.".into();
                self.remember_pattern_change(before);
            }
            Err(e) => self.pattern_message = e,
        }
    }

    fn delete_selected_vertex(&mut self) {
        let Some(PatternEditTarget::Anchor { piece, index }) = self.selected_target else { return; };
        let Some(piece_data) = self.pattern_project.pieces.get(piece) else { return; };
        let id = piece_data.id;
        let before = self.editor_snapshot();
        match delete_project_anchor(&mut self.pattern_project, id, index) {
            Ok(()) => {
                self.selected_target = None;
                self.dragging_target = None;
                self.pattern_message = "Vértice removido.".into();
                self.remember_pattern_change(before);
            }
            Err(e) => self.pattern_message = e,
        }
    }

    fn delete_selected_piece(&mut self) {
        if self.pattern_project.pieces.is_empty() { return; }
        let before = self.editor_snapshot();
        self.selected_piece = self.selected_piece.min(self.pattern_project.pieces.len() - 1);
        let id = self.pattern_project.pieces[self.selected_piece].id;
        self.pattern_project.pieces.remove(self.selected_piece);
        self.pattern_project.seams.retain(|seam| seam.a.piece_id != id && seam.b.piece_id != id);
        self.selected_piece = self.selected_piece.min(self.pattern_project.pieces.len().saturating_sub(1));
        self.selected_target = None;
        self.dragging_target = None;
        self.pattern_message = "Peça excluída; costuras dependentes também foram removidas.".into();
        self.remember_pattern_change(before);
    }

    fn primitive_origin(&self) -> (f64, f64) {
        if self.pattern_project.pieces.is_empty() {
            return (0.0, 0.0);
        }
        let mut max_x = f64::NEG_INFINITY;
        let mut min_y = f64::INFINITY;
        for piece in &self.pattern_project.pieces {
            for point in sample_curve(&piece.outline, 24) {
                max_x = max_x.max(point.x);
                min_y = min_y.min(point.y);
            }
        }
        if !max_x.is_finite() || !min_y.is_finite() { (0.0, 0.0) }
        else { (max_x + 0.10, min_y) }
    }

    fn create_primitive_piece(&mut self, outline: Curve2D, label: &str) {
        let points = sample_curve(&outline, 1);
        if points.len() < 3 {
            self.pattern_message = "A primitiva gerada não formou um contorno fechado válido.".into();
            return;
        }
        let before = self.editor_snapshot();
        let next_id = self.pattern_project.pieces.iter()
            .map(|piece| piece.id).max().unwrap_or(0) + 1;
        let name = format!("{label} {next_id}");
        match PatternPiece::from_polygon(next_id, name, &points) {
            Ok(mut piece) => {
                // Mantém a curva original em vez de converter os segmentos para uma polilinha.
                piece.outline = outline;
                self.pattern_project.pieces.push(piece);
                self.selected_piece = self.pattern_project.pieces.len() - 1;
                self.selected_target = None;
                self.pattern_tool = PatternTool::Select;
                self.pattern_fit_pending = true;
                self.pattern_message = format!("{label} criado como molde vetorial editável.");
                self.remember_pattern_change(before);
            }
            Err(error) => self.pattern_message = error,
        }
    }

    fn create_rectangle_primitive(&mut self) {
        let (x, y) = self.primitive_origin();
        match rectangle(
            PatternPoint2::new(x, y),
            PatternPoint2::new(x + 0.40, y + 0.60),
        ) {
            Ok(outline) => self.create_primitive_piece(outline, "Retângulo"),
            Err(error) => self.pattern_message = error.to_string(),
        }
    }

    fn create_regular_polygon_primitive(&mut self) {
        let (x, y) = self.primitive_origin();
        let sides = self.pattern_polygon_sides.clamp(3, 100);
        match regular_polygon(
            PatternPoint2::new(x + 0.20, y + 0.30),
            0.20,
            sides,
            -std::f64::consts::FRAC_PI_2,
        ) {
            Ok(outline) => self.create_primitive_piece(outline, "Polígono"),
            Err(error) => self.pattern_message = error.to_string(),
        }
    }

    fn drafted_internal_curve(&self) -> Result<(usize, Curve2D), String> {
        if self.draft_points.len() < 2 {
            return Err("Desenhe ao menos dois pontos para uma linha interna.".into());
        }
        if self.pattern_project.pieces.is_empty() {
            return Err("Crie ou selecione um molde antes de adicionar detalhes internos.".into());
        }
        let piece_index = self.selected_piece.min(self.pattern_project.pieces.len() - 1);
        let outline = sample_curve(&self.pattern_project.pieces[piece_index].outline, 24);
        let curve = Curve2D::polyline(&self.draft_points, false)?;
        // Evita gravar linhas que saiam do molde. Cada segmento é amostrado para detectar
        // trechos que atravessam o contorno, não apenas os seus pontos de controle.
        for pair in self.draft_points.windows(2) {
            for step in 0..=16 {
                let t = step as f64 / 16.0;
                let point = PatternPoint2::new(
                    pair[0].x + (pair[1].x - pair[0].x) * t,
                    pair[0].y + (pair[1].y - pair[0].y) * t,
                );
                if !point_in_polygon(point, &outline) {
                    return Err("A linha interna precisa permanecer dentro do contorno do molde.".into());
                }
            }
        }
        Ok((piece_index, curve))
    }

    fn create_internal_line_from_draft(&mut self) {
        let (piece_index, curve) = match self.drafted_internal_curve() {
            Ok(value) => value,
            Err(error) => { self.pattern_message = error; return; }
        };
        let before = self.editor_snapshot();
        self.pattern_project.pieces[piece_index].internal_lines.push(curve);
        self.draft_points.clear();
        self.pattern_tool = PatternTool::Select;
        self.pattern_message = "Linha interna adicionada ao molde.".into();
        self.remember_pattern_change(before);
    }

    fn create_fold_line_from_draft(&mut self) {
        let (piece_index, curve) = match self.drafted_internal_curve() {
            Ok(value) => value,
            Err(error) => { self.pattern_message = error; return; }
        };
        let before = self.editor_snapshot();
        self.pattern_project.pieces[piece_index].fold_lines.push(FoldLine {
            curve,
            kind: FoldKind::Valley,
            stiffness: 0.35,
        });
        self.draft_points.clear();
        self.pattern_tool = PatternTool::Select;
        self.pattern_message = "Linha de dobra (vale) adicionada; rigidez inicial 0,35.".into();
        self.remember_pattern_change(before);
    }

    fn create_pleat_from_draft(&mut self) {
        let (piece_index, axis) = match self.drafted_internal_curve() {
            Ok(value) => value,
            Err(error) => { self.pattern_message = error; return; }
        };
        let before = self.editor_snapshot();
        self.pattern_project.pieces[piece_index].pleats.push(Pleat {
            axis,
            depth_meters: 0.02,
            angle_radians: std::f64::consts::FRAC_PI_4,
        });
        self.draft_points.clear();
        self.pattern_tool = PatternTool::Select;
        self.pattern_message = "Prega criada com profundidade inicial de 2 cm e ângulo de 45°.".into();
        self.remember_pattern_change(before);
    }

    fn create_grainline_from_draft(&mut self) {
        if self.draft_points.len() != 2 {
            self.pattern_message = "A linha de fio exige exatamente dois pontos.".into();
            return;
        }
        let (piece_index, curve) = match self.drafted_internal_curve() {
            Ok(value) => value,
            Err(error) => { self.pattern_message = error; return; }
        };
        let start = curve.start;
        let Some(end) = curve.endpoint() else { return; };
        let before = self.editor_snapshot();
        self.pattern_project.pieces[piece_index].grain_line = Some(GrainLine { start, end });
        self.draft_points.clear();
        self.pattern_tool = PatternTool::Select;
        self.pattern_message = "Linha de fio atualizada.".into();
        self.remember_pattern_change(before);
    }

    fn cut_and_sew_from_draft(&mut self) {
        if self.draft_points.len() < 2 {
            self.pattern_message = "Para Cut & Sew, desenhe um caminho com pelo menos dois pontos.".into();
            return;
        }
        if self.pattern_project.pieces.is_empty() {
            self.pattern_message = "Crie um molde antes de executar Cut & Sew.".into();
            return;
        }
        let source_index = self.selected_piece.min(self.pattern_project.pieces.len() - 1);
        let source = self.pattern_project.pieces[source_index].clone();
        let cut_path = match Curve2D::polyline(&self.draft_points, false) {
            Ok(curve) => curve,
            Err(error) => { self.pattern_message = error; return; }
        };
        let cut_result = match cut_closed_outline(&source.outline, &cut_path, 0.035) {
            Ok(result) => result,
            Err(error) => {
                self.pattern_message = match error {
                    pattern_geom::GeometryError::SelfIntersection =>
                        "Corte inválido: o caminho se cruza. Use um caminho simples.".into(),
                    pattern_geom::GeometryError::InvalidParameter =>
                        "Corte inválido: mantenha o caminho dentro do molde e conecte as duas extremidades ao contorno (tolerância de 3,5 cm).".into(),
                    _ => format!("Não foi possível dividir o molde: {error}"),
                };
                return;
            }
        };

        let first_id = source.id;
        let second_id = self.pattern_project.pieces.iter()
            .map(|piece| piece.id).max().unwrap_or(0) + 1;
        let mut first = source.clone();
        let mut second = source.clone();
        first.name = format!("{} (A)", source.name);
        second.id = second_id;
        second.name = format!("{} (B)", source.name);
        first.outline = cut_result.first.clone();
        second.outline = cut_result.second.clone();

        let polygon_a = sample_curve(&first.outline, 24);
        let polygon_b = sample_curve(&second.outline, 24);
        let sewing_side = match split_piece_metadata(
            &source, &mut first, &mut second, &cut_result, &polygon_a, &polygon_b,
        ) {
            Ok(side) => side,
            Err(error) => {
                self.pattern_message = format!("Cut & Sew cancelado sem alterar o molde: {error}");
                return;
            }
        };

        // Remap existing seam references before replacing the source piece. If a referenced
        // interval spans both sides of the cut, refuse rather than silently corrupting it.
        let mut project = self.pattern_project.clone();
        for seam in &mut project.seams {
            for edge in [&mut seam.a, &mut seam.b] {
                if edge.piece_id != first_id { continue; }
                match edge.curve {
                    BoundaryCurve::Outline => {
                        if let Err(error) = remap_edge_after_cut(
                            edge, &cut_result.source_segment_fragments, first_id, second_id,
                        ) {
                            self.pattern_message = format!(
                                "Cut & Sew cancelado: {error} A costura original foi preservada."
                            );
                            return;
                        }
                    }
                    BoundaryCurve::SewingLine => {
                        let Some(side) = sewing_side else {
                            self.pattern_message = "Cut & Sew cancelado: uma costura existente referencia a linha de costura, mas ela não pôde ser atribuída a uma das peças.".into();
                            return;
                        };
                        edge.piece_id = if side == 0 { first_id } else { second_id };
                    }
                }
            }
        }

        project.pieces[source_index] = first;
        project.pieces.push(second);

        let next_seam_id = project.seams.iter()
            .map(|seam| seam.id).max().unwrap_or(0) + 1;
        let (a_start, a_end) = cut_result.first_cut_segment_range;
        let (b_start, b_end) = cut_result.second_cut_segment_range;
        project.seams.push(SeamDefinition {
            id: next_seam_id,
            name: format!("Cut & Sew {next_seam_id}"),
            a: EdgeRef {
                piece_id: first_id,
                curve: BoundaryCurve::Outline,
                segment_start: a_start as u32,
                segment_end: a_end as u32,
                reversed: false,
            },
            b: EdgeRef {
                piece_id: second_id,
                curve: BoundaryCurve::Outline,
                segment_start: b_start as u32,
                segment_end: b_end as u32,
                // Corresponding endpoints run in opposite directions around each outline.
                reversed: true,
            },
            ease_meters: 0.0,
        });

        if let Err(error) = project.validate() {
            self.pattern_message = format!("Cut & Sew cancelado pela validação do projeto: {error}");
            return;
        }

        let before = self.editor_snapshot();
        self.pattern_project = project;
        self.selected_piece = source_index;
        self.selected_target = Some(PatternEditTarget::Piece { piece: source_index });
        self.draft_points.clear();
        self.pattern_tool = PatternTool::Select;
        self.pattern_fit_pending = true;
        self.pattern_message = format!(
            "Cut & Sew concluído: '{}' foi dividido em duas peças, com a nova costura {}.",
            source.name, next_seam_id
        );
        self.remember_pattern_change(before);
    }

    fn create_piece_from_draft(&mut self) {
        if self.draft_points.len() < 3 {
            self.pattern_message = "Uma peça precisa de pelo menos 3 pontos.".into();
            return;
        }
        let before = self.editor_snapshot();
        let next_id = self.pattern_project.pieces.iter().map(|piece| piece.id).max().unwrap_or(0) + 1;
        let piece_name = format!("Peça {next_id}");
        match PatternPiece::from_polygon(next_id, piece_name, &self.draft_points) {
            Ok(piece) => {
                self.pattern_project.pieces.push(piece);
                self.selected_piece = self.pattern_project.pieces.len() - 1;
                self.selected_target = None;
                self.draft_points.clear();
                self.pattern_tool = PatternTool::Select;
                self.pattern_message = "Peça criada. Use Selecionar/editar para mover a peça, vértices e alças.".into();
                self.remember_pattern_change(before);
            }
            Err(e) => self.pattern_message = e,
        }
    }

    fn add_seam_from_controls(&mut self) {
        let count = self.pattern_project.pieces.len();
        if count < 2 { return; }
        let a_index = self.seam_a_piece.min(count - 1);
        let b_index = self.seam_b_piece.min(count - 1);
        let a_piece = &self.pattern_project.pieces[a_index];
        let b_piece = &self.pattern_project.pieces[b_index];
        let a_segment = self.seam_a_segment.min(a_piece.outline.segment_count().saturating_sub(1)) as u32;
        let b_segment = self.seam_b_segment.min(b_piece.outline.segment_count().saturating_sub(1)) as u32;
        let next_id = self.pattern_project.seams.iter().map(|seam| seam.id).max().unwrap_or(0) + 1;
        let mut edge_a = EdgeRef::single_segment(a_piece.id, a_segment);
        let mut edge_b = EdgeRef::single_segment(b_piece.id, b_segment);
        edge_a.reversed = self.seam_a_reversed;
        edge_b.reversed = self.seam_b_reversed;
        let seam = SeamDefinition {
            id: next_id,
            name: format!("Costura {next_id}"),
            a: edge_a,
            b: edge_b,
            ease_meters: self.seam_ease_m,
        };
        let before = self.editor_snapshot();
        self.pattern_project.seams.push(seam);
        if let Err(e) = self.pattern_project.validate() {
            self.pattern_project.seams.pop();
            self.pattern_message = e;
        } else {
            self.pattern_message = "Costura adicionada ao projeto.".into();
            self.remember_pattern_change(before);
        }
    }

    fn pattern_viewport(&mut self, ui: &mut egui::Ui) {
        let rect = ui.available_rect_before_wrap();
        if self.pattern_fit_pending {
            self.fit_pattern_view(rect);
            self.pattern_fit_pending = false;
        }
        let response = ui.allocate_rect(rect, egui::Sense::click_and_drag());
        let painter = ui.painter_at(rect);
        let center = rect.center();

        // O zoom é medido em pixels por metro; a translação fica em pixels da viewport.
        let origin = center + self.pattern_pan;
        let scale = self.pattern_zoom;
        let to_screen = |p: PatternPoint2| {
            egui::pos2(origin.x + p.x as f32 * scale, origin.y - p.y as f32 * scale)
        };
        let to_world = |p: egui::Pos2| {
            PatternPoint2::new(
                (p.x - origin.x) as f64 / scale as f64,
                (origin.y - p.y) as f64 / scale as f64,
            )
        };

        painter.rect_filled(rect, 0.0, egui::Color32::from_gray(23));
        let grid_step = (scale * 0.1).max(4.0);
        let first_x = origin.x + ((rect.left() - origin.x) / grid_step).ceil() * grid_step;
        let first_y = origin.y + ((rect.top() - origin.y) / grid_step).ceil() * grid_step;
        let mut x = first_x;
        while x <= rect.right() {
            painter.line_segment(
                [egui::pos2(x, rect.top()), egui::pos2(x, rect.bottom())],
                egui::Stroke::new(1.0, egui::Color32::from_gray(39)),
            );
            x += grid_step;
        }
        let mut y = first_y;
        while y <= rect.bottom() {
            painter.line_segment(
                [egui::pos2(rect.left(), y), egui::pos2(rect.right(), y)],
                egui::Stroke::new(1.0, egui::Color32::from_gray(39)),
            );
            y += grid_step;
        }
        painter.line_segment(
            [egui::pos2(rect.left(), origin.y), egui::pos2(rect.right(), origin.y)],
            egui::Stroke::new(1.0, egui::Color32::from_gray(88)),
        );
        painter.line_segment(
            [egui::pos2(origin.x, rect.top()), egui::pos2(origin.x, rect.bottom())],
            egui::Stroke::new(1.0, egui::Color32::from_gray(88)),
        );

        let tool_label = if self.pattern_tool == PatternTool::Draw {
            "DESENHAR · clique para adicionar pontos"
        } else {
            "EDITAR · arraste vértices/alças · botão direito/meio move a vista"
        };
        painter.text(
            rect.left_top() + egui::vec2(12.0, 12.0),
            egui::Align2::LEFT_TOP,
            tool_label,
            egui::FontId::proportional(14.0),
            egui::Color32::from_gray(205),
        );

        for (piece_index, piece) in self.pattern_project.pieces.iter().enumerate() {
            let sampled = sample_curve_with_ranges(&piece.outline, 12);
            let points = &sampled.points;
            if points.len() < 3 { continue; }
            let selected_piece = self.selected_piece == piece_index
                || self.selected_target.map(|t| t.piece()) == Some(piece_index);
            let outline_color = if selected_piece {
                egui::Color32::from_rgb(108, 205, 255)
            } else {
                egui::Color32::from_rgb(91, 183, 245)
            };

            // Contorno de corte em azul; margem de costura independente em verde.
            if piece.seam_allowance_meters > 0.0 {
                if let Ok(margin) = offset_polygon(points, piece.seam_allowance_meters) {
                    for i in 0..margin.len() {
                        painter.line_segment(
                            [to_screen(margin[i]), to_screen(margin[(i + 1) % margin.len()])],
                            egui::Stroke::new(1.5, egui::Color32::from_rgb(104, 196, 131)),
                        );
                    }
                }
            }

            if self.pattern_show_mesh {
                if let Ok(mesh) = triangulate_outline(points, self.pattern_spacing) {
                    for triangle in &mesh.triangles {
                        let a = to_screen(mesh.vertices[triangle[0] as usize]);
                        let b = to_screen(mesh.vertices[triangle[1] as usize]);
                        let c = to_screen(mesh.vertices[triangle[2] as usize]);
                        let stroke = egui::Stroke::new(0.7, egui::Color32::from_rgb(63, 91, 105));
                        painter.line_segment([a, b], stroke);
                        painter.line_segment([b, c], stroke);
                        painter.line_segment([c, a], stroke);
                    }
                }
            }

            for i in 0..points.len() {
                painter.line_segment(
                    [to_screen(points[i]), to_screen(points[(i + 1) % points.len()])],
                    egui::Stroke::new(2.0, outline_color),
                );
            }

            // Linhas internas, dobras, pregas e linha de fio têm estilos distintos do contorno.
            for line in &piece.internal_lines {
                let detail = sample_curve(line, 12);
                for pair in detail.windows(2) {
                    painter.line_segment(
                        [to_screen(pair[0]), to_screen(pair[1])],
                        egui::Stroke::new(1.7, egui::Color32::from_rgb(245, 178, 72)),
                    );
                }
            }
            for fold in &piece.fold_lines {
                let detail = sample_curve(&fold.curve, 12);
                let color = match fold.kind {
                    FoldKind::Valley => egui::Color32::from_rgb(185, 128, 255),
                    FoldKind::Mountain => egui::Color32::from_rgb(90, 220, 210),
                    FoldKind::Decorative => egui::Color32::from_rgb(230, 230, 230),
                };
                for pair in detail.windows(2) {
                    painter.line_segment(
                        [to_screen(pair[0]), to_screen(pair[1])],
                        egui::Stroke::new(1.8, color),
                    );
                }
            }
            for pleat in &piece.pleats {
                let detail = sample_curve(&pleat.axis, 12);
                for pair in detail.windows(2) {
                    painter.line_segment(
                        [to_screen(pair[0]), to_screen(pair[1])],
                        egui::Stroke::new(2.2, egui::Color32::from_rgb(255, 120, 170)),
                    );
                }
            }
            if let Some(grain) = &piece.grain_line {
                painter.line_segment(
                    [to_screen(grain.start), to_screen(grain.end)],
                    egui::Stroke::new(1.6, egui::Color32::from_rgb(120, 220, 135)),
                );
                let center = PatternPoint2::new(
                    (grain.start.x + grain.end.x) * 0.5,
                    (grain.start.y + grain.end.y) * 0.5,
                );
                painter.text(
                    to_screen(center), egui::Align2::CENTER_CENTER, "FIO",
                    egui::FontId::proportional(10.0),
                    egui::Color32::from_rgb(120, 220, 135),
                );
            }
            if let Some(PatternEditTarget::Edge { piece: selected_index, segment, .. }) = self.selected_target {
                if selected_index == piece_index {
                    if let Some(&(start, end)) = sampled.segment_vertex_ranges.get(segment) {
                        let sample_count = if end >= start { end - start } else { points.len() - start };
                        for step in 0..sample_count {
                            let ia = (start + step) % points.len();
                            let ib = (ia + 1) % points.len();
                            painter.line_segment(
                                [to_screen(points[ia]), to_screen(points[ib])],
                                egui::Stroke::new(4.5, egui::Color32::from_rgb(255, 196, 76)),
                            );
                        }
                    }
                }
            }

            let centroid = points.iter().fold(PatternPoint2::new(0.0, 0.0), |sum, point| {
                PatternPoint2::new(
                    sum.x + point.x / points.len() as f64,
                    sum.y + point.y / points.len() as f64,
                )
            });
            painter.text(
                to_screen(centroid),
                egui::Align2::CENTER_CENTER,
                &piece.name,
                egui::FontId::proportional(13.0),
                egui::Color32::WHITE,
            );

            if self.pattern_tool == PatternTool::Select {
                // Alças Bézier e linhas de controlo são exibidas apenas para a peça selecionada.
                if selected_piece {
                    let mut previous = piece.outline.start;
                    for (segment_index, segment) in piece.outline.segments.iter().enumerate() {
                        if let CurveSegment::CubicBezier { control1, control2, .. } = segment {
                            painter.line_segment(
                                [to_screen(previous), to_screen(*control1)],
                                egui::Stroke::new(1.0, egui::Color32::from_rgb(175, 119, 225)),
                            );
                            painter.line_segment(
                                [to_screen(segment.endpoint()), to_screen(*control2)],
                                egui::Stroke::new(1.0, egui::Color32::from_rgb(175, 119, 225)),
                            );
                            let c1 = to_screen(*control1);
                            let c2 = to_screen(*control2);
                            painter.circle_filled(c1, 4.0, egui::Color32::from_rgb(201, 151, 247));
                            painter.circle_filled(c2, 4.0, egui::Color32::from_rgb(201, 151, 247));
                            if self.selected_target == Some(PatternEditTarget::Control1 { piece: piece_index, segment: segment_index }) {
                                painter.circle_stroke(c1, 7.0, egui::Stroke::new(2.0, egui::Color32::WHITE));
                            }
                            if self.selected_target == Some(PatternEditTarget::Control2 { piece: piece_index, segment: segment_index }) {
                                painter.circle_stroke(c2, 7.0, egui::Stroke::new(2.0, egui::Color32::WHITE));
                            }
                        }
                        previous = segment.endpoint();
                    }
                }
                let anchors = outline_anchors(&piece.outline);
                for (anchor_index, anchor) in anchors.iter().enumerate() {
                    let screen = to_screen(*anchor);
                    let selected = self.selected_target == Some(PatternEditTarget::Anchor {
                        piece: piece_index,
                        index: anchor_index,
                    });
                    painter.circle_filled(
                        screen,
                        if selected { 6.0 } else { 4.0 },
                        if selected { egui::Color32::WHITE } else { egui::Color32::from_rgb(255, 190, 75) },
                    );
                    if selected {
                        painter.circle_stroke(screen, 9.0, egui::Stroke::new(1.5, egui::Color32::from_rgb(255, 190, 75)));
                    }
                }
            }
        }

        if self.draft_points.len() >= 2 {
            let polyline = self.draft_points.iter().copied().map(to_screen).collect::<Vec<_>>();
            painter.add(egui::Shape::line(
                polyline,
                egui::Stroke::new(2.0, egui::Color32::from_rgb(245, 180, 65)),
            ));
        }
        for point in &self.draft_points {
            painter.circle_filled(to_screen(*point), 4.5, egui::Color32::from_rgb(255, 194, 75));
        }

        // Botão direito/meio arrasta a viewport; a roda aproxima/afasta ao redor do cursor.
        if response.dragged_by(egui::PointerButton::Secondary)
            || response.dragged_by(egui::PointerButton::Middle)
        {
            self.pattern_pan += ui.input(|i| i.pointer.delta());
        }
        if response.hovered() {
            let (scroll, pointer) = ui.input(|i| (i.smooth_scroll_delta.y, i.pointer.hover_pos()));
            if scroll != 0.0 {
                let pointer = pointer.unwrap_or(rect.center());
                let before = to_world(pointer);
                self.pattern_zoom = (self.pattern_zoom * (scroll * 0.0015).exp()).clamp(80.0, 1800.0);
                self.pattern_pan.x = pointer.x - center.x - before.x as f32 * self.pattern_zoom;
                self.pattern_pan.y = pointer.y - center.y + before.y as f32 * self.pattern_zoom;
            }
        }

        if self.pattern_tool == PatternTool::Select {
            if response.drag_started_by(egui::PointerButton::Primary) {
                if let Some(pointer) = response.interact_pointer_pos() {
                    self.selected_target = self.hit_test_pattern(pointer, center);
                    if let Some(target) = self.selected_target {
                        self.selected_piece = target.piece().min(self.pattern_project.pieces.len().saturating_sub(1));
                        self.dragging_target = if target.is_movable() { Some(target) } else { None };
                        self.dragging_before = self.dragging_target.map(|_| self.editor_snapshot());
                        self.drag_last_world = Some(to_world(pointer));
                    } else {
                        self.dragging_target = None;
                        self.dragging_before = None;
                        self.drag_last_world = None;
                    }
                }
            }
            if response.dragged_by(egui::PointerButton::Primary) {
                if let (Some(target), Some(pointer)) = (self.dragging_target, response.interact_pointer_pos()) {
                    let mut world = to_world(pointer);
                    if self.pattern_snap {
                        world.x = (world.x * 100.0).round() / 100.0;
                        world.y = (world.y * 100.0).round() / 100.0;
                    }
                    if let PatternEditTarget::Piece { piece } = target {
                        if let (Some(previous), Some(piece_data)) = (
                            self.drag_last_world,
                            self.pattern_project.pieces.get(piece),
                        ) {
                            let id = piece_data.id;
                            translate_project_piece(
                                &mut self.pattern_project,
                                id,
                                world.x - previous.x,
                                world.y - previous.y,
                            );
                        }
                    } else {
                        self.move_pattern_target(target, world);
                    }
                    self.drag_last_world = Some(world);
                }
            }
            if response.drag_stopped_by(egui::PointerButton::Primary) {
                if let Some(before) = self.dragging_before.take() {
                    self.remember_pattern_change(before);
                }
                self.dragging_target = None;
                self.drag_last_world = None;
            }
            if response.clicked() && self.dragging_target.is_none() {
                if let Some(pointer) = response.interact_pointer_pos() {
                    self.selected_target = self.hit_test_pattern(pointer, center);
                    if let Some(target) = self.selected_target {
                        self.selected_piece = target.piece().min(self.pattern_project.pieces.len().saturating_sub(1));
                    }
                }
            }
            if response.hovered() && ui.input(|i| i.key_pressed(egui::Key::Delete)) {
                match self.selected_target {
                    Some(PatternEditTarget::Piece { .. }) => self.delete_selected_piece(),
                    Some(PatternEditTarget::Anchor { .. }) => self.delete_selected_vertex(),
                    _ => {}
                }
            }
        } else if response.clicked() {
            if let Some(pos) = response.interact_pointer_pos() {
                let before = self.editor_snapshot();
                let mut world = to_world(pos);
                if self.pattern_snap {
                    world.x = (world.x * 100.0).round() / 100.0;
                    world.y = (world.y * 100.0).round() / 100.0;
                }
                self.draft_points.push(world);
                self.remember_pattern_change(before);
            }
        }
    }

    fn start_pattern_simulation(&mut self) -> Result<(), String> {
        self.pattern_project.validate()?;
        if self.pattern_project.pieces.is_empty() {
            return Err("O projeto não contém peças.".into());
        }

        // Prepara as malhas antes de posicionar os painéis para preservar o layout relativo
        // que o usuário desenhou no editor 2D.
        let mut prepared = Vec::with_capacity(self.pattern_project.pieces.len());
        for piece in &self.pattern_project.pieces {
            let sampled = sample_curve_with_ranges(&piece.outline, 8);
            validate_simple_polygon(&sampled.points)
                .map_err(|e| format!("Contorno inválido em '{}': {e}", piece.name))?;
            let mesh = triangulate_outline(&sampled.points, self.pattern_spacing)
                .map_err(|e| format!("Não foi possível triangular '{}': {e}", piece.name))?;
            if mesh.boundary_vertex_count != sampled.points.len() {
                return Err(format!("Mapeamento de contorno inválido na peça '{}'.", piece.name));
            }
            prepared.push((piece.id, sampled.segment_vertex_ranges, mesh));
        }

        let min_x = prepared.iter()
            .flat_map(|(_, _, mesh)| mesh.vertices.iter().map(|p| p.x))
            .fold(f64::INFINITY, f64::min);
        let max_x = prepared.iter()
            .flat_map(|(_, _, mesh)| mesh.vertices.iter().map(|p| p.x))
            .fold(f64::NEG_INFINITY, f64::max);
        let min_y = prepared.iter()
            .flat_map(|(_, _, mesh)| mesh.vertices.iter().map(|p| p.y))
            .fold(f64::INFINITY, f64::min);

        let mut builder = ClothBuilder::new();
        builder.add_collider(Collider::ground());
        builder.add_collider(Collider::Cylinder {
            a: Vec3::ZERO,
            b: Vec3::new(0.0, 1.5, 0.0),
            radius: PATTERN_SIM_BODY_RADIUS,
        });

        struct PieceBuildInfo {
            id: u64,
            handle: cloth_sim::PanelHandle,
            ranges: Vec<(usize, usize)>,
            boundary_count: usize,
        }
        let mut infos: Vec<PieceBuildInfo> = Vec::new();
        let piece_count = prepared.len();
        let origin_x = (-0.5 * (min_x + max_x)) as f32;
        let origin_y = (0.9 - min_y) as f32;

        for (piece_index, (id, ranges, mesh)) in prepared.into_iter().enumerate() {
            // Mantém as coordenadas relativas do molde no plano XY. Os painéis ficam tangentes
            // ao lado frontal do colisor corporal, com folga para espessura e ruído inicial.
            // Camadas anteriores foram centradas em Z=0 e começavam atravessando o cilindro;
            // a correção de colisão em posições se convertia em velocidades enormes no primeiro frame.
            let uv = mesh.vertices.iter()
                .map(|p| Vec2::new(p.x as f32, p.y as f32))
                .collect();
            let panel_mesh = PanelMesh { uv, triangles: mesh.triangles };
            let z = pattern_sim_layer_z(piece_index, self.sim.params.thickness);
            let handle = builder.add_panel_with_grain(
                &panel_mesh,
                Placement::new(Vec3::new(origin_x, origin_y, z), Quat::IDENTITY),
                Vec2::Y,
            );
            infos.push(PieceBuildInfo {
                id,
                handle,
                ranges,
                boundary_count: mesh.boundary_vertex_count,
            });
        }

        for seam in &self.pattern_project.seams {
            if seam.a.curve != BoundaryCurve::Outline || seam.b.curve != BoundaryCurve::Outline {
                return Err(format!("A costura '{}' usa uma linha de costura interna; a simulação atual suporta bordas do contorno.", seam.name));
            }
            let info_a = infos.iter().find(|info| info.id == seam.a.piece_id)
                .ok_or_else(|| format!("Peça A ausente na costura '{}'.", seam.name))?;
            let info_b = infos.iter().find(|info| info.id == seam.b.piece_id)
                .ok_or_else(|| format!("Peça B ausente na costura '{}'.", seam.name))?;
            let locals_a = sampled_edge_indices(&seam.a, &info_a.ranges, info_a.boundary_count)?;
            let locals_b = sampled_edge_indices(&seam.b, &info_b.ranges, info_b.boundary_count)?;
            let globals_a = info_a.handle.globals(&locals_a);
            let globals_b = info_b.handle.globals(&locals_b);
            builder.sew(&globals_a, &globals_b);
        }

        builder.perturb(PATTERN_SIM_PERTURBATION);
        self.sim = builder.build(self.sim.params.clone());
        self.seam_pairs = self.sim.seam_pairs();
        self.diverged = false;
        self.accumulator = 0.0;
        self.playing = true;
        self.using_pattern_sim = true;
        self.workspace = WorkspaceMode::Simulation;
        self.upload_scene();
        Ok(())
    }

    fn viewport(&mut self, ui: &mut egui::Ui) {
        let rect = ui.available_rect_before_wrap();
        let response = ui.allocate_rect(rect, egui::Sense::click_and_drag());

        // Câmera
        let drag = response.drag_delta();
        if response.dragged_by(egui::PointerButton::Primary) {
            self.camera.orbit(drag.x, drag.y);
        } else if response.dragged_by(egui::PointerButton::Secondary)
            || response.dragged_by(egui::PointerButton::Middle)
        {
            self.camera.pan(drag.x, drag.y, rect.height());
        }
        if response.hovered() {
            let scroll = ui.input(|i| i.smooth_scroll_delta.y);
            if scroll != 0.0 {
                self.camera.zoom(scroll);
            }
        }

        // Dados do frame
        self.sim.compute_vertex_normals(&mut self.normals);
        let cloth_vertices: Vec<Vertex> = self
            .sim
            .pos
            .iter()
            .zip(&self.normals)
            .map(|(p, n)| Vertex { pos: p.to_array(), normal: n.to_array() })
            .collect();

        let aspect = rect.width() / rect.height().max(1.0);
        ui.painter().add(egui_wgpu::Callback::new_paint_callback(
            rect,
            ViewportCallback {
                view_proj: self.camera.view_proj(aspect),
                eye: self.camera.eye(),
                cloth_vertices,
                style: self.style.clone(),
            },
        ));
    }

    /// Modo de captura: simula N passos, pede um screenshot e salva em PPM.
    fn handle_capture(&mut self, ui: &egui::Ui) {
        let Some(cap) = &mut self.capture else {
            return;
        };
        if cap.frames > 0 {
            cap.frames -= 1;
            return;
        }
        if !cap.requested {
            cap.requested = true;
            ui.ctx().send_viewport_cmd(egui::ViewportCommand::Screenshot(egui::UserData::default()));
            return;
        }
        let image = ui.input(|i| {
            i.events.iter().find_map(|e| match e {
                egui::Event::Screenshot { image, .. } => Some(image.clone()),
                _ => None,
            })
        });
        if let Some(img) = image {
            let [w, h] = img.size;
            let mut data = format!("P6\n{w} {h}\n255\n").into_bytes();
            for px in &img.pixels {
                data.extend_from_slice(&[px.r(), px.g(), px.b()]);
            }
            match std::fs::write(&cap.path, data) {
                Ok(()) => eprintln!("screenshot salvo em {} ({w}x{h})", cap.path),
                Err(e) => eprintln!("falha ao salvar screenshot: {e}"),
            }
            ui.ctx().send_viewport_cmd(egui::ViewportCommand::Close);
        }
    }
}

fn translate_curve(curve: &mut Curve2D, dx: f64, dy: f64) {
    let shift = |p: &mut PatternPoint2| {
        p.x += dx;
        p.y += dy;
    };
    shift(&mut curve.start);
    for segment in &mut curve.segments {
        match segment {
            CurveSegment::Line { to } => shift(to),
            CurveSegment::CubicBezier { control1, control2, to } => {
                shift(control1);
                shift(control2);
                shift(to);
            }
        }
    }
}

fn translate_piece(piece: &mut PatternPiece, dx: f64, dy: f64) {
    translate_curve(&mut piece.outline, dx, dy);
    if let Some(curve) = &mut piece.sewing_line { translate_curve(curve, dx, dy); }
    for curve in &mut piece.internal_lines { translate_curve(curve, dx, dy); }
    for fold in &mut piece.fold_lines { translate_curve(&mut fold.curve, dx, dy); }
    for pleat in &mut piece.pleats { translate_curve(&mut pleat.axis, dx, dy); }
    if let Some(grain) = &mut piece.grain_line {
        grain.start.x += dx; grain.start.y += dy;
        grain.end.x += dx; grain.end.y += dy;
    }
    for dart in &mut piece.darts {
        dart.apex.x += dx; dart.apex.y += dy;
        dart.leg_a.x += dx; dart.leg_a.y += dy;
        dart.leg_b.x += dx; dart.leg_b.y += dy;
    }
    for hole in &mut piece.holes {
        hole.center.x += dx; hole.center.y += dy;
    }
    for annotation in &mut piece.annotations {
        annotation.position.x += dx; annotation.position.y += dy;
    }
}

fn translate_project_piece(project: &mut PatternProject, piece_id: u64, dx: f64, dy: f64) {
    if let Some(piece) = project.pieces.iter_mut().find(|p| p.id == piece_id) {
        translate_piece(piece, dx, dy);
    }
}

fn outline_anchors(curve: &Curve2D) -> Vec<PatternPoint2> {
    let mut anchors = vec![curve.start];
    let count = if curve.closed {
        curve.segments.len().saturating_sub(1)
    } else {
        curve.segments.len()
    };
    anchors.extend(curve.segments.iter().take(count).map(CurveSegment::endpoint));
    anchors
}

fn set_curve_anchor(curve: &mut Curve2D, index: usize, point: PatternPoint2) {
    if index == 0 {
        curve.start = point;
        if curve.closed {
            if let Some(last) = curve.segments.last_mut() {
                match last {
                    CurveSegment::Line { to } | CurveSegment::CubicBezier { to, .. } => *to = point,
                }
            }
        }
    } else if let Some(segment) = curve.segments.get_mut(index - 1) {
        match segment {
            CurveSegment::Line { to } | CurveSegment::CubicBezier { to, .. } => *to = point,
        }
    }
}

fn closest_segment_parameter(point: egui::Pos2, a: egui::Pos2, b: egui::Pos2) -> (f32, f32) {
    let ab = b - a;
    let length_squared = ab.length_sq();
    let t = if length_squared <= f32::EPSILON {
        0.0
    } else {
        ((point - a).dot(ab) / length_squared).clamp(0.0, 1.0)
    };
    (point.distance(a + ab * t), t)
}

fn split_project_segment(
    project: &mut PatternProject,
    piece_id: u64,
    segment_index: usize,
    t: f64,
) -> Result<usize, String> {
    let piece = project.pieces.iter_mut().find(|p| p.id == piece_id)
        .ok_or_else(|| "Peça não encontrada.".to_string())?;
    let curve = &mut piece.outline;
    if segment_index >= curve.segments.len() {
        return Err("Segmento selecionado fora do contorno.".into());
    }
    let t = t.clamp(0.001, 0.999);
    let p0 = if segment_index == 0 {
        curve.start
    } else {
        curve.segments[segment_index - 1].endpoint()
    };
    let original = curve.segments[segment_index].clone();
    let (left, right, split_point) = match original {
        CurveSegment::Line { to } => {
            let split = PatternPoint2::new(
                p0.x + (to.x - p0.x) * t,
                p0.y + (to.y - p0.y) * t,
            );
            (CurveSegment::Line { to: split }, CurveSegment::Line { to }, split)
        }
        CurveSegment::CubicBezier { control1, control2, to } => {
            let lerp = |a: PatternPoint2, b: PatternPoint2| PatternPoint2::new(
                a.x + (b.x - a.x) * t,
                a.y + (b.y - a.y) * t,
            );
            let p01 = lerp(p0, control1);
            let p12 = lerp(control1, control2);
            let p23 = lerp(control2, to);
            let p012 = lerp(p01, p12);
            let p123 = lerp(p12, p23);
            let split = lerp(p012, p123);
            (
                CurveSegment::CubicBezier { control1: p01, control2: p012, to: split },
                CurveSegment::CubicBezier { control1: p123, control2: p23, to },
                split,
            )
        }
    };
    curve.segments[segment_index] = left;
    curve.segments.insert(segment_index + 1, right);

    // Atualiza os intervalos de costura que referenciam a curva subdividida.
    for seam in &mut project.seams {
        for edge in [&mut seam.a, &mut seam.b] {
            if edge.piece_id != piece_id || edge.curve != BoundaryCurve::Outline { continue; }
            if edge.segment_start as usize > segment_index {
                edge.segment_start += 1;
                edge.segment_end += 1;
            } else if edge.segment_end as usize > segment_index {
                edge.segment_end += 1;
            }
        }
    }
    // O vértice novo é o endpoint do segmento à esquerda.
    let _ = split_point;
    project.validate()?;
    Ok(segment_index + 1)
}

fn convert_project_segment_to_bezier(
    project: &mut PatternProject,
    piece_id: u64,
    segment_index: usize,
) -> Result<(), String> {
    let piece = project.pieces.iter_mut().find(|p| p.id == piece_id)
        .ok_or_else(|| "Peça não encontrada.".to_string())?;
    let curve = &mut piece.outline;
    let segment = curve.segments.get(segment_index)
        .ok_or_else(|| "Segmento selecionado fora do contorno.".to_string())?;
    let end = segment.endpoint();
    let start = if segment_index == 0 {
        curve.start
    } else {
        curve.segments[segment_index - 1].endpoint()
    };
    let one_third = |a: PatternPoint2, b: PatternPoint2, t: f64| PatternPoint2::new(
        a.x + (b.x - a.x) * t,
        a.y + (b.y - a.y) * t,
    );
    curve.segments[segment_index] = CurveSegment::CubicBezier {
        control1: one_third(start, end, 1.0 / 3.0),
        control2: one_third(start, end, 2.0 / 3.0),
        to: end,
    };
    project.validate()?;
    Ok(())
}

fn delete_project_anchor(
    project: &mut PatternProject,
    piece_id: u64,
    anchor_index: usize,
) -> Result<(), String> {
    // Usa uma cópia transacional: se o remapeamento de costuras não puder ser representado
    // como intervalo contíguo, o projeto original permanece intacto.
    let mut staged = project.clone();
    let piece = staged.pieces.iter_mut().find(|p| p.id == piece_id)
        .ok_or_else(|| "Peça não encontrada.".to_string())?;
    let curve = &mut piece.outline;
    if !curve.closed || curve.segments.len() < 4 {
        return Err("O contorno deve ter pelo menos quatro vértices para excluir um ponto.".into());
    }
    if curve.segments.iter().any(|s| !matches!(s, CurveSegment::Line { .. })) {
        return Err("A exclusão de vértices está disponível para contornos lineares; em Bézier use Inserir vértice e edite as alças.".into());
    }
    let n = curve.segments.len();
    if anchor_index >= n { return Err("Vértice selecionado fora do contorno.".into()); }

    let map_segment = |old: usize| -> usize {
        if anchor_index == 0 {
            if old == 0 || old == n - 1 { n - 2 } else { old - 1 }
        } else if old < anchor_index - 1 {
            old
        } else if old <= anchor_index {
            anchor_index - 1
        } else {
            old - 1
        }
    };

    for seam in &mut staged.seams {
        for edge in [&mut seam.a, &mut seam.b] {
            if edge.piece_id != piece_id || edge.curve != BoundaryCurve::Outline { continue; }
            let original: Vec<usize> = (edge.segment_start as usize..edge.segment_end as usize)
                .filter(|&i| i < n)
                .map(map_segment)
                .collect();
            if original.is_empty() {
                return Err("Uma costura referencia o segmento que seria removido.".into());
            }
            let mut unique = original.clone();
            unique.sort_unstable();
            unique.dedup();
            let min = *unique.first().unwrap();
            let max = *unique.last().unwrap();
            if max - min + 1 != unique.len() {
                return Err("Não é possível excluir esse vértice sem fragmentar um intervalo de costura. Ajuste a costura primeiro.".into());
            }
            edge.segment_start = min as u32;
            edge.segment_end = (max + 1) as u32;
        }
    }

    let curve = &mut staged.pieces.iter_mut().find(|p| p.id == piece_id).unwrap().outline;
    if anchor_index == 0 {
        let new_start = curve.segments[0].endpoint();
        curve.start = new_start;
        curve.segments.remove(0);
        if let Some(last) = curve.segments.last_mut() {
            match last {
                CurveSegment::Line { to } | CurveSegment::CubicBezier { to, .. } => *to = new_start,
            }
        }
    } else {
        let next = curve.segments[anchor_index].endpoint();
        match &mut curve.segments[anchor_index - 1] {
            CurveSegment::Line { to } => *to = next,
            _ => unreachable!("contornos Bézier foram rejeitados acima"),
        }
        curve.segments.remove(anchor_index);
    }
    staged.validate()?;
    *project = staged;
    Ok(())
}

fn sampled_edge_indices(
    edge: &EdgeRef,
    ranges: &[(usize, usize)],
    boundary_count: usize,
  ) -> Result<Vec<u32>, String> {
    if boundary_count < 2 || edge.segment_start >= edge.segment_end {
        return Err("Intervalo de costura sem vértices suficientes.".into());
    }
    if edge.segment_end as usize > ranges.len() {
        return Err("Intervalo de costura fora da curva amostrada.".into());
    }
    let mut indices: Vec<u32> = Vec::new();
    for segment_index in edge.segment_start as usize..edge.segment_end as usize {
        let (start, end) = ranges[segment_index];
        if indices.last().copied() != Some(start as u32) {
            indices.push(start as u32);
        }
        if end >= start {
            for index in (start + 1)..=end {
                let wrapped = (index % boundary_count) as u32;
                if indices.last().copied() != Some(wrapped) { indices.push(wrapped); }
            }
        } else {
            for index in (start + 1)..boundary_count {
                let wrapped = index as u32;
                if indices.last().copied() != Some(wrapped) { indices.push(wrapped); }
            }
            let wrapped = (end % boundary_count) as u32;
            if indices.last().copied() != Some(wrapped) { indices.push(wrapped); }
        }
    }
    if edge.reversed { indices.reverse(); }
    indices.dedup();
    if indices.len() < 2 {
        return Err("A borda selecionada contém menos de dois vértices.".into());
    }
    Ok(indices)
}

impl eframe::App for ClothEvoApp {
    fn ui(&mut self, ui: &mut egui::Ui, _frame: &mut eframe::Frame) {
        if self.workspace == WorkspaceMode::Pattern {
            let (undo, redo) = ui.input(|i| (
                i.modifiers.ctrl && i.key_pressed(egui::Key::Z) && !i.modifiers.shift,
                (i.modifiers.ctrl && i.key_pressed(egui::Key::Y))
                    || (i.modifiers.ctrl && i.modifiers.shift && i.key_pressed(egui::Key::Z)),
            ));
            if undo { self.undo_pattern_edit(); }
            if redo { self.redo_pattern_edit(); }
        }
        // Em captura, simula em passos fixos independentemente do tempo real.
        let dt = if self.capture.is_some() { FIXED_DT } else { ui.input(|i| i.stable_dt) };
        let steps_before = self.sim.stats.steps;
        if self.workspace == WorkspaceMode::Simulation { self.advance(dt); }
        if self.sim.stats.steps != steps_before {
            self.last_step_ms = self.sim.stats.step_ms;
        }

        egui::Panel::left("controls").default_size(310.0).show(ui, |ui| {
            egui::ScrollArea::vertical().show(ui, |ui| self.controls(ui));
        });
        egui::CentralPanel::default().frame(egui::Frame::NONE).show(ui, |ui| {
            if self.workspace == WorkspaceMode::Pattern {
                self.pattern_viewport(ui);
            } else {
                self.viewport(ui);
            }
        });

        self.handle_capture(ui);
        if (self.playing && self.workspace == WorkspaceMode::Simulation) || self.capture.is_some() {
            ui.ctx().request_repaint();
        }
    }
}


#[cfg(test)]
mod pattern_editor_tests {
    use super::*;

    fn square_project() -> PatternProject {
        let mut project = PatternProject::new("Teste de edição");
        let square = [
            PatternPoint2::new(0.0, 0.0),
            PatternPoint2::new(1.0, 0.0),
            PatternPoint2::new(1.0, 1.0),
            PatternPoint2::new(0.0, 1.0),
        ];
        project.pieces.push(PatternPiece::from_polygon(1, "A", &square).unwrap());
        project.pieces.push(PatternPiece::from_polygon(
            2,
            "B",
            &[
                PatternPoint2::new(1.0, 0.0),
                PatternPoint2::new(2.0, 0.0),
                PatternPoint2::new(2.0, 1.0),
                PatternPoint2::new(1.0, 1.0),
            ],
        ).unwrap());
        project.seams.push(SeamDefinition {
            id: 1,
            name: "A-B".into(),
            a: EdgeRef::single_segment(1, 1),
            b: EdgeRef::single_segment(2, 3),
            ease_meters: 0.0,
        });
        project
    }

    #[test]
    fn pattern_sim_layers_start_outside_the_body_collision_shell() {
        let thickness = 0.005;
        for index in 0..12 {
            let layer_z = pattern_sim_layer_z(index, thickness);
            // perturb() pode deslocar Z para ambos os lados em até sua amplitude.
            let lowest_z = layer_z - PATTERN_SIM_PERTURBATION;
            assert!(
                lowest_z >= PATTERN_SIM_BODY_RADIUS + thickness - 1e-6,
                "a camada {index} começa dentro do colisor: z mínimo {lowest_z}"
            );
        }
    }

    #[test]
    fn translating_piece_moves_every_curve_point_without_changing_seam_ids() {
        let mut project = square_project();
        let before = project.pieces[0].clone();
        translate_project_piece(&mut project, 1, 2.0, -0.5);
        let moved = &project.pieces[0];
        assert_eq!(moved.outline.start, PatternPoint2::new(before.outline.start.x + 2.0, before.outline.start.y - 0.5));
        assert_eq!(moved.id, before.id);
        assert_eq!(project.seams[0].a.piece_id, 1);
        assert!((pattern_geom::curve_length(&moved.outline, 32) - pattern_geom::curve_length(&before.outline, 32)).abs() < 1.0e-6);
    }

    #[test]
    fn editor_snapshot_compares_project_and_draft_for_history() {
        let project = square_project();
        let a = PatternEditorSnapshot {
            project: project.clone(),
            draft_points: vec![PatternPoint2::new(0.0, 0.0)],
            selected_piece: 0,
            selected_target: None,
        };
        let mut b = a.clone();
        assert_eq!(a, b);
        b.draft_points.push(PatternPoint2::new(1.0, 0.0));
        assert_ne!(a, b);
    }

    #[test]
    fn moving_first_anchor_keeps_closed_curve_closed() {
        let mut curve = PatternPiece::from_polygon(
            1,
            "Quadrado",
            &[
                PatternPoint2::new(0.0, 0.0),
                PatternPoint2::new(1.0, 0.0),
                PatternPoint2::new(1.0, 1.0),
                PatternPoint2::new(0.0, 1.0),
            ],
        ).unwrap().outline;
        let moved = PatternPoint2::new(-0.25, 0.15);
        set_curve_anchor(&mut curve, 0, moved);
        assert_eq!(curve.start, moved);
        assert_eq!(curve.segments.last().unwrap().endpoint(), moved);
        assert_eq!(outline_anchors(&curve).len(), 4);
    }

    #[test]
    fn splitting_seam_edge_updates_segment_ranges_and_keeps_geometry_length() {
        let mut project = square_project();
        let before = pattern_geom::curve_length(&project.pieces[0].outline, 32);
        split_project_segment(&mut project, 1, 1, 0.35).unwrap();
        let after = pattern_geom::curve_length(&project.pieces[0].outline, 32);
        assert!((before - after).abs() < 1.0e-6);
        assert_eq!(project.pieces[0].outline.segments.len(), 5);
        assert_eq!(project.seams[0].a.segment_start, 1);
        assert_eq!(project.seams[0].a.segment_end, 3);
        assert_eq!(project.seams[0].b.segment_start, 3);
        assert_eq!(project.seams[0].b.segment_end, 4);
        project.validate().unwrap();
    }

    #[test]
    fn deleting_polygon_anchor_keeps_project_valid_and_seam_reference_valid() {
        let mut project = square_project();
        delete_project_anchor(&mut project, 1, 1).unwrap();
        assert_eq!(project.pieces[0].outline.segments.len(), 3);
        assert_eq!(project.seams[0].a.segment_start, 0);
        assert_eq!(project.seams[0].a.segment_end, 1);
        project.validate().unwrap();
    }

    #[test]
    fn converting_line_to_bezier_preserves_straight_shape_and_seam_indices() {
        let mut project = square_project();
        convert_project_segment_to_bezier(&mut project, 1, 1).unwrap();
        let segment = &project.pieces[0].outline.segments[1];
        assert!(matches!(segment, CurveSegment::CubicBezier { .. }));
        assert_eq!(project.seams[0].a.segment_start, 1);
        assert_eq!(project.seams[0].a.segment_end, 2);
        assert!((pattern_geom::curve_length(&project.pieces[0].outline, 64) - 4.0).abs() < 1.0e-4);
        project.validate().unwrap();
    }

    #[test]
    fn deleting_anchor_on_bezier_outline_is_rejected_without_mutating_project() {
        let mut project = square_project();
        project.pieces[0].outline.segments[0] = CurveSegment::CubicBezier {
            control1: PatternPoint2::new(0.2, -0.1),
            control2: PatternPoint2::new(0.7, -0.1),
            to: PatternPoint2::new(1.0, 0.0),
        };
        let before = project.clone();
        assert!(delete_project_anchor(&mut project, 1, 1).is_err());
        assert_eq!(project, before);
    }
}
