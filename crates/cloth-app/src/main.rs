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
    build_avatar_colliders, build_scene, AvatarConfig, ClothBuilder, Collider, FabricPreset,
    PanelMaterial, PanelMesh, Placement, SceneKind, Simulation, SimParams,
};
use eframe::{egui, egui_wgpu};
use glam::{EulerRot, Quat, Vec2, Vec3};
use pattern_core::{
    BoundaryCurve, Curve2D, CurveParam, CurveSegment, EdgePath, EdgeRef, FoldKind, FoldLine, GrainLine,
    Notch, PatternPiece, PatternProject, Pleat,
    Point2 as PatternPoint2, SeamDefinition, SymmetryLink,
};
use pattern_geom::{
    cut_closed_outline, mirror_curve_x, offset_polygon, point_in_polygon, rectangle,
    regular_polygon, sample_curve, sample_curve_with_forced_params, sample_curve_with_ranges, validate_simple_polygon,
    CutAndSewResult, SourceSegmentFragment,
};
use pattern_mesh::{triangulate_outline, triangulate_outline_with_forced_vertices};

const FIXED_DT: f32 = 1.0 / 60.0;
const DEFAULT_RES: u32 = 40;
const PATTERN_PIXELS_PER_METER: f32 = 360.0;
const PATTERN_SIM_BODY_RADIUS: f32 = 0.14;
const PATTERN_SIM_PERTURBATION: f32 = 0.001;
const PATTERN_SIM_LAYER_SPACING: f32 = 0.003;

/// Raio de montagem por painel, incluindo espessura, ruído inicial e folga entre camadas.
fn pattern_sim_shell_radius(piece_index: usize, thickness: f32) -> f32 {
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

#[derive(Clone, Debug)]
struct PieceSampleMapping {
    ranges: Vec<(usize, usize)>,
    /// (segment, t, índice de vértice na borda amostrada).
    forced_vertices: Vec<(u32, f64, usize)>,
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
    pattern_symmetry_axis_x: f64,
    pattern_show_mesh: bool,
    selected_piece: usize,
    seam_a_piece: usize,
    seam_b_piece: usize,
    seam_sides_a: Vec<EdgePath>,
    seam_sides_b: Vec<EdgePath>,
    seam_a_active_path: usize,
    seam_b_active_path: usize,
    seam_active_side_a: bool,
    seam_a_segment: usize,
    seam_b_segment: usize,
    /// Fim exclusivo do intervalo selecionado (na UI, o último segmento é inclusivo).
    seam_a_segment_end: usize,
    seam_b_segment_end: usize,
    seam_a_reversed: bool,
    seam_b_reversed: bool,
    seam_a_start_t: f64,
    seam_a_end_t: f64,
    seam_b_start_t: f64,
    seam_b_end_t: f64,
    seam_free_mode: bool,
    seam_snap_tolerance_px: f32,
    dragging_seam_handle: Option<(bool, bool)>,
    seam_drag_before: Option<PatternEditorSnapshot>,
    seam_ease_m: f64,
    seam_name: String,
    selected_seam_id: Option<u64>,
    notch_segment: usize,
    notch_t: f64,
    notch_depth_m: f64,
    notch_width_m: f64,
    selected_notch_index: usize,
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
            Collider::Ellipsoid { center, radii } => {
                out.push(primitives::ellipsoid(center, radii, 40, 20, [0.69, 0.73, 0.77]));
            }
            Collider::Capsule { a, b, radius } => {
                out.push(primitives::cylinder(a, b, radius, 32, [0.69, 0.73, 0.77]));
                out.push(primitives::uv_sphere(a, radius, 24, 12, [0.69, 0.73, 0.77]));
                out.push(primitives::uv_sphere(b, radius, 24, 12, [0.69, 0.73, 0.77]));
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
            pattern_symmetry_axis_x: 0.0,
            pattern_show_mesh: true,
            selected_piece: 0,
            seam_a_piece: 0,
            seam_b_piece: 1,
            seam_sides_a: Vec::new(),
            seam_sides_b: Vec::new(),
            seam_a_active_path: 0,
            seam_b_active_path: 0,
            seam_active_side_a: true,
            seam_a_segment: 0,
            seam_b_segment: 0,
            seam_a_segment_end: 1,
            seam_b_segment_end: 1,
            seam_a_reversed: false,
            seam_b_reversed: false,
            seam_a_start_t: 0.0,
            seam_a_end_t: 1.0,
            seam_b_start_t: 0.0,
            seam_b_end_t: 1.0,
            seam_free_mode: false,
            seam_snap_tolerance_px: 10.0,
            dragging_seam_handle: None,
            seam_drag_before: None,
            seam_ease_m: 0.0,
            seam_name: String::new(),
            selected_seam_id: None,
            notch_segment: 0,
            notch_t: 0.5,
            notch_depth_m: 0.008,
            notch_width_m: 0.012,
            selected_notch_index: 0,
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
            if !self.sim.is_finite() {
                self.playing = false;
                self.diverged = true;
                self.accumulator = 0.0;
                self.sim.reset();
                return;
            }
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
                if self.diverged {
                    self.sim.reset();
                    self.accumulator = 0.0;
                    self.diverged = false;
                    self.playing = true;
                } else {
                    self.playing = !self.playing;
                }
            }
            if ui.button("⏭ Passo único").clicked() {
                self.playing = false;
                self.accumulator = 0.0;
                self.sim.step(FIXED_DT);
                self.last_step_ms = self.sim.stats.step_ms;
                if !self.sim.is_finite() {
                    self.sim.reset();
                    self.last_step_ms = 0.0;
                    self.diverged = true;
                }
            }
            if ui.button("↺ Reiniciar").clicked() {
                self.sim.reset();
                self.last_step_ms = 0.0;
                self.playing = false;
                self.diverged = false;
                self.accumulator = 0.0;
            }
        });
        if self.diverged {
            ui.colored_label(
                egui::Color32::LIGHT_RED,
                "Instabilidade numérica detectada: simulação pausada e restaurada ao estado inicial. Aumente substeps/iterações ou reduza rigidez.",
            );
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
        ui.strong("Solver e estabilidade");
        ui.add(egui::Slider::new(&mut p.substeps, 1..=80).text("substeps por quadro"));
        ui.add(egui::Slider::new(&mut p.iterations, 1..=12).text("iterações por substep"));
        ui.add(egui::Slider::new(&mut p.max_particle_speed, 2.0..=60.0)
            .text("limite de velocidade (m/s)"));
        ui.checkbox(&mut p.self_collision, "Auto-colisão (mais custo)");
        ui.weak("A auto-colisão usa hash espacial vértice-vértice; vértices vizinhos e costuras são excluídos.");
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
        ui.label(format!("velocidade máx.: {:.2} m/s · deslocamento do substep: {:.1} mm",
            sim.max_particle_speed(), sim.max_particle_displacement() * 1000.0));
        let unstable = sim.unstable_particle_count(10.0);
        if unstable > 0 {
            ui.colored_label(
                egui::Color32::from_rgb(255, 92, 92),
                format!("Partículas instáveis: {} acima de 10 m/s ou com dados inválidos", unstable),
            );
        } else {
            ui.colored_label(egui::Color32::LIGHT_GREEN, "Partículas: estáveis (limiar de 10 m/s)");
        }
        if !self.seam_pairs.is_empty() {
            ui.label(format!("abertura máx. das costuras: {:.1} mm", self.seam_gap_max() * 1000.0));
            let seam_load = sim.seam_tension_estimate_max_n();
            let load_color = if seam_load > 100.0 { egui::Color32::from_rgb(255, 92, 92) } else { egui::Color32::LIGHT_GREEN };
            ui.colored_label(load_color, format!("carga XPBD estimada máx. da costura: {:.2} N", seam_load));
            ui.weak("Estimativa do multiplicador XPBD; indicador comparativo, não força experimental calibrada.");
        }
        let critical_seams = self.sim.seam_warning_pairs(0.015).len();
        if critical_seams > 0 {
            ui.colored_label(
                egui::Color32::from_rgb(255, 92, 92),
                format!("Abertura crítica: {} ligações de costura acima de 15 mm", critical_seams),
            );
        } else if !self.seam_pairs.is_empty() {
            ui.colored_label(egui::Color32::LIGHT_GREEN, "Costuras sem abertura crítica (> 15 mm)");
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
        ui.separator();
        ui.strong("Simetria vinculada");
        ui.add(egui::DragValue::new(&mut self.pattern_symmetry_axis_x)
            .speed(0.005).prefix("Eixo X (m) "));
        ui.horizontal(|ui| {
            if ui.button("Usar X mínimo do molde").clicked() {
                self.set_symmetry_axis_to_min_x();
            }
            if ui.button("Criar par espelhado").clicked() {
                self.create_mirrored_pair();
            }
            if ui.button("Desvincular par").clicked() {
                self.unlink_selected_symmetry();
            }
        });
        ui.weak("A edição de vértices e alças é refletida no molde vinculado.");
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
        ui.strong("Pré-visualização da malha 2D");
        ui.add(egui::Slider::new(&mut self.pattern_spacing, 0.025..=0.20).text("espaçamento visual (m)"));
        ui.checkbox(&mut self.pattern_show_mesh, "Mostrar triangulação");
        ui.weak("A resolução da simulação 3D é definida separadamente em Física e montagem 3D por molde.");
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
                let piece_id = self.pattern_project.pieces[self.selected_piece].id;
                self.sync_symmetric_mate(piece_id);
                self.remember_pattern_change(allowance_before);
            }
        }

        if !self.pattern_project.pieces.is_empty() {
            ui.separator();
            ui.collapsing("Física e montagem 3D por molde", |ui| {
                let index = self.selected_piece.min(self.pattern_project.pieces.len() - 1);
                let piece_id = self.pattern_project.pieces[index].id;
                let before = self.editor_snapshot();
                let mut changed = false;
                {
                    let piece = &mut self.pattern_project.pieces[index];
                    ui.label(format!("Material: {}", piece.name));
                    changed |= ui.add(egui::Slider::new(
                        &mut piece.fabric.density_kg_m2, 0.02..=1.20,
                    ).text("Densidade (kg/m²)")).changed();
                    changed |= ui.add(egui::Slider::new(
                        &mut piece.fabric.stretch_warp_n_m, 100.0..=30000.0,
                    ).text("Tração no urdume (N/m)")).changed();
                    changed |= ui.add(egui::Slider::new(
                        &mut piece.fabric.stretch_weft_n_m, 100.0..=30000.0,
                    ).text("Tração na trama (N/m)")).changed();
                    changed |= ui.add(egui::Slider::new(
                        &mut piece.fabric.shear_n_m, 10.0..=5000.0,
                    ).text("Cisalhamento (N/m)")).changed();
                    changed |= ui.add(egui::Slider::new(
                        &mut piece.fabric.bend_nm, 0.000001..=0.001,
                    ).logarithmic(true).text("Flexão (N·m)")).changed();
                    changed |= ui.add(egui::Slider::new(
                        &mut piece.fabric.thickness_meters, 0.0001..=0.01,
                    ).text("Espessura (m)")).changed();
                    changed |= ui.add(egui::Slider::new(
                        &mut piece.fabric.friction_coefficient, 0.0..=1.5,
                    ).text("Atrito deste tecido")).changed();
                    changed |= ui.add(egui::Slider::new(
                        &mut piece.fabric.particle_distance_meters, 0.01..=0.20,
                    ).text("Distância entre partículas (m)")).changed();

                    ui.separator();
                    ui.label("Posição 3D (deslocamento relativo ao layout 2D)");
                    ui.horizontal(|ui| {
                        changed |= ui.add(egui::DragValue::new(
                            &mut piece.assembly.translation_meters[0],
                        ).speed(0.01).prefix("X ")).changed();
                        changed |= ui.add(egui::DragValue::new(
                            &mut piece.assembly.translation_meters[1],
                        ).speed(0.01).prefix("Y ")).changed();
                        changed |= ui.add(egui::DragValue::new(
                            &mut piece.assembly.translation_meters[2],
                        ).speed(0.01).prefix("Z ")).changed();
                    });
                    ui.label("Rotação local (graus)");
                    changed |= ui.add(egui::Slider::new(
                        &mut piece.assembly.rotation_degrees[0], -180.0..=180.0,
                    ).text("Rotação X")).changed();
                    changed |= ui.add(egui::Slider::new(
                        &mut piece.assembly.rotation_degrees[1], -180.0..=180.0,
                    ).text("Rotação Y")).changed();
                    changed |= ui.add(egui::Slider::new(
                        &mut piece.assembly.rotation_degrees[2], -180.0..=180.0,
                    ).text("Rotação Z")).changed();
                }
                if changed {
                    self.sync_symmetric_mate(piece_id);
                    self.remember_pattern_change(before);
                }

                let selected_anchor = match self.selected_target {
                    Some(PatternEditTarget::Anchor { piece, index: anchor }) if piece == index => Some(anchor),
                    _ => None,
                };
                if let Some(anchor) = selected_anchor {
                    let pinned = self.pattern_project.pieces[index].assembly.pinned_anchors.contains(&(anchor as u32));
                    if ui.button(if pinned {
                        format!("Desafixar vértice {}", anchor + 1)
                    } else {
                        format!("Fixar vértice {} na simulação", anchor + 1)
                    }).clicked() {
                        let before = self.editor_snapshot();
                        let assembly = &mut self.pattern_project.pieces[index].assembly;
                        if pinned {
                            assembly.pinned_anchors.retain(|candidate| *candidate != anchor as u32);
                        } else {
                            assembly.pinned_anchors.push(anchor as u32);
                            assembly.pinned_anchors.sort_unstable();
                            assembly.pinned_anchors.dedup();
                        }
                        self.sync_symmetric_mate(piece_id);
                        self.remember_pattern_change(before);
                    }
                } else {
                    ui.weak("Selecione um vértice do contorno para fixá-lo ou liberá-lo na simulação.");
                }
                if !self.pattern_project.pieces[index].assembly.pinned_anchors.is_empty() {
                    ui.label(format!(
                        "Vértices fixados: {}",
                        self.pattern_project.pieces[index].assembly.pinned_anchors.iter()
                            .map(|id| (id + 1).to_string()).collect::<Vec<_>>().join(", ")
                    ));
                }
            });
        }

        if self.pattern_project.pieces.len() >= 2 {
            ui.separator();
            ui.strong("Sistema avançado de costura");

            let mut requested_seam = self.selected_seam_id;
            egui::ComboBox::from_id_salt("selected_seam")
                .selected_text(self.selected_seam_id
                    .and_then(|id| self.pattern_project.seams.iter().find(|seam| seam.id == id))
                    .map(|seam| format!("Editar: {}", seam.name))
                    .unwrap_or_else(|| "Nova costura".to_string()))
                .width(245.0)
                .show_ui(ui, |ui| {
                    ui.selectable_value(&mut requested_seam, None, "Nova costura");
                    for seam in &self.pattern_project.seams {
                        ui.selectable_value(
                            &mut requested_seam, Some(seam.id),
                            format!("{} · #{}", seam.name, seam.id),
                        );
                    }
                });
            if requested_seam != self.selected_seam_id {
                self.selected_seam_id = requested_seam;
                if let Some(id) = requested_seam {
                    if let Some(seam) = self.pattern_project.seams.iter().find(|seam| seam.id == id).cloned() {
                        self.seam_sides_a = seam.sides_a.clone();
                        self.seam_sides_b = seam.sides_b.clone();
                        self.seam_a_active_path = 0;
                        self.seam_b_active_path = 0;
                        self.seam_active_side_a = true;
                        if let (Some(edge_a), Some(edge_b)) = (seam.sides_a.first(), seam.sides_b.first()) {
                            self.load_seam_edge_to_controls(true, edge_a);
                            self.load_seam_edge_to_controls(false, edge_b);
                            self.seam_free_mode = [edge_a, edge_b].iter().any(|edge|
                                edge.start.t > 1.0e-9 || edge.end.t < 1.0 - 1.0e-9);
                        }
                        self.seam_ease_m = seam.ease_meters;
                        self.seam_name = seam.name;
                    }
                } else {
                    self.reset_seam_controls();
                }
            }

            let options: Vec<(u64, String)> = self.pattern_project.pieces.iter()
                .map(|piece| (piece.id, piece.name.clone())).collect();
            self.seam_a_piece = self.seam_a_piece.min(options.len() - 1);
            self.seam_b_piece = self.seam_b_piece.min(options.len() - 1);
            egui::ComboBox::from_id_salt("seam_piece_a")
                .selected_text(format!("A: {}", options[self.seam_a_piece].1))
                .width(245.0)
                .show_ui(ui, |ui| {
                    for (index, (_, name)) in options.iter().enumerate() {
                        ui.selectable_value(&mut self.seam_a_piece, index, name);
                    }
                });
            egui::ComboBox::from_id_salt("seam_piece_b")
                .selected_text(format!("B: {}", options[self.seam_b_piece].1))
                .width(245.0)
                .show_ui(ui, |ui| {
                    for (index, (_, name)) in options.iter().enumerate() {
                        ui.selectable_value(&mut self.seam_b_piece, index, name);
                    }
                });

            let a_count = self.pattern_project.pieces[self.seam_a_piece].outline.segment_count().max(1);
            let b_count = self.pattern_project.pieces[self.seam_b_piece].outline.segment_count().max(1);
            self.seam_a_segment = self.seam_a_segment.min(a_count - 1);
            self.seam_b_segment = self.seam_b_segment.min(b_count - 1);
            self.seam_a_segment_end = self.seam_a_segment_end.clamp(self.seam_a_segment + 1, a_count);
            self.seam_b_segment_end = self.seam_b_segment_end.clamp(self.seam_b_segment + 1, b_count);

            ui.horizontal(|ui| {
                egui::ComboBox::from_id_salt("seam_a_start")
                    .selected_text(format!("Segmentos {} a {}", self.seam_a_segment + 1, self.seam_a_segment_end))
                    .show_ui(ui, |ui| {
                        for index in 0..a_count {
                            ui.selectable_value(&mut self.seam_a_segment, index, format!("Segmentos {} a {}", index + 1, self.seam_a_segment_end));
                        }
                    });
                egui::ComboBox::from_id_salt("seam_a_end")
                    .selected_text(format!("Segmentos {} a {}", self.seam_a_segment + 1, self.seam_a_segment_end))
                    .show_ui(ui, |ui| {
                        for end in (self.seam_a_segment + 1)..=a_count {
                            ui.selectable_value(&mut self.seam_a_segment_end, end, format!("Segmentos {} a {}", self.seam_a_segment + 1, end));
                        }
                    });
            });
            self.seam_a_segment_end = self.seam_a_segment_end.clamp(self.seam_a_segment + 1, a_count);
            ui.horizontal(|ui| {
                egui::ComboBox::from_id_salt("seam_b_start")
                    .selected_text(format!("Segmentos {} a {}", self.seam_b_segment + 1, self.seam_b_segment_end))
                    .show_ui(ui, |ui| {
                        for index in 0..b_count {
                            ui.selectable_value(&mut self.seam_b_segment, index, format!("Segmentos {} a {}", index + 1, self.seam_b_segment_end));
                        }
                    });
                egui::ComboBox::from_id_salt("seam_b_end")
                    .selected_text(format!("Segmentos {} a {}", self.seam_b_segment + 1, self.seam_b_segment_end))
                    .show_ui(ui, |ui| {
                        for end in (self.seam_b_segment + 1)..=b_count {
                            ui.selectable_value(&mut self.seam_b_segment_end, end, format!("Segmentos {} a {}", self.seam_b_segment + 1, end));
                        }
                    });
            });
            self.seam_b_segment_end = self.seam_b_segment_end.clamp(self.seam_b_segment + 1, b_count);

            ui.checkbox(&mut self.seam_free_mode, "Costura livre (extremidades parciais)");
            if self.seam_free_mode {
                ui.label("A — parâmetros das extremidades");
                ui.horizontal(|ui| {
                    ui.add(egui::Slider::new(&mut self.seam_a_start_t, 0.0..=1.0).text("Início t"));
                    ui.add(egui::Slider::new(&mut self.seam_a_end_t, 0.0..=1.0).text("Fim t"));
                });
                ui.label("B — parâmetros das extremidades");
                ui.horizontal(|ui| {
                    ui.add(egui::Slider::new(&mut self.seam_b_start_t, 0.0..=1.0).text("Início t"));
                    ui.add(egui::Slider::new(&mut self.seam_b_end_t, 0.0..=1.0).text("Fim t"));
                });
                ui.add(egui::Slider::new(&mut self.seam_snap_tolerance_px, 4.0..=32.0)
                    .text("Tolerância de encaixe (px)"));
                ui.weak("Arraste os pontos de início/fim no canvas; eles encaixam em vértices e piques.");
            }

            ui.horizontal(|ui| {
                if ui.button("Usar borda selecionada como A").clicked() {
                    if let Some(PatternEditTarget::Edge { piece, segment, .. }) = self.selected_target {
                        self.seam_a_piece = piece.min(options.len() - 1);
                        self.seam_a_segment = segment;
                        self.seam_a_segment_end = segment + 1;
                        self.seam_a_start_t = 0.0;
                        self.seam_a_end_t = 1.0;
                    } else {
                        self.pattern_message = "Selecione uma borda no canvas antes de defini-la como A.".into();
                    }
                }
                if ui.button("Usar borda selecionada como B").clicked() {
                    if let Some(PatternEditTarget::Edge { piece, segment, .. }) = self.selected_target {
                        self.seam_b_piece = piece.min(options.len() - 1);
                        self.seam_b_segment = segment;
                        self.seam_b_segment_end = segment + 1;
                        self.seam_b_start_t = 0.0;
                        self.seam_b_end_t = 1.0;
                    } else {
                        self.pattern_message = "Selecione uma borda no canvas antes de defini-la como B.".into();
                    }
                }
            });

            ui.add(egui::TextEdit::singleline(&mut self.seam_name).hint_text("Nome da costura"));
            let mut ease_cm = self.seam_ease_m * 100.0;
            if ui.add(egui::Slider::new(&mut ease_cm, -50.0..=50.0)
                .text("Diferença-alvo A − B (cm)")).changed()
            {
                self.seam_ease_m = ease_cm / 100.0;
            }
            ui.weak("Folga assinada: positivo = A mais longa; negativo = B mais longa. O valor é aplicado à simulação.");
            ui.checkbox(&mut self.seam_a_reversed, "Inverter direção da borda A");
            ui.checkbox(&mut self.seam_b_reversed, "Inverter direção da borda B");

            let piece_a = &self.pattern_project.pieces[self.seam_a_piece];
            let piece_b = &self.pattern_project.pieces[self.seam_b_piece];
            let edge_a = EdgePath {
                piece_id: piece_a.id, curve: BoundaryCurve::Outline,
                start: CurveParam {
                    segment: self.seam_a_segment as u32,
                    t: if self.seam_free_mode { self.seam_a_start_t } else { 0.0 },
                },
                end: CurveParam {
                    segment: self.seam_a_segment_end.saturating_sub(1) as u32,
                    t: if self.seam_free_mode { self.seam_a_end_t } else { 1.0 },
                },
                reversed: self.seam_a_reversed,
            };
            let edge_b = EdgePath {
                piece_id: piece_b.id, curve: BoundaryCurve::Outline,
                start: CurveParam {
                    segment: self.seam_b_segment as u32,
                    t: if self.seam_free_mode { self.seam_b_start_t } else { 0.0 },
                },
                end: CurveParam {
                    segment: self.seam_b_segment_end.saturating_sub(1) as u32,
                    t: if self.seam_free_mode { self.seam_b_end_t } else { 1.0 },
                },
                reversed: self.seam_b_reversed,
            };
            let length_a = edge_length_meters(&self.pattern_project, &edge_a);
            let length_b = edge_length_meters(&self.pattern_project, &edge_b);
            let difference = length_a - length_b;
            ui.label(format!("Comprimento A: {:.1} cm · B: {:.1} cm", length_a * 100.0, length_b * 100.0));
            ui.label(format!("Diferença geométrica A − B: {:+.1} cm · alvo: {:+.1} cm",
                difference * 100.0, self.seam_ease_m * 100.0));
            let shortest = length_a.min(length_b);
            let longest = length_a.max(length_b);
            if shortest > 1.0e-9 && longest / shortest > 3.0 {
                ui.colored_label(
                    egui::Color32::from_rgb(255, 175, 80),
                    format!("Aviso: a razão A/B é {:.2}:1 (acima de 3:1). Confira os segmentos; a costura continuará permitida.",
                        longest / shortest),
                );
            }
            if (difference - self.seam_ease_m).abs() > 0.02 {
                ui.colored_label(
                    egui::Color32::from_rgb(255, 190, 80),
                    "A diferença medida não coincide com o alvo; a simulação ajustará o repouso da borda excedente.",
                );
            }

            let notches_a = edge_notch_positions(&self.pattern_project, &edge_a);
            let notches_b = edge_notch_positions(&self.pattern_project, &edge_b);
            if notches_a.is_empty() && notches_b.is_empty() {
                ui.weak("Piques pareados: nenhum pique nas bordas selecionadas.");
            } else if notches_a.len() != notches_b.len() {
                ui.colored_label(
                    egui::Color32::from_rgb(255, 150, 90),
                    format!("Piques incompatíveis: A tem {}, B tem {}. Os que existirem serão pareados por ordem.",
                        notches_a.len(), notches_b.len()),
                );
            } else {
                let max_delta = notches_a.iter().zip(&notches_b)
                    .map(|(a, b)| (a - b).abs()).fold(0.0_f32, f32::max);
                if max_delta > 0.03 {
                    ui.colored_label(egui::Color32::from_rgb(255, 150, 90),
                        format!("Piques desalinhados: diferença de posição normalizada máxima {:.1}%.", max_delta * 100.0));
                } else {
                    ui.colored_label(egui::Color32::LIGHT_GREEN,
                        format!("Piques alinhados: {} correspondências.", notches_a.len()));
                }
            }

            ui.horizontal(|ui| {
                if self.selected_seam_id.is_some() {
                    if ui.button("Salvar alterações").clicked() {
                        self.update_selected_seam_from_controls();
                    }
                    if ui.button("Descosturar").clicked() {
                        self.remove_selected_seam();
                    }
                } else if ui.button("Criar costura").clicked() {
                    self.add_seam_from_controls();
                }
                if ui.button("Nova costura").clicked() {
                    self.reset_seam_controls();
                }
            });
            ui.label(format!("Costuras no projeto: {}", self.pattern_project.seams.len()));
        } else {
            ui.weak("Crie ao menos duas peças para configurar uma costura.");
        }

        if !self.pattern_project.pieces.is_empty() {
            ui.separator();
            ui.collapsing("Piques e correspondência", |ui| {
                let piece_index = self.selected_piece.min(self.pattern_project.pieces.len() - 1);
                self.selected_piece = piece_index;
                let piece_id = self.pattern_project.pieces[piece_index].id;
                let segment_count = self.pattern_project.pieces[piece_index].outline.segment_count().max(1);
                self.notch_segment = self.notch_segment.min(segment_count - 1);
                egui::ComboBox::from_id_salt("notch_segment")
                    .selected_text(format!("Borda: segmento {}", self.notch_segment + 1))
                    .show_ui(ui, |ui| {
                        for index in 0..segment_count {
                            ui.selectable_value(&mut self.notch_segment, index, format!("Segmento {}", index + 1));
                        }
                    });
                ui.add(egui::Slider::new(&mut self.notch_t, 0.01..=0.99).text("Posição no segmento"));
                ui.add(egui::Slider::new(&mut self.notch_depth_m, 0.002..=0.04).text("Profundidade (m)"));
                ui.add(egui::Slider::new(&mut self.notch_width_m, 0.003..=0.05).text("Largura (m)"));

                let notches = self.pattern_project.pieces[piece_index].notches.clone();
                if !notches.is_empty() {
                    self.selected_notch_index = self.selected_notch_index.min(notches.len() - 1);
                    let mut selected_notch = self.selected_notch_index;
                    egui::ComboBox::from_id_salt("selected_notch")
                        .selected_text(format!("Pique {}", notches[selected_notch].id))
                        .show_ui(ui, |ui| {
                            for (index, notch) in notches.iter().enumerate() {
                                ui.selectable_value(&mut selected_notch, index,
                                    format!("Pique {} · seg. {} · t={:.2}", notch.id, notch.segment_index + 1, notch.t));
                            }
                        });
                    if selected_notch != self.selected_notch_index {
                        self.selected_notch_index = selected_notch;
                        let notch = notches[selected_notch].clone();
                        self.notch_segment = notch.segment_index as usize;
                        self.notch_t = notch.t;
                        self.notch_depth_m = notch.depth_meters;
                        self.notch_width_m = notch.width_meters;
                    }
                    ui.horizontal(|ui| {
                        if ui.button("Aplicar edição do pique").clicked() {
                            self.update_notch_from_controls();
                        }
                        if ui.button("Remover pique").clicked() {
                            self.remove_selected_notch();
                        }
                    });
                } else {
                    ui.weak("Esta peça ainda não possui piques.");
                }
                if ui.button("Adicionar pique à borda").clicked() {
                    self.add_notch_from_controls();
                }
                let _ = piece_id;
                ui.weak("Piques posicionados no mesmo progresso de arco são pareados automaticamente entre as bordas da costura.");
            });
        }

        ui.separator();
        let avatar_before = self.pattern_project.avatar.clone();
        let mut avatar_changed = false;
        ui.collapsing("Avatar 3D e colisão corporal", |ui| {
            ui.label("Manequim procedural usado na montagem 3D e na colisão tecido-corpo.");
            avatar_changed |= ui.add(egui::Slider::new(&mut self.pattern_project.avatar.height_meters, 1.20..=2.20)
                .text("Altura (m)")).changed();
            avatar_changed |= ui.add(egui::Slider::new(&mut self.pattern_project.avatar.shoulder_width_meters, 0.25..=0.70)
                .text("Largura dos ombros (m)")).changed();
            avatar_changed |= ui.add(egui::Slider::new(&mut self.pattern_project.avatar.waist_width_meters, 0.18..=0.60)
                .text("Largura da cintura (m)")).changed();
            avatar_changed |= ui.add(egui::Slider::new(&mut self.pattern_project.avatar.hip_width_meters, 0.22..=0.70)
                .text("Largura do quadril (m)")).changed();
            ui.weak("Usa 18 colisores SDF de elipsoides e cápsulas. Importação de malhas OBJ/FBX/glTF ainda não está implementada.");
        });
        if avatar_changed {
            let mut before = self.editor_snapshot();
            before.project.avatar = avatar_before;
            self.remember_pattern_change(before);
        }

        ui.separator();
        if ui.button("Montar e simular na viewport 3D").clicked() {
            match self.start_pattern_simulation() {
                Ok(()) => {},
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
        let piece_index = target.piece();
        let Some(piece_id) = self.pattern_project.pieces.get(piece_index).map(|piece| piece.id) else {
            return;
        };
        {
            let Some(piece) = self.pattern_project.pieces.get_mut(piece_index) else { return; };
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
        self.sync_symmetric_mate(piece_id);
    }

    fn insert_selected_vertex(&mut self) {
        let Some(PatternEditTarget::Edge { piece, segment, t }) = self.selected_target else {
            self.pattern_message = "Selecione uma borda no modo Editar antes de inserir um vértice.".into();
            return;
        };
        let Some(id) = self.pattern_project.pieces.get(piece).map(|piece| piece.id) else { return; };
        let before = self.editor_snapshot();
        let mut project = self.pattern_project.clone();
        let mate_id = symmetry_counterpart_id(&project, id);
        let result = split_project_segment(&mut project, id, segment, t);
        let new_anchor = match result {
            Ok(index) => index,
            Err(error) => { self.pattern_message = error; return; }
        };
        if let Some(mate_id) = mate_id {
            if let Err(error) = split_project_segment(&mut project, mate_id, segment, t) {
                self.pattern_message = format!("Não foi possível inserir o vértice nos dois moldes; nenhuma alteração foi aplicada: {error}");
                return;
            }
        }
        self.pattern_project = project;
        self.selected_target = Some(PatternEditTarget::Anchor { piece, index: new_anchor });
        self.pattern_message = if mate_id.is_some() {
            "Vértice inserido e refletido no molde vinculado."
        } else {
            "Vértice inserido; a curva mantém sua forma original."
        }.into();
        self.remember_pattern_change(before);
    }

    fn convert_selected_edge_to_bezier(&mut self) {
        let Some(PatternEditTarget::Edge { piece, segment, .. }) = self.selected_target else {
            self.pattern_message = "Selecione um segmento para convertê-lo em Bézier.".into();
            return;
        };
        let Some(id) = self.pattern_project.pieces.get(piece).map(|piece| piece.id) else { return; };
        let before = self.editor_snapshot();
        let mut project = self.pattern_project.clone();
        let mate_id = symmetry_counterpart_id(&project, id);
        if let Err(error) = convert_project_segment_to_bezier(&mut project, id, segment) {
            self.pattern_message = error;
            return;
        }
        if let Some(mate_id) = mate_id {
            if let Err(error) = convert_project_segment_to_bezier(&mut project, mate_id, segment) {
                self.pattern_message = format!("Conversão cancelada para preservar a simetria: {error}");
                return;
            }
        }
        self.pattern_project = project;
        self.selected_target = Some(PatternEditTarget::Control1 { piece, segment });
        self.pattern_message = if mate_id.is_some() {
            "Segmento convertido em Bézier nos dois moldes. Arraste as alças para editar."
        } else {
            "Segmento convertido. Arraste as alças roxas para editar a curva."
        }.into();
        self.remember_pattern_change(before);
    }

    fn delete_selected_vertex(&mut self) {
        let Some(PatternEditTarget::Anchor { piece, index }) = self.selected_target else { return; };
        let Some(id) = self.pattern_project.pieces.get(piece).map(|piece| piece.id) else { return; };
        let before = self.editor_snapshot();
        let mut project = self.pattern_project.clone();
        let mate_id = symmetry_counterpart_id(&project, id);
        if let Err(error) = delete_project_anchor(&mut project, id, index) {
            self.pattern_message = error;
            return;
        }
        if let Some(mate_id) = mate_id {
            if let Err(error) = delete_project_anchor(&mut project, mate_id, index) {
                self.pattern_message = format!("Exclusão cancelada para preservar a simetria: {error}");
                return;
            }
        }
        self.pattern_project = project;
        self.selected_target = None;
        self.dragging_target = None;
        self.pattern_message = if mate_id.is_some() {
            "Vértice removido nos dois moldes vinculados."
        } else {
            "Vértice removido."
        }.into();
        self.remember_pattern_change(before);
    }

    fn delete_selected_piece(&mut self) {
        if self.pattern_project.pieces.is_empty() { return; }
        let before = self.editor_snapshot();
        self.selected_piece = self.selected_piece.min(self.pattern_project.pieces.len() - 1);
        let id = self.pattern_project.pieces[self.selected_piece].id;
        self.pattern_project.pieces.remove(self.selected_piece);
        self.pattern_project.seams.retain(|seam| !seam.sides_a.iter().chain(seam.sides_b.iter()).any(|edge| edge.piece_id == id));
        self.pattern_project.symmetry_links.retain(|link| {
            link.piece_a_id != id && link.piece_b_id != id
        });
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

    fn set_symmetry_axis_to_min_x(&mut self) {
        let Some(piece) = self.pattern_project.pieces.get(self.selected_piece) else {
            self.pattern_message = "Crie ou selecione um molde primeiro.".into();
            return;
        };
        let points = sample_curve(&piece.outline, 24);
        if let Some(min_x) = points.iter().map(|p| p.x).reduce(f64::min) {
            self.pattern_symmetry_axis_x = min_x;
            self.pattern_message = format!("Eixo de simetria definido em X = {min_x:.3} m.");
        }
    }

    fn create_mirrored_pair(&mut self) {
        if !self.pattern_symmetry_axis_x.is_finite() {
            self.pattern_message = "O eixo de simetria precisa ser um valor finito.".into();
            return;
        }
        let Some(source) = self.pattern_project.pieces.get(self.selected_piece).cloned() else {
            self.pattern_message = "Crie ou selecione um molde primeiro.".into();
            return;
        };
        if self.pattern_project.symmetry_links.iter().any(|link| {
            link.piece_a_id == source.id || link.piece_b_id == source.id
        }) {
            self.pattern_message = "O molde selecionado já possui um par vinculado.".into();
            return;
        }

        let before = self.editor_snapshot();
        let new_id = self.pattern_project.pieces.iter().map(|piece| piece.id)
            .max().unwrap_or(0) + 1;
        let mirrored_name = format!("{} (espelho)", source.name);
        let mirrored = match mirror_pattern_piece(
            &source, new_id, mirrored_name, self.pattern_symmetry_axis_x,
        ) {
            Ok(piece) => piece,
            Err(error) => { self.pattern_message = error; return; }
        };

        let mut project = self.pattern_project.clone();
        project.pieces.push(mirrored);
        project.symmetry_links.push(SymmetryLink {
            piece_a_id: source.id,
            piece_b_id: new_id,
            axis_x: self.pattern_symmetry_axis_x,
        });

        // Replica a costura M:N apenas quando todos os moldes dos dois lados têm contrapartes.
        let original_seams = project.seams.clone();
        let mut next_seam_id = project.seams.iter().map(|seam| seam.id).max().unwrap_or(0) + 1;
        for seam in original_seams {
            let mut mirrored_seam = seam.clone();
            let mut changed = false;
            let mut complete = true;
            for edge in mirrored_seam.sides_a.iter_mut().chain(mirrored_seam.sides_b.iter_mut()) {
                let Some(mate) = symmetry_counterpart_id(&project, edge.piece_id) else {
                    complete = false; break;
                };
                changed |= mate != edge.piece_id;
                edge.piece_id = mate;
            }
            if !complete || !changed { continue; }
            mirrored_seam.id = next_seam_id;
            mirrored_seam.name = format!("{} (espelho)", seam.name);
            project.seams.push(mirrored_seam);
            next_seam_id += 1;
        }

        if let Err(error) = project.validate() {
            self.pattern_message = format!("Não foi possível criar o par espelhado: {error}");
            return;
        }
        self.pattern_project = project;
        self.selected_piece = self.pattern_project.pieces.len() - 1;
        self.selected_target = Some(PatternEditTarget::Piece { piece: self.selected_piece });
        self.pattern_fit_pending = true;
        self.pattern_message = format!(
            "Par espelhado criado para '{}'; edições de geometria serão sincronizadas nos dois sentidos.",
            source.name
        );
        self.remember_pattern_change(before);
    }

    fn unlink_selected_symmetry(&mut self) {
        let Some(piece) = self.pattern_project.pieces.get(self.selected_piece) else { return; };
        let id = piece.id;
        let before = self.editor_snapshot();
        let old_len = self.pattern_project.symmetry_links.len();
        self.pattern_project.symmetry_links.retain(|link| {
            link.piece_a_id != id && link.piece_b_id != id
        });
        if self.pattern_project.symmetry_links.len() == old_len {
            self.pattern_message = "O molde selecionado não possui vínculo de simetria.".into();
            return;
        }
        self.pattern_message = "Par desvinculado. Os dois moldes permanecem na cena como peças independentes.".into();
        self.remember_pattern_change(before);
    }

    fn sync_symmetric_mate(&mut self, piece_id: u64) {
        let Some(link) = self.pattern_project.symmetry_links.iter()
            .find(|link| link.piece_a_id == piece_id || link.piece_b_id == piece_id)
            .cloned() else { return; };
        let mate_id = if link.piece_a_id == piece_id { link.piece_b_id } else { link.piece_a_id };
        let Some(source) = self.pattern_project.pieces.iter().find(|piece| piece.id == piece_id).cloned() else {
            return;
        };
        let Some(target_name) = self.pattern_project.pieces.iter()
            .find(|piece| piece.id == mate_id).map(|piece| piece.name.clone()) else {
            return;
        };
        match mirror_pattern_piece(&source, mate_id, target_name, link.axis_x) {
            Ok(mirrored) => {
                if let Some(target) = self.pattern_project.pieces.iter_mut().find(|piece| piece.id == mate_id) {
                    *target = mirrored;
                }
            }
            Err(error) => self.pattern_message = format!("Falha ao sincronizar a simetria: {error}"),
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
        let piece_id = self.pattern_project.pieces[piece_index].id;
        self.pattern_project.pieces[piece_index].internal_lines.push(curve);
        self.sync_symmetric_mate(piece_id);
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
        let piece_id = self.pattern_project.pieces[piece_index].id;
        self.pattern_project.pieces[piece_index].fold_lines.push(FoldLine {
            curve,
            kind: FoldKind::Valley,
            stiffness: 0.35,
        });
        self.sync_symmetric_mate(piece_id);
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
        let piece_id = self.pattern_project.pieces[piece_index].id;
        self.pattern_project.pieces[piece_index].pleats.push(Pleat {
            axis,
            depth_meters: 0.02,
            angle_radians: std::f64::consts::FRAC_PI_4,
        });
        self.sync_symmetric_mate(piece_id);
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
        let piece_id = self.pattern_project.pieces[piece_index].id;
        self.pattern_project.pieces[piece_index].grain_line = Some(GrainLine { start, end });
        self.sync_symmetric_mate(piece_id);
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

        let symmetry_link = self.pattern_project.symmetry_links.iter()
            .find(|link| link.piece_a_id == source.id || link.piece_b_id == source.id)
            .cloned();
        let max_piece_id = self.pattern_project.pieces.iter()
            .map(|piece| piece.id).max().unwrap_or(0);
        let second_id = max_piece_id + 1;

        // If the source has a linked mirror, cut the mate with the reflected path in the same
        // transaction. This keeps both sides topologically paired instead of silently diverging.
        let paired_cut: Option<(usize, PatternPiece, CutAndSewResult, u64, f64)> =
            if let Some(link) = &symmetry_link {
                let mate_id = if link.piece_a_id == source.id { link.piece_b_id } else { link.piece_a_id };
                let Some(mate_index) = self.pattern_project.pieces.iter()
                    .position(|piece| piece.id == mate_id) else {
                    self.pattern_message = "O par espelhado referencia um molde inexistente.".into();
                    return;
                };
                let mate_source = self.pattern_project.pieces[mate_index].clone();
                let mirrored_path = match mirror_curve_x(&cut_path, link.axis_x) {
                    Ok(path) => path,
                    Err(error) => { self.pattern_message = error.to_string(); return; }
                };
                let mate_result = match cut_closed_outline(&mate_source.outline, &mirrored_path, 0.035) {
                    Ok(result) => result,
                    Err(error) => {
                        self.pattern_message = format!(
                            "O corte do par espelhado não pôde ser validado ({error}); nenhum dos moldes foi alterado."
                        );
                        return;
                    }
                };
                Some((mate_index, mate_source, mate_result, max_piece_id + 2, link.axis_x))
            } else { None };

        let first_id = source.id;
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

        let mut mate_pieces: Option<(PatternPiece, PatternPiece, usize, u64, Option<usize>)> = None;
        if let Some((mate_index, mate_source, mate_result, mate_second_id, _)) = &paired_cut {
            let mut mate_first = mate_source.clone();
            let mut mate_second = mate_source.clone();
            mate_first.name = format!("{} (A)", mate_source.name);
            mate_second.id = *mate_second_id;
            mate_second.name = format!("{} (B)", mate_source.name);
            mate_first.outline = mate_result.first.clone();
            mate_second.outline = mate_result.second.clone();
            let mate_polygon_a = sample_curve(&mate_first.outline, 24);
            let mate_polygon_b = sample_curve(&mate_second.outline, 24);
            let mate_sewing_side = match split_piece_metadata(
                mate_source, &mut mate_first, &mut mate_second, mate_result,
                &mate_polygon_a, &mate_polygon_b,
            ) {
                Ok(side) => side,
                Err(error) => {
                    self.pattern_message = format!(
                        "Cut & Sew cancelado: não foi possível particionar os metadados do molde espelhado: {error}"
                    );
                    return;
                }
            };
            mate_pieces = Some((
                mate_first,
                mate_second,
                *mate_index,
                mate_source.id,
                mate_sewing_side,
            ));
        }

        let mut project = self.pattern_project.clone();
        // Remap existing references on both source pieces before replacing them. If a previous
        // seam crosses a segment split by the new cut, reject atomically rather than corrupt it.
        for seam in &mut project.seams {
            for edge in seam.sides_a.iter_mut().chain(seam.sides_b.iter_mut()) {
                let target = if edge.piece_id == first_id {
                    Some((&cut_result, first_id, second_id, sewing_side))
                } else if let Some((_, mate_source, mate_result, mate_second_id, _)) = &paired_cut {
                    if edge.piece_id == mate_source.id {
                        Some((mate_result, mate_source.id, *mate_second_id,
                            mate_pieces.as_ref().and_then(|entry| entry.4)))
                    } else { None }
                } else { None };
                let Some((result, base_id, new_id, source_sewing_side)) = target else { continue; };
                match edge.curve {
                    BoundaryCurve::Outline => {
                        if let Err(error) = remap_edge_after_cut(
                            edge, &result.source_segment_fragments, base_id, new_id,
                        ) {
                            self.pattern_message = format!(
                                "Cut & Sew cancelado: {error} A costura original foi preservada."
                            );
                            return;
                        }
                    }
                    BoundaryCurve::SewingLine => {
                        let Some(side) = source_sewing_side else {
                            self.pattern_message = "Cut & Sew cancelado: uma costura existente usa uma linha de costura que não pôde ser atribuída a um dos lados.".into();
                            return;
                        };
                        edge.piece_id = if side == 0 { base_id } else { new_id };
                    }
                }
            }
        }

        project.pieces[source_index] = first;
        project.pieces.push(second);
        if let Some((mate_first, mate_second, mate_index, mate_source_id, _)) = mate_pieces {
            project.pieces[mate_index] = mate_first;
            project.pieces.push(mate_second);

            if let Some((_, _, _, mate_second_id, axis_x)) = &paired_cut {
                project.symmetry_links.push(SymmetryLink {
                    piece_a_id: second_id,
                    piece_b_id: *mate_second_id,
                    axis_x: *axis_x,
                });
            }
            let _ = mate_source_id;
        }

        let mut next_seam_id = project.seams.iter()
            .map(|seam| seam.id).max().unwrap_or(0) + 1;
        let (a_start, a_end) = cut_result.first_cut_segment_range;
        let (b_start, b_end) = cut_result.second_cut_segment_range;
        project.seams.push(SeamDefinition {
            id: next_seam_id,
            name: format!("Cut & Sew {next_seam_id}"),
            a: EdgeRef {
                piece_id: first_id, curve: BoundaryCurve::Outline,
                segment_start: a_start as u32, segment_end: a_end as u32, reversed: false,
            }.into(),
            b: EdgeRef {
                piece_id: second_id, curve: BoundaryCurve::Outline,
                segment_start: b_start as u32, segment_end: b_end as u32, reversed: true,
            }.into(),
            ease_meters: 0.0,
        });
        next_seam_id += 1;

        if let Some((_, mate_source, mate_result, mate_second_id, _)) = &paired_cut {
            let (mate_a_start, mate_a_end) = mate_result.first_cut_segment_range;
            let (mate_b_start, mate_b_end) = mate_result.second_cut_segment_range;
            project.seams.push(SeamDefinition {
                id: next_seam_id,
                name: format!("Cut & Sew {next_seam_id}"),
                a: EdgeRef {
                    piece_id: mate_source.id, curve: BoundaryCurve::Outline,
                    segment_start: mate_a_start as u32, segment_end: mate_a_end as u32, reversed: false,
                }.into(),
                b: EdgeRef {
                    piece_id: *mate_second_id, curve: BoundaryCurve::Outline,
                    segment_start: mate_b_start as u32, segment_end: mate_b_end as u32, reversed: true,
                }.into(),
                ease_meters: 0.0,
            });
        }

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
        self.pattern_message = if paired_cut.is_some() {
            format!(
                "Cut & Sew concluído nos dois moldes espelhados: '{}' foi dividido em quatro peças, com duas novas costuras e o vínculo de simetria preservado.",
                source.name
            )
        } else {
            format!(
                "Cut & Sew concluído: '{}' foi dividido em duas peças, com a nova costura {}.",
                source.name, next_seam_id - 1
            )
        };
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

    fn reset_seam_controls(&mut self) {
        self.selected_seam_id = None;
        self.seam_name.clear();
        self.seam_ease_m = 0.0;
        self.seam_a_segment = 0;
        self.seam_b_segment = 0;
        self.seam_a_segment_end = 1;
        self.seam_b_segment_end = 1;
        self.seam_sides_a.clear();
        self.seam_sides_b.clear();
        self.seam_a_active_path = 0;
        self.seam_b_active_path = 0;
        self.seam_active_side_a = true;
        self.seam_a_reversed = false;
        self.seam_b_reversed = false;
        self.seam_a_start_t = 0.0;
        self.seam_a_end_t = 1.0;
        self.seam_b_start_t = 0.0;
        self.seam_b_end_t = 1.0;
        self.seam_free_mode = false;
        self.dragging_seam_handle = None;
        self.seam_drag_before = None;
    }

    fn load_seam_edge_to_controls(&mut self, is_a: bool, edge: &EdgePath) {
        let Some(index) = self.pattern_project.pieces.iter().position(|piece| piece.id == edge.piece_id) else {
            return;
        };
        let segment_count = self.pattern_project.pieces[index].outline.segment_count().max(1);
        let start = (edge.start.segment as usize).min(segment_count - 1);
        let end = (edge.end.segment as usize
            + usize::from(edge.end.t >= 1.0 - 1.0e-9)).clamp(start + 1, segment_count);
        if is_a {
            self.seam_a_piece = index;
            self.seam_a_segment = start;
            self.seam_a_segment_end = end;
            self.seam_a_start_t = edge.start.t;
            self.seam_a_end_t = edge.end.t;
            self.seam_a_reversed = edge.reversed;
        } else {
            self.seam_b_piece = index;
            self.seam_b_segment = start;
            self.seam_b_segment_end = end;
            self.seam_b_start_t = edge.start.t;
            self.seam_b_end_t = edge.end.t;
            self.seam_b_reversed = edge.reversed;
        }
        self.seam_free_mode |= edge.start.t > 1.0e-9 || edge.end.t < 1.0 - 1.0e-9;
    }

    fn load_active_chain_edge_to_controls(&mut self, is_a: bool) {
        let index = if is_a { self.seam_a_active_path } else { self.seam_b_active_path };
        let edge = if is_a { self.seam_sides_a.get(index) } else { self.seam_sides_b.get(index) }.cloned();
        if let Some(edge) = edge { self.load_seam_edge_to_controls(is_a, &edge); }
    }

    fn current_seam_sides_with_draft(&self) -> Result<(Vec<EdgePath>, Vec<EdgePath>), String> {
        let (draft_a, draft_b) = self.seam_edges_from_controls()?;
        let merge = |stored: &[EdgePath], active: usize, draft: EdgePath| {
            let mut result = stored.to_vec();
            if result.is_empty() { result.push(draft); }
            else if active < result.len() { result[active] = draft; }
            else { result.push(draft); }
            result
        };
        Ok((
            merge(&self.seam_sides_a, self.seam_a_active_path, draft_a),
            merge(&self.seam_sides_b, self.seam_b_active_path, draft_b),
        ))
    }

    fn add_draft_edge_to_side(&mut self, is_a: bool, edge: EdgePath) {
        let target = if is_a { &mut self.seam_sides_a } else { &mut self.seam_sides_b };
        if target.iter().any(|existing| existing.piece_id == edge.piece_id
            && existing.curve == edge.curve
            && existing.start_position() < edge.end_position() - 1.0e-9
            && edge.start_position() < existing.end_position() - 1.0e-9)
        {
            self.pattern_message = "Essa borda já está presente ou sobreposta no lado da costura.".into();
            return;
        }
        target.push(edge);
        if is_a { self.seam_a_active_path = target.len() - 1; }
        else { self.seam_b_active_path = target.len() - 1; }
        self.seam_active_side_a = is_a;
        self.load_active_chain_edge_to_controls(is_a);
        self.pattern_message = format!("Borda adicionada ao lado {}.", if is_a { "A" } else { "B" });
    }

    fn seam_edges_from_controls(&self) -> Result<(EdgePath, EdgePath), String> {
        let count = self.pattern_project.pieces.len();
        if count < 2 { return Err("Crie pelo menos duas peças antes de costurá-las.".into()); }
        let a_piece = &self.pattern_project.pieces[self.seam_a_piece.min(count - 1)];
        let b_piece = &self.pattern_project.pieces[self.seam_b_piece.min(count - 1)];
        let make_edge = |piece: &PatternPiece, start: usize, end: usize,
                         start_t: f64, end_t: f64, reversed: bool| {
            let segment_count = piece.outline.segment_count();
            let start = start.min(segment_count.saturating_sub(1));
            let end = end.clamp(start + 1, segment_count);
            EdgePath {
                piece_id: piece.id,
                curve: BoundaryCurve::Outline,
                start: CurveParam {
                    segment: start as u32,
                    t: if self.seam_free_mode { start_t.clamp(0.0, 1.0) } else { 0.0 },
                },
                end: CurveParam {
                    segment: (end - 1) as u32,
                    t: if self.seam_free_mode { end_t.clamp(0.0, 1.0) } else { 1.0 },
                },
                reversed,
            }
        };
        Ok((
            make_edge(a_piece, self.seam_a_segment, self.seam_a_segment_end,
                self.seam_a_start_t, self.seam_a_end_t, self.seam_a_reversed),
            make_edge(b_piece, self.seam_b_segment, self.seam_b_segment_end,
                self.seam_b_start_t, self.seam_b_end_t, self.seam_b_reversed),
        ))
    }

    fn hit_test_seam_handle<F: Fn(PatternPoint2) -> egui::Pos2>(
        &self,
        pointer: egui::Pos2,
        to_screen: &F,
    ) -> Option<(bool, bool)> {
        if !self.seam_free_mode { return None; }
        let (edge_a, edge_b) = self.seam_edges_from_controls().ok()?;
        for (is_a, edge) in [(true, edge_a), (false, edge_b)] {
            let curve = edge_curve(&self.pattern_project, &edge)?;
            let start = curve_segment_point(curve, edge.start.segment as usize, edge.start.t)?;
            let end = curve_segment_point(curve, edge.end.segment as usize, edge.end.t)?;
            if pointer.distance(to_screen(start)) <= 13.0 { return Some((is_a, true)); }
            if pointer.distance(to_screen(end)) <= 13.0 { return Some((is_a, false)); }
        }
        None
    }

    fn update_seam_handle_from_pointer(&mut self, is_a: bool, is_start: bool, world: PatternPoint2) {
        let piece_index = if is_a { self.seam_a_piece } else { self.seam_b_piece };
        let Some(piece) = self.pattern_project.pieces.get(piece_index) else { return; };
        let curve = &piece.outline;
        let Some((mut param, _)) = closest_curve_param(curve, world) else { return; };
        param = snap_curve_param_to_features(
            curve, &piece.notches, param, world,
            self.seam_snap_tolerance_px as f64 / self.pattern_zoom.max(1.0) as f64,
        );
        let (start_t, end_t, start_segment, end_exclusive) = if is_a {
            (self.seam_a_start_t, self.seam_a_end_t,
             self.seam_a_segment, self.seam_a_segment_end)
        } else {
            (self.seam_b_start_t, self.seam_b_end_t,
             self.seam_b_segment, self.seam_b_segment_end)
        };
        let current_start = CurveParam { segment: start_segment as u32, t: start_t };
        let current_end = CurveParam {
            segment: end_exclusive.saturating_sub(1) as u32, t: end_t,
        };
        let segment_count = curve.segment_count();
        let new_param = if is_start {
            let maximum = current_end.segment as f64 + current_end.t - 1.0e-4;
            param = param_at_curve_position(param.segment as f64 + param.t, segment_count, maximum);
            if param.segment as f64 + param.t >= current_end.segment as f64 + current_end.t {
                return;
            }
            param
        } else {
            let minimum = current_start.segment as f64 + current_start.t + 1.0e-4;
            param = param_at_curve_position(param.segment as f64 + param.t, segment_count, f64::INFINITY);
            if param.segment as f64 + param.t <= minimum {
                param = param_at_curve_position(minimum, segment_count, f64::INFINITY);
            }
            param
        };

        if is_a {
            if is_start {
                self.seam_a_segment = new_param.segment as usize;
                self.seam_a_start_t = new_param.t;
            } else {
                self.seam_a_segment_end = new_param.segment as usize + 1;
                self.seam_a_end_t = new_param.t;
            }
            self.seam_a_segment_end = self.seam_a_segment_end.clamp(self.seam_a_segment + 1, segment_count);
        } else {
            if is_start {
                self.seam_b_segment = new_param.segment as usize;
                self.seam_b_start_t = new_param.t;
            } else {
                self.seam_b_segment_end = new_param.segment as usize + 1;
                self.seam_b_end_t = new_param.t;
            }
            self.seam_b_segment_end = self.seam_b_segment_end.clamp(self.seam_b_segment + 1, segment_count);
        }
    }

    fn commit_seam_handle_drag(&mut self) {
        let Some(before) = self.seam_drag_before.take() else { return; };
        let Some(id) = self.selected_seam_id else { return; };
        let (sides_a, sides_b) = match self.current_seam_sides_with_draft() {
            Ok(sides) => sides,
            Err(error) => { self.pattern_message = error; return; }
        };
        let mut staged = self.pattern_project.clone();
        let Some(seam) = staged.seams.iter_mut().find(|seam| seam.id == id) else { return; };
        seam.sides_a = sides_a;
        seam.sides_b = sides_b;
        if let Err(error) = staged.validate() {
            self.pattern_message = format!("A extremidade não foi aplicada: {error}");
            return;
        }
        self.pattern_project = staged;
        self.remember_pattern_change(before);
        self.pattern_message = format!("Extremidades da costura '{id}' atualizadas.");
    }

    fn add_seam_from_controls(&mut self) {
        let (sides_a, sides_b) = match self.current_seam_sides_with_draft() {
            Ok(sides) => sides,
            Err(error) => { self.pattern_message = error; return; }
        };
        let next_id = self.pattern_project.seams.iter().map(|seam| seam.id).max().unwrap_or(0) + 1;
        let name = if self.seam_name.trim().is_empty() {
            format!("Costura {next_id}")
        } else { self.seam_name.trim().to_owned() };
        let seam = SeamDefinition {
            id: next_id,
            name: name.clone(),
            sides_a,
            sides_b,
            ease_meters: self.seam_ease_m,
        };
        let before = self.editor_snapshot();
        let mut staged = self.pattern_project.clone();
        staged.seams.push(seam);
        if let Err(error) = staged.validate() {
            self.pattern_message = error;
            return;
        }
        self.pattern_project = staged;
        self.selected_seam_id = Some(next_id);
        self.seam_name = name;
        self.pattern_message = "Costura adicionada. Pareamento por arco, folga-alvo e piques serão usados na simulação.".into();
        self.remember_pattern_change(before);
    }

    fn update_selected_seam_from_controls(&mut self) {
        let Some(id) = self.selected_seam_id else {
            self.add_seam_from_controls();
            return;
        };
        let (sides_a, sides_b) = match self.current_seam_sides_with_draft() {
            Ok(sides) => sides,
            Err(error) => { self.pattern_message = error; return; }
        };
        let before = self.editor_snapshot();
        let mut staged = self.pattern_project.clone();
        let Some(seam) = staged.seams.iter_mut().find(|seam| seam.id == id) else {
            self.pattern_message = "A costura selecionada não existe mais.".into();
            self.selected_seam_id = None;
            return;
        };
        seam.name = if self.seam_name.trim().is_empty() {
            format!("Costura {id}")
        } else { self.seam_name.trim().to_owned() };
        seam.sides_a = sides_a;
        seam.sides_b = sides_b;
        seam.ease_meters = self.seam_ease_m;
        if let Err(error) = staged.validate() {
            self.pattern_message = error;
            return;
        }
        self.pattern_project = staged;
        self.pattern_message = format!("Costura '{id}' atualizada.");
        self.remember_pattern_change(before);
    }

    fn remove_selected_seam(&mut self) {
        let Some(id) = self.selected_seam_id else { return; };
        let before = self.editor_snapshot();
        let previous_len = self.pattern_project.seams.len();
        self.pattern_project.seams.retain(|seam| seam.id != id);
        if self.pattern_project.seams.len() == previous_len {
            self.pattern_message = "A costura selecionada não foi encontrada.".into();
            self.selected_seam_id = None;
            return;
        }
        self.selected_seam_id = None;
        self.seam_name.clear();
        self.pattern_message = format!("Costura '{id}' removida do projeto.");
        self.remember_pattern_change(before);
    }

    fn add_notch_from_controls(&mut self) {
        if self.pattern_project.pieces.is_empty() { return; }
        let index = self.selected_piece.min(self.pattern_project.pieces.len() - 1);
        let segment_count = self.pattern_project.pieces[index].outline.segment_count();
        if segment_count == 0 { return; }
        let before = self.editor_snapshot();
        let piece = &mut self.pattern_project.pieces[index];
        let id = piece.notches.iter().map(|notch| notch.id).max().unwrap_or(0) + 1;
        piece.notches.push(Notch {
            id,
            segment_index: self.notch_segment.min(segment_count - 1) as u32,
            t: self.notch_t.clamp(0.01, 0.99),
            depth_meters: self.notch_depth_m.clamp(0.001, 0.10),
            width_meters: self.notch_width_m.clamp(0.001, 0.10),
        });
        self.selected_notch_index = piece.notches.len() - 1;
        self.pattern_message = format!("Pique {id} adicionado ao molde '{}'.", piece.name);
        self.remember_pattern_change(before);
    }

    fn update_notch_from_controls(&mut self) {
        if self.pattern_project.pieces.is_empty() { return; }
        let index = self.selected_piece.min(self.pattern_project.pieces.len() - 1);
        let segment_count = self.pattern_project.pieces[index].outline.segment_count();
        let before = self.editor_snapshot();
        let piece = &mut self.pattern_project.pieces[index];
        let Some(notch) = piece.notches.get_mut(self.selected_notch_index) else { return; };
        notch.segment_index = self.notch_segment.min(segment_count.saturating_sub(1)) as u32;
        notch.t = self.notch_t.clamp(0.01, 0.99);
        notch.depth_meters = self.notch_depth_m.clamp(0.001, 0.10);
        notch.width_meters = self.notch_width_m.clamp(0.001, 0.10);
        self.pattern_message = format!("Pique {} atualizado.", notch.id);
        self.remember_pattern_change(before);
    }

    fn remove_selected_notch(&mut self) {
        if self.pattern_project.pieces.is_empty() { return; }
        let index = self.selected_piece.min(self.pattern_project.pieces.len() - 1);
        if self.selected_notch_index >= self.pattern_project.pieces[index].notches.len() { return; }
        let before = self.editor_snapshot();
        let notch = self.pattern_project.pieces[index].notches.remove(self.selected_notch_index);
        self.selected_notch_index = self.selected_notch_index.min(
            self.pattern_project.pieces[index].notches.len().saturating_sub(1)
        );
        self.pattern_message = format!("Pique {} removido.", notch.id);
        self.remember_pattern_change(before);
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
        let symmetry_screen_x = origin.x + self.pattern_symmetry_axis_x as f32 * scale;
        painter.line_segment(
            [egui::pos2(symmetry_screen_x, rect.top()), egui::pos2(symmetry_screen_x, rect.bottom())],
            egui::Stroke::new(1.0, egui::Color32::from_rgb(154, 111, 72)),
        );
        painter.text(
            egui::pos2(symmetry_screen_x + 4.0, rect.top() + 34.0),
            egui::Align2::LEFT_TOP, "EIXO SIMETRIA",
            egui::FontId::proportional(10.0),
            egui::Color32::from_rgb(198, 153, 108),
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

            // Piques do molde: traço transversal à borda e ponto dourado na posição paramétrica.
            for notch in &piece.notches {
                let segment_index = notch.segment_index as usize;
                if segment_index >= piece.outline.segments.len() { continue; }
                let t = notch.t.clamp(0.0, 1.0);
                let Some(point) = curve_segment_point(&piece.outline, segment_index, t) else { continue; };
                let before = curve_segment_point(&piece.outline, segment_index, (t - 0.005).max(0.0)).unwrap_or(point);
                let after = curve_segment_point(&piece.outline, segment_index, (t + 0.005).min(1.0)).unwrap_or(point);
                let dx = after.x - before.x;
                let dy = after.y - before.y;
                let length = (dx * dx + dy * dy).sqrt().max(1e-9);
                let normal = PatternPoint2::new(-dy / length, dx / length);
                let half_width = (notch.width_meters * 0.5).clamp(0.004, 0.035);
                let a = PatternPoint2::new(point.x - normal.x * half_width, point.y - normal.y * half_width);
                let b = PatternPoint2::new(point.x + normal.x * half_width, point.y + normal.y * half_width);
                let color = egui::Color32::from_rgb(255, 198, 72);
                painter.line_segment([to_screen(a), to_screen(b)], egui::Stroke::new(2.4, color));
                painter.circle_filled(to_screen(point), 3.0, color);
                if selected_piece {
                    painter.text(to_screen(point), egui::Align2::LEFT_BOTTOM,
                        format!("P{}", notch.id), egui::FontId::proportional(9.0), color);
                }
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
                    let pinned = piece.assembly.pinned_anchors.contains(&(anchor_index as u32));
                    let color = if pinned {
                        egui::Color32::from_rgb(255, 80, 95)
                    } else if selected {
                        egui::Color32::WHITE
                    } else {
                        egui::Color32::from_rgb(255, 190, 75)
                    };
                    painter.circle_filled(screen, if selected || pinned { 6.0 } else { 4.0 }, color);
                    if selected || pinned {
                        painter.circle_stroke(
                            screen, 9.0,
                            egui::Stroke::new(1.5, if pinned { color } else { egui::Color32::from_rgb(255, 190, 75) }),
                        );
                    }
                }
            }
        }

        // As costuras são sobrepostas às bordas. Ao passar o cursor sobre qualquer lado da
        // costura selecionada, ambos os lados ganham destaque, setas de direção e ligações entre piques.
        let hovered_pointer = ui.input(|input| input.pointer.hover_pos());
        for seam in &self.pattern_project.seams {
            let selected = self.selected_seam_id == Some(seam.id);
            let (draw_a, draw_b) = if selected {
                self.seam_edges_from_controls().unwrap_or((seam.a.clone(), seam.b.clone()))
            } else {
                (seam.a.clone(), seam.b.clone())
            };
            let la = edge_length_meters(&self.pattern_project, &draw_a);
            let lb = edge_length_meters(&self.pattern_project, &draw_b);
            let mismatch = (la - lb - seam.ease_meters).abs();
            let path_a = edge_path_points(&self.pattern_project, &draw_a, 24);
            let path_b = edge_path_points(&self.pattern_project, &draw_b, 24);
            let screen_a = path_a.iter().copied().map(to_screen).collect::<Vec<_>>();
            let screen_b = path_b.iter().copied().map(to_screen).collect::<Vec<_>>();
            let hovering_selected_seam = selected && response.hovered()
                && hovered_pointer.map(|pointer| {
                    distance_to_screen_path(pointer, &screen_a).min(distance_to_screen_path(pointer, &screen_b)) <= 9.0
                }).unwrap_or(false);
            let color = if mismatch > 0.02 {
                egui::Color32::from_rgb(255, 92, 92)
            } else if selected || hovering_selected_seam {
                egui::Color32::from_rgb(255, 220, 110)
            } else {
                egui::Color32::from_rgb(64, 221, 206)
            };
            for path in [&screen_a, &screen_b] {
                for pair in path.windows(2) {
                    painter.line_segment(
                        [pair[0], pair[1]],
                        egui::Stroke::new(if selected || hovering_selected_seam { 4.0 } else { 2.6 }, color),
                    );
                }
            }
            if hovering_selected_seam {
                draw_path_direction_arrow(&painter, &screen_a, color);
                draw_path_direction_arrow(&painter, &screen_b, color);
                for (progress_a, progress_b) in paired_notch_markers(&self.pattern_project, &draw_a, &draw_b) {
                    if let (Some(a), Some(b)) = (
                        point_on_path_at_progress(&path_a, progress_a),
                        point_on_path_at_progress(&path_b, progress_b),
                    ) {
                        painter.line_segment(
                            [to_screen(a), to_screen(b)],
                            egui::Stroke::new(1.1, egui::Color32::from_rgba_unmultiplied(255, 225, 130, 180)),
                        );
                    }
                }
            }
            if selected {
                if let (Some(a), Some(b)) = (path_a.get(path_a.len()/2), path_b.get(path_b.len()/2)) {
                    painter.text(to_screen(*a), egui::Align2::CENTER_BOTTOM,
                        format!("C{} · {:.1} cm", seam.id, mismatch * 100.0),
                        egui::FontId::proportional(10.0), color);
                    painter.text(to_screen(*b), egui::Align2::CENTER_TOP,
                        format!("C{}", seam.id), egui::FontId::proportional(10.0), color);
                }
            }
        }

        // Preview da nova costura e alças arrastáveis do Free Sewing, inclusive antes de salvar.
        if self.seam_free_mode {
            if let Ok((edge_a, edge_b)) = self.seam_edges_from_controls() {
                let tint_a = egui::Color32::from_rgb(255, 190, 74);
                let tint_b = egui::Color32::from_rgb(94, 224, 207);
                for (is_a, edge, color) in [(true, edge_a, tint_a), (false, edge_b, tint_b)] {
                    let path = edge_path_points(&self.pattern_project, &edge, 24);
                    let screen_path = path.iter().copied().map(to_screen).collect::<Vec<_>>();
                    for pair in screen_path.windows(2) {
                        painter.line_segment([pair[0], pair[1]], egui::Stroke::new(3.0, color));
                    }
                    if let Some(curve) = edge_curve(&self.pattern_project, &edge) {
                        for (is_start, param) in [(true, edge.start), (false, edge.end)] {
                            if let Some(point) = curve_segment_point(curve, param.segment as usize, param.t) {
                                let screen = to_screen(point);
                                painter.circle_filled(screen, 6.0, color);
                                painter.circle_stroke(screen, 9.0, egui::Stroke::new(1.5, egui::Color32::WHITE));
                                painter.text(screen + egui::vec2(8.0, if is_start { -8.0 } else { 8.0 }),
                                    egui::Align2::LEFT_CENTER,
                                    format!("{}{}", if is_a { "A" } else { "B" }, if is_start { " início" } else { " fim" }),
                                    egui::FontId::proportional(9.0), color);
                            }
                        }
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
                    if let Some(handle) = self.hit_test_seam_handle(pointer, &to_screen) {
                        self.dragging_seam_handle = Some(handle);
                        self.seam_drag_before = Some(self.editor_snapshot());
                        self.dragging_target = None;
                        self.dragging_before = None;
                        self.drag_last_world = None;
                    } else {
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
            }
            if response.dragged_by(egui::PointerButton::Primary) {
                if let (Some((is_a, is_start)), Some(pointer)) = (self.dragging_seam_handle, response.interact_pointer_pos()) {
                    self.update_seam_handle_from_pointer(is_a, is_start, to_world(pointer));
                } else if let (Some(target), Some(pointer)) = (self.dragging_target, response.interact_pointer_pos()) {
                    let mut world = to_world(pointer);
                    if self.pattern_snap {
                        world.x = (world.x * 100.0).round() / 100.0;
                        world.y = (world.y * 100.0).round() / 100.0;
                    }
                    if let PatternEditTarget::Piece { piece } = target {
                        if let Some(previous) = self.drag_last_world {
                            if let Some(id) = self.pattern_project.pieces.get(piece).map(|data| data.id) {
                                translate_project_piece(
                                    &mut self.pattern_project,
                                    id,
                                    world.x - previous.x,
                                    world.y - previous.y,
                                );
                                self.sync_symmetric_mate(id);
                            }
                        }
                    } else {
                        self.move_pattern_target(target, world);
                    }
                    self.drag_last_world = Some(world);
                }
            }
            if response.drag_stopped_by(egui::PointerButton::Primary) {
                if self.dragging_seam_handle.take().is_some() {
                    self.commit_seam_handle_drag();
                    self.seam_drag_before = None;
                } else if let Some(before) = self.dragging_before.take() {
                    self.remember_pattern_change(before);
                }
                self.dragging_target = None;
                self.drag_last_world = None;
            }
            if response.clicked() && self.dragging_target.is_none() && self.dragging_seam_handle.is_none() {
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

        // Força todos os piques e extremos de costura para dentro da amostragem de contorno
        // antes da triangulação. As extremidades não são aproximadas ao vértice vizinho.
        let mut forced_by_piece: std::collections::HashMap<u64, Vec<(u32, f64)>> =
            std::collections::HashMap::new();
        for piece in &self.pattern_project.pieces {
            forced_by_piece.entry(piece.id).or_default()
                .extend(piece.notches.iter().map(|notch| (notch.segment_index, notch.t)));
        }
        for seam in &self.pattern_project.seams {
            for edge in [&seam.a, &seam.b] {
                if edge.curve != BoundaryCurve::Outline { continue; }
                let points = forced_by_piece.entry(edge.piece_id).or_default();
                points.push((edge.start.segment, edge.start.t));
                points.push((edge.end.segment, edge.end.t));
            }
        }
        for params in forced_by_piece.values_mut() {
            params.sort_by(|a, b| a.0.cmp(&b.0).then_with(|| {
                a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal)
            }));
            params.dedup_by(|a, b| a.0 == b.0 && (a.1 - b.1).abs() <= 1.0e-9);
        }

        // Cada molde usa a sua própria distância entre partículas. As coordenadas da malha
        // são recentradas por peça, para que a transformação 3D gire em torno do centro do painel.
        let mut prepared = Vec::with_capacity(self.pattern_project.pieces.len());
        for piece in &self.pattern_project.pieces {
            let particle_distance = piece.fabric.particle_distance_meters;
            if !particle_distance.is_finite() || particle_distance <= 0.0 {
                return Err(format!("Distância entre partículas inválida em '{}'.", piece.name));
            }
            let params = forced_by_piece.get(&piece.id).cloned().unwrap_or_default();
            let (sampled, forced_indices) = sample_curve_with_forced_params(&piece.outline, 8, &params)
                .map_err(|e| format!("Não foi possível amostrar as costuras/piques de '{}': {e}", piece.name))?;
            validate_simple_polygon(&sampled.points)
                .map_err(|e| format!("Contorno inválido em '{}': {e}", piece.name))?;
            let mesh = triangulate_outline_with_forced_vertices(
                &sampled.points, particle_distance, &forced_indices,
            ).map_err(|e| format!("Não foi possível triangular '{}': {e}", piece.name))?;
            if mesh.boundary_vertex_count != sampled.points.len() {
                return Err(format!("Mapeamento de contorno inválido na peça '{}'.", piece.name));
            }
            let mapping = PieceSampleMapping {
                ranges: sampled.segment_vertex_ranges,
                forced_vertices: params.iter().copied().zip(forced_indices.iter().copied())
                    .map(|((segment, t), index)| (segment, t, index)).collect(),
            };
            let grain = piece.grain_line.as_ref().map(|line| {
                Vec2::new((line.end.x - line.start.x) as f32, (line.end.y - line.start.y) as f32)
            }).unwrap_or(Vec2::Y);
            prepared.push((
                piece.id,
                mapping,
                mesh,
                piece.fabric.clone(),
                piece.assembly.clone(),
                grain,
                piece.name.clone(),
            ));
        }

        let min_x = prepared.iter()
            .flat_map(|(_, _, mesh, _, _, _, _)| mesh.vertices.iter().map(|p| p.x))
            .fold(f64::INFINITY, f64::min);
        let max_x = prepared.iter()
            .flat_map(|(_, _, mesh, _, _, _, _)| mesh.vertices.iter().map(|p| p.x))
            .fold(f64::NEG_INFINITY, f64::max);
        let min_y = prepared.iter()
            .flat_map(|(_, _, mesh, _, _, _, _)| mesh.vertices.iter().map(|p| p.y))
            .fold(f64::INFINITY, f64::min);

        let avatar = AvatarConfig {
            height_m: self.pattern_project.avatar.height_meters as f32,
            shoulder_width_m: self.pattern_project.avatar.shoulder_width_meters as f32,
            waist_width_m: self.pattern_project.avatar.waist_width_meters as f32,
            hip_width_m: self.pattern_project.avatar.hip_width_meters as f32,
        };
        let body_colliders = build_avatar_colliders(&avatar)?;
        let mut builder = ClothBuilder::new();
        builder.add_collider(Collider::ground());
        for collider in body_colliders {
            builder.add_collider(collider);
        }

        struct PieceBuildInfo {
            id: u64,
            handle: cloth_sim::PanelHandle,
            sample: PieceSampleMapping,
            boundary_count: usize,
        }
        let mut infos: Vec<PieceBuildInfo> = Vec::new();
        let piece_count = prepared.len();
        let x_span = (max_x - min_x).max(1.0e-6);
        let origin_y = (0.9 - min_y) as f32;

        for (piece_index, (id, sample, mesh, fabric, assembly, grain, name)) in
            prepared.into_iter().enumerate()
        {
            let mesh_min_x = mesh.vertices.iter().map(|p| p.x).fold(f64::INFINITY, f64::min);
            let mesh_max_x = mesh.vertices.iter().map(|p| p.x).fold(f64::NEG_INFINITY, f64::max);
            let mesh_min_y = mesh.vertices.iter().map(|p| p.y).fold(f64::INFINITY, f64::min);
            let mesh_max_y = mesh.vertices.iter().map(|p| p.y).fold(f64::NEG_INFINITY, f64::max);
            let center_x = (mesh_min_x + mesh_max_x) * 0.5;
            let center_y = (mesh_min_y + mesh_max_y) * 0.5;

            let uv = mesh.vertices.iter()
                .map(|p| Vec2::new((p.x - center_x) as f32, (p.y - center_y) as f32))
                .collect();
            let panel_mesh = PanelMesh { uv, triangles: mesh.triangles };
            // Distribui o layout 2D pela semicircunferência frontal do colisor corporal:
            // moldes mais à esquerda/direita começam tangentes às laterais, e costuras os unem.
            let angle = (((center_x - min_x) / x_span).clamp(0.0, 1.0)
                * std::f64::consts::PI - std::f64::consts::FRAC_PI_2) as f32;
            let shell_radius = pattern_sim_shell_radius(piece_index, fabric.thickness_meters as f32);
            let auto_origin = Vec3::new(
                angle.sin() * shell_radius,
                origin_y + center_y as f32,
                angle.cos() * shell_radius,
            );
            let translation = auto_origin + Vec3::new(
                assembly.translation_meters[0] as f32,
                assembly.translation_meters[1] as f32,
                assembly.translation_meters[2] as f32,
            );
            let local_rotation = Quat::from_euler(
                EulerRot::XYZ,
                assembly.rotation_degrees[0].to_radians() as f32,
                assembly.rotation_degrees[1].to_radians() as f32,
                assembly.rotation_degrees[2].to_radians() as f32,
            );
            let rotation = Quat::from_rotation_y(angle) * local_rotation;
            let material = PanelMaterial {
                density_kg_m2: fabric.density_kg_m2 as f32,
                stretch_warp_n_m: fabric.stretch_warp_n_m as f32,
                stretch_weft_n_m: fabric.stretch_weft_n_m as f32,
                shear_n_m: fabric.shear_n_m as f32,
                bend_nm: fabric.bend_nm as f32,
                thickness_meters: fabric.thickness_meters as f32,
                friction_coefficient: fabric.friction_coefficient as f32,
            };
            let handle = builder.add_panel_with_grain_and_material(
                &panel_mesh,
                Placement::new(translation, rotation),
                grain,
                material,
            ).map_err(|error| format!("Material inválido em '{}': {error}", name))?;

            // Âncoras de contorno: 0 é o início da curva; as demais correspondem ao fim do
            // segmento anterior. Os índices são os mesmos que o editor exibe ao selecionar nós.
            for anchor in &assembly.pinned_anchors {
                let anchor_index = *anchor as usize;
                if anchor_index >= sample.ranges.len() {
                    return Err(format!(
                        "O vértice fixado {} não existe no contorno de '{}'.",
                        anchor_index + 1, name
                    ));
                }
                let local_index = if anchor_index == 0 {
                    0
                } else {
                    sample.ranges[anchor_index - 1].1 % mesh.boundary_vertex_count
                };
                builder.pin(handle.global(local_index as u32));
            }

            infos.push(PieceBuildInfo {
                id,
                handle,
                sample,
                boundary_count: mesh.boundary_vertex_count,
            });
        }

        for seam in &self.pattern_project.seams {
            if seam.a.curve != BoundaryCurve::Outline || seam.b.curve != BoundaryCurve::Outline {
                return Err(format!(
                    "A costura '{}' usa uma linha de costura interna; a simulação atual suporta bordas do contorno.",
                    seam.name
                ));
            }
            let info_a = infos.iter().find(|info| info.id == seam.a.piece_id)
                .ok_or_else(|| format!("Peça A ausente na costura '{}'.", seam.name))?;
            let info_b = infos.iter().find(|info| info.id == seam.b.piece_id)
                .ok_or_else(|| format!("Peça B ausente na costura '{}'.", seam.name))?;
            let locals_a = sampled_edge_indices(
                &seam.a, &info_a.sample.ranges, &info_a.sample.forced_vertices, info_a.boundary_count,
            )?;
            let locals_b = sampled_edge_indices(
                &seam.b, &info_b.sample.ranges, &info_b.sample.forced_vertices, info_b.boundary_count,
            )?;
            let globals_a = info_a.handle.globals(&locals_a);
            let globals_b = info_b.handle.globals(&locals_b);
            let notch_markers = paired_notch_markers(&self.pattern_project, &seam.a, &seam.b);
            builder.sew_with_markers(
                &globals_a, &globals_b, seam.ease_meters as f32, &notch_markers,
            );
        }

        // A simulação mantém o espaçamento real de cada molde; o ruído inicial é menor que
        // a folga aplicada contra o corpo para evitar penetração artificial no primeiro frame.
        builder.perturb(PATTERN_SIM_PERTURBATION);
        self.sim = builder.build(self.sim.params.clone());
        self.seam_pairs = self.sim.seam_pairs();
        self.diverged = false;
        self.accumulator = 0.0;
        self.playing = true;
        self.using_pattern_sim = true;
        self.workspace = WorkspaceMode::Simulation;
        self.upload_scene();
        let pinned = self.pattern_project.pieces.iter()
            .map(|piece| piece.assembly.pinned_anchors.len())
            .sum::<usize>();
        let body_collider_count = self.sim.colliders.len().saturating_sub(1);
        self.pattern_message = format!(
            "Montagem 3D pronta: {} moldes, {} costuras, {} âncoras fixadas e manequim com {} volumes de colisão. Avatar: {:.2} m; ombros {:.2} m, cintura {:.2} m e quadril {:.2} m.",
            piece_count, self.pattern_project.seams.len(), pinned, body_collider_count,
            self.pattern_project.avatar.height_meters,
            self.pattern_project.avatar.shoulder_width_meters,
            self.pattern_project.avatar.waist_width_meters,
            self.pattern_project.avatar.hip_width_meters,
        );
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
                warning_seam_pairs: self.sim.seam_warning_pairs(0.015),
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

/// Início geométrico de um segmento na curva.
fn curve_segment_start(curve: &Curve2D, segment_index: usize) -> Option<PatternPoint2> {
    if segment_index >= curve.segments.len() { return None; }
    if segment_index == 0 { Some(curve.start) }
    else { Some(curve.segments[segment_index - 1].endpoint()) }
}

/// Avalia um segmento linear ou Bézier cúbico no parâmetro local t ∈ [0, 1].
fn curve_segment_point(curve: &Curve2D, segment_index: usize, t: f64) -> Option<PatternPoint2> {
    let start = curve_segment_start(curve, segment_index)?;
    let segment = curve.segments.get(segment_index)?;
    let t = t.clamp(0.0, 1.0);
    let lerp = |a: PatternPoint2, b: PatternPoint2, t: f64| PatternPoint2::new(
        a.x + (b.x - a.x) * t,
        a.y + (b.y - a.y) * t,
    );
    Some(match segment {
        CurveSegment::Line { to } => lerp(start, *to, t),
        CurveSegment::CubicBezier { control1, control2, to } => {
            let a = lerp(start, *control1, t);
            let b = lerp(*control1, *control2, t);
            let c = lerp(*control2, *to, t);
            lerp(lerp(a, b, t), lerp(b, c, t), t)
        }
    })
}

/// Comprimento de arco entre dois parâmetros locais do mesmo segmento.
fn curve_segment_arc_between(curve: &Curve2D, segment_index: usize, start_t: f64, end_t: f64) -> f64 {
    let start_t = start_t.clamp(0.0, 1.0);
    let end_t = end_t.clamp(0.0, 1.0);
    if end_t <= start_t { return 0.0; }
    if matches!(curve.segments.get(segment_index), Some(CurveSegment::Line { .. })) {
        let a = curve_segment_point(curve, segment_index, start_t);
        let b = curve_segment_point(curve, segment_index, end_t);
        return a.zip(b).map(|(a, b)| a.distance(b)).unwrap_or(0.0);
    }
    let steps = 32usize;
    let mut previous = match curve_segment_point(curve, segment_index, start_t) {
        Some(point) => point,
        None => return 0.0,
    };
    let mut length = 0.0;
    for step in 1..=steps {
        let t = start_t + (end_t - start_t) * step as f64 / steps as f64;
        let Some(point) = curve_segment_point(curve, segment_index, t) else { continue; };
        length += previous.distance(point);
        previous = point;
    }
    length
}

fn curve_segment_arc_length(curve: &Curve2D, segment_index: usize, end_t: f64) -> f64 {
    curve_segment_arc_between(curve, segment_index, 0.0, end_t)
}

fn param_at_curve_position(position: f64, segment_count: usize, upper_bound: f64) -> CurveParam {
    let max_position = segment_count as f64;
    let upper = if upper_bound.is_finite() { upper_bound.min(max_position) } else { max_position };
    let position = position.clamp(0.0, upper.max(0.0));
    if segment_count == 0 { return CurveParam { segment: 0, t: 0.0 }; }
    if position >= max_position {
        return CurveParam { segment: (segment_count - 1) as u32, t: 1.0 };
    }
    let segment = position.floor() as usize;
    CurveParam { segment: segment as u32, t: (position - segment as f64).clamp(0.0, 1.0) }
}

/// Projeta um ponto sobre a curva amostrando cada segmento; retorna o parâmetro com menor erro.
fn closest_curve_param(curve: &Curve2D, point: PatternPoint2) -> Option<(CurveParam, f64)> {
    const STEPS: usize = 64;
    if curve.segments.is_empty() { return None; }
    let mut best: Option<(CurveParam, f64)> = None;
    for segment in 0..curve.segments.len() {
        for step in 0..STEPS {
            let t0 = step as f64 / STEPS as f64;
            let t1 = (step + 1) as f64 / STEPS as f64;
            let a = curve_segment_point(curve, segment, t0)?;
            let b = curve_segment_point(curve, segment, t1)?;
            let dx = b.x - a.x;
            let dy = b.y - a.y;
            let length_sq = dx * dx + dy * dy;
            let alpha = if length_sq > 1.0e-18 {
                (((point.x - a.x) * dx + (point.y - a.y) * dy) / length_sq).clamp(0.0, 1.0)
            } else { 0.0 };
            let projected = PatternPoint2::new(a.x + dx * alpha, a.y + dy * alpha);
            let distance = point.distance(projected);
            let param = CurveParam { segment: segment as u32, t: t0 + (t1 - t0) * alpha };
            if best.as_ref().map(|(_, d)| distance < *d).unwrap_or(true) {
                best = Some((param, distance));
            }
        }
    }
    best
}

fn snap_curve_param_to_features(
    curve: &Curve2D,
    notches: &[Notch],
    fallback: CurveParam,
    pointer: PatternPoint2,
    tolerance_world: f64,
) -> CurveParam {
    let mut best = (fallback, pointer.distance(
        curve_segment_point(curve, fallback.segment as usize, fallback.t).unwrap_or(pointer)
    ));
    let mut consider = |param: CurveParam| {
        if let Some(point) = curve_segment_point(curve, param.segment as usize, param.t) {
            let distance = pointer.distance(point);
            if distance < best.1 { best = (param, distance); }
        }
    };
    for segment in 0..curve.segments.len() {
        consider(CurveParam { segment: segment as u32, t: 0.0 });
        consider(CurveParam { segment: segment as u32, t: 1.0 });
    }
    for notch in notches {
        if notch.segment_index as usize < curve.segments.len() && notch.t.is_finite() {
            consider(CurveParam {
                segment: notch.segment_index,
                t: notch.t.clamp(0.0, 1.0),
            });
        }
    }
    if best.1 <= tolerance_world { best.0 } else { fallback }
}

fn edge_curve<'a>(project: &'a PatternProject, edge: &EdgePath) -> Option<&'a Curve2D> {
    let piece = project.pieces.iter().find(|piece| piece.id == edge.piece_id)?;
    match edge.curve {
        BoundaryCurve::Outline => Some(&piece.outline),
        BoundaryCurve::SewingLine => piece.sewing_line.as_ref(),
    }
}

/// Mede a extensão real do intervalo de segmentos, inclusive para intervalos Bézier.
fn edge_length_meters(project: &PatternProject, edge: &EdgePath) -> f64 {
    project.edge_length_meters(edge).unwrap_or(0.0)
}

/// Posição de cada pique como progresso de arco normalizado na borda de costura.
fn edge_notch_positions(project: &PatternProject, edge: &EdgePath) -> Vec<f32> {
    if edge.curve != BoundaryCurve::Outline { return Vec::new(); }
    let Some(piece) = project.pieces.iter().find(|piece| piece.id == edge.piece_id) else {
        return Vec::new();
    };
    let curve = &piece.outline;
    let start = edge.start;
    let end = edge.end;
    let total = edge_length_meters(project, edge);
    if total <= 1.0e-9 { return Vec::new(); }

    let mut out = piece.notches.iter().filter_map(|notch| {
        let index = notch.segment_index;
        let scalar = index as f64 + notch.t;
        if scalar < edge.start_position() - 1.0e-9
            || scalar > edge.end_position() + 1.0e-9
            || !notch.t.is_finite()
        { return None; }
        let mut prefix = 0.0;
        for segment in start.segment..=index {
            let seg = segment as usize;
            let t0 = if segment == start.segment { start.t } else { 0.0 };
            let t1 = if segment == index { notch.t } else { 1.0 };
            if t1 > t0 {
                prefix += curve_segment_arc_between(curve, seg, t0, t1);
            }
        }
        let mut progress = (prefix / total).clamp(0.0, 1.0);
        if edge.reversed { progress = 1.0 - progress; }
        Some(progress as f32)
    }).collect::<Vec<_>>();
    out.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
    out
}

/// Gera marcadores de notch para que cada pique pareado receba uma restrição explícita do solver.
fn paired_notch_markers(
    project: &PatternProject,
    edge_a: &EdgePath,
    edge_b: &EdgePath,
) -> Vec<(f32, f32)> {
    edge_notch_positions(project, edge_a).into_iter()
        .zip(edge_notch_positions(project, edge_b))
        .collect()
}

/// Amostra apenas o intervalo de segmentos referenciado pela costura, para o overlay do canvas.
fn edge_path_points(project: &PatternProject, edge: &EdgePath, steps: usize) -> Vec<PatternPoint2> {
    let Some(curve) = edge_curve(project, edge) else { return Vec::new(); };
    if edge.start_position() >= edge.end_position()
        || edge.end.segment as usize >= curve.segments.len()
    { return Vec::new(); }
    let steps = steps.max(1);
    let mut points = Vec::new();
    for index in edge.start.segment as usize..=edge.end.segment as usize {
        let t0 = if index == edge.start.segment as usize { edge.start.t } else { 0.0 };
        let t1 = if index == edge.end.segment as usize { edge.end.t } else { 1.0 };
        if t1 < t0 { return Vec::new(); }
        for step in 0..=steps {
            if index > edge.start.segment as usize && step == 0 { continue; }
            let t = t0 + (t1 - t0) * step as f64 / steps as f64;
            if let Some(point) = curve_segment_point(curve, index, t) { points.push(point); }
        }
    }
    if edge.reversed { points.reverse(); }
    points
}

fn point_on_path_at_progress(points: &[PatternPoint2], progress: f32) -> Option<PatternPoint2> {
    if points.len() < 2 { return points.first().copied(); }
    let lengths = points.windows(2).map(|pair| pair[0].distance(pair[1])).collect::<Vec<_>>();
    let total: f64 = lengths.iter().sum();
    if total <= 1.0e-9 { return points.first().copied(); }
    let mut target = total * progress.clamp(0.0, 1.0) as f64;
    for (index, length) in lengths.iter().copied().enumerate() {
        if target <= length || index + 1 == lengths.len() {
            let t = if length > 1.0e-9 { (target / length).clamp(0.0, 1.0) } else { 0.0 };
            return Some(PatternPoint2::new(
                points[index].x + (points[index + 1].x - points[index].x) * t,
                points[index].y + (points[index + 1].y - points[index].y) * t,
            ));
        }
        target -= length;
    }
    points.last().copied()
}

fn distance_to_screen_path(point: egui::Pos2, points: &[egui::Pos2]) -> f32 {
    let mut best = f32::INFINITY;
    for pair in points.windows(2) {
        let delta = pair[1] - pair[0];
        let length_sq = delta.length_sq();
        let t = if length_sq > 1.0e-6 {
            ((point - pair[0]).dot(delta) / length_sq).clamp(0.0, 1.0)
        } else { 0.0 };
        best = best.min(point.distance(pair[0] + delta * t));
    }
    best
}

fn draw_path_direction_arrow(
    painter: &egui::Painter,
    points: &[egui::Pos2],
    color: egui::Color32,
) {
    if points.len() < 3 { return; }
    let index = points.len() / 2;
    let tip = points[index];
    let direction = (points[index] - points[index - 1]).normalized();
    if direction.length_sq() <= 1.0e-6 { return; }
    let normal = egui::vec2(-direction.y, direction.x);
    let base = tip - direction * 9.0;
    painter.line_segment([tip, base + normal * 4.0], egui::Stroke::new(1.8, color));
    painter.line_segment([tip, base - normal * 4.0], egui::Stroke::new(1.8, color));
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

fn symmetry_counterpart_id(project: &PatternProject, piece_id: u64) -> Option<u64> {
    project.symmetry_links.iter().find_map(|link| {
        if link.piece_a_id == piece_id { Some(link.piece_b_id) }
        else if link.piece_b_id == piece_id { Some(link.piece_a_id) }
        else { None }
    })
}

fn mirror_pattern_piece(
    source: &PatternPiece,
    target_id: u64,
    target_name: String,
    axis_x: f64,
) -> Result<PatternPiece, String> {
    let mirror = |curve: &Curve2D| {
        mirror_curve_x(curve, axis_x).map_err(|error| error.to_string())
    };
    let mut piece = source.clone();
    piece.id = target_id;
    piece.name = target_name;
    piece.outline = mirror(&source.outline)?;
    piece.sewing_line = source.sewing_line.as_ref().map(mirror).transpose()?;
    piece.internal_lines = source.internal_lines.iter().map(mirror).collect::<Result<Vec<_>, _>>()?;
    for fold in &mut piece.fold_lines {
        fold.curve = mirror(&fold.curve)?;
    }
    for pleat in &mut piece.pleats {
        pleat.axis = mirror(&pleat.axis)?;
    }
    if let Some(grain) = &mut piece.grain_line {
        grain.start.x = 2.0 * axis_x - grain.start.x;
        grain.end.x = 2.0 * axis_x - grain.end.x;
    }
    for dart in &mut piece.darts {
        dart.apex.x = 2.0 * axis_x - dart.apex.x;
        dart.leg_a.x = 2.0 * axis_x - dart.leg_a.x;
        dart.leg_b.x = 2.0 * axis_x - dart.leg_b.x;
    }
    for hole in &mut piece.holes {
        hole.center.x = 2.0 * axis_x - hole.center.x;
    }
    for annotation in &mut piece.annotations {
        annotation.position.x = 2.0 * axis_x - annotation.position.x;
        annotation.rotation_degrees = 180.0 - annotation.rotation_degrees;
    }
    // Conjugate the 3D transform by reflection across X: translation X and Euler Y/Z change
    // sign while X rotation remains unchanged. Topology indices stay stable through reflection.
    piece.assembly.translation_meters[0] = -source.assembly.translation_meters[0];
    piece.assembly.rotation_degrees[0] = source.assembly.rotation_degrees[0];
    piece.assembly.rotation_degrees[1] = -source.assembly.rotation_degrees[1];
    piece.assembly.rotation_degrees[2] = -source.assembly.rotation_degrees[2];
    Ok(piece)
}

fn dense_curve_samples(curve: &Curve2D, steps: usize) -> Vec<PatternPoint2> {
    let steps = steps.max(1);
    let mut points = vec![curve.start];
    let mut start = curve.start;
    for segment in &curve.segments {
        for step in 1..=steps {
            let t = step as f64 / steps as f64;
            let point = match segment {
                CurveSegment::Line { to } => PatternPoint2::new(
                    start.x + (to.x - start.x) * t,
                    start.y + (to.y - start.y) * t,
                ),
                CurveSegment::CubicBezier { control1, control2, to } => {
                    let mt = 1.0 - t;
                    let a = mt * mt * mt;
                    let b = 3.0 * mt * mt * t;
                    let c = 3.0 * mt * t * t;
                    let d = t * t * t;
                    PatternPoint2::new(
                        a * start.x + b * control1.x + c * control2.x + d * to.x,
                        a * start.y + b * control1.y + c * control2.y + d * to.y,
                    )
                }
            };
            points.push(point);
        }
        start = segment.endpoint();
    }
    points
}

fn classify_detail_point(
    point: PatternPoint2,
    polygon_a: &[PatternPoint2],
    polygon_b: &[PatternPoint2],
) -> Result<usize, String> {
    match (
        point_in_polygon(point, polygon_a),
        point_in_polygon(point, polygon_b),
    ) {
        (true, false) => Ok(0),
        (false, true) => Ok(1),
        (true, true) => Err("um detalhe toca a linha de corte; separe-o antes de cortar".into()),
        (false, false) => Err("um detalhe está fora das duas novas peças".into()),
    }
}

fn classify_points_side(
    points: &[PatternPoint2],
    polygon_a: &[PatternPoint2],
    polygon_b: &[PatternPoint2],
) -> Result<usize, String> {
    let mut chosen = None;
    for point in points {
        let side = classify_detail_point(*point, polygon_a, polygon_b)?;
        if chosen.is_some_and(|previous| previous != side) {
            return Err("o corte atravessa um detalhe interno; divida ou reposicione esse detalhe antes de cortar".into());
        }
        chosen = Some(side);
    }
    chosen.ok_or_else(|| "não foi possível determinar a peça que contém o detalhe".to_string())
}

fn classify_curve_side(
    curve: &Curve2D,
    polygon_a: &[PatternPoint2],
    polygon_b: &[PatternPoint2],
) -> Result<usize, String> {
    classify_points_side(&dense_curve_samples(curve, 24), polygon_a, polygon_b)
}

fn assign_curve_to_side(
    curve: Curve2D,
    side: usize,
    first: &mut Vec<Curve2D>,
    second: &mut Vec<Curve2D>,
) {
    if side == 0 { first.push(curve); } else { second.push(curve); }
}

fn split_piece_metadata(
    source: &PatternPiece,
    first: &mut PatternPiece,
    second: &mut PatternPiece,
    cut: &CutAndSewResult,
    polygon_a: &[PatternPoint2],
    polygon_b: &[PatternPoint2],
) -> Result<Option<usize>, String> {
    first.internal_lines.clear();
    second.internal_lines.clear();
    first.fold_lines.clear();
    second.fold_lines.clear();
    first.pleats.clear();
    second.pleats.clear();
    first.darts.clear();
    second.darts.clear();
    first.notches.clear();
    second.notches.clear();
    first.holes.clear();
    second.holes.clear();
    first.annotations.clear();
    second.annotations.clear();
    first.grain_line = None;
    second.grain_line = None;
    first.sewing_line = None;
    second.sewing_line = None;

    for line in &source.internal_lines {
        let side = classify_curve_side(line, polygon_a, polygon_b)
            .map_err(|error| format!("linha interna: {error}"))?;
        assign_curve_to_side(line.clone(), side, &mut first.internal_lines, &mut second.internal_lines);
    }
    for fold in &source.fold_lines {
        let side = classify_curve_side(&fold.curve, polygon_a, polygon_b)
            .map_err(|error| format!("linha de dobra: {error}"))?;
        if side == 0 { first.fold_lines.push(fold.clone()); }
        else { second.fold_lines.push(fold.clone()); }
    }
    for pleat in &source.pleats {
        let side = classify_curve_side(&pleat.axis, polygon_a, polygon_b)
            .map_err(|error| format!("prega: {error}"))?;
        if side == 0 { first.pleats.push(pleat.clone()); }
        else { second.pleats.push(pleat.clone()); }
    }

    if let Some(grain) = &source.grain_line {
        let line = Curve2D::polyline(&[grain.start, grain.end], false)
            .map_err(|_| "linha de fio inválida".to_string())?;
        let side = classify_curve_side(&line, polygon_a, polygon_b)
            .map_err(|error| format!("linha de fio: {error}"))?;
        if side == 0 { first.grain_line = Some(grain.clone()); }
        else { second.grain_line = Some(grain.clone()); }
    }

    let sewing_side = if let Some(line) = &source.sewing_line {
        let side = classify_curve_side(line, polygon_a, polygon_b)
            .map_err(|error| format!("linha de costura: {error}"))?;
        if side == 0 { first.sewing_line = Some(line.clone()); }
        else { second.sewing_line = Some(line.clone()); }
        Some(side)
    } else { None };

    for dart in &source.darts {
        let leg_a = Curve2D::polyline(&[dart.apex, dart.leg_a], false)
            .map_err(|_| "perna A da pence inválida".to_string())?;
        let leg_b = Curve2D::polyline(&[dart.apex, dart.leg_b], false)
            .map_err(|_| "perna B da pence inválida".to_string())?;
        let side_a = classify_curve_side(&leg_a, polygon_a, polygon_b)
            .map_err(|error| format!("pence: {error}"))?;
        let side_b = classify_curve_side(&leg_b, polygon_a, polygon_b)
            .map_err(|error| format!("pence: {error}"))?;
        if side_a != side_b {
            return Err("o corte atravessa uma pence; feche ou recrie a pence antes de cortar".into());
        }
        if side_a == 0 { first.darts.push(dart.clone()); }
        else { second.darts.push(dart.clone()); }
    }

    for notch in &source.notches {
        let source_index = notch.segment_index as usize;
        let fragments = cut.source_segment_fragments.get(source_index)
            .ok_or_else(|| "um entalhe referencia um segmento inexistente".to_string())?;
        let matches = fragments.iter().filter(|fragment| {
            notch.t >= fragment.t_start - 1.0e-8 && notch.t <= fragment.t_end + 1.0e-8
        }).collect::<Vec<&SourceSegmentFragment>>();
        if matches.len() != 1 {
            return Err("um entalhe está exatamente sobre a linha de corte; mova-o antes de cortar".into());
        }
        let fragment = matches[0];
        let span = fragment.t_end - fragment.t_start;
        let mut moved = notch.clone();
        moved.segment_index = fragment.new_segment_index as u32;
        moved.t = if span.abs() <= 1.0e-12 { 0.0 }
            else { ((notch.t - fragment.t_start) / span).clamp(0.0, 1.0) };
        if fragment.piece_index == 0 { first.notches.push(moved); }
        else { second.notches.push(moved); }
    }

    for hole in &source.holes {
        if !hole.radius_meters.is_finite() || hole.radius_meters < 0.0 {
            return Err("um furo possui raio inválido".into());
        }
        let mut boundary = Vec::with_capacity(25);
        for index in 0..24 {
            let angle = std::f64::consts::TAU * index as f64 / 24.0;
            boundary.push(PatternPoint2::new(
                hole.center.x + hole.radius_meters * angle.cos(),
                hole.center.y + hole.radius_meters * angle.sin(),
            ));
        }
        let side = classify_detail_point(hole.center, polygon_a, polygon_b)
            .map_err(|error| format!("furo: {error}"))?;
        if boundary.iter().any(|point| {
            classify_detail_point(*point, polygon_a, polygon_b)
                .map(|boundary_side| boundary_side != side).unwrap_or(true)
        }) {
            return Err("o corte atravessa um furo; reposicione ou remova o furo antes de cortar".into());
        }
        if side == 0 { first.holes.push(hole.clone()); }
        else { second.holes.push(hole.clone()); }
    }
    for annotation in &source.annotations {
        let side = classify_detail_point(annotation.position, polygon_a, polygon_b)
            .map_err(|error| format!("anotação: {error}"))?;
        if side == 0 { first.annotations.push(annotation.clone()); }
        else { second.annotations.push(annotation.clone()); }
    }

    Ok(sewing_side)
}

fn remap_edge_after_cut(
    edge: &mut EdgePath,
    mapping: &[Vec<SourceSegmentFragment>],
    first_id: u64,
    second_id: u64,
) -> Result<(), String> {
    let start = edge.start.segment as usize;
    let end = edge.end.segment as usize;
    if start > end || end >= mapping.len() || edge.start_position() >= edge.end_position() {
        return Err("uma costura existente referencia um intervalo inválido".into());
    }
    let mut side = None;
    let mut first_new_index: Option<usize> = None;
    for source_index in start..=end {
        let fragments = &mapping[source_index];
        if fragments.len() != 1 {
            return Err(format!(
                "a costura existente usa o segmento {} que foi dividido pelo corte; remova ou refaça essa costura primeiro.",
                source_index + 1
            ));
        }
        let fragment = &fragments[0];
        if fragment.t_start.abs() > 1.0e-8 || (fragment.t_end - 1.0).abs() > 1.0e-8 {
            return Err(format!("a costura existente usa o segmento {} parcialmente cortado.", source_index + 1));
        }
        if side.is_some_and(|previous| previous != fragment.piece_index) {
            return Err("a costura existente atravessaria as duas novas peças; reconfigure essa costura antes do corte".into());
        }
        side = Some(fragment.piece_index);
        if let Some(first_index) = first_new_index {
            if fragment.new_segment_index != first_index + source_index - start {
                return Err("o intervalo da costura deixaria de ser contínuo depois do corte; refaça essa costura após dividir".into());
            }
        } else {
            first_new_index = Some(fragment.new_segment_index);
        }
    }
    let side = side.ok_or_else(|| "intervalo de costura vazio".to_string())?;
    edge.piece_id = if side == 0 { first_id } else { second_id };
    edge.start.segment = first_new_index.unwrap() as u32;
    edge.end.segment = edge.start.segment + (end - start) as u32;
    Ok(())
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

    // Inserir um vértice desloca os índices das âncoras posteriores, inclusive as que estão
    // fixadas para a montagem 3D.
    let new_anchor_index = segment_index + 1;
    for anchor in &mut piece.assembly.pinned_anchors {
        if *anchor as usize >= new_anchor_index {
            *anchor += 1;
        }
    }

    // Reparametriza as extremidades sobre os segmentos Bézier/lineares recém-criados.
    for seam in &mut project.seams {
        for edge in [&mut seam.a, &mut seam.b] {
            if edge.piece_id != piece_id || edge.curve != BoundaryCurve::Outline { continue; }
            remap_param_after_segment_split(&mut edge.start, segment_index, t);
            remap_param_after_segment_split(&mut edge.end, segment_index, t);
        }
    }
    // O vértice novo é o endpoint do segmento à esquerda.
    let _ = split_point;
    project.validate()?;
    Ok(segment_index + 1)
}

fn remap_param_after_segment_split(param: &mut CurveParam, segment_index: usize, split_t: f64) {
    let index = segment_index as u32;
    if param.segment > index {
        param.segment += 1;
    } else if param.segment == index {
        if param.t < split_t - 1.0e-9 {
            param.t = (param.t / split_t).clamp(0.0, 1.0);
        } else if param.t > split_t + 1.0e-9 {
            param.segment += 1;
            param.t = ((param.t - split_t) / (1.0 - split_t)).clamp(0.0, 1.0);
        } else {
            param.t = 1.0;
        }
    }
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
            let start_segment = edge.start.segment as usize;
            let end_segment = edge.end.segment as usize;
            if start_segment >= n || end_segment >= n || start_segment > end_segment {
                return Err("Uma costura referencia o segmento que seria removido.".into());
            }
            let affected = if anchor_index == 0 {
                [Some(0usize), Some(n - 1)]
            } else {
                [Some(anchor_index - 1), Some(anchor_index)]
            };
            for param in [&edge.start, &edge.end] {
                if affected.iter().flatten().any(|&segment| segment == param.segment as usize)
                    && param.t > 1.0e-9 && param.t < 1.0 - 1.0e-9
                {
                    return Err("Não é possível excluir o vértice porque uma extremidade de costura livre está em um dos segmentos que seriam unidos.".into());
                }
            }
            let original: Vec<usize> = (start_segment..=end_segment)
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
            edge.start.segment = map_segment(start_segment) as u32;
            edge.end.segment = map_segment(end_segment) as u32;
            if edge.start.segment > edge.end.segment {
                return Err("A exclusão fragmentaria a ordem dos parâmetros da costura.".into());
            }
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
    // O ponto que deixa de existir perde a fixação; os vértices posteriores mudam de índice.
    {
        let piece = staged.pieces.iter_mut().find(|piece| piece.id == piece_id).unwrap();
        piece.assembly.pinned_anchors.retain(|anchor| *anchor as usize != anchor_index);
        for anchor in &mut piece.assembly.pinned_anchors {
            if *anchor as usize > anchor_index {
                *anchor -= 1;
            }
        }
    }
    staged.validate()?;
    *project = staged;
    Ok(())
}

fn sampled_edge_indices(
    edge: &EdgePath,
    ranges: &[(usize, usize)],
    forced_vertices: &[(u32, f64, usize)],
    boundary_count: usize,
) -> Result<Vec<u32>, String> {
    if boundary_count < 2 || edge.start_position() >= edge.end_position() {
        return Err("Intervalo de costura sem vértices suficientes.".into());
    }
    if edge.end.segment as usize >= ranges.len() {
        return Err("Intervalo de costura fora da curva amostrada.".into());
    }

    let vertex_for_param = |param: CurveParam| -> Option<usize> {
        if param.t.abs() <= 1.0e-9 {
            return ranges.get(param.segment as usize).map(|range| range.0);
        }
        if (param.t - 1.0).abs() <= 1.0e-9 {
            return ranges.get(param.segment as usize).map(|range| range.1 % boundary_count);
        }
        forced_vertices.iter().find_map(|(segment, t, index)| {
            (*segment == param.segment && (*t - param.t).abs() <= 1.0e-9).then_some(*index)
        })
    };
    let start_index = vertex_for_param(edge.start)
        .ok_or_else(|| "O início da costura livre não está presente como vértice forçado na malha.".to_string())?;
    let end_index = vertex_for_param(edge.end)
        .ok_or_else(|| "O fim da costura livre não está presente como vértice forçado na malha.".to_string())?;

    let append = |indices: &mut Vec<u32>, index: usize| {
        let wrapped = (index % boundary_count) as u32;
        if indices.last().copied() != Some(wrapped) { indices.push(wrapped); }
    };
    let mut indices: Vec<u32> = Vec::new();
    for segment_index in edge.start.segment as usize..=edge.end.segment as usize {
        let (range_start, range_end) = ranges[segment_index];
        let begin = if segment_index == edge.start.segment as usize { start_index } else { range_start };
        let finish = if segment_index == edge.end.segment as usize { end_index } else { range_end % boundary_count };
        if begin <= finish {
            for index in begin..=finish { append(&mut indices, index); }
        } else {
            for index in begin..boundary_count { append(&mut indices, index); }
            for index in 0..=finish { append(&mut indices, index); }
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
        // Avança depois de processar os controles deste frame. Assim, "Passo único" pausa
        // antes do solver rodar e nunca recebe um passo automático adicional no mesmo frame.
        let dt = if self.capture.is_some() { FIXED_DT } else { ui.input(|i| i.stable_dt) };
        let was_simulation_workspace = self.workspace == WorkspaceMode::Simulation;
        let diverged_before = self.diverged;
        let steps_before = self.sim.stats.steps;

        egui::Panel::left("controls").default_size(310.0).show(ui, |ui| {
            egui::ScrollArea::vertical().show(ui, |ui| self.controls(ui));
        });
        if was_simulation_workspace {
            self.advance(dt);
        }
        if self.sim.stats.steps != steps_before {
            self.last_step_ms = self.sim.stats.step_ms;
        }

        egui::CentralPanel::default().frame(egui::Frame::NONE).show(ui, |ui| {
            if self.workspace == WorkspaceMode::Pattern {
                self.pattern_viewport(ui);
            } else {
                self.viewport(ui);
            }
        });

        self.handle_capture(ui);
        if self.diverged && !diverged_before {
            // O diagnóstico foi definido após renderizar o painel de controles deste frame.
            ui.ctx().request_repaint();
        }
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
            a: EdgeRef::single_segment(1, 1).into(),
            b: EdgeRef::single_segment(2, 3).into(),
            ease_meters: 0.0,
        });
        project
    }

    #[test]
    fn seam_measurements_and_notch_markers_follow_edge_arc_and_direction() {
        let mut project = square_project();
        project.pieces[0].notches.push(Notch {
            id: 1, segment_index: 1, t: 0.5, depth_meters: 0.008, width_meters: 0.012,
        });
        project.pieces[1].notches.push(Notch {
            id: 1, segment_index: 3, t: 0.5, depth_meters: 0.008, width_meters: 0.012,
        });
        let seam = project.seams[0].clone();
        assert!((edge_length_meters(&project, &seam.a) - 1.0).abs() < 1e-6);
        assert!((edge_length_meters(&project, &seam.b) - 1.0).abs() < 1e-6);
        assert_eq!(paired_notch_markers(&project, &seam.a, &seam.b), vec![(0.5, 0.5)]);

        let mut reversed = seam.b.clone();
        reversed.reversed = true;
        assert_eq!(edge_notch_positions(&project, &reversed), vec![0.5]);
    }

    #[test]
    fn pattern_sim_shell_layers_start_outside_the_body_collision_shell() {
        let thickness = 0.005;
        for index in 0..12 {
            let shell_radius = pattern_sim_shell_radius(index, thickness);
            // perturb() pode deslocar Z para ambos os lados em até sua amplitude.
            let lowest_radius = shell_radius - PATTERN_SIM_PERTURBATION;
            assert!(
                lowest_radius >= PATTERN_SIM_BODY_RADIUS + thickness - 1e-6,
                "a camada {index} começa dentro do colisor: raio mínimo {lowest_radius}"
            );
        }
    }

    #[test]
    fn cut_seam_remapping_keeps_untouched_edges_and_rejects_split_edges() {
        let outline = rectangle(
            PatternPoint2::new(0.0, 0.0),
            PatternPoint2::new(2.0, 1.0),
        ).unwrap();
        let path = Curve2D::polyline(&[
            PatternPoint2::new(1.0, 0.0),
            PatternPoint2::new(1.0, 1.0),
        ], false).unwrap();
        let cut = cut_closed_outline(&outline, &path, 0.02).unwrap();

        let mut untouched: EdgePath = EdgeRef::single_segment(1, 1).into();
        remap_edge_after_cut(&mut untouched, &cut.source_segment_fragments, 1, 10).unwrap();
        assert_eq!(untouched.piece_id, 1);
        assert_eq!(untouched.start.segment, 1);
        assert_eq!(untouched.end.segment, 1);

        let mut divided: EdgePath = EdgeRef::single_segment(1, 0).into();
        assert!(remap_edge_after_cut(&mut divided, &cut.source_segment_fragments, 1, 10).is_err());
    }

    #[test]
    fn mirrored_pair_keeps_topology_in_sync_after_anchor_insertion() {
        let mut project = square_project();
        let source = project.pieces[0].clone();
        let mirror = mirror_pattern_piece(&source, 3, "A espelho".into(), 1.0).unwrap();
        project.pieces.push(mirror);
        project.symmetry_links.push(SymmetryLink {
            piece_a_id: 1,
            piece_b_id: 3,
            axis_x: 1.0,
        });
        let mirrored_id = symmetry_counterpart_id(&project, 1).unwrap();
        split_project_segment(&mut project, 1, 1, 0.35).unwrap();
        split_project_segment(&mut project, mirrored_id, 1, 0.35).unwrap();
        assert_eq!(project.pieces[0].outline.segments.len(), project.pieces[2].outline.segments.len());
        project.validate().unwrap();
    }

    #[test]
    fn mirror_pattern_piece_reflects_outline_and_internal_metadata() {
        let mut source = PatternPiece::from_polygon(10, "Frente", &[
            PatternPoint2::new(0.0, 0.0),
            PatternPoint2::new(1.0, 0.0),
            PatternPoint2::new(1.0, 1.0),
            PatternPoint2::new(0.0, 1.0),
        ]).unwrap();
        source.internal_lines.push(Curve2D::polyline(&[
            PatternPoint2::new(0.2, 0.2),
            PatternPoint2::new(0.4, 0.5),
        ], false).unwrap());
        source.grain_line = Some(GrainLine {
            start: PatternPoint2::new(0.3, 0.1),
            end: PatternPoint2::new(0.3, 0.9),
        });
        source.darts.push(pattern_core::Dart {
            apex: PatternPoint2::new(0.5, 0.6),
            leg_a: PatternPoint2::new(0.4, 0.5),
            leg_b: PatternPoint2::new(0.6, 0.5),
            intake_meters: 0.02,
        });
        let mirrored = mirror_pattern_piece(&source, 11, "Costas".into(), 0.0).unwrap();
        assert_eq!(mirrored.id, 11);
        assert_eq!(mirrored.outline.start, PatternPoint2::new(0.0, 0.0));
        assert_eq!(mirrored.outline.segments[0].endpoint(), PatternPoint2::new(-1.0, 0.0));
        assert_eq!(mirrored.internal_lines[0].start, PatternPoint2::new(-0.2, 0.2));
        assert_eq!(mirrored.grain_line.as_ref().unwrap().start.x, -0.3);
        assert_eq!(mirrored.darts[0].apex.x, -0.5);
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
    fn contour_edits_keep_fixed_anchors_attached_to_the_same_vertices() {
        let mut project = square_project();
        project.pieces[0].assembly.pinned_anchors = vec![1, 3];
        split_project_segment(&mut project, 1, 0, 0.5).unwrap();
        assert_eq!(project.pieces[0].assembly.pinned_anchors, vec![2, 4]);

        delete_project_anchor(&mut project, 1, 2).unwrap();
        assert_eq!(project.pieces[0].assembly.pinned_anchors, vec![3]);
        project.validate().unwrap();
    }

    #[test]
    fn splitting_seam_edge_updates_segment_ranges_and_keeps_geometry_length() {
        let mut project = square_project();
        let before = pattern_geom::curve_length(&project.pieces[0].outline, 32);
        split_project_segment(&mut project, 1, 1, 0.35).unwrap();
        let after = pattern_geom::curve_length(&project.pieces[0].outline, 32);
        assert!((before - after).abs() < 1.0e-6);
        assert_eq!(project.pieces[0].outline.segments.len(), 5);
        assert_eq!(project.seams[0].a.start.segment, 1);
        assert_eq!(project.seams[0].a.end.segment, 2);
        assert_eq!(project.seams[0].b.start.segment, 3);
        assert_eq!(project.seams[0].b.end.segment, 3);
        project.validate().unwrap();
    }

    #[test]
    fn deleting_polygon_anchor_keeps_project_valid_and_seam_reference_valid() {
        let mut project = square_project();
        delete_project_anchor(&mut project, 1, 1).unwrap();
        assert_eq!(project.pieces[0].outline.segments.len(), 3);
        assert_eq!(project.seams[0].a.start.segment, 0);
        assert_eq!(project.seams[0].a.end.segment, 0);
        project.validate().unwrap();
    }

    #[test]
    fn converting_line_to_bezier_preserves_straight_shape_and_seam_indices() {
        let mut project = square_project();
        convert_project_segment_to_bezier(&mut project, 1, 1).unwrap();
        let segment = &project.pieces[0].outline.segments[1];
        assert!(matches!(segment, CurveSegment::CubicBezier { .. }));
        assert_eq!(project.seams[0].a.start.segment, 1);
        assert_eq!(project.seams[0].a.end.segment, 1);
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
