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
    BoundaryCurve, EdgeRef, PatternPiece, PatternProject, Point2 as PatternPoint2, SeamDefinition,
};
use pattern_geom::{offset_polygon, sample_curve, sample_curve_with_ranges, validate_simple_polygon};
use pattern_mesh::triangulate_outline;

const FIXED_DT: f32 = 1.0 / 60.0;
const DEFAULT_RES: u32 = 40;
const PATTERN_PIXELS_PER_METER: f32 = 360.0;

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
    pattern_show_mesh: bool,
    selected_piece: usize,
    seam_a_piece: usize,
    seam_b_piece: usize,
    seam_a_segment: usize,
    seam_b_segment: usize,
    seam_ease_m: f64,
    pattern_message: String,
    using_pattern_sim: bool,
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
            pattern_show_mesh: true,
            selected_piece: 0,
            seam_a_piece: 0,
            seam_b_piece: 1,
            seam_a_segment: 0,
            seam_b_segment: 0,
            seam_ease_m: 0.0,
            pattern_message: String::new(),
            using_pattern_sim: false,
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

    fn pattern_controls(&mut self, ui: &mut egui::Ui) {
        ui.heading("Pattern Studio");
        ui.label("Desenhe contornos clicando na área 2D. As medidas internas são em metros.");
        ui.horizontal(|ui| {
            if ui.button("Desfazer ponto").clicked() { self.draft_points.pop(); }
            if ui.button("Limpar rascunho").clicked() { self.draft_points.clear(); }
        });
        if ui.button("Fechar peça a partir dos pontos").clicked() {
            self.create_piece_from_draft();
        }
        ui.label(format!("Pontos no rascunho: {}", self.draft_points.len()));

        ui.separator();
        ui.strong("Malha 2D");
        ui.add(egui::Slider::new(&mut self.pattern_spacing, 0.025..=0.20).text("espaçamento (m)"));
        ui.checkbox(&mut self.pattern_show_mesh, "Mostrar triangulação");
        if ui.button("Salvar projeto JSON").clicked() {
            match self.pattern_project.to_json().and_then(|json| {
                std::fs::write("clothevo-pattern.json", json)
                    .map_err(|e| format!("Falha ao salvar: {e}"))
            }) {
                Ok(()) => self.pattern_message = "Projeto salvo em clothevo-pattern.json.".into(),
                Err(e) => self.pattern_message = e,
            }
        }
        if ui.button("Abrir projeto JSON").clicked() {
            match std::fs::read_to_string("clothevo-pattern.json")
                .map_err(|e| format!("Falha ao abrir arquivo: {e}"))
                .and_then(|json| PatternProject::from_json(&json))
            {
                Ok(project) => {
                    self.pattern_project = project;
                    self.draft_points.clear();
                    self.pattern_message = "Projeto carregado.".into();
                }
                Err(e) => self.pattern_message = e,
            }
        }

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
            ui.add(egui::Slider::new(
                &mut self.pattern_project.pieces[self.selected_piece].seam_allowance_meters,
                0.0..=0.05,
            ).text("margem (m)"));
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
            ui.add(egui::DragValue::new(&mut self.seam_ease_m).speed(0.001).prefix("Folga (m) "));
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

    fn create_piece_from_draft(&mut self) {
        if self.draft_points.len() < 3 {
            self.pattern_message = "Uma peça precisa de pelo menos 3 pontos.".into();
            return;
        }
        let next_id = self.pattern_project.pieces.iter().map(|piece| piece.id).max().unwrap_or(0) + 1;
        let piece_name = format!("Peça {next_id}");
        match PatternPiece::from_polygon(next_id, piece_name, &self.draft_points) {
            Ok(piece) => {
                self.pattern_project.pieces.push(piece);
                self.draft_points.clear();
                self.pattern_message = "Peça criada. Configure costuras e salve o projeto.".into();
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
        let seam = SeamDefinition {
            id: next_id,
            name: format!("Costura {next_id}"),
            a: EdgeRef::single_segment(a_piece.id, a_segment),
            b: EdgeRef::single_segment(b_piece.id, b_segment),
            ease_meters: self.seam_ease_m,
        };
        self.pattern_project.seams.push(seam);
        if let Err(e) = self.pattern_project.validate() {
            self.pattern_project.seams.pop();
            self.pattern_message = e;
        } else {
            self.pattern_message = "Costura adicionada ao projeto.".into();
        }
    }

    fn pattern_viewport(&mut self, ui: &mut egui::Ui) {
        let rect = ui.available_rect_before_wrap();
        let response = ui.allocate_rect(rect, egui::Sense::click());
        let painter = ui.painter_at(rect);
        let center = rect.center();
        let scale = PATTERN_PIXELS_PER_METER;

        painter.rect_filled(rect, 0.0, egui::Color32::from_gray(23));
        for i in -30..=30 {
            let offset = i as f32 * 0.1 * scale;
            let x = center.x + offset;
            let y = center.y + offset;
            if x >= rect.left() && x <= rect.right() {
                painter.line_segment([egui::pos2(x, rect.top()), egui::pos2(x, rect.bottom())],
                    egui::Stroke::new(1.0, egui::Color32::from_gray(39)));
            }
            if y >= rect.top() && y <= rect.bottom() {
                painter.line_segment([egui::pos2(rect.left(), y), egui::pos2(rect.right(), y)],
                    egui::Stroke::new(1.0, egui::Color32::from_gray(39)));
            }
        }
        painter.line_segment([egui::pos2(rect.left(), center.y), egui::pos2(rect.right(), center.y)],
            egui::Stroke::new(1.0, egui::Color32::from_gray(88)));
        painter.line_segment([egui::pos2(center.x, rect.top()), egui::pos2(center.x, rect.bottom())],
            egui::Stroke::new(1.0, egui::Color32::from_gray(88)));
        painter.text(rect.left_top() + egui::vec2(12.0, 12.0), egui::Align2::LEFT_TOP,
            "PATTERN 2D · clique para criar pontos", egui::FontId::proportional(14.0),
            egui::Color32::from_gray(205));

        let to_screen = |p: PatternPoint2| {
            egui::pos2(center.x + p.x as f32 * scale, center.y - p.y as f32 * scale)
        };

        for piece in &self.pattern_project.pieces {
            let points = sample_curve(&piece.outline, 10);
            if points.len() < 3 { continue; }
            // Contorno de corte em azul; margem de costura independente em verde.
            if piece.seam_allowance_meters > 0.0 {
                if let Ok(margin) = offset_polygon(&points, piece.seam_allowance_meters) {
                    for i in 0..margin.len() {
                        painter.line_segment(
                            [to_screen(margin[i]), to_screen(margin[(i + 1) % margin.len()])],
                            egui::Stroke::new(1.5, egui::Color32::from_rgb(104, 196, 131)),
                        );
                    }
                }
            }

            if self.pattern_show_mesh {
                if let Ok(mesh) = triangulate_outline(&points, self.pattern_spacing) {
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
                let a = to_screen(points[i]);
                let b = to_screen(points[(i + 1) % points.len()]);
                painter.line_segment([a, b], egui::Stroke::new(2.0, egui::Color32::from_rgb(91, 183, 245)));
            }
            let centroid = points.iter().fold(PatternPoint2::new(0.0, 0.0), |sum, point| {
                PatternPoint2::new(sum.x + point.x / points.len() as f64, sum.y + point.y / points.len() as f64)
            });
            painter.text(to_screen(centroid), egui::Align2::CENTER_CENTER, &piece.name,
                egui::FontId::proportional(13.0), egui::Color32::WHITE);
        }

        if self.draft_points.len() >= 2 {
            let polyline = self.draft_points.iter().copied().map(to_screen).collect::<Vec<_>>();
            painter.add(egui::Shape::line(polyline, egui::Stroke::new(2.0, egui::Color32::from_rgb(245, 180, 65))));
        }
        for point in &self.draft_points {
            painter.circle_filled(to_screen(*point), 4.5, egui::Color32::from_rgb(255, 194, 75));
        }

        if response.clicked() {
            if let Some(pos) = response.interact_pointer_pos() {
                self.draft_points.push(PatternPoint2::new(
                    (pos.x - center.x) as f64 / scale as f64,
                    (center.y - pos.y) as f64 / scale as f64,
                ));
            }
        }
    }

    fn start_pattern_simulation(&mut self) -> Result<(), String> {
        self.pattern_project.validate()?;
        if self.pattern_project.pieces.is_empty() {
            return Err("O projeto não contém peças.".into());
        }

        let mut builder = ClothBuilder::new();
        builder.add_collider(Collider::ground());
        builder.add_collider(Collider::Cylinder {
            a: Vec3::new(0.0, 0.0, 0.0),
            b: Vec3::new(0.0, 1.5, 0.0),
            radius: 0.14,
        });

        struct PieceBuildInfo {
            id: u64,
            handle: cloth_sim::PanelHandle,
            ranges: Vec<(usize, usize)>,
            boundary_count: usize,
        }
        let mut infos: Vec<PieceBuildInfo> = Vec::new();
        let piece_count = self.pattern_project.pieces.len();

        for (piece_index, piece) in self.pattern_project.pieces.iter().enumerate() {
            let sampled = sample_curve_with_ranges(&piece.outline, 8);
            validate_simple_polygon(&sampled.points)
                .map_err(|e| format!("Contorno inválido em '{}': {e}", piece.name))?;
            let mesh = triangulate_outline(&sampled.points, self.pattern_spacing)
                .map_err(|e| format!("Não foi possível triangular '{}': {e}", piece.name))?;
            if mesh.boundary_vertex_count != sampled.points.len() {
                return Err(format!("Mapeamento de contorno inválido na peça '{}'.", piece.name));
            }

            let min_x = mesh.vertices.iter().map(|p| p.x).fold(f64::INFINITY, f64::min);
            let min_y = mesh.vertices.iter().map(|p| p.y).fold(f64::INFINITY, f64::min);
            let max_x = mesh.vertices.iter().map(|p| p.x).fold(f64::NEG_INFINITY, f64::max);
            let width = (max_x - min_x) as f32;
            let uv = mesh.vertices.iter()
                .map(|p| Vec2::new((p.x - min_x) as f32, (p.y - min_y) as f32))
                .collect();
            let panel_mesh = PanelMesh { uv, triangles: mesh.triangles.clone() };
            let centered_x = -width * 0.5;
            let z = (piece_index as f32 - (piece_count.saturating_sub(1)) as f32 * 0.5) * 0.24;
            let handle = builder.add_panel_with_grain(
                &panel_mesh,
                Placement::new(Vec3::new(centered_x, 0.9, z), Quat::IDENTITY),
                Vec2::Y,
            );
            infos.push(PieceBuildInfo {
                id: piece.id,
                handle,
                ranges: sampled.segment_vertex_ranges,
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

        builder.perturb(0.001);
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
