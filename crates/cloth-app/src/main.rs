//! ClothEvo — Fase 0: protótipo do simulador de tecido XPBD (egui + wgpu).
//!
//! Modo de captura (para validar o render sem interação), via variáveis de ambiente:
//! `CLOTHEVO_SCENE=1|2|3`, `CLOTHEVO_FRAMES=<n passos simulados>` e
//! `CLOTHEVO_SCREENSHOT=<arquivo.ppm>` — salva o viewport e fecha o app.

use cloth_render::{
    primitives, ClothTopology, OrbitCamera, SceneRenderer, StaticMeshData, Vertex,
    ViewStyle, ViewportCallback,
};
use cloth_sim::{build_scene, Collider, FabricPreset, SceneKind, Simulation, SimParams};
use eframe::{egui, egui_wgpu};
use glam::Vec3;

const FIXED_DT: f32 = 1.0 / 60.0;
const DEFAULT_RES: u32 = 40;

fn main() -> eframe::Result {
    let options = eframe::NativeOptions {
        viewport: egui::ViewportBuilder::default()
            .with_title("ClothEvo — Fase 0 (simulação XPBD)")
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
        };
        app.upload_scene();
        app
    }

    /// Reconstrói a cena preservando os parâmetros atuais.
    fn load_scene(&mut self, kind: SceneKind, res: u32) {
        let params = self.sim.params.clone();
        self.kind = kind;
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
        ui.label("Fase 0 · validação do solver XPBD");
        ui.separator();

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

impl eframe::App for ClothEvoApp {
    fn ui(&mut self, ui: &mut egui::Ui, _frame: &mut eframe::Frame) {
        // Em captura, simula em passos fixos independentemente do tempo real.
        let dt = if self.capture.is_some() { FIXED_DT } else { ui.input(|i| i.stable_dt) };
        let steps_before = self.sim.stats.steps;
        self.advance(dt);
        if self.sim.stats.steps != steps_before {
            self.last_step_ms = self.sim.stats.step_ms;
        }

        egui::Panel::left("controls").default_size(310.0).show(ui, |ui| {
            egui::ScrollArea::vertical().show(ui, |ui| self.controls(ui));
        });
        egui::CentralPanel::default().frame(egui::Frame::NONE).show(ui, |ui| self.viewport(ui));

        self.handle_capture(ui);
        if self.playing || self.capture.is_some() {
            ui.ctx().request_repaint();
        }
    }
}
