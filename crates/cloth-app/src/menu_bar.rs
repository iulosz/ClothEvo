//! Barra de menus principal: File, Edit, 3D Garment, 2D Pattern, Sewing, Materials, Avatar,
//! Display, Preferences, Settings e Help.
//!
//! Cada item chama uma função que já existe no aplicativo. Itens cujo recurso ainda não existe
//! aparecem desabilitados, para o layout ficar completo sem prometer o que não funciona.

use super::*;

const REPOSITORY_URL: &str = "https://github.com/iulosz/ClothEvo";

/// Item de menu com atalho opcional; devolve `true` quando foi clicado.
fn item(ui: &mut egui::Ui, enabled: bool, label: &str, shortcut: &str) -> bool {
    let mut button = egui::Button::new(label);
    if !shortcut.is_empty() {
        button = button.shortcut_text(shortcut);
    }
    ui.add_enabled(enabled, button).clicked()
}

/// Item ainda não implementado: sempre desabilitado.
fn coming_soon(ui: &mut egui::Ui, label: &str) {
    ui.add_enabled(false, egui::Button::new(label))
        .on_disabled_hover_text("Ainda não disponível no ClothEvo.");
}

impl ClothEvoApp {
    /// Atalhos globais dos menus File (Ctrl+N/O/S/Shift+S). Desfazer/refazer continuam tratados
    /// no editor de moldes.
    pub(super) fn handle_menu_shortcuts(&mut self, ui: &egui::Ui) {
        use egui::{Key, KeyboardShortcut, Modifiers};
        let new = KeyboardShortcut::new(Modifiers::COMMAND, Key::N);
        let open = KeyboardShortcut::new(Modifiers::COMMAND, Key::O);
        let save = KeyboardShortcut::new(Modifiers::COMMAND, Key::S);
        let save_as = KeyboardShortcut::new(Modifiers::COMMAND | Modifiers::SHIFT, Key::S);
        // O atalho mais específico (com Shift) é consumido primeiro.
        let (do_save_as, do_save, do_new, do_open) = ui.ctx().input_mut(|i| (
            i.consume_shortcut(&save_as),
            i.consume_shortcut(&save),
            i.consume_shortcut(&new),
            i.consume_shortcut(&open),
        ));
        if do_new { self.new_pattern_project(); }
        if do_open { self.open_pattern_project_dialog(); }
        if do_save_as { self.save_pattern_project_as(); }
        else if do_save { self.save_pattern_project(); }
    }

    pub(super) fn menu_bar(&mut self, ui: &mut egui::Ui) {
        egui::MenuBar::new().ui(ui, |ui| {
            self.file_menu(ui);
            self.edit_menu(ui);
            self.garment_menu(ui);
            self.pattern_menu(ui);
            self.sewing_menu(ui);
            self.materials_menu(ui);
            self.avatar_menu(ui);
            self.display_menu(ui);
            self.preferences_menu(ui);
            self.settings_menu(ui);
            self.help_menu(ui);
        });
    }

    /// Janelas abertas pelo menu Help.
    pub(super) fn menu_windows(&mut self, ctx: &egui::Context) {
        egui::Window::new("Atalhos de teclado")
            .open(&mut self.show_shortcuts_window)
            .collapsible(false)
            .resizable(false)
            .show(ctx, |ui| {
                egui::Grid::new("shortcuts_grid").num_columns(2).spacing([24.0, 4.0]).show(ui, |ui| {
                    for (keys, action) in [
                        ("Ctrl+N", "Novo projeto"),
                        ("Ctrl+O", "Abrir projeto"),
                        ("Ctrl+S", "Salvar projeto"),
                        ("Ctrl+Shift+S", "Salvar como…"),
                        ("Ctrl+Z / Ctrl+Y", "Desfazer / refazer (Moldes 2D)"),
                        ("W / E / R", "Mover / girar / escalar peça (Moldes 2D)"),
                        ("Delete", "Excluir seleção (Moldes 2D)"),
                        ("Esc", "Cancelar o desenho em andamento"),
                        ("1 / 2 / 3", "Câmera frontal / lateral / topo (viewport 3D)"),
                    ] {
                        ui.label(egui::RichText::new(keys).monospace().strong());
                        ui.label(action);
                        ui.end_row();
                    }
                });
            });
        egui::Window::new("Sobre o ClothEvo")
            .open(&mut self.show_about_window)
            .collapsible(false)
            .resizable(false)
            .show(ctx, |ui| {
                ui.heading("ClothEvo");
                ui.label(format!("Versão {}", env!("CARGO_PKG_VERSION")));
                ui.label("Editor de moldes 2D e simulação de tecido XPBD 3D, em Rust (egui + wgpu).");
                ui.label("Os projetos são armazenados em JSON no formato nativo .clothevo.");
                ui.hyperlink_to("Repositório no GitHub", REPOSITORY_URL);
            });
    }

    fn editing_patterns(&self) -> bool {
        matches!(self.workspace, WorkspaceMode::Pattern | WorkspaceMode::Split)
    }

    fn show_pattern_workspace(&mut self) {
        self.playing = false;
        self.workspace = WorkspaceMode::Pattern;
    }

    // ----- File -------------------------------------------------------------------------

    fn file_menu(&mut self, ui: &mut egui::Ui) {
        ui.menu_button("File", |ui| {
            if item(ui, true, "Novo projeto", "Ctrl+N") { self.new_pattern_project(); }
            if item(ui, true, "Abrir…", "Ctrl+O") { self.open_pattern_project_dialog(); }
            if item(ui, true, "Salvar", "Ctrl+S") { self.save_pattern_project(); }
            if item(ui, true, "Salvar como…", "Ctrl+Shift+S") { self.save_pattern_project_as(); }
            ui.separator();
            let import_pending = self.avatar_import_pending_path.is_some();
            if item(ui, !import_pending, "Importar avatar OBJ/FBX…", "") { self.import_avatar_dialog(); }
            coming_soon(ui, "Importar moldes DXF (AAMA/ASTM)…");
            coming_soon(ui, "Exportar moldes…");
            coming_soon(ui, "Exportar roupa 3D…");
            ui.separator();
            if item(ui, true, "Sair", "") && self.confirm_discard_unsaved_changes() {
                ui.ctx().send_viewport_cmd(egui::ViewportCommand::Close);
            }
        });
    }

    // ----- Edit -------------------------------------------------------------------------

    fn edit_menu(&mut self, ui: &mut egui::Ui) {
        ui.menu_button("Edit", |ui| {
            let editing = self.editing_patterns();
            let has_piece = !self.pattern_project.pieces.is_empty();
            if item(ui, editing && !self.pattern_undo.is_empty(), "Desfazer", "Ctrl+Z") { self.undo_pattern_edit(); }
            if item(ui, editing && !self.pattern_redo.is_empty(), "Refazer", "Ctrl+Y") { self.redo_pattern_edit(); }
            ui.separator();
            if item(ui, editing && has_piece, "Duplicar peça selecionada", "") { self.duplicate_selected_piece(); }
            if item(ui, editing && has_piece, "Excluir peça selecionada", "") { self.delete_selected_piece(); }
            if item(ui, editing && self.selected_target.is_some(), "Excluir vértice selecionado", "Del") {
                self.delete_selected_vertex();
            }
            if item(ui, editing && has_piece, "Inserir vértice na aresta selecionada", "") { self.insert_selected_vertex(); }
            if item(ui, editing && has_piece, "Converter aresta em curva Bézier", "") { self.convert_selected_edge_to_bezier(); }
            if !editing {
                ui.separator();
                ui.weak("Edição de moldes disponível em 2D Pattern.");
            }
        });
    }

    // ----- 3D Garment -------------------------------------------------------------------

    fn garment_menu(&mut self, ui: &mut egui::Ui) {
        ui.menu_button("3D Garment", |ui| {
            if ui.selectable_label(self.workspace == WorkspaceMode::Simulation, "Ambiente 3D").clicked() {
                self.workspace = WorkspaceMode::Simulation;
            }
            if item(ui, true, "Posicionamento 3D dos painéis", "") { self.enter_placement_workspace(); }
            ui.separator();
            let label = if self.playing { "Pausar simulação" } else { "Continuar simulação" };
            if item(ui, true, label, "") {
                if self.diverged {
                    self.sim.reset();
                    self.accumulator = 0.0;
                    self.diverged = false;
                    self.playing = true;
                } else {
                    self.playing = !self.playing;
                }
            }
            if item(ui, true, "Passo único", "") {
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
            if item(ui, true, "Reiniciar simulação", "") {
                self.sim.reset();
                self.last_step_ms = 0.0;
                self.playing = false;
                self.diverged = false;
                self.accumulator = 0.0;
            }
            ui.separator();
            ui.checkbox(&mut self.sim.params.self_collision, "Auto-colisão (mais custo)");
            coming_soon(ui, "Acessórios e hardware (zíper, botões)");
        });
    }

    // ----- 2D Pattern -------------------------------------------------------------------

    fn pattern_menu(&mut self, ui: &mut egui::Ui) {
        ui.menu_button("2D Pattern", |ui| {
            if ui.selectable_label(self.workspace == WorkspaceMode::Pattern, "Editor de moldes 2D").clicked() {
                self.show_pattern_workspace();
            }
            if ui.selectable_label(self.workspace == WorkspaceMode::Split, "3D + 2D").clicked() {
                self.workspace = WorkspaceMode::Split;
                self.pattern_fit_pending = true;
            }
            ui.separator();
            if ui.selectable_label(self.pattern_tool == PatternTool::Draw, "Ferramenta: desenhar").clicked() {
                self.pattern_tool = PatternTool::Draw;
            }
            if ui.selectable_label(self.pattern_tool == PatternTool::Select, "Ferramenta: selecionar").clicked() {
                self.pattern_tool = PatternTool::Select;
            }
            ui.separator();
            let editing = self.editing_patterns();
            if item(ui, editing, "Novo retângulo", "") { self.create_rectangle_primitive(); }
            if item(ui, editing, "Novo polígono regular", "") { self.create_regular_polygon_primitive(); }
            if item(ui, editing && !self.pattern_project.pieces.is_empty(), "Criar par espelhado", "") {
                self.create_mirrored_pair();
            }
            if item(ui, editing && !self.pattern_project.pieces.is_empty(), "Desfazer simetria da peça", "") {
                self.unlink_selected_symmetry();
            }
            if item(ui, editing && !self.pattern_project.pieces.is_empty(), "Eixo de simetria no limite esquerdo", "") {
                self.set_symmetry_axis_to_min_x();
            }
            ui.separator();
            ui.checkbox(&mut self.pattern_snap, "Ímã (snap) de pontos");
            ui.checkbox(&mut self.pattern_show_mesh, "Mostrar malha dos moldes");
            ui.checkbox(&mut self.show_avatar_guide, "Guia anatômico T-Pose");
            coming_soon(ui, "Cotas e margens de costura");
        });
    }

    // ----- Sewing -----------------------------------------------------------------------

    fn sewing_menu(&mut self, ui: &mut egui::Ui) {
        ui.menu_button("Sewing", |ui| {
            let editing = self.editing_patterns();
            let has_piece = !self.pattern_project.pieces.is_empty();
            if item(ui, editing && has_piece, "Adicionar costura (painel de controles)", "") { self.add_seam_from_controls(); }
            if item(ui, editing && self.selected_seam_id.is_some(), "Atualizar costura selecionada", "") {
                self.update_selected_seam_from_controls();
            }
            if item(ui, editing && self.selected_seam_id.is_some(), "Remover costura selecionada", "") {
                self.remove_selected_seam();
            }
            if item(ui, editing && !self.auto_seam_suggestions.is_empty(), "Aplicar costuras automáticas sugeridas", "") {
                self.apply_selected_auto_seams();
            }
            ui.separator();
            ui.checkbox(&mut self.seam_pick_mode, "Selecionar arestas no canvas");
            ui.checkbox(&mut self.seam_pick_pair_mode, "Modo rápido (A e B por cliques)");
            ui.checkbox(&mut self.seam_free_mode, "Costura livre (parâmetro t)");
            ui.separator();
            if item(ui, editing && has_piece, "Cut & Sew a partir do desenho", "") { self.cut_and_sew_from_draft(); }
            if item(ui, editing && has_piece, "Linha interna a partir do desenho", "") { self.create_internal_line_from_draft(); }
            if item(ui, editing && has_piece, "Linha de dobra (vale)", "") { self.create_fold_line_from_draft(); }
            if item(ui, editing && has_piece, "Linha de dobra (montanha)", "") { self.create_mountain_fold_line_from_draft(); }
            if item(ui, editing && has_piece, "Prega", "") { self.create_pleat_from_draft(); }
            if item(ui, editing && has_piece, "Linha de grão", "") { self.create_grainline_from_draft(); }
            ui.separator();
            if item(ui, editing && has_piece, "Adicionar pique (notch)", "") { self.add_notch_from_controls(); }
            if item(ui, editing && has_piece, "Remover pique selecionado", "") { self.remove_selected_notch(); }
            ui.separator();
            let mut seams_changed = false;
            seams_changed |= ui.checkbox(&mut self.style.seams, "Mostrar linhas de costura no 3D").changed();
            if seams_changed { self.upload_scene(); }
        });
    }

    // ----- Materials --------------------------------------------------------------------

    fn materials_menu(&mut self, ui: &mut egui::Ui) {
        ui.menu_button("Materials", |ui| {
            ui.menu_button("Tecido (preset)", |ui| {
                for preset in FabricPreset::ALL {
                    if ui.selectable_label(self.preset == preset, preset.name()).clicked() {
                        self.preset = preset;
                        preset.apply(&mut self.sim.params);
                    }
                }
            });
            ui.separator();
            let mut colors_changed = false;
            ui.horizontal(|ui| {
                colors_changed |= ui.color_edit_button_rgb(&mut self.style.cloth_front).changed();
                ui.label("Cor da face externa");
            });
            ui.horizontal(|ui| {
                colors_changed |= ui.color_edit_button_rgb(&mut self.style.cloth_back).changed();
                ui.label("Cor da face interna");
            });
            if colors_changed { self.upload_scene(); }
            ui.separator();
            coming_soon(ui, "Texturas e mapas PBR");
            coming_soon(ui, "Biblioteca de tecidos personalizados");
        });
    }

    // ----- Avatar -----------------------------------------------------------------------

    fn avatar_menu(&mut self, ui: &mut egui::Ui) {
        ui.menu_button("Avatar", |ui| {
            let import_pending = self.avatar_import_pending_path.is_some();
            let has_model = self.avatar_mesh_path.is_some();
            if item(ui, !import_pending, "Importar avatar OBJ/FBX…", "") { self.import_avatar_dialog(); }
            if item(ui, !import_pending && has_model, "Aplicar escala / colisão", "") {
                if let Some(path) = self.avatar_mesh_path.clone() {
                    self.schedule_avatar_import(&path, true);
                }
            }
            if item(ui, !import_pending && has_model, "Remover modelo importado", "") { self.remove_avatar_model(); }
            ui.separator();
            let mut view_changed = false;
            view_changed |= ui.checkbox(&mut self.avatar_visible, "Exibir avatar").changed();
            view_changed |= ui.add_enabled(
                self.avatar_mesh.is_some(),
                egui::Checkbox::new(&mut self.avatar_wireframe, "Wireframe de inspeção"),
            ).changed();
            if view_changed { self.upload_scene(); }
            if ui.add_enabled(
                has_model,
                egui::Checkbox::new(&mut self.avatar_collision_enabled, "Usar avatar como colisão"),
            ).changed() {
                if let Err(error) = self.sync_pattern_avatar_colliders() {
                    self.pattern_message = format!("Não foi possível atualizar a colisão: {error}");
                }
                self.upload_scene();
            }
            ui.separator();
            ui.checkbox(&mut self.show_avatar_guide, "Guia anatômico T-Pose");
            if let Some(name) = self.avatar_mesh_name.as_ref() {
                ui.weak(format!("Modelo: {name}"));
            } else {
                ui.weak("Avatar procedural");
            }
            coming_soon(ui, "Poses e animação do avatar");
        });
    }

    // ----- Display ----------------------------------------------------------------------

    fn display_menu(&mut self, ui: &mut egui::Ui) {
        ui.menu_button("Display", |ui| {
            if ui.selectable_label(self.workspace == WorkspaceMode::Simulation, "Ambiente 3D").clicked() {
                self.workspace = WorkspaceMode::Simulation;
            }
            if ui.selectable_label(self.workspace == WorkspaceMode::Pattern, "Moldes 2D").clicked() {
                self.show_pattern_workspace();
            }
            if ui.selectable_label(self.workspace == WorkspaceMode::Split, "3D + 2D").clicked() {
                self.workspace = WorkspaceMode::Split;
                self.pattern_fit_pending = true;
            }
            if self.workspace == WorkspaceMode::Split {
                ui.checkbox(&mut self.split_swapped, "Trocar lados");
                ui.checkbox(&mut self.split_vertical, "Empilhar");
            }
            ui.separator();
            ui.menu_button("Câmera 3D", |ui| {
                for (label, preset) in [
                    ("Frontal (1)", CameraPreset::Front),
                    ("Lateral (2)", CameraPreset::Side),
                    ("Topo (3)", CameraPreset::Top),
                ] {
                    if ui.button(label).clicked() {
                        self.focus_camera_preset(preset, 1360.0 / 840.0);
                    }
                }
            });
            if item(ui, self.workspace == WorkspaceMode::Pattern, "Ajustar vista 2D", "") {
                self.pattern_fit_pending = true;
            }
            ui.separator();
            let mut render_changed = false;
            render_changed |= ui.checkbox(&mut self.style.wireframe, "Wireframe da malha").changed();
            render_changed |= ui.checkbox(&mut self.style.seams, "Linhas de costura").changed();
            if render_changed { self.upload_scene(); }
            ui.checkbox(&mut self.pattern_show_mesh, "Malha dos moldes 2D");
            ui.checkbox(&mut self.show_avatar_guide, "Guia anatômico T-Pose");
            ui.separator();
            ui.checkbox(&mut self.show_metrics, "Métricas do solver");
        });
    }

    // ----- Preferences ------------------------------------------------------------------

    fn preferences_menu(&mut self, ui: &mut egui::Ui) {
        ui.menu_button("Preferences", |ui| {
            ui.label("Escala da interface");
            let before = self.ui_scale;
            ui.add(egui::Slider::new(&mut self.ui_scale, 0.75..=1.75).step_by(0.05).fixed_decimals(2));
            if (self.ui_scale - before).abs() > f32::EPSILON {
                ui.ctx().set_zoom_factor(self.ui_scale);
            }
            if ui.small_button("Restaurar 100 %").clicked() {
                self.ui_scale = 1.0;
                ui.ctx().set_zoom_factor(1.0);
            }
            ui.separator();
            ui.label("Editor de moldes");
            ui.add(egui::Slider::new(&mut self.seam_snap_tolerance_px, 4.0..=32.0).text("tolerância de snap (px)"));
            ui.add(egui::Slider::new(&mut self.avatar_guide_backdrop, 0.0..=0.6).text("fundo do guia"));
            ui.separator();
            coming_soon(ui, "Atalhos personalizados");
            coming_soon(ui, "Idioma");
        });
    }

    // ----- Settings ---------------------------------------------------------------------

    fn settings_menu(&mut self, ui: &mut egui::Ui) {
        ui.menu_button("Settings", |ui| {
            ui.label("Solver de simulação");
            {
                let p = &mut self.sim.params;
                ui.add(egui::Slider::new(&mut p.substeps, 1..=80).text("substeps por quadro"));
                ui.add(egui::Slider::new(&mut p.iterations, 1..=12).text("iterações por substep"));
                ui.add(egui::Slider::new(&mut p.max_particle_speed, 2.0..=60.0).text("limite de velocidade (m/s)"));
                ui.horizontal(|ui| {
                    if ui.button("Equilibrado").on_hover_text("20 substeps · 2 iterações").clicked() {
                        p.substeps = 20;
                        p.iterations = 2;
                        p.max_particle_speed = 20.0;
                    }
                    if ui.button("Mais estável").on_hover_text("40 substeps · 4 iterações").clicked() {
                        p.substeps = 40;
                        p.iterations = 4;
                        p.max_particle_speed = 12.0;
                    }
                });
                ui.checkbox(&mut p.parallel, "Paralelo (rayon)");
                ui.checkbox(&mut p.self_collision, "Auto-colisão");
            }
            ui.separator();
            ui.label("Física do tecido");
            {
                let p = &mut self.sim.params;
                ui.add(egui::Slider::new(&mut p.damping, 0.0..=5.0).text("amortecimento 1/s"));
                ui.add(egui::Slider::new(&mut p.friction, 0.0..=1.5).text("atrito"));
                ui.add(egui::Slider::new(&mut p.thickness, 0.0..=0.03).text("espessura m"));
            }
            ui.separator();
            ui.label("Avatar e colisão");
            ui.add(egui::Slider::new(&mut self.avatar_collision_spacing_m, 0.004..=0.04).text("espaçamento da colisão (m)"));
            ui.add(egui::Slider::new(&mut self.avatar_heading_degrees, -180.0..=180.0).text("orientação (°)"));
            ui.weak("Use Avatar › Aplicar escala / colisão para reprocessar o modelo.");
        });
    }

    // ----- Help -------------------------------------------------------------------------

    fn help_menu(&mut self, ui: &mut egui::Ui) {
        ui.menu_button("Help", |ui| {
            if ui.button("Atalhos de teclado").clicked() { self.show_shortcuts_window = true; }
            if ui.button("Repositório no GitHub").clicked() {
                ui.ctx().open_url(egui::OpenUrl::new_tab(REPOSITORY_URL));
            }
            ui.separator();
            if ui.button("Sobre o ClothEvo").clicked() { self.show_about_window = true; }
        });
    }
}
