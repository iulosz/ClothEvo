//! Pré-visualização da malha 2D dos moldes.
//!
//! A malha mostrada aqui é a mesma que a simulação 3D usa: cada molde é amostrado com a sua
//! própria distância entre partículas e a triangulação preserva, como vértices fixos, os piques e
//! as extremidades das costuras. Os resultados ficam em cache e só são recalculados quando algo que
//! afeta a malha muda; falhas são guardadas por molde para serem mostradas ao usuário.

use super::*;
use pattern_geom::SampledCurve;
use pattern_mesh::PatternMesh;
use std::collections::hash_map::DefaultHasher;
use std::hash::{Hash, Hasher};

/// Ângulo mínimo (graus) a partir do qual um triângulo é considerado bom.
pub(crate) const GOOD_MIN_ANGLE_DEG: f32 = 30.0;
/// Abaixo deste ângulo mínimo (graus) o triângulo é considerado ruim para o solver XPBD.
pub(crate) const POOR_MIN_ANGLE_DEG: f32 = 20.0;

// ---------------------------------------------------------------------------------------------
// Caminho compartilhado com a simulação 3D
// ---------------------------------------------------------------------------------------------

/// Parâmetros (segmento, t) que precisam existir como vértices do contorno: piques e extremos
/// de costura. Ordenados e sem duplicatas.
pub(crate) fn forced_parameters_by_piece(project: &PatternProject) -> HashMap<u64, Vec<(u32, f64)>> {
    let mut forced_by_piece: HashMap<u64, Vec<(u32, f64)>> = HashMap::new();
    for piece in &project.pieces {
        forced_by_piece.entry(piece.id).or_default()
            .extend(piece.notches.iter().map(|notch| (notch.segment_index, notch.t)));
    }
    for seam in &project.seams {
        for edge in seam.sides_a.iter().chain(seam.sides_b.iter()) {
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
    forced_by_piece
}

/// Triangula um molde já amostrado exatamente como a simulação faz. Devolve a malha e os índices
/// dos vértices forçados, na ordem de `params`.
pub(crate) fn triangulate_piece(
    piece: &PatternPiece,
    sampled: &SampledCurve,
    params: &[(u32, f64)],
) -> Result<(PatternMesh, Vec<usize>), String> {
    let particle_distance = piece.fabric.particle_distance_meters;
    if !particle_distance.is_finite() || particle_distance <= 0.0 {
        return Err(format!("Distância entre partículas inválida em '{}'.", piece.name));
    }
    let forced_indices = forced_parameter_indices(piece, sampled, params)
        .map_err(|error| format!("Não foi possível mapear os piques/extremos de '{}': {error}", piece.name))?;
    validate_simple_polygon(&sampled.points)
        .map_err(|e| format!("Contorno inválido em '{}': {e}", piece.name))?;
    let mesh = triangulate_outline_with_forced_vertices(&sampled.points, particle_distance, &forced_indices)
        .map_err(|e| format!("Não foi possível triangular '{}': {e}", piece.name))?;
    if mesh.boundary_vertex_count != sampled.points.len() {
        return Err(format!("Mapeamento de contorno inválido na peça '{}'.", piece.name));
    }
    Ok((mesh, forced_indices))
}

// ---------------------------------------------------------------------------------------------
// Qualidade
// ---------------------------------------------------------------------------------------------

/// Menor ângulo interno do triângulo, em graus. Triângulos degenerados devolvem 0.
pub(crate) fn triangle_min_angle_deg(a: PatternPoint2, b: PatternPoint2, c: PatternPoint2) -> f32 {
    let angle_at = |p: PatternPoint2, q: PatternPoint2, r: PatternPoint2| -> f64 {
        let (ux, uy) = (q.x - p.x, q.y - p.y);
        let (vx, vy) = (r.x - p.x, r.y - p.y);
        let (lu, lv) = ((ux * ux + uy * uy).sqrt(), (vx * vx + vy * vy).sqrt());
        if lu <= 1.0e-12 || lv <= 1.0e-12 { return 0.0; }
        ((ux * vx + uy * vy) / (lu * lv)).clamp(-1.0, 1.0).acos().to_degrees()
    };
    angle_at(a, b, c).min(angle_at(b, c, a)).min(angle_at(c, a, b)) as f32
}

#[derive(Clone, Debug, Default, PartialEq)]
pub(crate) struct MeshStats {
    pub vertices: usize,
    pub triangles: usize,
    pub min_angle_deg: f32,
    pub mean_min_angle_deg: f32,
    pub poor_triangles: usize,
    pub mean_edge_m: f32,
}

#[derive(Clone, Debug)]
pub(crate) struct MeshData {
    pub mesh: PatternMesh,
    /// Menor ângulo de cada triângulo, alinhado com `mesh.triangles`.
    pub triangle_min_angles: Vec<f32>,
    pub stats: MeshStats,
}

impl MeshData {
    fn new(mesh: PatternMesh) -> Self {
        let mut angles = Vec::with_capacity(mesh.triangles.len());
        let mut edge_sum = 0.0_f64;
        for triangle in &mesh.triangles {
            let a = mesh.vertices[triangle[0] as usize];
            let b = mesh.vertices[triangle[1] as usize];
            let c = mesh.vertices[triangle[2] as usize];
            angles.push(triangle_min_angle_deg(a, b, c));
            edge_sum += a.distance(b) + b.distance(c) + c.distance(a);
        }
        let triangles = angles.len();
        let stats = MeshStats {
            vertices: mesh.vertices.len(),
            triangles,
            min_angle_deg: angles.iter().copied().fold(f32::INFINITY, f32::min),
            mean_min_angle_deg: if triangles == 0 { 0.0 } else { angles.iter().sum::<f32>() / triangles as f32 },
            poor_triangles: angles.iter().filter(|angle| **angle < POOR_MIN_ANGLE_DEG).count(),
            mean_edge_m: if triangles == 0 { 0.0 } else { (edge_sum / (3 * triangles) as f64) as f32 },
        };
        Self { mesh, triangle_min_angles: angles, stats }
    }
}

/// Cor de preenchimento de um triângulo segundo o seu menor ângulo.
pub(crate) fn quality_color(min_angle_deg: f32) -> egui::Color32 {
    if min_angle_deg >= GOOD_MIN_ANGLE_DEG {
        egui::Color32::from_rgba_unmultiplied(70, 175, 115, 70)
    } else if min_angle_deg >= POOR_MIN_ANGLE_DEG {
        egui::Color32::from_rgba_unmultiplied(235, 195, 60, 120)
    } else {
        egui::Color32::from_rgba_unmultiplied(225, 70, 70, 165)
    }
}

// ---------------------------------------------------------------------------------------------
// Cache
// ---------------------------------------------------------------------------------------------

#[derive(Clone, Debug)]
pub(crate) struct PieceMeshPreview {
    pub piece_id: u64,
    pub name: String,
    pub particle_distance_m: f64,
    pub result: Result<MeshData, String>,
}

#[derive(Default)]
pub(crate) struct MeshPreview {
    fingerprint: Option<u64>,
    pieces: Vec<PieceMeshPreview>,
    /// Falha ao sincronizar as bordas de costura; a malha mostrada é então aproximada.
    pub sync_warning: Option<String>,
    /// Quantas vezes a malha foi recalculada (usado nos testes).
    pub rebuilds: u32,
}

fn hash_point(hasher: &mut DefaultHasher, point: PatternPoint2) {
    point.x.to_bits().hash(hasher);
    point.y.to_bits().hash(hasher);
}

/// Resume tudo o que influencia a malha: contornos, piques, espaçamento e extremos de costura.
fn project_mesh_fingerprint(project: &PatternProject) -> u64 {
    let mut hasher = DefaultHasher::new();
    for piece in &project.pieces {
        piece.id.hash(&mut hasher);
        hash_point(&mut hasher, piece.outline.start);
        piece.outline.closed.hash(&mut hasher);
        for segment in &piece.outline.segments {
            match segment {
                CurveSegment::Line { to } => {
                    0u8.hash(&mut hasher);
                    hash_point(&mut hasher, *to);
                }
                CurveSegment::CubicBezier { control1, control2, to } => {
                    1u8.hash(&mut hasher);
                    hash_point(&mut hasher, *control1);
                    hash_point(&mut hasher, *control2);
                    hash_point(&mut hasher, *to);
                }
            }
        }
        for notch in &piece.notches {
            notch.segment_index.hash(&mut hasher);
            notch.t.to_bits().hash(&mut hasher);
        }
        piece.fabric.particle_distance_meters.to_bits().hash(&mut hasher);
    }
    for seam in &project.seams {
        for edge in seam.sides_a.iter().chain(seam.sides_b.iter()) {
            if edge.curve != BoundaryCurve::Outline { continue; }
            edge.piece_id.hash(&mut hasher);
            edge.start.segment.hash(&mut hasher);
            edge.start.t.to_bits().hash(&mut hasher);
            edge.end.segment.hash(&mut hasher);
            edge.end.t.to_bits().hash(&mut hasher);
        }
    }
    hasher.finish()
}

impl MeshPreview {
    pub fn pieces(&self) -> &[PieceMeshPreview] { &self.pieces }

    pub fn piece(&self, piece_id: u64) -> Option<&PieceMeshPreview> {
        self.pieces.iter().find(|piece| piece.piece_id == piece_id)
    }

    pub fn error_count(&self) -> usize {
        self.pieces.iter().filter(|piece| piece.result.is_err()).count()
    }

    pub fn totals(&self) -> MeshStats {
        let mut totals = MeshStats { min_angle_deg: f32::INFINITY, ..MeshStats::default() };
        let mut weighted_angle = 0.0_f32;
        for data in self.pieces.iter().filter_map(|piece| piece.result.as_ref().ok()) {
            totals.vertices += data.stats.vertices;
            totals.triangles += data.stats.triangles;
            totals.poor_triangles += data.stats.poor_triangles;
            totals.min_angle_deg = totals.min_angle_deg.min(data.stats.min_angle_deg);
            weighted_angle += data.stats.mean_min_angle_deg * data.stats.triangles as f32;
        }
        if totals.triangles > 0 {
            totals.mean_min_angle_deg = weighted_angle / totals.triangles as f32;
        } else {
            totals.min_angle_deg = 0.0;
        }
        totals
    }

    /// Recalcula a malha somente se algo que a afeta mudou desde a última chamada.
    pub fn refresh(&mut self, project: &PatternProject) {
        let fingerprint = project_mesh_fingerprint(project);
        if self.fingerprint == Some(fingerprint) { return; }
        self.fingerprint = Some(fingerprint);
        self.rebuilds += 1;
        self.sync_warning = None;
        self.pieces.clear();

        let mut samples = match sample_pattern_outlines_with_seam_sync(&project.pieces, &project.seams) {
            Ok(samples) => samples,
            Err(error) => {
                self.sync_warning = Some(format!(
                    "Não foi possível sincronizar as bordas de costura ({error}); a malha mostrada é aproximada."
                ));
                HashMap::new()
            }
        };
        let forced_by_piece = forced_parameters_by_piece(project);

        for piece in &project.pieces {
            let sampled = samples.remove(&piece.id).or_else(|| {
                // Sem sincronização de costura: amostra o molde isoladamente.
                sample_pattern_outlines_with_seam_sync(std::slice::from_ref(piece), &[])
                    .ok()
                    .and_then(|mut isolated| isolated.remove(&piece.id))
            });
            let result = match sampled {
                None => Err(format!("Não foi possível amostrar o contorno de '{}'.", piece.name)),
                Some(sampled) => {
                    let params = forced_by_piece.get(&piece.id).cloned().unwrap_or_default();
                    triangulate_piece(piece, &sampled, &params).map(|(mesh, _)| MeshData::new(mesh))
                }
            };
            self.pieces.push(PieceMeshPreview {
                piece_id: piece.id,
                name: piece.name.clone(),
                particle_distance_m: piece.fabric.particle_distance_meters,
                result,
            });
        }
    }
}

// ---------------------------------------------------------------------------------------------
// Painel
// ---------------------------------------------------------------------------------------------

impl ClothEvoApp {
    pub(super) fn mesh_preview_panel(&mut self, ui: &mut egui::Ui) {
        self.mesh_preview.refresh(&self.pattern_project);
        let errors = self.mesh_preview.error_count();
        let title = if errors > 0 {
            egui::RichText::new("Pré-visualização da malha 2D  ⚠").strong()
                .color(egui::Color32::from_rgb(255, 170, 53))
        } else {
            egui::RichText::new("Pré-visualização da malha 2D").strong()
        };
        egui::CollapsingHeader::new(title)
            .id_salt("mesh_preview_panel")
            .default_open(false)
            .show(ui, |ui| {
                ui.checkbox(&mut self.pattern_show_mesh, "Mostrar triangulação");
                ui.add_enabled(
                    self.pattern_show_mesh,
                    egui::Checkbox::new(&mut self.mesh_quality_colors, "Colorir por qualidade dos triângulos"),
                ).on_hover_text("Verde: ângulo mínimo ≥ 30°. Amarelo: 20–30°. Vermelho: < 20° (triângulos finos que prejudicam a estabilidade do XPBD).");

                if self.pattern_project.pieces.is_empty() {
                    ui.weak("Crie um molde para ver a malha que será simulada.");
                    return;
                }

                ui.separator();
                self.mesh_spacing_controls(ui);

                ui.separator();
                self.mesh_statistics(ui);

                ui.separator();
                ui.weak("Esta é a mesma malha usada na simulação 3D: espaçamento por molde e vértices fixos nos piques e nas pontas das costuras.");
            });
    }

    fn mesh_spacing_controls(&mut self, ui: &mut egui::Ui) {
        let index = self.selected_piece.min(self.pattern_project.pieces.len() - 1);
        let piece_id = self.pattern_project.pieces[index].id;
        let name = self.pattern_project.pieces[index].name.clone();
        ui.label(format!("Molde selecionado: {name}"));

        let before = self.editor_snapshot();
        let mut spacing = self.pattern_project.pieces[index].fabric.particle_distance_meters;
        let changed = ui.add(
            egui::Slider::new(&mut spacing, 0.01..=0.20).logarithmic(true)
                .text("distância entre partículas (m)"),
        ).on_hover_text("Define a resolução da malha deste molde. Valores menores dão mais detalhe e custam mais na simulação.").changed();
        if changed {
            self.pattern_project.pieces[index].fabric.particle_distance_meters = spacing;
            self.sync_symmetric_mate(piece_id);
            self.remember_pattern_change(before);
        }

        if self.pattern_project.pieces.len() > 1
            && ui.button("Aplicar este espaçamento a todos os moldes").clicked()
        {
            let before = self.editor_snapshot();
            for piece in &mut self.pattern_project.pieces {
                piece.fabric.particle_distance_meters = spacing;
            }
            self.remember_pattern_change(before);
        }
    }

    fn mesh_statistics(&mut self, ui: &mut egui::Ui) {
        if let Some(warning) = &self.mesh_preview.sync_warning {
            ui.colored_label(egui::Color32::from_rgb(255, 170, 53), warning);
        }
        let totals = self.mesh_preview.totals();
        ui.strong(format!(
            "Total: {} moldes · {} partículas · {} triângulos",
            self.mesh_preview.pieces().len(), totals.vertices, totals.triangles,
        ));
        if totals.triangles > 0 {
            ui.label(format!(
                "Qualidade: ângulo mínimo {:.1}° · médio {:.1}° · {} triângulos ruins (< {:.0}°)",
                totals.min_angle_deg, totals.mean_min_angle_deg, totals.poor_triangles, POOR_MIN_ANGLE_DEG,
            ));
            if totals.poor_triangles > 0 {
                ui.colored_label(
                    egui::Color32::from_rgb(255, 120, 120),
                    "Há triângulos muito finos. Ajuste o espaçamento ou simplifique o contorno perto das pontas.",
                );
            }
        }

        for preview in self.mesh_preview.pieces().iter().filter(|piece| piece.result.is_err()) {
            if let Err(message) = &preview.result {
                ui.colored_label(egui::Color32::from_rgb(255, 120, 120), format!("⚠ {}: {message}", preview.name));
            }
        }

        egui::CollapsingHeader::new("Detalhes por molde")
            .id_salt("mesh_preview_details")
            .show(ui, |ui| {
                egui::Grid::new("mesh_preview_grid").striped(true).num_columns(5).show(ui, |ui| {
                    for header in ["Molde", "Esp. (m)", "Partículas", "Triângulos", "Âng. mín."] {
                        ui.strong(header);
                    }
                    ui.end_row();
                    for preview in self.mesh_preview.pieces() {
                        ui.label(&preview.name);
                        ui.label(format!("{:.3}", preview.particle_distance_m));
                        match &preview.result {
                            Ok(data) => {
                                ui.label(data.stats.vertices.to_string());
                                ui.label(data.stats.triangles.to_string());
                                ui.label(format!("{:.1}°", data.stats.min_angle_deg));
                            }
                            Err(_) => {
                                ui.label("—");
                                ui.label("—");
                                ui.colored_label(egui::Color32::from_rgb(255, 120, 120), "erro");
                            }
                        }
                        ui.end_row();
                    }
                });
            });
    }

    /// Desenha a malha em cache de um molde no canvas 2D.
    pub(super) fn draw_piece_mesh_preview(
        &self,
        painter: &egui::Painter,
        piece: &PatternPiece,
        to_screen: &impl Fn(PatternPoint2) -> egui::Pos2,
    ) {
        let Some(preview) = self.mesh_preview.piece(piece.id) else { return; };
        let data = match &preview.result {
            Ok(data) => data,
            Err(_) => {
                // Sinaliza no canvas que este molde não pôde ser triangulado.
                let center = piece_pivot(piece);
                painter.text(
                    to_screen(center),
                    egui::Align2::CENTER_CENTER,
                    "⚠ malha inválida",
                    egui::FontId::proportional(12.0),
                    egui::Color32::from_rgb(255, 170, 53),
                );
                return;
            }
        };
        if self.mesh_quality_colors {
            let mut fill = egui::Mesh::default();
            for (triangle, angle) in data.mesh.triangles.iter().zip(&data.triangle_min_angles) {
                let color = quality_color(*angle);
                let base = fill.vertices.len() as u32;
                for &index in triangle {
                    fill.colored_vertex(to_screen(data.mesh.vertices[index as usize]), color);
                }
                fill.add_triangle(base, base + 1, base + 2);
            }
            painter.add(egui::Shape::mesh(fill));
        }
        let stroke = egui::Stroke::new(0.7, egui::Color32::from_rgb(63, 91, 105));
        for triangle in &data.mesh.triangles {
            let a = to_screen(data.mesh.vertices[triangle[0] as usize]);
            let b = to_screen(data.mesh.vertices[triangle[1] as usize]);
            let c = to_screen(data.mesh.vertices[triangle[2] as usize]);
            painter.line_segment([a, b], stroke);
            painter.line_segment([b, c], stroke);
            painter.line_segment([c, a], stroke);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn square(id: u64, name: &str, x0: f64, side: f64) -> PatternPiece {
        PatternPiece::from_polygon(id, name, &[
            PatternPoint2::new(x0, 0.0),
            PatternPoint2::new(x0 + side, 0.0),
            PatternPoint2::new(x0 + side, side),
            PatternPoint2::new(x0, side),
        ]).unwrap()
    }

    fn project_with_square(spacing: f64) -> PatternProject {
        let mut project = PatternProject::new("malha");
        let mut piece = square(1, "A", 0.0, 0.5);
        piece.fabric.particle_distance_meters = spacing;
        project.pieces.push(piece);
        project
    }

    #[test]
    fn min_angle_of_known_triangles() {
        let right = triangle_min_angle_deg(
            PatternPoint2::new(0.0, 0.0), PatternPoint2::new(1.0, 0.0), PatternPoint2::new(0.0, 1.0),
        );
        assert!((right - 45.0).abs() < 1.0e-3, "{right}");
        let equilateral = triangle_min_angle_deg(
            PatternPoint2::new(0.0, 0.0), PatternPoint2::new(1.0, 0.0),
            PatternPoint2::new(0.5, 0.866_025_4),
        );
        assert!((equilateral - 60.0).abs() < 1.0e-2, "{equilateral}");
        let degenerate = triangle_min_angle_deg(
            PatternPoint2::new(0.0, 0.0), PatternPoint2::new(1.0, 0.0), PatternPoint2::new(2.0, 0.0),
        );
        assert!(degenerate.abs() < 1.0e-3);
    }

    #[test]
    fn quality_colors_follow_the_thresholds() {
        assert_eq!(quality_color(45.0), quality_color(GOOD_MIN_ANGLE_DEG));
        assert_ne!(quality_color(25.0), quality_color(45.0));
        assert_ne!(quality_color(10.0), quality_color(25.0));
    }

    #[test]
    fn preview_uses_each_pieces_own_spacing() {
        let mut project = project_with_square(0.20);
        let mut fine = square(2, "B", 1.0, 0.5);
        fine.fabric.particle_distance_meters = 0.05;
        project.pieces.push(fine);
        let mut preview = MeshPreview::default();
        preview.refresh(&project);
        let coarse = preview.piece(1).unwrap().result.as_ref().unwrap().stats.triangles;
        let dense = preview.piece(2).unwrap().result.as_ref().unwrap().stats.triangles;
        assert!(dense > coarse * 4, "malha fina ({dense}) deveria ter muito mais triângulos que a grossa ({coarse})");
    }

    #[test]
    fn preview_mesh_matches_the_simulation_path() {
        let mut project = project_with_square(0.08);
        project.pieces[0].notches.push(pattern_core::Notch {
            id: 1, segment_index: 0, t: 0.37, depth_meters: 0.005, width_meters: 0.005,
        });
        let mut preview = MeshPreview::default();
        preview.refresh(&project);
        let shown = &preview.piece(1).unwrap().result.as_ref().unwrap().mesh;

        let piece = &project.pieces[0];
        let samples = sample_pattern_outlines_with_seam_sync(&project.pieces, &project.seams).unwrap();
        let forced = forced_parameters_by_piece(&project);
        let (expected, indices) = triangulate_piece(piece, &samples[&1], &forced[&1]).unwrap();
        assert_eq!(shown, &expected);
        assert_eq!(indices.len(), 1, "o pique precisa virar um vértice fixo");
    }

    #[test]
    fn cache_rebuilds_only_when_the_mesh_inputs_change() {
        let mut project = project_with_square(0.10);
        let mut preview = MeshPreview::default();
        preview.refresh(&project);
        preview.refresh(&project);
        preview.refresh(&project);
        assert_eq!(preview.rebuilds, 1, "sem mudanças não deve recalcular");

        // Propriedades que não afetam a malha não invalidam o cache.
        project.pieces[0].name = "Frente".into();
        project.pieces[0].fabric.density_kg_m2 += 0.1;
        preview.refresh(&project);
        assert_eq!(preview.rebuilds, 1);

        let before = preview.piece(1).unwrap().result.as_ref().unwrap().stats.triangles;
        project.pieces[0].fabric.particle_distance_meters = 0.05;
        preview.refresh(&project);
        assert_eq!(preview.rebuilds, 2);
        let after = preview.piece(1).unwrap().result.as_ref().unwrap().stats.triangles;
        assert!(after > before);

        project.pieces[0].outline.start.x += 0.01;
        preview.refresh(&project);
        assert_eq!(preview.rebuilds, 3, "mover o contorno invalida o cache");
    }

    #[test]
    fn invalid_piece_reports_an_error_instead_of_disappearing() {
        // Gravata-borboleta: contorno que se auto-intercepta.
        let mut project = PatternProject::new("inválido");
        let bow_tie = [
            PatternPoint2::new(0.0, 0.0),
            PatternPoint2::new(1.0, 1.0),
            PatternPoint2::new(1.0, 0.0),
            PatternPoint2::new(0.0, 1.0),
        ];
        project.pieces.push(PatternPiece::from_polygon(1, "Ruim", &bow_tie).unwrap());
        project.pieces.push(square(2, "Boa", 2.0, 0.5));
        let mut preview = MeshPreview::default();
        preview.refresh(&project);
        assert_eq!(preview.error_count(), 1);
        let message = preview.piece(1).unwrap().result.as_ref().unwrap_err();
        assert!(message.contains("Ruim"), "{message}");
        assert!(preview.piece(2).unwrap().result.is_ok(), "um molde ruim não derruba os outros");
    }

    #[test]
    fn invalid_spacing_is_reported() {
        let project = project_with_square(0.0);
        let mut preview = MeshPreview::default();
        preview.refresh(&project);
        assert_eq!(preview.error_count(), 1);
    }

    #[test]
    fn totals_sum_every_piece() {
        let mut project = project_with_square(0.10);
        project.pieces.push(square(2, "B", 1.0, 0.5));
        let mut preview = MeshPreview::default();
        preview.refresh(&project);
        let totals = preview.totals();
        let sum: usize = preview.pieces().iter()
            .map(|piece| piece.result.as_ref().unwrap().stats.triangles).sum();
        assert_eq!(totals.triangles, sum);
        assert!(totals.min_angle_deg > 0.0 && totals.min_angle_deg <= 60.0);
    }
}
