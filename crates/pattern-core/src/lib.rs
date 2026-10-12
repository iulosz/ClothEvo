//! Modelo de domínio para moldes de vestuário.
//! Esta crate não depende de UI, renderização, geometria computacional ou física.
//! As medidas lineares internas são armazenadas em metros.

use serde::{Deserialize, Serialize};
use std::collections::HashSet;

pub type PieceId = u64;
pub type SeamId = u64;

#[derive(Clone, Copy, Debug, Default, PartialEq, Serialize, Deserialize)]
pub struct Point2 { pub x: f64, pub y: f64 }

impl Point2 {
    pub const fn new(x: f64, y: f64) -> Self { Self { x, y } }
    pub fn distance(self, other: Self) -> f64 {
        let dx = self.x - other.x;
        let dy = self.y - other.y;
        (dx * dx + dy * dy).sqrt()
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub enum LengthUnit { Millimeter, Centimeter, Meter, Inch }

impl Default for LengthUnit {
    fn default() -> Self { Self::Centimeter }
}

impl LengthUnit {
    pub fn meters_per_unit(self) -> f64 {
        match self {
            Self::Millimeter => 0.001,
            Self::Centimeter => 0.01,
            Self::Meter => 1.0,
            Self::Inch => 0.0254,
        }
    }
    pub fn label(self) -> &'static str {
        match self {
            Self::Millimeter => "mm",
            Self::Centimeter => "cm",
            Self::Meter => "m",
            Self::Inch => "in",
        }
    }
}

/// Segmento cujo ponto inicial é o fim do segmento anterior (ou Curve2D::start).
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub enum CurveSegment {
    Line { to: Point2 },
    CubicBezier { control1: Point2, control2: Point2, to: Point2 },
}

impl CurveSegment {
    pub fn endpoint(&self) -> Point2 {
        match self {
            Self::Line { to } | Self::CubicBezier { to, .. } => *to,
        }
    }
}

/// Caminho 2D com segmentos explícitos. Contornos fechados devem terminar no ponto inicial.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct Curve2D {
    pub start: Point2,
    pub segments: Vec<CurveSegment>,
    pub closed: bool,
}

impl Curve2D {
    pub fn polyline(points: &[Point2], closed: bool) -> Result<Self, String> {
        let minimum = if closed { 3 } else { 2 };
        if points.len() < minimum {
            return Err(format!("São necessários ao menos {minimum} pontos."));
        }
        if points.iter().any(|p| !p.x.is_finite() || !p.y.is_finite()) {
            return Err("A curva contém coordenadas não finitas.".into());
        }
        let mut segments = points.iter().skip(1)
            .map(|p| CurveSegment::Line { to: *p })
            .collect::<Vec<_>>();
        if closed {
            segments.push(CurveSegment::Line { to: points[0] });
        }
        Ok(Self { start: points[0], segments, closed })
    }

    pub fn segment_count(&self) -> usize { self.segments.len() }
    pub fn endpoint(&self) -> Option<Point2> {
        self.segments.last().map(CurveSegment::endpoint).or(Some(self.start))
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub enum BoundaryCurve { Outline, SewingLine }

/// Intervalo de segmentos; segment_end é exclusivo.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct EdgeRef {
    pub piece_id: PieceId,
    pub curve: BoundaryCurve,
    pub segment_start: u32,
    pub segment_end: u32,
    pub reversed: bool,
}

impl EdgeRef {
    pub fn single_segment(piece_id: PieceId, segment_index: u32) -> Self {
        Self {
            piece_id,
            curve: BoundaryCurve::Outline,
            segment_start: segment_index,
            segment_end: segment_index + 1,
            reversed: false,
        }
    }
}

/// Duas bordas costuradas. A folga é armazenada em metros; sua aplicação física fica a cargo
/// do sistema de costura, não do formato de dados.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct SeamDefinition {
    pub id: SeamId,
    pub name: String,
    pub a: EdgeRef,
    pub b: EdgeRef,
    pub ease_meters: f64,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct GrainLine { pub start: Point2, pub end: Point2 }

/// Tipo semântico de uma linha de dobra usada pelo preview e pela simulação.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
pub enum FoldKind {
    #[default]
    Valley,
    Mountain,
    Decorative,
}

/// Linha interna com parâmetros mecânicos; stiffness é um coeficiente normalizado em [0, 1].
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct FoldLine {
    pub curve: Curve2D,
    pub kind: FoldKind,
    pub stiffness: f64,
}

/// Metadados de uma prega. A curva localiza o eixo; depth e angle descrevem sua forma-alvo.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct Pleat {
    pub axis: Curve2D,
    pub depth_meters: f64,
    pub angle_radians: f64,
}

/// Propriedades físicas específicas do molde, em unidades SI.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct FabricProperties {
    pub density_kg_m2: f64,
    pub stretch_warp_n_m: f64,
    pub stretch_weft_n_m: f64,
    pub shear_n_m: f64,
    pub bend_nm: f64,
    pub thickness_meters: f64,
    pub particle_distance_meters: f64,
}

impl Default for FabricProperties {
    fn default() -> Self {
        Self {
            density_kg_m2: 0.18,
            stretch_warp_n_m: 8_000.0,
            stretch_weft_n_m: 6_000.0,
            shear_n_m: 1_200.0,
            bend_nm: 0.0001,
            thickness_meters: 0.001,
            particle_distance_meters: 0.04,
        }
    }
}

/// Transformação afim 2D de gradação para um tamanho comercial.
/// Matriz armazenada como [m00, m01, tx, m10, m11, ty].
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct GradingRule {
    pub label: String,
    pub matrix: [f64; 6],
}

impl GradingRule {
    pub fn transform(&self, point: Point2) -> Point2 {
        Point2::new(
            self.matrix[0] * point.x + self.matrix[1] * point.y + self.matrix[2],
            self.matrix[3] * point.x + self.matrix[4] * point.y + self.matrix[5],
        )
    }
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct Dart {
    pub apex: Point2,
    pub leg_a: Point2,
    pub leg_b: Point2,
    pub intake_meters: f64,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct Notch {
    pub id: u64,
    pub segment_index: u32,
    pub t: f64,
    pub depth_meters: f64,
    pub width_meters: f64,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct Hole { pub center: Point2, pub radius_meters: f64 }

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct Annotation {
    pub text: String,
    pub position: Point2,
    pub rotation_degrees: f64,
}

/// Transformação de montagem no espaço 3D. A origem é um deslocamento relativo ao
/// posicionamento automático do painel; a rotação é expressa em graus XYZ.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct AssemblyTransform {
    pub translation_meters: [f64; 3],
    pub rotation_degrees: [f64; 3],
    /// Índices de âncoras do contorno a fixar na simulação. Vazio significa peça livre.
    #[serde(default)]
    pub pinned_anchors: Vec<u32>,
}

impl Default for AssemblyTransform {
    fn default() -> Self {
        Self {
            translation_meters: [0.0; 3],
            rotation_degrees: [0.0; 3],
            pinned_anchors: Vec::new(),
        }
    }
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct PatternPiece {
    pub id: PieceId,
    pub name: String,
    /// Contorno de corte, separado da linha de costura e da margem.
    pub outline: Curve2D,
    pub sewing_line: Option<Curve2D>,
    pub seam_allowance_meters: f64,
    pub grain_line: Option<GrainLine>,
    #[serde(default)]
    pub internal_lines: Vec<Curve2D>,
    #[serde(default)]
    pub fold_lines: Vec<FoldLine>,
    #[serde(default)]
    pub pleats: Vec<Pleat>,
    #[serde(default)]
    pub fabric: FabricProperties,
    #[serde(default)]
    pub assembly: AssemblyTransform,
    pub darts: Vec<Dart>,
    pub notches: Vec<Notch>,
    pub holes: Vec<Hole>,
    pub annotations: Vec<Annotation>,
}

impl PatternPiece {
    pub fn from_polygon(id: PieceId, name: impl Into<String>, points: &[Point2]) -> Result<Self, String> {
        Ok(Self {
            id,
            name: name.into(),
            outline: Curve2D::polyline(points, true)?,
            sewing_line: None,
            seam_allowance_meters: 0.01,
            grain_line: None,
            internal_lines: Vec::new(),
            fold_lines: Vec::new(),
            pleats: Vec::new(),
            fabric: FabricProperties::default(),
            assembly: AssemblyTransform::default(),
            darts: Vec::new(),
            notches: Vec::new(),
            holes: Vec::new(),
            annotations: Vec::new(),
        })
    }
}

/// Par de moldes espelhados em torno do eixo vertical x = axis_x.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct SymmetryLink {
    pub piece_a_id: PieceId,
    pub piece_b_id: PieceId,
    pub axis_x: f64,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct PatternProject {
    pub schema_version: u32,
    pub name: String,
    pub units: LengthUnit,
    pub pieces: Vec<PatternPiece>,
    pub seams: Vec<SeamDefinition>,
    #[serde(default)]
    pub grading_rules: Vec<GradingRule>,
    #[serde(default)]
    pub symmetry_links: Vec<SymmetryLink>,
}

impl PatternProject {
    pub const CURRENT_SCHEMA_VERSION: u32 = 1;

    pub fn new(name: impl Into<String>) -> Self {
        Self {
            schema_version: Self::CURRENT_SCHEMA_VERSION,
            name: name.into(),
            units: LengthUnit::Centimeter,
            pieces: Vec::new(),
            seams: Vec::new(),
            grading_rules: Vec::new(),
            symmetry_links: Vec::new(),
        }
    }

    pub fn validate(&self) -> Result<(), String> {
        if self.schema_version != Self::CURRENT_SCHEMA_VERSION {
            return Err(format!(
                "Versão de projeto {} não suportada (esperada {}).",
                self.schema_version, Self::CURRENT_SCHEMA_VERSION
            ));
        }
        let mut piece_ids = HashSet::new();
        for piece in &self.pieces {
            if !piece_ids.insert(piece.id) {
                return Err(format!("ID de peça duplicado: {}.", piece.id));
            }
            if !piece.seam_allowance_meters.is_finite() || piece.seam_allowance_meters < 0.0 {
                return Err(format!("Margem de costura inválida na peça '{}'.", piece.name));
            }
            if !piece.outline.closed || piece.outline.segments.len() < 3 {
                return Err(format!("O contorno da peça '{}' precisa ser fechado e ter ao menos 3 segmentos.", piece.name));
            }
            if piece.outline.segments.iter().any(|s| {
                let points = match s {
                    CurveSegment::Line { to } => vec![*to],
                    CurveSegment::CubicBezier { control1, control2, to } => vec![*control1, *control2, *to],
                };
                points.iter().any(|p| !p.x.is_finite() || !p.y.is_finite())
            }) || !piece.outline.start.x.is_finite() || !piece.outline.start.y.is_finite() {
                return Err(format!("O contorno da peça '{}' contém coordenadas não finitas.", piece.name));
            }
            let fabric_values = [
                piece.fabric.density_kg_m2,
                piece.fabric.stretch_warp_n_m,
                piece.fabric.stretch_weft_n_m,
                piece.fabric.shear_n_m,
                piece.fabric.bend_nm,
                piece.fabric.thickness_meters,
                piece.fabric.particle_distance_meters,
            ];
            if fabric_values.iter().any(|value| !value.is_finite() || *value <= 0.0) {
                return Err(format!("As propriedades físicas do molde '{}' precisam ser finitas e positivas.", piece.name));
            }
            if piece.assembly.translation_meters.iter().chain(piece.assembly.rotation_degrees.iter())
                .any(|value| !value.is_finite())
            {
                return Err(format!("A transformação 3D do molde '{}' contém valores não finitos.", piece.name));
            }
            let anchor_count = piece.outline.segments.len();
            let mut pinned = HashSet::new();
            for anchor in &piece.assembly.pinned_anchors {
                if *anchor as usize >= anchor_count {
                    return Err(format!("O molde '{}' referencia o vértice fixado {} que não existe.", piece.name, anchor + 1));
                }
                if !pinned.insert(*anchor) {
                    return Err(format!("O vértice {} do molde '{}' está fixado mais de uma vez.", anchor + 1, piece.name));
                }
            }
        }

        let mut symmetry_pairs = HashSet::new();
        let mut symmetry_members = HashSet::new();
        for link in &self.symmetry_links {
            if link.piece_a_id == link.piece_b_id {
                return Err("Um molde não pode ser vinculado à própria cópia espelhada.".into());
            }
            if !link.axis_x.is_finite() {
                return Err("O eixo de simetria contém coordenada inválida.".into());
            }
            if !piece_ids.contains(&link.piece_a_id) || !piece_ids.contains(&link.piece_b_id) {
                return Err("Uma relação de simetria referencia um molde inexistente.".into());
            }
            if !symmetry_members.insert(link.piece_a_id)
                || !symmetry_members.insert(link.piece_b_id)
            {
                return Err("Cada molde só pode participar de um único par de simetria.".into());
            }
            let pair = if link.piece_a_id < link.piece_b_id {
                (link.piece_a_id, link.piece_b_id)
            } else {
                (link.piece_b_id, link.piece_a_id)
            };
            if !symmetry_pairs.insert(pair) {
                return Err("Há mais de uma relação de simetria entre o mesmo par de moldes.".into());
            }
        }

        let mut seam_ids = HashSet::new();
        for seam in &self.seams {
            if !seam_ids.insert(seam.id) {
                return Err(format!("ID de costura duplicado: {}.", seam.id));
            }
            if !seam.ease_meters.is_finite() {
                return Err(format!("Folga inválida na costura '{}'.", seam.name));
            }
            for edge in [&seam.a, &seam.b] {
                let piece = self.pieces.iter().find(|p| p.id == edge.piece_id)
                    .ok_or_else(|| format!("A costura '{}' referencia uma peça inexistente.", seam.name))?;
                let curve = match edge.curve {
                    BoundaryCurve::Outline => Some(&piece.outline),
                    BoundaryCurve::SewingLine => piece.sewing_line.as_ref(),
                }.ok_or_else(|| format!("A costura '{}' referencia uma linha de costura inexistente.", seam.name))?;
                if edge.segment_start >= edge.segment_end
                    || edge.segment_end as usize > curve.segments.len()
                {
                    return Err(format!("Intervalo de borda inválido na costura '{}'.", seam.name));
                }
            }
        }
        Ok(())
    }

    pub fn to_json(&self) -> Result<String, String> {
        self.validate()?;
        serde_json::to_string_pretty(self).map_err(|e| format!("Falha ao serializar projeto: {e}"))
    }

    pub fn from_json(json: &str) -> Result<Self, String> {
        let project: Self = serde_json::from_str(json)
            .map_err(|e| format!("Arquivo de projeto inválido: {e}"))?;
        project.validate()?;
        Ok(project)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn project_round_trip_preserves_piece_and_units() {
        let mut project = PatternProject::new("Camisa");
        project.units = LengthUnit::Millimeter;
        project.pieces.push(PatternPiece::from_polygon(
            1, "Frente",
            &[
                Point2::new(0.0, 0.0), Point2::new(0.4, 0.0),
                Point2::new(0.35, 0.6), Point2::new(0.05, 0.6),
            ],
        ).unwrap());
        let decoded = PatternProject::from_json(&project.to_json().unwrap()).unwrap();
        assert_eq!(decoded, project);
    }

    #[test]
    fn old_project_json_loads_with_defaults_for_new_pattern_metadata() {
        let json = r#"{
            "schema_version":1,
            "name":"Legado",
            "units":"Centimeter",
            "pieces":[{
                "id":1,"name":"Frente",
                "outline":{"start":{"x":0.0,"y":0.0},"segments":[
                    {"Line":{"to":{"x":1.0,"y":0.0}}},
                    {"Line":{"to":{"x":1.0,"y":1.0}}},
                    {"Line":{"to":{"x":0.0,"y":1.0}}},
                    {"Line":{"to":{"x":0.0,"y":0.0}}}
                ],"closed":true},
                "sewing_line":null,"seam_allowance_meters":0.01,"grain_line":null,
                "darts":[],"notches":[],"holes":[],"annotations":[]
            }],
            "seams":[]
        }"#;
        let project = PatternProject::from_json(json).unwrap();
        assert!(project.grading_rules.is_empty());
        assert!(project.pieces[0].fold_lines.is_empty());
        assert!(project.pieces[0].pleats.is_empty());
        assert_eq!(project.pieces[0].fabric, FabricProperties::default());
    }

    #[test]
    fn old_project_piece_json_defaults_assembly_transform() {
        let json = r#"{ "schema_version":1, "name":"Legacy", "units":"Centimeter",
            "pieces":[{"id":1,"name":"A","outline":{"start":{"x":0.0,"y":0.0},
            "segments":[{"Line":{"to":{"x":1.0,"y":0.0}}},
            {"Line":{"to":{"x":1.0,"y":1.0}}},
            {"Line":{"to":{"x":0.0,"y":1.0}}},
            {"Line":{"to":{"x":0.0,"y":0.0}}}],"closed":true},
            "sewing_line":null,"seam_allowance_meters":0.01,"grain_line":null,
            "internal_lines":[],"fold_lines":[],"pleats":[],"fabric":{"density_kg_m2":0.18,
            "stretch_warp_n_m":8000.0,"stretch_weft_n_m":6000.0,"shear_n_m":1200.0,
            "bend_nm":0.0001,"thickness_meters":0.001,"particle_distance_meters":0.01},
            "darts":[],"notches":[],"holes":[],"annotations":[]}],"seams":[]}"#;
        let project = PatternProject::from_json(json).unwrap();
        assert_eq!(project.pieces[0].assembly, AssemblyTransform::default());
    }

    #[test]
    fn old_project_json_defaults_symmetry_links_to_empty() {
        let json = r#"{
            "schema_version":1,"name":"Legado","units":"Centimeter",
            "pieces":[],"seams":[]
        }"#;
        let project = PatternProject::from_json(json).unwrap();
        assert!(project.symmetry_links.is_empty());
    }

    #[test]
    fn symmetry_links_validate_piece_ids_and_axis() {
        let mut project = PatternProject::new("Simetria");
        project.pieces.push(PatternPiece::from_polygon(1, "A", &[
            Point2::new(0.0, 0.0), Point2::new(1.0, 0.0),
            Point2::new(1.0, 1.0), Point2::new(0.0, 1.0),
        ]).unwrap());
        project.pieces.push(PatternPiece::from_polygon(2, "B", &[
            Point2::new(2.0, 0.0), Point2::new(3.0, 0.0),
            Point2::new(3.0, 1.0), Point2::new(2.0, 1.0),
        ]).unwrap());
        project.symmetry_links.push(SymmetryLink {
            piece_a_id: 1, piece_b_id: 2, axis_x: 1.5,
        });
        assert!(project.validate().is_ok());
        project.symmetry_links[0].axis_x = f64::NAN;
        assert!(project.validate().is_err());
    }

    #[test]
    fn grading_rule_applies_affine_transform() {
        let rule = GradingRule {
            label: "Tamanho M".into(),
            matrix: [1.0, 0.0, 0.1, 0.0, 1.5, -0.02],
        };
        let result = rule.transform(Point2::new(0.2, 0.4));
        assert!((result.x - 0.3).abs() < 1e-9);
        assert!((result.y - 0.58).abs() < 1e-9);
    }

    #[test]
    fn validation_rejects_seam_with_missing_piece() {
        let mut project = PatternProject::new("Teste");
        project.seams.push(SeamDefinition {
            id: 1, name: "Costura inválida".into(),
            a: EdgeRef::single_segment(7, 0),
            b: EdgeRef::single_segment(8, 0), ease_meters: 0.0,
        });
        assert!(project.validate().is_err());
    }
}
