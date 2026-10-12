//! Modelo de domínio para moldes de vestuário.
//! Esta crate não depende de UI, renderização, geometria computacional ou física.
//! As medidas lineares internas são armazenadas em metros.

use serde::{Deserialize, Deserializer, Serialize};
use std::collections::HashSet;

/// Limite defensivo contra projetos com geometria patológica.
const MAX_PATTERN_SEGMENTS: usize = 4_096;

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

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, Serialize, Deserialize)]
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

/// Comprimento de uma polilinha aberta, em metros. Usado pelo editor e pelo solver para
/// que ambos meçam cadeias de pontos com exatamente a mesma regra.
pub fn polyline_length_meters(points: &[Point2]) -> f64 {
    points.windows(2).map(|pair| pair[0].distance(pair[1])).sum()
}

/// Parâmetro de uma curva. t pertence ao segmento indexado e está no intervalo [0, 1].
#[derive(Clone, Copy, Debug, PartialEq, Serialize, Deserialize)]
pub struct CurveParam {
    pub segment: u32,
    pub t: f64,
}

/// Caminho parcial ou completo ao longo de uma curva de contorno/linha de costura.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct EdgePath {
    pub piece_id: PieceId,
    pub curve: BoundaryCurve,
    pub start: CurveParam,
    pub end: CurveParam,
    #[serde(default)]
    pub reversed: bool,
}

impl From<EdgeRef> for EdgePath {
    fn from(edge: EdgeRef) -> Self {
        Self {
            piece_id: edge.piece_id,
            curve: edge.curve,
            start: CurveParam { segment: edge.segment_start, t: 0.0 },
            end: CurveParam { segment: edge.segment_end.saturating_sub(1), t: 1.0 },
            reversed: edge.reversed,
        }
    }
}

impl EdgePath {
    pub fn from_segments(
        piece_id: PieceId,
        curve: BoundaryCurve,
        segment_start: u32,
        segment_end: u32,
        reversed: bool,
    ) -> Self {
        EdgeRef { piece_id, curve, segment_start, segment_end, reversed }.into()
    }

    pub fn start_position(&self) -> f64 { self.start.segment as f64 + self.start.t }
    pub fn end_position(&self) -> f64 { self.end.segment as f64 + self.end.t }

    pub fn to_edge_ref_if_full_segments(&self) -> Option<EdgeRef> {
        if self.start.t.abs() > 1.0e-9 || (self.end.t - 1.0).abs() > 1.0e-9 {
            return None;
        }
        Some(EdgeRef {
            piece_id: self.piece_id,
            curve: self.curve,
            segment_start: self.start.segment,
            segment_end: self.end.segment.checked_add(1)?,
            reversed: self.reversed,
        })
    }
}

/// Duas bordas costuradas. O desserializador migra o antigo EdgeRef automaticamente.
#[derive(Clone, Debug, PartialEq, Serialize)]
pub struct SeamDefinition {
    pub id: SeamId,
    pub name: String,
    pub sides_a: Vec<EdgePath>,
    pub sides_b: Vec<EdgePath>,
    pub ease_meters: f64,
}

impl<'de> Deserialize<'de> for SeamDefinition {
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where D: Deserializer<'de> {
        #[derive(Deserialize)]
        #[serde(untagged)]
        enum LegacyOrPath {
            Path(EdgePath),
            Legacy(EdgeRef),
        }
        #[derive(Deserialize)]
        struct RawSeam {
            id: SeamId,
            name: String,
            #[serde(default)]
            a: Option<LegacyOrPath>,
            #[serde(default)]
            b: Option<LegacyOrPath>,
            #[serde(default)]
            sides_a: Option<Vec<LegacyOrPath>>,
            #[serde(default)]
            sides_b: Option<Vec<LegacyOrPath>>,
            ease_meters: f64,
        }
        let raw = RawSeam::deserialize(deserializer)?;
        let convert = |edge| match edge {
            LegacyOrPath::Path(path) => path,
            LegacyOrPath::Legacy(edge) => edge.into(),
        };
        let migrate_side = |paths: Option<Vec<LegacyOrPath>>, legacy: Option<LegacyOrPath>| {
            paths.filter(|paths| !paths.is_empty())
                .map(|paths| paths.into_iter().map(convert).collect())
                .or_else(|| legacy.map(|edge| vec![convert(edge)]))
                .unwrap_or_default()
        };
        Ok(Self {
            id: raw.id,
            name: raw.name,
            sides_a: migrate_side(raw.sides_a, raw.a),
            sides_b: migrate_side(raw.sides_b, raw.b),
            ease_meters: raw.ease_meters,
        })
    }
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

/// Linha interna com parâmetros mecânicos; stiffness é a fração da rigidez de flexão local.
fn default_fold_angle() -> f64 { std::f64::consts::FRAC_PI_6 }

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct FoldLine {
    pub curve: Curve2D,
    pub kind: FoldKind,
    /// Fração da rigidez de flexão em [0, 1]; valores baixos facilitam a dobra.
    pub stiffness: f64,
    /// Ângulo-alvo assinado da dobra, em radianos. Padrão compatível com arquivos antigos: 30°.
    #[serde(default = "default_fold_angle")]
    pub angle_radians: f64,
}

/// Metadados de uma prega. A curva localiza o eixo; depth e angle descrevem sua forma-alvo.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct Pleat {
    pub axis: Curve2D,
    pub depth_meters: f64,
    pub angle_radians: f64,
}

/// Propriedades físicas específicas do molde, em unidades SI.
fn default_fabric_friction() -> f64 { 0.4 }

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct FabricProperties {
    pub density_kg_m2: f64,
    pub stretch_warp_n_m: f64,
    pub stretch_weft_n_m: f64,
    pub shear_n_m: f64,
    pub bend_nm: f64,
    pub thickness_meters: f64,
    /// Atrito do tecido em contato com avatar e outras partes do próprio tecido.
    #[serde(default = "default_fabric_friction")]
    pub friction_coefficient: f64,
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
            friction_coefficient: default_fabric_friction(),
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
    /// Escala local por eixo. O valor 1.0 preserva a dimensão original.
    #[serde(default = "default_assembly_scale")]
    pub scale: [f64; 3],
    /// Índices de âncoras do contorno a fixar na simulação. Vazio significa peça livre.
    #[serde(default)]
    pub pinned_anchors: Vec<u32>,
}

fn default_assembly_scale() -> [f64; 3] { [1.0; 3] }

impl Default for AssemblyTransform {
    fn default() -> Self {
        Self {
            translation_meters: [0.0; 3],
            rotation_degrees: [0.0; 3],
            scale: default_assembly_scale(),
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

/// Medidas do manequim procedural associado ao projeto 2D→3D.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct AvatarDimensions {
    #[serde(default = "default_avatar_height")]
    pub height_meters: f64,
    #[serde(default = "default_avatar_shoulders")]
    pub shoulder_width_meters: f64,
    #[serde(default = "default_avatar_waist")]
    pub waist_width_meters: f64,
    #[serde(default = "default_avatar_hips")]
    pub hip_width_meters: f64,
}
fn default_avatar_height() -> f64 { 1.72 }
fn default_avatar_shoulders() -> f64 { 0.42 }
fn default_avatar_waist() -> f64 { 0.29 }
fn default_avatar_hips() -> f64 { 0.36 }

impl Default for AvatarDimensions {
    fn default() -> Self {
        Self { height_meters: default_avatar_height(), shoulder_width_meters: default_avatar_shoulders(),
            waist_width_meters: default_avatar_waist(), hip_width_meters: default_avatar_hips() }
    }
}

fn default_project_schema_version() -> u32 { 1 }

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct PatternProject {
    #[serde(default = "default_project_schema_version")]
    pub schema_version: u32,
    pub name: String,
    pub units: LengthUnit,
    pub pieces: Vec<PatternPiece>,
    pub seams: Vec<SeamDefinition>,
    #[serde(default)]
    pub avatar: AvatarDimensions,
    #[serde(default)]
    pub grading_rules: Vec<GradingRule>,
    #[serde(default)]
    pub symmetry_links: Vec<SymmetryLink>,
}

fn point_has_non_finite(point: &Point2) -> bool {
    !point.x.is_finite() || !point.y.is_finite()
}

fn curve_has_non_finite(curve: &Curve2D) -> bool {
    if point_has_non_finite(&curve.start) {
        return true;
    }
    curve.segments.iter().any(|segment| match segment {
        CurveSegment::Line { to } => point_has_non_finite(to),
        CurveSegment::CubicBezier { control1, control2, to } => {
            point_has_non_finite(control1)
                || point_has_non_finite(control2)
                || point_has_non_finite(to)
        }
    })
}

impl PatternProject {
    pub const CURRENT_SCHEMA_VERSION: u32 = 2;

    pub fn new(name: impl Into<String>) -> Self {
        Self {
            schema_version: Self::CURRENT_SCHEMA_VERSION,
            name: name.into(),
            units: LengthUnit::Centimeter,
            pieces: Vec::new(),
            seams: Vec::new(),
            avatar: AvatarDimensions::default(),
            grading_rules: Vec::new(),
            symmetry_links: Vec::new(),
        }
    }

    /// Mede a mesma polilinha de contorno que entra no builder: Béziers em 8 passos e
    /// todos os piques/extremos parciais inseridos como vértices forçados.
    pub fn edge_length_meters(&self, edge: &EdgePath) -> Option<f64> {
        let piece = self.pieces.iter().find(|piece| piece.id == edge.piece_id)?;
        let curve = match edge.curve {
            BoundaryCurve::Outline => Some(&piece.outline),
            BoundaryCurve::SewingLine => piece.sewing_line.as_ref(),
        }?;
        if !edge.start.t.is_finite() || !edge.end.t.is_finite()
            || !(0.0..=1.0).contains(&edge.start.t) || !(0.0..=1.0).contains(&edge.end.t)
            || edge.start_position() >= edge.end_position()
        { return Some(0.0); }
        let first = edge.start.segment as usize;
        let last = edge.end.segment as usize;
        if last >= curve.segments.len() { return None; }

        let mut points = Vec::new();
        for index in first..=last {
            let t0 = if index == first { edge.start.t } else { 0.0 };
            let t1 = if index == last { edge.end.t } else { 1.0 };
            if t1 < t0 { return Some(0.0); }
            let p0 = if index == 0 { curve.start } else { curve.segments[index - 1].endpoint() };
            let segment = &curve.segments[index];
            let eval = |t: f64| match segment {
                CurveSegment::Line { to } => Point2::new(
                    p0.x + (to.x - p0.x) * t,
                    p0.y + (to.y - p0.y) * t,
                ),
                CurveSegment::CubicBezier { control1, control2, to } => {
                    let u = 1.0 - t;
                    Point2::new(
                        u*u*u*p0.x + 3.0*u*u*t*control1.x + 3.0*u*t*t*control2.x + t*t*t*to.x,
                        u*u*u*p0.y + 3.0*u*u*t*control1.y + 3.0*u*t*t*control2.y + t*t*t*to.y,
                    )
                }
            };
            let mut ts = Vec::new();
            if matches!(segment, CurveSegment::CubicBezier { .. }) {
                ts.extend((1..8).map(|step| step as f64 / 8.0));
            }
            if edge.curve == BoundaryCurve::Outline {
                ts.extend(piece.notches.iter().filter_map(|notch| {
                    (notch.segment_index as usize == index && notch.t > t0 && notch.t < t1)
                        .then_some(notch.t)
                }));
                for seam in &self.seams {
                    for path in seam.sides_a.iter().chain(seam.sides_b.iter()) {
                        if path.piece_id == piece.id && path.curve == edge.curve {
                            for param in [path.start, path.end] {
                                if param.segment as usize == index && param.t > t0 && param.t < t1 {
                                    ts.push(param.t);
                                }
                            }
                        }
                    }
                }
            }
            ts.retain(|t| *t > t0 + 1.0e-10 && *t < t1 - 1.0e-10);
            ts.push(t1);
            ts.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
            ts.dedup_by(|a, b| (*a - *b).abs() <= 1.0e-9);

            if points.is_empty() { points.push(eval(t0)); }
            for t in ts { points.push(eval(t)); }
        }
        if edge.reversed { points.reverse(); }
        Some(polyline_length_meters(&points))
    }

    /// Diagnósticos não bloqueantes que o painel pode apresentar ao usuário.
    pub fn validation_warnings(&self) -> Vec<String> {
        let mut warnings = Vec::new();
        for seam in &self.seams {
            let length = |paths: &[EdgePath]| paths.iter()
                .filter_map(|edge| self.edge_length_meters(edge)).sum::<f64>();
            let a = length(&seam.sides_a);
            let b = length(&seam.sides_b);
            let shortest = a.min(b);
            let longest = a.max(b);
            if shortest > 1.0e-9 && longest / shortest > 3.0 {
                warnings.push(format!(
                    "A costura '{}' tem razão de comprimentos A/B de {:.2}:1; confira os segmentos selecionados.",
                    seam.name, longest / shortest
                ));
            }
            for (label, paths) in [("A", &seam.sides_a), ("B", &seam.sides_b)] {
                for pair in paths.windows(2) {
                    let left_end = if pair[0].reversed { pair[0].start_position() } else { pair[0].end_position() };
                    let right_start = if pair[1].reversed { pair[1].end_position() } else { pair[1].start_position() };
                    let has_gap = if pair[0].piece_id == pair[1].piece_id && pair[0].curve == pair[1].curve {
                        (left_end - right_start).abs() > 1.0e-6
                    } else {
                        let end_param = if pair[0].reversed { pair[0].start } else { pair[0].end };
                        let start_param = if pair[1].reversed { pair[1].end } else { pair[1].start };
                        match (
                            edge_param_world(self, &pair[0], end_param),
                            edge_param_world(self, &pair[1], start_param),
                        ) {
                            (Some(a), Some(b)) => {
                                ((a[0]-b[0]).powi(2) + (a[1]-b[1]).powi(2) + (a[2]-b[2]).powi(2)).sqrt() > 0.01
                            }
                            _ => true,
                        }
                    };
                    if has_gap {
                        warnings.push(format!(
                            "A costura '{}' tem uma lacuna entre bordas consecutivas do lado {}.",
                            seam.name, label
                        ));
                    }
                }
            }
        }
        warnings
    }

    pub fn validate(&self) -> Result<(), String> {
        if self.schema_version != Self::CURRENT_SCHEMA_VERSION {
            return Err(format!(
                "Versão de projeto {} não suportada (esperada {}).",
                self.schema_version, Self::CURRENT_SCHEMA_VERSION
            ));
        }
        let avatar_values = [self.avatar.height_meters, self.avatar.shoulder_width_meters,
            self.avatar.waist_width_meters, self.avatar.hip_width_meters];
        if avatar_values.iter().any(|v| !v.is_finite()) {
            return Err("As medidas do avatar precisam ser finitas.".into());
        }
        if !(1.0..=2.5).contains(&self.avatar.height_meters)
            || !(0.20..=0.90).contains(&self.avatar.shoulder_width_meters)
            || !(0.15..=0.70).contains(&self.avatar.waist_width_meters)
            || !(0.18..=0.80).contains(&self.avatar.hip_width_meters)
        {
            return Err("As medidas do avatar estão fora dos limites físicos aceitos.".into());
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
            if piece.outline.segments.len() > MAX_PATTERN_SEGMENTS {
                return Err(format!("O contorno da peça '{}' excede o limite seguro de {} segmentos.", piece.name, MAX_PATTERN_SEGMENTS));
            }
            if curve_has_non_finite(&piece.outline) {
                return Err(format!("O contorno da peça '{}' contém coordenadas não finitas.", piece.name));
            }
            if piece.sewing_line.as_ref().is_some_and(|curve| {
                curve.segments.is_empty() || curve.segments.len() > MAX_PATTERN_SEGMENTS || curve_has_non_finite(curve)
            }) {
                return Err(format!("A linha de costura da peça '{}' contém geometria inválida ou excede o limite seguro de segmentos.", piece.name));
            }
            if piece.internal_lines.iter().any(|curve| {
                curve.segments.is_empty() || curve.segments.len() > MAX_PATTERN_SEGMENTS || curve_has_non_finite(curve)
            }) {
                return Err(format!("Uma linha interna da peça '{}' contém geometria inválida ou excede o limite seguro de segmentos.", piece.name));
            }
            if piece.grain_line.as_ref().is_some_and(|line| {
                point_has_non_finite(&line.start) || point_has_non_finite(&line.end)
            }) {
                return Err(format!("A linha de fio da peça '{}' contém coordenadas não finitas.", piece.name));
            }
            if piece.darts.iter().any(|dart| {
                point_has_non_finite(&dart.apex)
                    || point_has_non_finite(&dart.leg_a)
                    || point_has_non_finite(&dart.leg_b)
                    || !dart.intake_meters.is_finite()
                    || dart.intake_meters < 0.0
            }) {
                return Err(format!("Uma pence da peça '{}' contém coordenadas ou medidas inválidas.", piece.name));
            }
            if piece.holes.iter().any(|hole| {
                point_has_non_finite(&hole.center)
                    || !hole.radius_meters.is_finite()
                    || hole.radius_meters < 0.0
            }) {
                return Err(format!("Um furo da peça '{}' contém coordenadas ou raio inválido.", piece.name));
            }
            if piece.annotations.iter().any(|annotation| {
                point_has_non_finite(&annotation.position)
                    || !annotation.rotation_degrees.is_finite()
            }) {
                return Err(format!("Uma anotação da peça '{}' contém valores não finitos.", piece.name));
            }
            for fold in &piece.fold_lines {
                if fold.curve.segments.is_empty() || fold.curve.segments.len() > MAX_PATTERN_SEGMENTS || curve_has_non_finite(&fold.curve) {
                    return Err(format!("A linha de dobra da peça '{}' contém geometria inválida.", piece.name));
                }
                if !fold.stiffness.is_finite() || !(0.0..=1.0).contains(&fold.stiffness)
                    || !fold.angle_radians.is_finite()
                    || !(0.0..=std::f64::consts::PI).contains(&fold.angle_radians)
                {
                    return Err(format!("Parâmetros inválidos na linha de dobra do molde '{}'.", piece.name));
                }
            }
            for pleat in &piece.pleats {
                if pleat.axis.segments.is_empty() || pleat.axis.segments.len() > MAX_PATTERN_SEGMENTS || curve_has_non_finite(&pleat.axis) {
                    return Err(format!("O eixo de prega da peça '{}' contém geometria inválida.", piece.name));
                }
                if !pleat.depth_meters.is_finite() || pleat.depth_meters <= 0.0
                    || !pleat.angle_radians.is_finite()
                    || !(0.0..=std::f64::consts::PI).contains(&pleat.angle_radians)
                {
                    return Err(format!("Parâmetros inválidos na prega do molde '{}'.", piece.name));
                }
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
            if !piece.fabric.friction_coefficient.is_finite()
                || !(0.0..=5.0).contains(&piece.fabric.friction_coefficient)
            {
                return Err(format!("O atrito do tecido '{}' precisa estar entre 0 e 5.", piece.name));
            }
            if piece.assembly.translation_meters.iter()
                .any(|value| !value.is_finite() || value.abs() > 100.0)
            {
                return Err(format!("A translação do molde '{}' precisa ser finita e permanecer entre -100 m e 100 m.", piece.name));
            }
            if piece.assembly.rotation_degrees.iter()
                .any(|value| !value.is_finite() || value.abs() > 360_000.0)
            {
                return Err(format!("A rotação 3D do molde '{}' excede o limite seguro de ±360.000°.", piece.name));
            }
            if piece.assembly.scale.iter()
                .any(|value| !value.is_finite() || *value <= 0.0 || *value > 100.0)
            {
                return Err(format!("A escala 3D do molde '{}' precisa ser finita, positiva e no máximo 100.", piece.name));
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
            let mut notch_ids = HashSet::new();
            for notch in &piece.notches {
                if !notch_ids.insert(notch.id) {
                    return Err(format!("ID de pique duplicado {} no molde '{}'.", notch.id, piece.name));
                }
                if notch.segment_index as usize >= anchor_count
                    || !notch.t.is_finite() || !(0.0..=1.0).contains(&notch.t)
                    || !notch.depth_meters.is_finite() || notch.depth_meters <= 0.0
                    || !notch.width_meters.is_finite() || notch.width_meters <= 0.0
                {
                    return Err(format!("Pique {} inválido no molde '{}'.", notch.id, piece.name));
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

        if self.grading_rules.iter().any(|rule| {
            rule.matrix.iter().any(|value| !value.is_finite())
        }) {
            return Err("Uma regra de graduação contém coeficientes não finitos.".into());
        }

        let mut seam_ids = HashSet::new();
        let mut occupied_paths: Vec<(PieceId, BoundaryCurve, f64, f64, SeamId, bool)> = Vec::new();
        for seam in &self.seams {
            if !seam_ids.insert(seam.id) {
                return Err(format!("ID de costura duplicado: {}.", seam.id));
            }
            if !seam.ease_meters.is_finite() {
                return Err(format!("Folga inválida na costura '{}'.", seam.name));
            }
            if seam.sides_a.is_empty() || seam.sides_b.is_empty() {
                return Err(format!("A costura '{}' precisa ter pelo menos uma borda em cada lado.", seam.name));
            }
            for (is_a, paths) in [(true, &seam.sides_a), (false, &seam.sides_b)] {
                for edge in paths {
                    let piece = self.pieces.iter().find(|p| p.id == edge.piece_id)
                        .ok_or_else(|| format!("A costura '{}' referencia uma peça inexistente.", seam.name))?;
                    let curve = match edge.curve {
                        BoundaryCurve::Outline => Some(&piece.outline),
                        BoundaryCurve::SewingLine => piece.sewing_line.as_ref(),
                    }.ok_or_else(|| format!("A costura '{}' referencia uma linha de costura inexistente.", seam.name))?;
                    if !edge.start.t.is_finite() || !edge.end.t.is_finite()
                        || !(0.0..=1.0).contains(&edge.start.t) || !(0.0..=1.0).contains(&edge.end.t)
                        || edge.start.segment as usize >= curve.segments.len()
                        || edge.end.segment as usize >= curve.segments.len()
                        || edge.start_position() >= edge.end_position()
                    {
                        return Err(format!("Intervalo de borda inválido na costura '{}'.", seam.name));
                    }
                    let start = edge.start_position();
                    let end = edge.end_position();
                    if let Some((_, _, _, _, previous_seam, previous_side_a)) = occupied_paths.iter().find(|(piece_id, curve, lo, hi, _, _)| {
                        *piece_id == edge.piece_id && *curve == edge.curve
                            && start < *hi - 1.0e-9 && *lo < end - 1.0e-9
                    }) {
                        if *previous_seam == seam.id {
                            let relation = if *previous_side_a == is_a {
                                "A mesma borda está repetida/ sobreposta dentro de um dos lados"
                            } else {
                                "Os lados A e B da mesma peça possuem bordas sobrepostas"
                            };
                            return Err(format!("{relation} da costura '{}'.", seam.name));
                        }
                        return Err(format!(
                            "A borda da peça {} já está vinculada à costura {}; remova a sobreposição antes de continuar.",
                            edge.piece_id, previous_seam
                        ));
                    }
                    occupied_paths.push((edge.piece_id, edge.curve, start, end, seam.id, is_a));
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
        let mut project: Self = serde_json::from_str(json)
            .map_err(|e| format!("Arquivo de projeto inválido: {e}"))?;
        // Versão 1 armazenava uma costura com campos singulares a/b (EdgeRef).
        // O desserializador já migrou esses campos para sides_a/sides_b, portanto
        // podemos atualizar a versão de forma segura antes da validação.
        if project.schema_version == 1 {
            project.schema_version = Self::CURRENT_SCHEMA_VERSION;
        }
        project.validate()?;
        Ok(project)
    }
}

/// Parâmetros para gerar sugestões; a função é pura e não altera o projeto.
#[derive(Clone, Debug, PartialEq)]
pub struct SeamSuggestionParams {
    pub minimum_score: f64,
    pub maximum_length_ratio: f64,
    pub placement_distance_meters: f64,
}

impl Default for SeamSuggestionParams {
    fn default() -> Self {
        Self {
            minimum_score: 0.35,
            maximum_length_ratio: 4.0,
            placement_distance_meters: 0.75,
        }
    }
}

/// Candidata à aplicação pelo usuário; gerar sugestões nunca cria costuras no projeto.
#[derive(Clone, Debug, PartialEq)]
pub struct SeamSuggestion {
    pub a: EdgePath,
    pub b: EdgePath,
    pub score: f64,
    pub reasons: Vec<String>,
}

fn curve_point_at(curve: &Curve2D, segment_index: usize, t: f64) -> Option<Point2> {
    let segment = curve.segments.get(segment_index)?;
    let start = if segment_index == 0 { curve.start } else { curve.segments[segment_index - 1].endpoint() };
    let t = t.clamp(0.0, 1.0);
    match segment {
        CurveSegment::Line { to } => Some(Point2::new(
            start.x + (to.x - start.x) * t,
            start.y + (to.y - start.y) * t,
        )),
        CurveSegment::CubicBezier { control1, control2, to } => {
            let u = 1.0 - t;
            Some(Point2::new(
                u*u*u*start.x + 3.0*u*u*t*control1.x + 3.0*u*t*t*control2.x + t*t*t*to.x,
                u*u*u*start.y + 3.0*u*u*t*control1.y + 3.0*u*t*t*control2.y + t*t*t*to.y,
            ))
        }
    }
}

fn curve_segment_length_between(curve: &Curve2D, index: usize, start_t: f64, end_t: f64) -> f64 {
    if end_t <= start_t { return 0.0; }
    let Some(mut previous) = curve_point_at(curve, index, start_t) else { return 0.0; };
    let mut length = 0.0;
    for step in 1..=24 {
        let t = start_t + (end_t - start_t) * step as f64 / 24.0;
        let Some(point) = curve_point_at(curve, index, t) else { continue; };
        length += previous.distance(point);
        previous = point;
    }
    length
}

fn rotate_euler_xyz(point: [f64; 3], degrees: [f64; 3]) -> [f64; 3] {
    let (sx, cx) = degrees[0].to_radians().sin_cos();
    let (sy, cy) = degrees[1].to_radians().sin_cos();
    let (sz, cz) = degrees[2].to_radians().sin_cos();
    let x1 = point[0];
    let y1 = cx * point[1] - sx * point[2];
    let z1 = sx * point[1] + cx * point[2];
    let x2 = cy * x1 + sy * z1;
    let y2 = y1;
    let z2 = -sy * x1 + cy * z1;
    [cz * x2 - sz * y2, sz * x2 + cz * y2, z2]
}

fn edge_param_world(
    project: &PatternProject,
    edge: &EdgePath,
    param: CurveParam,
) -> Option<[f64; 3]> {
    let piece = project.pieces.iter().find(|piece| piece.id == edge.piece_id)?;
    let curve = match edge.curve {
        BoundaryCurve::Outline => Some(&piece.outline),
        BoundaryCurve::SewingLine => piece.sewing_line.as_ref(),
    }?;
    let point = curve_point_at(curve, param.segment as usize, param.t)?;
    let rotated = rotate_euler_xyz([point.x, point.y, 0.0], piece.assembly.rotation_degrees);
    Some([
        rotated[0] + piece.assembly.translation_meters[0],
        rotated[1] + piece.assembly.translation_meters[1],
        rotated[2] + piece.assembly.translation_meters[2],
    ])
}

fn edge_midpoint_3d(piece: &PatternPiece, segment: usize) -> Option<[f64; 3]> {
    let midpoint = curve_point_at(&piece.outline, segment, 0.5)?;
    let rotated = rotate_euler_xyz([midpoint.x, midpoint.y, 0.0], piece.assembly.rotation_degrees);
    Some([
        rotated[0] + piece.assembly.translation_meters[0],
        rotated[1] + piece.assembly.translation_meters[1],
        rotated[2] + piece.assembly.translation_meters[2],
    ])
}

fn edge_tangent_3d(piece: &PatternPiece, segment: usize) -> Option<[f64; 3]> {
    let p0 = curve_point_at(&piece.outline, segment, 0.49)?;
    let p1 = curve_point_at(&piece.outline, segment, 0.51)?;
    let direction = rotate_euler_xyz([p1.x - p0.x, p1.y - p0.y, 0.0], piece.assembly.rotation_degrees);
    let length = (direction[0]*direction[0] + direction[1]*direction[1] + direction[2]*direction[2]).sqrt();
    if length <= 1.0e-12 { return None; }
    Some([direction[0]/length, direction[1]/length, direction[2]/length])
}

fn notches_for_segment(piece: &PatternPiece, segment: usize) -> Vec<f64> {
    let mut positions = piece.notches.iter()
        .filter(|notch| notch.segment_index as usize == segment && notch.t.is_finite() && (0.0..=1.0).contains(&notch.t))
        .map(|notch| {
            let total = curve_segment_length_between(&piece.outline, segment, 0.0, 1.0);
            if total <= 1.0e-12 { notch.t } else {
                curve_segment_length_between(&piece.outline, segment, 0.0, notch.t) / total
            }
        }).collect::<Vec<_>>();
    positions.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
    positions
}

fn edge_was_sewn(project: &PatternProject, piece_id: PieceId, curve: BoundaryCurve, segment: u32) -> bool {
    project.seams.iter().any(|seam| seam.sides_a.iter().chain(seam.sides_b.iter()).any(|edge| {
        if edge.piece_id != piece_id || edge.curve != curve { return false; }
        let segment_start = segment as f64;
        let segment_end = segment_start + 1.0;
        segment_start < edge.end_position() - 1.0e-9
            && edge.start_position() < segment_end - 1.0e-9
    }))
}

/// Retorna pares de bordas candidatas ordenados por score e escolhidos de forma gulosa.
/// Não modifica `project`: o chamador deve mostrar sugestões e pedir aplicação explícita.
pub fn suggest_seams(
    project: &PatternProject,
    params: &SeamSuggestionParams,
) -> Vec<SeamSuggestion> {
    if !params.minimum_score.is_finite()
        || !params.maximum_length_ratio.is_finite()
        || !params.placement_distance_meters.is_finite()
        || params.maximum_length_ratio < 1.0
        || params.placement_distance_meters <= 0.0
    {
        return Vec::new();
    }

    let mut candidates: Vec<SeamSuggestion> = Vec::new();
    for (index_a, piece_a) in project.pieces.iter().enumerate() {
        for piece_b in project.pieces.iter().skip(index_a + 1) {
            for segment_a in 0..piece_a.outline.segments.len() {
                if edge_was_sewn(project, piece_a.id, BoundaryCurve::Outline, segment_a as u32) { continue; }
                let edge_a = EdgePath {
                    piece_id: piece_a.id, curve: BoundaryCurve::Outline,
                    start: CurveParam { segment: segment_a as u32, t: 0.0 },
                    end: CurveParam { segment: segment_a as u32, t: 1.0 },
                    reversed: false,
                };
                let Some(length_a) = project.edge_length_meters(&edge_a) else { continue; };
                if length_a <= 1.0e-9 { continue; }
                let Some(mid_a) = edge_midpoint_3d(piece_a, segment_a) else { continue; };
                let Some(tangent_a) = edge_tangent_3d(piece_a, segment_a) else { continue; };
                let notches_a = notches_for_segment(piece_a, segment_a);

                for segment_b in 0..piece_b.outline.segments.len() {
                    if edge_was_sewn(project, piece_b.id, BoundaryCurve::Outline, segment_b as u32) { continue; }
                    let edge_b = EdgePath {
                        piece_id: piece_b.id, curve: BoundaryCurve::Outline,
                        start: CurveParam { segment: segment_b as u32, t: 0.0 },
                        end: CurveParam { segment: segment_b as u32, t: 1.0 },
                        reversed: false,
                    };
                    let Some(length_b) = project.edge_length_meters(&edge_b) else { continue; };
                    if length_b <= 1.0e-9 { continue; }
                    let ratio = length_a.max(length_b) / length_a.min(length_b);
                    if ratio > params.maximum_length_ratio { continue; }
                    let Some(mid_b) = edge_midpoint_3d(piece_b, segment_b) else { continue; };
                    let distance = ((mid_a[0]-mid_b[0]).powi(2)
                        + (mid_a[1]-mid_b[1]).powi(2)
                        + (mid_a[2]-mid_b[2]).powi(2)).sqrt();
                    let proximity_score = (-distance / params.placement_distance_meters).exp();
                    let ratio_score = 1.0 / ratio;
                    let tangent_b = edge_tangent_3d(piece_b, segment_b).unwrap_or([0.0, 0.0, 0.0]);
                    let tangent_dot = (tangent_a[0]*tangent_b[0] + tangent_a[1]*tangent_b[1]
                        + tangent_a[2]*tangent_b[2]).clamp(-1.0, 1.0);
                    let orientation_score = (1.0 - tangent_dot) * 0.5;

                    let notches_b = notches_for_segment(piece_b, segment_b);
                    let count_similarity = 1.0 - notches_a.len().abs_diff(notches_b.len()) as f64
                        / notches_a.len().max(notches_b.len()).max(1) as f64;
                    let position_similarity = if notches_a.is_empty() && notches_b.is_empty() {
                        1.0
                    } else if notches_a.is_empty() || notches_b.is_empty() {
                        0.35
                    } else {
                        let deltas = notches_a.iter().map(|a| notches_b.iter()
                            .map(|b| (a - b).abs()).fold(f64::INFINITY, f64::min)).collect::<Vec<_>>();
                        let mean_delta = deltas.iter().sum::<f64>() / deltas.len() as f64;
                        (1.0 - mean_delta).clamp(0.0, 1.0)
                    };
                    let notch_score = if notches_a.is_empty() && notches_b.is_empty() {
                        0.5 // sem piques, esse critério é neutro em vez de premiar ausência de informação
                    } else {
                        0.5 * count_similarity + 0.5 * position_similarity
                    };
                    let score = (0.30 * notch_score + 0.25 * ratio_score
                        + 0.25 * proximity_score + 0.20 * orientation_score).clamp(0.0, 1.0);
                    if score < params.minimum_score { continue; }

                    let mut reasons = Vec::new();
                    reasons.push(format!("Comprimentos: {:.2} cm e {:.2} cm (razão {:.2}:1).",
                        length_a * 100.0, length_b * 100.0, ratio));
                    reasons.push(format!("Proximidade de montagem: {:.1} cm.", distance * 100.0));
                    reasons.push(format!("Orientação de bordas: {:.0}% de compatibilidade.", orientation_score * 100.0));
                    if !notches_a.is_empty() || !notches_b.is_empty() {
                        reasons.push(format!("Piques: {} em A e {} em B; correspondência normalizada {:.0}%.",
                            notches_a.len(), notches_b.len(), notch_score * 100.0));
                    } else {
                        reasons.push("Nenhum pique nas bordas; pontuação neutra por contagem/posição.".into());
                    }
                    candidates.push(SeamSuggestion { a: edge_a.clone(), b: edge_b, score, reasons });
                }
            }
        }
    }
    candidates.sort_by(|a,b| b.score.partial_cmp(&a.score).unwrap_or(std::cmp::Ordering::Equal));
    let mut used: HashSet<(PieceId, u32)> = HashSet::new();
    let mut result = Vec::new();
    for candidate in candidates {
        let key_a = (candidate.a.piece_id, candidate.a.start.segment);
        let key_b = (candidate.b.piece_id, candidate.b.start.segment);
        if used.contains(&key_a) || used.contains(&key_b) { continue; }
        used.insert(key_a);
        used.insert(key_b);
        result.push(candidate);
    }
    result
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
        assert_eq!(project.schema_version, PatternProject::CURRENT_SCHEMA_VERSION);
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
        // Novos parâmetros físicos de versões posteriores assumem seus defaults em projetos legados.
        assert_eq!(project.pieces[0].fabric.friction_coefficient, 0.4);
    }

    #[test]
    fn legacy_fold_line_defaults_physical_target_angle() {
        let json = r#"{
            "curve":{"start":{"x":0.1,"y":0.1},"segments":[
                {"Line":{"to":{"x":0.9,"y":0.9}}}
            ],"closed":false},
            "kind":"Valley",
            "stiffness":0.35
        }"#;
        let fold: FoldLine = serde_json::from_str(json).unwrap();
        assert!((fold.angle_radians - std::f64::consts::FRAC_PI_6).abs() < 1.0e-12);
    }

    #[test]
    fn old_project_json_defaults_symmetry_links_to_empty() {
        let json = r#"{
            "schema_version":1,"name":"Legado","units":"Centimeter",
            "pieces":[],"seams":[]
        }"#;
        let project = PatternProject::from_json(json).unwrap();
        assert!(project.symmetry_links.is_empty());
        assert_eq!(project.avatar, AvatarDimensions::default());
    }

    #[test]
    fn zero_fabric_friction_is_valid_but_negative_friction_is_rejected() {
        let mut project = PatternProject::new("Atrito");
        project.pieces.push(PatternPiece::from_polygon(1, "A", &[
            Point2::new(0.0, 0.0), Point2::new(1.0, 0.0),
            Point2::new(1.0, 1.0), Point2::new(0.0, 1.0),
        ]).unwrap());
        project.pieces[0].fabric.friction_coefficient = 0.0;
        assert!(project.validate().is_ok());
        project.pieces[0].fabric.friction_coefficient = -0.01;
        assert!(project.validate().is_err());
    }

    #[test]
    fn notches_validate_segment_position_dimensions_and_unique_ids() {
        let mut project = PatternProject::new("Piques");
        project.pieces.push(PatternPiece::from_polygon(1, "A", &[
            Point2::new(0.0, 0.0), Point2::new(1.0, 0.0),
            Point2::new(1.0, 1.0), Point2::new(0.0, 1.0),
        ]).unwrap());
        project.pieces[0].notches.push(Notch {
            id: 1, segment_index: 0, t: 0.5, depth_meters: 0.008, width_meters: 0.012,
        });
        assert!(project.validate().is_ok());
        project.pieces[0].notches[0].t = f64::NAN;
        assert!(project.validate().is_err());
        project.pieces[0].notches[0].t = 0.5;
        let duplicate = project.pieces[0].notches[0].clone();
        project.pieces[0].notches.push(duplicate);
        assert!(project.validate().is_err());
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
    fn legacy_edge_refs_migrate_to_single_edge_paths() {
        let json = r#"{
            "schema_version":1,"name":"Legacy seams","units":"Centimeter",
            "pieces":[],"seams":[{
                "id":1,"name":"Old seam",
                "a":{"piece_id":10,"curve":"Outline","segment_start":2,"segment_end":5,"reversed":true},
                "b":{"piece_id":11,"curve":"Outline","segment_start":0,"segment_end":1,"reversed":false},
                "ease_meters":0.02
            }]
        }"#;
        let project: PatternProject = serde_json::from_str(json).unwrap();
        let seam = &project.seams[0];
        assert_eq!(seam.sides_a.len(), 1);
        assert_eq!(seam.sides_b.len(), 1);
        assert_eq!(seam.sides_a[0].start, CurveParam { segment: 2, t: 0.0 });
        assert_eq!(seam.sides_a[0].end, CurveParam { segment: 4, t: 1.0 });
        assert!(seam.sides_a[0].reversed);
    }

    #[test]
    fn validation_rejects_seam_with_missing_piece() {
        let mut project = PatternProject::new("Teste");
        project.seams.push(SeamDefinition {
            id: 1, name: "Costura inválida".into(),
            sides_a: vec![EdgeRef::single_segment(7, 0).into()], sides_b: vec![EdgeRef::single_segment(8, 0).into()], ease_meters: 0.0,
        });
        assert!(project.validate().is_err());
    }
    #[test]
    fn validation_rejects_non_finite_internal_curve_geometry() {
        let mut project = PatternProject::new("Detalhes");
        let mut piece = PatternPiece::from_polygon(1, "Frente", &[
            Point2::new(0.0, 0.0), Point2::new(1.0, 0.0),
            Point2::new(1.0, 1.0), Point2::new(0.0, 1.0),
        ]).unwrap();
        piece.internal_lines.push(Curve2D {
            start: Point2::new(f64::NAN, 0.5),
            segments: vec![CurveSegment::Line { to: Point2::new(0.8, 0.5) }],
            closed: false,
        });
        project.pieces.push(piece);
        let error = project.validate().unwrap_err();
        assert!(error.contains("linha interna"));
    }

    #[test]
    fn validation_rejects_invalid_assembly_scale_and_dart_points() {
        let mut project = PatternProject::new("Transformações");
        project.pieces.push(PatternPiece::from_polygon(1, "Frente", &[
            Point2::new(0.0, 0.0), Point2::new(1.0, 0.0),
            Point2::new(1.0, 1.0), Point2::new(0.0, 1.0),
        ]).unwrap());
        project.pieces[0].assembly.scale[0] = 0.0;
        assert!(project.validate().unwrap_err().contains("escala 3D"));

        project.pieces[0].assembly.scale = [1.0; 3];
        project.pieces[0].darts.push(Dart {
            apex: Point2::new(f64::NAN, 0.5),
            leg_a: Point2::new(0.2, 0.7),
            leg_b: Point2::new(0.8, 0.7),
            intake_meters: 0.1,
        });
        assert!(project.validate().unwrap_err().contains("pence"));
    }

    #[test]
    fn validation_rejects_pathological_outline_segment_count() {
        let mut project = PatternProject::new("Muitos segmentos");
        let mut piece = PatternPiece::from_polygon(1, "Frente", &[
            Point2::new(0.0, 0.0), Point2::new(1.0, 0.0),
            Point2::new(1.0, 1.0), Point2::new(0.0, 1.0),
        ]).unwrap();
        piece.outline.segments = vec![
            CurveSegment::Line { to: Point2::new(0.5, 0.5) };
            MAX_PATTERN_SEGMENTS + 1
        ];
        project.pieces.push(piece);
        assert!(project.validate().unwrap_err().contains("limite seguro"));
    }

    #[test]
    fn validation_rejects_out_of_range_transforms_before_f32_conversion() {
        let mut project = PatternProject::new("Limites 3D");
        project.pieces.push(PatternPiece::from_polygon(1, "Frente", &[
            Point2::new(0.0, 0.0), Point2::new(1.0, 0.0),
            Point2::new(1.0, 1.0), Point2::new(0.0, 1.0),
        ]).unwrap());
        project.pieces[0].assembly.translation_meters[0] = 1.0e100;
        assert!(project.validate().unwrap_err().contains("translação"));
        project.pieces[0].assembly.translation_meters = [0.0; 3];
        project.pieces[0].assembly.rotation_degrees[1] = 1.0e100;
        assert!(project.validate().unwrap_err().contains("rotação"));
        project.pieces[0].assembly.rotation_degrees = [0.0; 3];
        project.pieces[0].assembly.scale[2] = 101.0;
        assert!(project.validate().unwrap_err().contains("escala"));
    }

    #[test]
    fn validation_rejects_non_finite_grading_coefficients() {
        let mut project = PatternProject::new("Graduação");
        project.grading_rules.push(GradingRule {
            label: "M".into(),
            matrix: [1.0, 0.0, 0.0, 0.0, f64::INFINITY, 0.0],
        });
        assert!(project.validate().unwrap_err().contains("graduação"));
    }


    #[test]
    fn auto_sewing_suggests_facing_edges_for_rectangles_side_by_side() {
        let mut project = PatternProject::new("Retângulos lado a lado");
        project.pieces.push(PatternPiece::from_polygon(1, "Esquerda", &[
            Point2::new(0.0, 0.0), Point2::new(1.0, 0.0),
            Point2::new(1.0, 1.0), Point2::new(0.0, 1.0),
        ]).unwrap());
        project.pieces.push(PatternPiece::from_polygon(2, "Direita", &[
            Point2::new(1.0, 0.0), Point2::new(2.0, 0.0),
            Point2::new(2.0, 1.0), Point2::new(1.0, 1.0),
        ]).unwrap());
        let suggestions = suggest_seams(&project, &SeamSuggestionParams {
            minimum_score: 0.0,
            ..SeamSuggestionParams::default()
        });
        assert!(suggestions.iter().any(|suggestion|
            suggestion.a.piece_id == 1 && suggestion.a.start.segment == 1
                && suggestion.b.piece_id == 2 && suggestion.b.start.segment == 3
        ));
    }

    #[test]
    fn auto_sewing_detects_misaligned_notch_positions() {
        let mut aligned = PatternProject::new("Piques alinhados");
        aligned.pieces.push(PatternPiece::from_polygon(1, "A", &[
            Point2::new(0.0, 0.0), Point2::new(1.0, 0.0),
            Point2::new(1.0, 1.0), Point2::new(0.0, 1.0),
        ]).unwrap());
        aligned.pieces.push(PatternPiece::from_polygon(2, "B", &[
            Point2::new(1.0, 0.0), Point2::new(2.0, 0.0),
            Point2::new(2.0, 1.0), Point2::new(1.0, 1.0),
        ]).unwrap());
        for piece in &mut aligned.pieces {
            let segment_index = if piece.id == 1 { 1 } else { 3 };
            piece.notches.push(Notch {
                id: 1, segment_index, t: 0.5,
                depth_meters: 0.008, width_meters: 0.012,
            });
        }
        let matching_notches = notches_for_segment(&aligned.pieces[0], 1);
        assert_eq!(matching_notches.len(), 1);
        let aligned_suggestions = suggest_seams(&aligned, &SeamSuggestionParams {
            minimum_score: 0.0,
            ..SeamSuggestionParams::default()
        });
        let aligned_score = aligned_suggestions.iter().find(|suggestion|
            suggestion.a.piece_id == 1 && suggestion.a.start.segment == 1
                && suggestion.b.piece_id == 2 && suggestion.b.start.segment == 3
        ).map(|suggestion| suggestion.score).expect("facing edges should be suggested when notches align");

        let mut misaligned = aligned.clone();
        misaligned.pieces[0].notches[0].t = 0.2;
        misaligned.pieces[1].notches[0].t = 0.8;
        let left = notches_for_segment(&misaligned.pieces[0], 1);
        let right = notches_for_segment(&misaligned.pieces[1], 3);
        assert!((left[0] - right[0]).abs() > 0.5);

        let suggestions = suggest_seams(&misaligned, &SeamSuggestionParams {
            minimum_score: 0.0,
            ..SeamSuggestionParams::default()
        });
        let misaligned_score = suggestions.iter().find(|suggestion|
            suggestion.a.piece_id == 1 && suggestion.a.start.segment == 1
                && suggestion.b.piece_id == 2 && suggestion.b.start.segment == 3
        ).map(|suggestion| suggestion.score).expect("facing edges remain a candidate with misaligned notches");
        assert!(misaligned_score < aligned_score);
    }

    #[test]
    fn auto_sewing_never_suggests_edges_that_are_already_sewn() {
        let mut project = PatternProject::new("Bordas já costuradas");
        project.pieces.push(PatternPiece::from_polygon(1, "Esquerda", &[
            Point2::new(0.0, 0.0), Point2::new(1.0, 0.0),
            Point2::new(1.0, 1.0), Point2::new(0.0, 1.0),
        ]).unwrap());
        project.pieces.push(PatternPiece::from_polygon(2, "Direita", &[
            Point2::new(1.0, 0.0), Point2::new(2.0, 0.0),
            Point2::new(2.0, 1.0), Point2::new(1.0, 1.0),
        ]).unwrap());
        project.seams.push(SeamDefinition {
            id: 1,
            name: "Existente".into(),
            sides_a: vec![EdgeRef::single_segment(1, 1).into()],
            sides_b: vec![EdgeRef { piece_id: 2, curve: BoundaryCurve::Outline,
                segment_start: 3, segment_end: 4, reversed: true }.into()],
            ease_meters: 0.0,
        });
        let suggestions = suggest_seams(&project, &SeamSuggestionParams {
            minimum_score: 0.0,
            ..SeamSuggestionParams::default()
        });
        assert!(suggestions.iter().all(|suggestion|
            !((suggestion.a.piece_id == 1 && suggestion.a.start.segment == 1)
                || (suggestion.b.piece_id == 1 && suggestion.b.start.segment == 1)
                || (suggestion.a.piece_id == 2 && suggestion.a.start.segment == 3)
                || (suggestion.b.piece_id == 2 && suggestion.b.start.segment == 3))
        ));
    }


}
