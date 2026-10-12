//! Algoritmos geométricos sem dependência da UI ou do renderizador.

use pattern_core::{Curve2D, CurveSegment, Point2};
use std::fmt;

pub mod editor_ops;
pub use editor_ops::{
    extract_segment_range, insert_anchor, mirror_curve_x, project_point_to_curve,
    rectangle, regular_polygon, resample_curve, translate_curve, CurveProjection,
};

const EPS: f64 = 1.0e-9;

#[derive(Clone, Debug, PartialEq, Eq)]
pub enum GeometryError {
    TooFewPoints,
    DegenerateEdge,
    SelfIntersection,
    InvalidOffset,
    InvalidSegment,
    InvalidParameter,
}

impl fmt::Display for GeometryError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::TooFewPoints => write!(f, "O polígono precisa ter ao menos 3 pontos distintos."),
            Self::DegenerateEdge => write!(f, "O polígono contém uma aresta de comprimento zero."),
            Self::SelfIntersection => write!(f, "O contorno se cruza ou não é simples."),
            Self::InvalidOffset => write!(f, "O offset solicitado é geometricamente inválido."),
            Self::InvalidSegment => write!(f, "O intervalo ou índice de segmento é inválido."),
            Self::InvalidParameter => write!(f, "Um parâmetro geométrico é inválido."),
        }
    }
}
impl std::error::Error for GeometryError {}

#[derive(Clone, Debug, PartialEq)]
pub struct SampledCurve {
    pub points: Vec<Point2>,
    /// Índices inclusivos (início, fim) na lista de pontos amostrados por segmento.
    /// O fim pode ser 0 para o segmento de fechamento.
    pub segment_vertex_ranges: Vec<(usize, usize)>,
}

fn cubic(p0: Point2, p1: Point2, p2: Point2, p3: Point2, t: f64) -> Point2 {
    let mt = 1.0 - t;
    let a = mt * mt * mt;
    let b = 3.0 * mt * mt * t;
    let c = 3.0 * mt * t * t;
    let d = t * t * t;
    Point2::new(
        a * p0.x + b * p1.x + c * p2.x + d * p3.x,
        a * p0.y + b * p1.y + c * p2.y + d * p3.y,
    )
}

/// Amostra uma curva e preserva o mapeamento entre segmentos da curva original e vértices.
pub fn sample_curve_with_ranges(curve: &Curve2D, cubic_steps: usize) -> SampledCurve {
    let steps = cubic_steps.max(1);
    let mut points = vec![curve.start];
    let mut ranges = Vec::with_capacity(curve.segments.len());
    let mut current = curve.start;

    for segment in &curve.segments {
        let start_index = points.len() - 1;
        match segment {
            CurveSegment::Line { to } => points.push(*to),
            CurveSegment::CubicBezier { control1, control2, to } => {
                for step in 1..=steps {
                    let t = step as f64 / steps as f64;
                    points.push(cubic(current, *control1, *control2, *to, t));
                }
            }
        }
        let end_index = points.len() - 1;
        ranges.push((start_index, end_index));
        current = segment.endpoint();
    }

    if curve.closed && points.len() > 1 && points[0].distance(*points.last().unwrap()) <= EPS {
        let duplicated_index = points.len() - 1;
        points.pop();
        for range in &mut ranges {
            if range.1 == duplicated_index { range.1 = 0; }
        }
    }
    SampledCurve { points, segment_vertex_ranges: ranges }
}

/// Amostra curvas Bézier cúbicas e segmentos lineares. Em curvas fechadas, remove o ponto
/// final duplicado para que o polígono represente cada vértice apenas uma vez.
pub fn sample_curve(curve: &Curve2D, cubic_steps: usize) -> Vec<Point2> {
    sample_curve_with_ranges(curve, cubic_steps).points
}

/// Comprimento aproximado de uma curva. A precisão das Bézier é controlada por cubic_steps.
pub fn curve_length(curve: &Curve2D, cubic_steps: usize) -> f64 {
    let sampled = sample_curve(curve, cubic_steps.max(1));
    sampled.windows(2).map(|w| w[0].distance(w[1])).sum::<f64>()
        + if curve.closed && sampled.len() > 2 {
            sampled[sampled.len() - 1].distance(sampled[0])
        } else { 0.0 }
}

pub fn signed_area(polygon: &[Point2]) -> f64 {
    if polygon.len() < 3 { return 0.0; }
    let mut sum = 0.0;
    for i in 0..polygon.len() {
        let a = polygon[i];
        let b = polygon[(i + 1) % polygon.len()];
        sum += a.x * b.y - b.x * a.y;
    }
    0.5 * sum
}

/// Ray casting. Pontos sobre a borda são considerados dentro.
pub fn point_in_polygon(point: Point2, polygon: &[Point2]) -> bool {
    if polygon.len() < 3 { return false; }
    let mut inside = false;
    let mut j = polygon.len() - 1;
    for i in 0..polygon.len() {
        let a = polygon[i];
        let b = polygon[j];
        if point_segment_distance(point, a, b) <= EPS { return true; }
        let crosses = (a.y > point.y) != (b.y > point.y)
            && point.x < (b.x - a.x) * (point.y - a.y) / (b.y - a.y) + a.x;
        if crosses { inside = !inside; }
        j = i;
    }
    inside
}

fn point_segment_distance(p: Point2, a: Point2, b: Point2) -> f64 {
    let dx = b.x - a.x;
    let dy = b.y - a.y;
    let len2 = dx * dx + dy * dy;
    if len2 <= EPS * EPS { return p.distance(a); }
    let t = (((p.x - a.x) * dx + (p.y - a.y) * dy) / len2).clamp(0.0, 1.0);
    p.distance(Point2::new(a.x + t * dx, a.y + t * dy))
}

fn orient(a: Point2, b: Point2, c: Point2) -> f64 {
    (b.x - a.x) * (c.y - a.y) - (b.y - a.y) * (c.x - a.x)
}

fn on_segment(a: Point2, b: Point2, p: Point2) -> bool {
    p.x >= a.x.min(b.x) - EPS && p.x <= a.x.max(b.x) + EPS
        && p.y >= a.y.min(b.y) - EPS && p.y <= a.y.max(b.y) + EPS
}

fn segments_intersect(a: Point2, b: Point2, c: Point2, d: Point2) -> bool {
    let o1 = orient(a, b, c);
    let o2 = orient(a, b, d);
    let o3 = orient(c, d, a);
    let o4 = orient(c, d, b);
    if ((o1 > EPS && o2 < -EPS) || (o1 < -EPS && o2 > EPS))
        && ((o3 > EPS && o4 < -EPS) || (o3 < -EPS && o4 > EPS)) {
        return true;
    }
    (o1.abs() <= EPS && on_segment(a, b, c))
        || (o2.abs() <= EPS && on_segment(a, b, d))
        || (o3.abs() <= EPS && on_segment(c, d, a))
        || (o4.abs() <= EPS && on_segment(c, d, b))
}

/// Valida uma borda fechada sem autointerseções. A entrada não precisa repetir o ponto inicial.
pub fn validate_simple_polygon(polygon: &[Point2]) -> Result<(), GeometryError> {
    if polygon.len() < 3 { return Err(GeometryError::TooFewPoints); }
    for i in 0..polygon.len() {
        if polygon[i].distance(polygon[(i + 1) % polygon.len()]) <= EPS {
            return Err(GeometryError::DegenerateEdge);
        }
    }
    for i in 0..polygon.len() {
        let a = polygon[i];
        let b = polygon[(i + 1) % polygon.len()];
        for j in (i + 1)..polygon.len() {
            if j == i + 1 || (i == 0 && j == polygon.len() - 1) { continue; }
            let c = polygon[j];
            let d = polygon[(j + 1) % polygon.len()];
            if segments_intersect(a, b, c, d) { return Err(GeometryError::SelfIntersection); }
        }
    }
    if signed_area(polygon).abs() <= EPS { return Err(GeometryError::TooFewPoints); }
    Ok(())
}

/// Offset de polígono fechado por interseção das linhas offset adjacentes (miter join).
/// Coordenadas em metros; distância positiva desloca para fora nos dois sentidos de contorno.
pub fn offset_polygon(polygon: &[Point2], distance: f64) -> Result<Vec<Point2>, GeometryError> {
    validate_simple_polygon(polygon)?;
    if !distance.is_finite() { return Err(GeometryError::InvalidOffset); }
    if distance.abs() <= EPS { return Ok(polygon.to_vec()); }

    let orientation = if signed_area(polygon) >= 0.0 { 1.0 } else { -1.0 };
    let mut result = Vec::with_capacity(polygon.len());
    for i in 0..polygon.len() {
        let prev = polygon[(i + polygon.len() - 1) % polygon.len()];
        let current = polygon[i];
        let next = polygon[(i + 1) % polygon.len()];

        let mut d0 = (current.x - prev.x, current.y - prev.y);
        let mut d1 = (next.x - current.x, next.y - current.y);
        let l0 = (d0.0 * d0.0 + d0.1 * d0.1).sqrt();
        let l1 = (d1.0 * d1.0 + d1.1 * d1.1).sqrt();
        if l0 <= EPS || l1 <= EPS { return Err(GeometryError::DegenerateEdge); }
        d0.0 /= l0; d0.1 /= l0;
        d1.0 /= l1; d1.1 /= l1;

        let n0 = (orientation * d0.1, -orientation * d0.0);
        let n1 = (orientation * d1.1, -orientation * d1.0);
        let denom = 1.0 + n0.0 * n1.0 + n0.1 * n1.1;
        if denom.abs() <= 1.0e-7 { return Err(GeometryError::InvalidOffset); }
        let mx = (n0.0 + n1.0) * distance / denom;
        let my = (n0.1 + n1.1) * distance / denom;
        let miter_len = (mx * mx + my * my).sqrt();
        if miter_len > distance.abs() * 20.0 + EPS { return Err(GeometryError::InvalidOffset); }
        result.push(Point2::new(current.x + mx, current.y + my));
    }
    validate_simple_polygon(&result)?;
    Ok(result)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn pattern_primitives_create_closed_vector_outlines() {
        let rect = rectangle(Point2::new(0.0, 0.0), Point2::new(2.0, 1.0)).unwrap();
        assert!(rect.closed);
        assert_eq!(rect.segment_count(), 4);

        let pentagon = regular_polygon(Point2::new(0.0, 0.0), 1.0, 5, 0.0).unwrap();
        assert!(pentagon.closed);
        assert_eq!(pentagon.segment_count(), 5);
        assert!(regular_polygon(Point2::new(0.0, 0.0), 1.0, 2, 0.0).is_err());
    }

    #[test]
    fn bezier_anchor_insertion_preserves_shape_and_increases_topology() {
        let curve = Curve2D {
            start: Point2::new(0.0, 0.0),
            segments: vec![CurveSegment::CubicBezier {
                control1: Point2::new(0.0, 1.0),
                control2: Point2::new(1.0, 1.0),
                to: Point2::new(1.0, 0.0),
            }],
            closed: false,
        };
        let divided = insert_anchor(&curve, 0, 0.5).unwrap();
        assert_eq!(divided.segment_count(), 2);
        let original_length = curve_length(&curve, 128);
        let divided_length = curve_length(&divided, 128);
        assert!((original_length - divided_length).abs() < 0.002);
        assert!(insert_anchor(&curve, 0, 0.0).is_err());
    }

    #[test]
    fn extracting_segment_range_can_reverse_direction() {
        let curve = Curve2D::polyline(&[
            Point2::new(0.0, 0.0),
            Point2::new(1.0, 0.0),
            Point2::new(1.0, 1.0),
        ], false).unwrap();
        let forward = extract_segment_range(&curve, 0, 2, false).unwrap();
        let backward = extract_segment_range(&curve, 0, 2, true).unwrap();
        assert_eq!(forward.start, Point2::new(0.0, 0.0));
        assert_eq!(forward.endpoint(), Some(Point2::new(1.0, 1.0)));
        assert_eq!(backward.start, Point2::new(1.0, 1.0));
        assert_eq!(backward.endpoint(), Some(Point2::new(0.0, 0.0)));
    }

    #[test]
    fn projection_finds_nearest_line_position() {
        let curve = Curve2D::polyline(&[
            Point2::new(0.0, 0.0), Point2::new(2.0, 0.0)
        ], false).unwrap();
        let projected = project_point_to_curve(&curve, Point2::new(0.5, 0.25)).unwrap();
        assert_eq!(projected.segment_index, 0);
        assert!((projected.t - 0.25).abs() < 1.0e-9);
        assert!((projected.distance - 0.25).abs() < 1.0e-9);
    }

    #[test]
    fn curve_resampling_respects_target_spacing() {
        let curve = rectangle(Point2::new(0.0, 0.0), Point2::new(2.0, 1.0)).unwrap();
        let resampled = resample_curve(&curve, 0.25).unwrap();
        let points = sample_curve(&resampled, 1);
        for index in 0..points.len() {
            let next = points[(index + 1) % points.len()];
            assert!(points[index].distance(next) <= 0.251);
        }
    }

    #[test]
    fn cubic_sampling_preserves_segment_mapping_and_length() {
        let curve = Curve2D {
            start: Point2::new(0.0, 0.0),
            segments: vec![CurveSegment::CubicBezier {
                control1: Point2::new(0.0, 1.0),
                control2: Point2::new(1.0, 1.0),
                to: Point2::new(1.0, 0.0),
            }],
            closed: false,
        };
        let sampled = sample_curve_with_ranges(&curve, 8);
        assert_eq!(sampled.points.len(), 9);
        assert_eq!(sampled.segment_vertex_ranges, vec![(0, 8)]);
        assert!(curve_length(&curve, 16) > 1.0);
    }

    #[test]
    fn polygon_offset_expands_rectangle() {
        let rect = [
            Point2::new(0.0, 0.0), Point2::new(2.0, 0.0),
            Point2::new(2.0, 1.0), Point2::new(0.0, 1.0),
        ];
        let offset = offset_polygon(&rect, 0.1).unwrap();
        assert!((signed_area(&offset).abs() - 2.64).abs() < 1e-6);
    }

    #[test]
    fn self_intersection_is_rejected() {
        let bow_tie = [
            Point2::new(0.0, 0.0), Point2::new(1.0, 1.0),
            Point2::new(0.0, 1.0), Point2::new(1.0, 0.0),
        ];
        assert_eq!(validate_simple_polygon(&bow_tie), Err(GeometryError::SelfIntersection));
    }
}
