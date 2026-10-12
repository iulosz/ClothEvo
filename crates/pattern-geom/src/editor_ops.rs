//! Operações de edição vetorial independentes da interface.
//!
//! As funções trabalham diretamente em Curve2D: subdivisão Bézier exata, criação de primitivas,
//! extração de intervalos de aresta, projeção e reamostragem para discretização da malha.

use pattern_core::{Curve2D, CurveSegment, Point2};

use crate::{sample_curve, GeometryError};

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct CurveProjection {
    pub segment_index: usize,
    /// Parâmetro local no segmento original, no intervalo [0, 1].
    pub t: f64,
    pub point: Point2,
    pub distance: f64,
}

fn finite_point(point: Point2) -> bool {
    point.x.is_finite() && point.y.is_finite()
}

fn lerp(a: Point2, b: Point2, t: f64) -> Point2 {
    Point2::new(a.x + (b.x - a.x) * t, a.y + (b.y - a.y) * t)
}

fn cubic(p0: Point2, p1: Point2, p2: Point2, p3: Point2, t: f64) -> Point2 {
    let a = lerp(p0, p1, t);
    let b = lerp(p1, p2, t);
    let c = lerp(p2, p3, t);
    lerp(lerp(a, b, t), lerp(b, c, t), t)
}

fn segment_start(curve: &Curve2D, segment_index: usize) -> Option<Point2> {
    if segment_index >= curve.segments.len() { return None; }
    if segment_index == 0 { Some(curve.start) }
    else { Some(curve.segments[segment_index - 1].endpoint()) }
}

fn eval_segment(start: Point2, segment: &CurveSegment, t: f64) -> Point2 {
    match segment {
        CurveSegment::Line { to } => lerp(start, *to, t),
        CurveSegment::CubicBezier { control1, control2, to } => {
            cubic(start, *control1, *control2, *to, t)
        }
    }
}

/// Cria um retângulo fechado alinhado aos eixos, usando min/max em metros.
pub fn rectangle(min: Point2, max: Point2) -> Result<Curve2D, GeometryError> {
    if !finite_point(min) || !finite_point(max)
        || max.x - min.x <= 1.0e-9 || max.y - min.y <= 1.0e-9
    {
        return Err(GeometryError::InvalidParameter);
    }
    Curve2D::polyline(&[
        min,
        Point2::new(max.x, min.y),
        max,
        Point2::new(min.x, max.y),
    ], true).map_err(|_| GeometryError::InvalidParameter)
}

/// Cria um polígono regular fechado.
pub fn regular_polygon(
    center: Point2,
    radius: f64,
    sides: usize,
    rotation_radians: f64,
) -> Result<Curve2D, GeometryError> {
    if !finite_point(center) || !radius.is_finite() || radius <= 1.0e-9
        || !rotation_radians.is_finite() || sides < 3 || sides > 100_000
    {
        return Err(GeometryError::InvalidParameter);
    }
    let points = (0..sides).map(|i| {
        let angle = rotation_radians + std::f64::consts::TAU * i as f64 / sides as f64;
        Point2::new(center.x + radius * angle.cos(), center.y + radius * angle.sin())
    }).collect::<Vec<_>>();
    Curve2D::polyline(&points, true).map_err(|_| GeometryError::InvalidParameter)
}

/// Translada todos os pontos de controle e endpoints sem alterar a topologia.
pub fn translate_curve(curve: &Curve2D, delta: Point2) -> Result<Curve2D, GeometryError> {
    if !finite_point(delta) { return Err(GeometryError::InvalidParameter); }
    let map = |p: Point2| Point2::new(p.x + delta.x, p.y + delta.y);
    let segments = curve.segments.iter().map(|segment| match segment {
        CurveSegment::Line { to } => CurveSegment::Line { to: map(*to) },
        CurveSegment::CubicBezier { control1, control2, to } => CurveSegment::CubicBezier {
            control1: map(*control1), control2: map(*control2), to: map(*to),
        },
    }).collect();
    Ok(Curve2D { start: map(curve.start), segments, closed: curve.closed })
}

/// Espelha uma curva em relação à reta vertical x = axis_x.
pub fn mirror_curve_x(curve: &Curve2D, axis_x: f64) -> Result<Curve2D, GeometryError> {
    if !axis_x.is_finite() { return Err(GeometryError::InvalidParameter); }
    let map = |p: Point2| Point2::new(2.0 * axis_x - p.x, p.y);
    let segments = curve.segments.iter().map(|segment| match segment {
        CurveSegment::Line { to } => CurveSegment::Line { to: map(*to) },
        CurveSegment::CubicBezier { control1, control2, to } => CurveSegment::CubicBezier {
            control1: map(*control1), control2: map(*control2), to: map(*to),
        },
    }).collect();
    Ok(Curve2D { start: map(curve.start), segments, closed: curve.closed })
}

/// Insere um novo nó em uma aresta. Curvas cúbicas são subdivididas exatamente por De Casteljau.
pub fn insert_anchor(
    curve: &Curve2D,
    segment_index: usize,
    t: f64,
) -> Result<Curve2D, GeometryError> {
    if !t.is_finite() || t <= 1.0e-9 || t >= 1.0 - 1.0e-9 {
        return Err(GeometryError::InvalidParameter);
    }
    let start = segment_start(curve, segment_index).ok_or(GeometryError::InvalidSegment)?;
    let original = curve.segments[segment_index].clone();
    let (first, second) = match original {
        CurveSegment::Line { to } => {
            let split = lerp(start, to, t);
            (CurveSegment::Line { to: split }, CurveSegment::Line { to })
        }
        CurveSegment::CubicBezier { control1, control2, to } => {
            let p01 = lerp(start, control1, t);
            let p12 = lerp(control1, control2, t);
            let p23 = lerp(control2, to, t);
            let p012 = lerp(p01, p12, t);
            let p123 = lerp(p12, p23, t);
            let split = lerp(p012, p123, t);
            (
                CurveSegment::CubicBezier { control1: p01, control2: p012, to: split },
                CurveSegment::CubicBezier { control1: p123, control2: p23, to },
            )
        }
    };
    let mut result = curve.clone();
    result.segments[segment_index] = first;
    result.segments.insert(segment_index + 1, second);
    Ok(result)
}

/// Clona um intervalo [start, end) de segmentos, preservando as Bézier e opcionalmente revertendo
/// o sentido da curva. É a primitiva base para Trace e para mapear intervalos de costura.
pub fn extract_segment_range(
    curve: &Curve2D,
    start: usize,
    end: usize,
    reversed: bool,
) -> Result<Curve2D, GeometryError> {
    if start >= end || end > curve.segments.len() {
        return Err(GeometryError::InvalidSegment);
    }
    let starts = (0..curve.segments.len())
        .map(|i| segment_start(curve, i).unwrap())
        .collect::<Vec<_>>();
    let (new_start, segments) = if reversed {
        let mut out = Vec::with_capacity(end - start);
        for index in (start..end).rev() {
            let old_start = starts[index];
            let reversed_segment = match &curve.segments[index] {
                CurveSegment::Line { .. } => CurveSegment::Line { to: old_start },
                CurveSegment::CubicBezier { control1, control2, .. } => CurveSegment::CubicBezier {
                    control1: *control2,
                    control2: *control1,
                    to: old_start,
                },
            };
            out.push(reversed_segment);
        }
        (curve.segments[end - 1].endpoint(), out)
    } else {
        (starts[start], curve.segments[start..end].to_vec())
    };
    Ok(Curve2D { start: new_start, segments, closed: false })
}

/// Projeta um ponto para a curva. Linhas são exatas; Bézier usa subdivisão adaptativa uniforme
/// (64 amostras por segmento) e retorna t no segmento original para posterior edição/Trace.
pub fn project_point_to_curve(curve: &Curve2D, query: Point2) -> Option<CurveProjection> {
    if !finite_point(query) || curve.segments.is_empty() { return None; }
    let mut best: Option<CurveProjection> = None;
    for (index, segment) in curve.segments.iter().enumerate() {
        let start = segment_start(curve, index)?;
        let steps = match segment {
            CurveSegment::Line { .. } => 1,
            CurveSegment::CubicBezier { .. } => 64,
        };
        let mut previous_t = 0.0;
        let mut previous = start;
        for step in 1..=steps {
            let current_t = step as f64 / steps as f64;
            let current = eval_segment(start, segment, current_t);
            let dx = current.x - previous.x;
            let dy = current.y - previous.y;
            let len2 = dx * dx + dy * dy;
            let local = if len2 <= 1.0e-18 { 0.0 } else {
                (((query.x - previous.x) * dx + (query.y - previous.y) * dy) / len2).clamp(0.0, 1.0)
            };
            let projected = lerp(previous, current, local);
            let t = previous_t + (current_t - previous_t) * local;
            let distance = query.distance(projected);
            if best.map(|b| distance < b.distance).unwrap_or(true) {
                best = Some(CurveProjection { segment_index: index, t, point: projected, distance });
            }
            previous_t = current_t;
            previous = current;
        }
    }
    best
}

/// Reamostra uma curva em intervalos aproximadamente uniformes. Útil para definir a distância
/// entre partículas da malha; o contorno fechado não duplica o ponto inicial no fim da lista.
pub fn resample_curve(curve: &Curve2D, spacing: f64) -> Result<Curve2D, GeometryError> {
    if !spacing.is_finite() || spacing <= 1.0e-9 || curve.segments.is_empty()
        || !finite_point(curve.start)
    {
        return Err(GeometryError::InvalidParameter);
    }
    let sampled = sample_curve(curve, 64);
    if sampled.len() < 2 { return Err(GeometryError::InvalidParameter); }
    let mut path = sampled;
    if curve.closed {
        path.push(path[0]);
    }
    let mut cumulative = Vec::with_capacity(path.len());
    cumulative.push(0.0);
    for pair in path.windows(2) {
        let step = pair[0].distance(pair[1]);
        if !step.is_finite() { return Err(GeometryError::InvalidParameter); }
        cumulative.push(cumulative.last().copied().unwrap_or(0.0) + step);
    }
    let total = *cumulative.last().unwrap_or(&0.0);
    if total <= 1.0e-9 { return Err(GeometryError::InvalidParameter); }
    let mut points = Vec::new();
    let mut distance = 0.0;
    let limit = if curve.closed { total - 1.0e-9 } else { total };
    while distance < limit && points.len() < 1_000_000 {
        let upper = cumulative.partition_point(|value| *value < distance);
        let segment = upper.saturating_sub(1).min(path.len() - 2);
        let length = cumulative[segment + 1] - cumulative[segment];
        let t = if length <= 1.0e-12 { 0.0 } else {
            (distance - cumulative[segment]) / length
        };
        points.push(lerp(path[segment], path[segment + 1], t));
        distance += spacing;
    }
    if !curve.closed && points.last().map(|p| p.distance(*path.last().unwrap()) > 1.0e-8).unwrap_or(true) {
        points.push(*path.last().unwrap());
    }
    let minimum = if curve.closed { 3 } else { 2 };
    if points.len() < minimum { return Err(GeometryError::InvalidParameter); }
    Curve2D::polyline(&points, curve.closed).map_err(|_| GeometryError::InvalidParameter)
}
