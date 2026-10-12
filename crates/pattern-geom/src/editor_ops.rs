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


/// Identifies one fragment of a source boundary segment after the cut.
#[derive(Clone, Debug, PartialEq)]
pub struct SourceSegmentFragment {
    pub piece_index: usize,
    pub new_segment_index: usize,
    pub t_start: f64,
    pub t_end: f64,
}

/// Result of splitting one closed pattern outline with an open cut path.
/// Cut segment ranges use half-open indexing, matching the pattern core edge references.
#[derive(Clone, Debug, PartialEq)]
pub struct CutAndSewResult {
    pub first: Curve2D,
    pub second: Curve2D,
    pub first_cut_segment_range: (usize, usize),
    pub second_cut_segment_range: (usize, usize),
    /// Maps every source segment to the resulting boundary fragments in parametric order.
    pub source_segment_fragments: Vec<Vec<SourceSegmentFragment>>,
}

#[derive(Clone, Debug)]
struct BoundaryFragment {
    source_segment: usize,
    t_start: f64,
    t_end: f64,
    scalar_start: f64,
    scalar_end: f64,
    geometry: CurveSegment,
}

#[derive(Clone, Copy, Debug)]
struct CutLocation {
    scalar: f64,
    point: Point2,
}

fn split_cubic(
    p0: Point2,
    p1: Point2,
    p2: Point2,
    p3: Point2,
    t: f64,
) -> ([Point2; 4], [Point2; 4]) {
    let p01 = lerp(p0, p1, t);
    let p12 = lerp(p1, p2, t);
    let p23 = lerp(p2, p3, t);
    let p012 = lerp(p01, p12, t);
    let p123 = lerp(p12, p23, t);
    let split = lerp(p012, p123, t);
    ([p0, p01, p012, split], [split, p123, p23, p3])
}

fn segment_slice(start: Point2, segment: &CurveSegment, t0: f64, t1: f64) -> CurveSegment {
    match segment {
        CurveSegment::Line { to } => CurveSegment::Line { to: lerp(start, *to, t1) },
        CurveSegment::CubicBezier { control1, control2, to } => {
            let (left, _) = split_cubic(start, *control1, *control2, *to, t1);
            let sub = if t0 <= 1.0e-12 {
                left
            } else {
                let relative = (t0 / t1).clamp(0.0, 1.0);
                let (_, right) = split_cubic(left[0], left[1], left[2], left[3], relative);
                right
            };
            CurveSegment::CubicBezier {
                control1: sub[1],
                control2: sub[2],
                to: sub[3],
            }
        }
    }
}

fn locate_cut_endpoint(curve: &Curve2D, query: Point2) -> Result<CutLocation, GeometryError> {
    let projection = project_point_to_curve(curve, query)
        .ok_or(GeometryError::InvalidParameter)?;
    let count = curve.segments.len() as f64;
    let mut scalar = projection.segment_index as f64 + projection.t.clamp(0.0, 1.0);
    let nearest_integer = scalar.round();
    if (scalar - nearest_integer).abs() <= 1.0e-5 {
        scalar = nearest_integer;
    }
    if scalar >= count - 1.0e-8 { scalar = 0.0; }
    if scalar < 0.0 { scalar = 0.0; }
    let point = point_at_scalar(curve, scalar);
    Ok(CutLocation { scalar, point })
}

fn point_at_scalar(curve: &Curve2D, scalar: f64) -> Point2 {
    let count = curve.segments.len();
    if count == 0 { return curve.start; }
    let s = scalar.clamp(0.0, count as f64);
    let nearest = s.round();
    if (s - nearest).abs() <= 1.0e-8 {
        let vertex = (nearest as usize) % count;
        if vertex == 0 { curve.start }
        else { curve.segments[vertex - 1].endpoint() }
    } else {
        let index = (s.floor() as usize).min(count - 1);
        eval_segment(
            segment_start(curve, index).unwrap_or(curve.start),
            &curve.segments[index],
            s - index as f64,
        )
    }
}

fn path_with_snapped_endpoints(path: &Curve2D, start: Point2, end: Point2) -> Curve2D {
    let mut snapped = path.clone();
    snapped.start = start;
    if let Some(last) = snapped.segments.last_mut() {
        match last {
            CurveSegment::Line { to } | CurveSegment::CubicBezier { to, .. } => *to = end,
        }
    }
    snapped
}

fn orientation(a: Point2, b: Point2, c: Point2) -> f64 {
    (b.x - a.x) * (c.y - a.y) - (b.y - a.y) * (c.x - a.x)
}

fn segment_intersects(a: Point2, b: Point2, c: Point2, d: Point2) -> bool {
    let cross1 = orientation(a, b, c);
    let cross2 = orientation(a, b, d);
    let cross3 = orientation(c, d, a);
    let cross4 = orientation(c, d, b);
    let eps = 1.0e-9;
    ((cross1 > eps && cross2 < -eps) || (cross1 < -eps && cross2 > eps))
        && ((cross3 > eps && cross4 < -eps) || (cross3 < -eps && cross4 > eps))
}

fn validate_cut_path(outline: &Curve2D, path: &Curve2D) -> Result<(), GeometryError> {
    if path.closed || path.segments.is_empty()
        || !finite_point(path.start)
        || path.segments.iter().any(|segment| {
            let end = segment.endpoint();
            !finite_point(end) || match segment {
                CurveSegment::Line { .. } => false,
                CurveSegment::CubicBezier { control1, control2, .. } =>
                    !finite_point(*control1) || !finite_point(*control2),
            }
        })
    {
        return Err(GeometryError::InvalidParameter);
    }

    let polygon = sample_curve(outline, 24);
    let sampled = sample_curve(path, 24);
    if sampled.len() < 2 { return Err(GeometryError::InvalidParameter); }
    for (index, point) in sampled.iter().enumerate() {
        if index == 0 || index + 1 == sampled.len() { continue; }
        if !crate::point_in_polygon(*point, &polygon) {
            return Err(GeometryError::InvalidParameter);
        }
        if project_point_to_curve(outline, *point)
            .map(|projection| projection.distance <= 1.0e-5).unwrap_or(true)
        {
            return Err(GeometryError::InvalidParameter);
        }
    }

    // Reject self-crossing sampled cut paths.
    for i in 0..sampled.len() - 1 {
        for j in i + 2..sampled.len() - 1 {
            if segment_intersects(sampled[i], sampled[i + 1], sampled[j], sampled[j + 1]) {
                return Err(GeometryError::SelfIntersection);
            }
        }
    }
    Ok(())
}

fn build_boundary_fragments(
    curve: &Curve2D,
    locations: [CutLocation; 2],
) -> Vec<BoundaryFragment> {
    let mut fragments = Vec::new();
    let count = curve.segments.len();
    for index in 0..count {
        let mut breaks = vec![0.0, 1.0];
        for location in locations {
            let segment_index = location.scalar.floor() as usize;
            let t = location.scalar - segment_index as f64;
            if segment_index == index && t > 1.0e-8 && t < 1.0 - 1.0e-8 {
                breaks.push(t);
            }
        }
        breaks.sort_by(|a, b| a.total_cmp(b));
        breaks.dedup_by(|a, b| (*a - *b).abs() <= 1.0e-9);
        let start = segment_start(curve, index).unwrap_or(curve.start);
        for pair in breaks.windows(2) {
            let t0 = pair[0];
            let t1 = pair[1];
            if t1 - t0 <= 1.0e-10 { continue; }
            fragments.push(BoundaryFragment {
                source_segment: index,
                t_start: t0,
                t_end: t1,
                scalar_start: index as f64 + t0,
                scalar_end: index as f64 + t1,
                geometry: segment_slice(start, &curve.segments[index], t0, t1),
            });
        }
    }
    fragments
}

fn boundary_route(
    start: f64,
    end: f64,
    count: usize,
    fragments: &[BoundaryFragment],
) -> Vec<BoundaryFragment> {
    let mut end_unwrapped = end;
    if end_unwrapped <= start + 1.0e-9 { end_unwrapped += count as f64; }
    let mut route = Vec::new();
    for cycle in 0..=1 {
        let shift = cycle as f64 * count as f64;
        for fragment in fragments {
            let a = fragment.scalar_start + shift;
            let b = fragment.scalar_end + shift;
            if a >= start - 1.0e-8 && b <= end_unwrapped + 1.0e-8 {
                route.push(fragment.clone());
            }
        }
    }
    route
}

fn reverse_open_curve(curve: &Curve2D) -> Curve2D {
    let mut current = curve.start;
    let mut starts = Vec::with_capacity(curve.segments.len());
    for segment in &curve.segments {
        starts.push(current);
        current = segment.endpoint();
    }
    let mut segments = Vec::with_capacity(curve.segments.len());
    for index in (0..curve.segments.len()).rev() {
        let start = starts[index];
        let reversed = match &curve.segments[index] {
            CurveSegment::Line { .. } => CurveSegment::Line { to: start },
            CurveSegment::CubicBezier { control1, control2, .. } => CurveSegment::CubicBezier {
                control1: *control2,
                control2: *control1,
                to: start,
            },
        };
        segments.push(reversed);
    }
    Curve2D {
        start: curve.endpoint().unwrap_or(curve.start),
        segments,
        closed: false,
    }
}

/// Splits a closed outline along an open cut path whose endpoints lie on the outline.
/// The function preserves original line/Bézier segments, snaps both endpoints to the outline,
/// validates the resulting contours, and returns cut-edge ranges for automatic sewing.
/// Endpoint tolerance and coordinates are expressed in meters.
pub fn cut_closed_outline(
    outline: &Curve2D,
    cut_path: &Curve2D,
    endpoint_tolerance: f64,
) -> Result<CutAndSewResult, GeometryError> {
    if !outline.closed || outline.segments.len() < 3
        || !endpoint_tolerance.is_finite() || endpoint_tolerance <= 0.0
    {
        return Err(GeometryError::InvalidParameter);
    }
    let polygon = sample_curve(outline, 32);
    crate::validate_simple_polygon(&polygon)?;
    validate_cut_path(outline, cut_path)?;

    let start = locate_cut_endpoint(outline, cut_path.start)?;
    let end = locate_cut_endpoint(outline, cut_path.endpoint().ok_or(GeometryError::InvalidParameter)?)?;
    if cut_path.start.distance(start.point) > endpoint_tolerance
        || cut_path.endpoint().unwrap().distance(end.point) > endpoint_tolerance
    {
        return Err(GeometryError::InvalidParameter);
    }
    let count = outline.segments.len();
    let mut delta = (end.scalar - start.scalar).abs();
    delta = delta.min(count as f64 - delta);
    if delta <= 1.0e-5 {
        return Err(GeometryError::InvalidParameter);
    }

    // Snap only endpoints; intermediate cut geometry remains unchanged.
    let cut = path_with_snapped_endpoints(cut_path, start.point, end.point);
    let fragments = build_boundary_fragments(outline, [start, end]);
    let route_a = boundary_route(start.scalar, end.scalar, count, &fragments);
    let route_b = boundary_route(end.scalar, start.scalar, count, &fragments);
    if route_a.is_empty() || route_b.is_empty()
        || route_a.len() + route_b.len() != fragments.len()
    {
        return Err(GeometryError::InvalidParameter);
    }

    let mut source_map: Vec<Vec<SourceSegmentFragment>> =
        (0..count).map(|_| Vec::new()).collect();
    let mut first_segments = Vec::new();
    for fragment in &route_a {
        let new_index = first_segments.len();
        first_segments.push(fragment.geometry.clone());
        source_map[fragment.source_segment].push(SourceSegmentFragment {
            piece_index: 0,
            new_segment_index: new_index,
            t_start: fragment.t_start,
            t_end: fragment.t_end,
        });
    }
    let first_cut_start = first_segments.len();
    first_segments.extend(reverse_open_curve(&cut).segments);
    let first_cut_end = first_segments.len();

    let mut second_segments = Vec::new();
    for fragment in &route_b {
        let new_index = second_segments.len();
        second_segments.push(fragment.geometry.clone());
        source_map[fragment.source_segment].push(SourceSegmentFragment {
            piece_index: 1,
            new_segment_index: new_index,
            t_start: fragment.t_start,
            t_end: fragment.t_end,
        });
    }
    let second_cut_start = second_segments.len();
    second_segments.extend(cut.segments.clone());
    let second_cut_end = second_segments.len();

    let first = Curve2D { start: start.point, segments: first_segments, closed: true };
    let second = Curve2D { start: end.point, segments: second_segments, closed: true };

    crate::validate_simple_polygon(&sample_curve(&first, 32))?;
    crate::validate_simple_polygon(&sample_curve(&second, 32))?;

    for mapped in &mut source_map {
        mapped.sort_by(|a, b| a.t_start.total_cmp(&b.t_start));
    }

    Ok(CutAndSewResult {
        first,
        second,
        first_cut_segment_range: (first_cut_start, first_cut_end),
        second_cut_segment_range: (second_cut_start, second_cut_end),
        source_segment_fragments: source_map,
    })
}
