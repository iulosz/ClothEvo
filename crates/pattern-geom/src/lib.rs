//! Algoritmos geométricos sem dependência da UI ou do renderizador.

use pattern_core::{
    BoundaryCurve, Curve2D, CurveParam, CurveSegment, EdgePath, PatternPiece, Point2, SeamDefinition,
};
use std::collections::HashMap;
use std::fmt;

pub mod editor_ops;
pub use editor_ops::{
    cut_closed_outline, extract_segment_range, insert_anchor, mirror_curve_x,
    project_point_to_curve, rectangle, regular_polygon, resample_curve, translate_curve,
    CutAndSewResult, CurveProjection, SourceSegmentFragment,
};

const EPS: f64 = 1.0e-9;
/// Orçamento global por curva para manter amostragem/sincronização limitada.
const MAX_CURVE_SAMPLES: usize = 4_096;

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
/// Variante do amostrador que injeta exatamente parâmetros indicados na borda.
/// Retorna os índices dos vértices forçados na mesma ordem de `forced_params`; não faz
/// projeção para um vértice vizinho. Os parâmetros precisam identificar segmentos existentes.
pub fn sample_curve_with_forced_params(
    curve: &Curve2D,
    cubic_steps: usize,
    forced_params: &[(u32, f64)],
) -> Result<(SampledCurve, Vec<usize>), GeometryError> {
    if forced_params.iter().any(|(segment, t)| {
        *segment as usize >= curve.segments.len() || !t.is_finite() || !(0.0..=1.0).contains(t)
    }) {
        return Err(GeometryError::InvalidParameter);
    }
    let steps = cubic_steps.max(1);
    let mut points = vec![curve.start];
    let mut ranges = Vec::with_capacity(curve.segments.len());
    let mut forced_indices = vec![None; forced_params.len()];
    let mut current = curve.start;

    for (segment_index, segment) in curve.segments.iter().enumerate() {
        let start_index = points.len() - 1;
        let mut ts = Vec::new();
        if matches!(segment, CurveSegment::CubicBezier { .. }) {
            ts.extend((1..steps).map(|step| step as f64 / steps as f64));
        }
        ts.extend(forced_params.iter().filter_map(|(forced_segment, t)| {
            (*forced_segment as usize == segment_index && *t > EPS && *t < 1.0 - EPS).then_some(*t)
        }));
        ts.push(1.0);
        ts.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
        ts.dedup_by(|a, b| (*a - *b).abs() <= EPS);

        for (forced_index, (forced_segment, t)) in forced_params.iter().enumerate() {
            if *forced_segment as usize == segment_index && t.abs() <= EPS {
                forced_indices[forced_index] = Some(start_index);
            }
        }
        for t in ts {
            let point = match segment {
                CurveSegment::Line { to } => Point2::new(
                    current.x + (to.x - current.x) * t,
                    current.y + (to.y - current.y) * t,
                ),
                CurveSegment::CubicBezier { control1, control2, to } => {
                    cubic(current, *control1, *control2, *to, t)
                }
            };
            points.push(point);
            let vertex_index = points.len() - 1;
            for (forced_index, (forced_segment, forced_t)) in forced_params.iter().enumerate() {
                if *forced_segment as usize == segment_index && (*forced_t - t).abs() <= EPS {
                    forced_indices[forced_index] = Some(vertex_index);
                }
            }
        }
        ranges.push((start_index, points.len() - 1));
        current = segment.endpoint();
    }

    if curve.closed && points.len() > 1 && points[0].distance(*points.last().unwrap()) <= EPS {
        let duplicated_index = points.len() - 1;
        points.pop();
        for range in &mut ranges {
            if range.1 == duplicated_index { range.1 = 0; }
        }
        for index in &mut forced_indices {
            if *index == Some(duplicated_index) { *index = Some(0); }
        }
    }
    let forced_indices = forced_indices.into_iter()
        .map(|index| index.ok_or(GeometryError::InvalidParameter))
        .collect::<Result<Vec<_>, _>>()?;
    Ok((SampledCurve { points, segment_vertex_ranges: ranges }, forced_indices))
}

pub fn sample_curve_with_ranges(curve: &Curve2D, cubic_steps: usize) -> SampledCurve {
    let cubic_count = curve.segments.iter()
        .filter(|segment| matches!(segment, CurveSegment::CubicBezier { .. }))
        .count();
    let line_count = curve.segments.len().saturating_sub(cubic_count);
    let extra_budget = MAX_CURVE_SAMPLES.saturating_sub(1 + line_count);
    let safe_steps = if cubic_count == 0 { 1 } else {
        (extra_budget / cubic_count).max(1)
    };
    let steps = cubic_steps.max(1).min(safe_steps);
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

/// Parâmetros t amostrados por segmento da curva original. Cada lista inclui 0 e 1.
type SegmentParameters = Vec<Vec<f64>>;

fn point_on_segment(curve: &Curve2D, segment_index: usize, t: f64) -> Point2 {
    let start = if segment_index == 0 {
        curve.start
    } else {
        curve.segments[segment_index - 1].endpoint()
    };
    match &curve.segments[segment_index] {
        CurveSegment::Line { to } => Point2::new(
            start.x + (to.x - start.x) * t,
            start.y + (to.y - start.y) * t,
        ),
        CurveSegment::CubicBezier { control1, control2, to } => {
            cubic(start, *control1, *control2, *to, t)
        }
    }
}

fn cubic_parameters(
    curve: &Curve2D,
    segment_index: usize,
    t0: f64,
    t1: f64,
    max_spacing: f64,
    depth: u32,
    out: &mut Vec<f64>,
) -> Result<(), GeometryError> {
    let p0 = point_on_segment(curve, segment_index, t0);
    let p1 = point_on_segment(curve, segment_index, t1);
    let span = t1 - t0;
    let q1 = point_on_segment(curve, segment_index, t0 + span * 0.25);
    let q2 = point_on_segment(curve, segment_index, t0 + span * 0.50);
    let q3 = point_on_segment(curve, segment_index, t0 + span * 0.75);
    let polyline_length = p0.distance(q1) + q1.distance(q2)
        + q2.distance(q3) + q3.distance(p1);
    let chord_length = p0.distance(p1);
    let sufficiently_short = polyline_length <= max_spacing;
    let sufficiently_flat = polyline_length - chord_length <= max_spacing * 0.04;

    if depth >= 18 || (sufficiently_short && sufficiently_flat) {
        if out.len() >= MAX_CURVE_SAMPLES {
            return Err(GeometryError::InvalidParameter);
        }
        out.push(t1);
        return Ok(());
    }
    let mid = (t0 + t1) * 0.5;
    cubic_parameters(curve, segment_index, t0, mid, max_spacing, depth + 1, out)?;
    cubic_parameters(curve, segment_index, mid, t1, max_spacing, depth + 1, out)?;
    Ok(())
}

fn segment_parameters(curve: &Curve2D, max_spacing: f64) -> Result<SegmentParameters, GeometryError> {
    if !max_spacing.is_finite() || max_spacing <= 0.0
        || curve.segments.len().saturating_add(1) > MAX_CURVE_SAMPLES
    {
        return Err(GeometryError::InvalidParameter);
    }
    // Count all vertices across the whole curve, not just per individual segment.
    let mut total_samples = 1usize;
    let mut result = Vec::with_capacity(curve.segments.len());
    for (index, segment) in curve.segments.iter().enumerate() {
        match segment {
            CurveSegment::Line { to } => {
                let start = if index == 0 {
                    curve.start
                } else {
                    curve.segments[index - 1].endpoint()
                };
                let length = start.distance(*to);
                let count = (length / max_spacing).ceil().max(1.0);
                if !count.is_finite() || count > MAX_CURVE_SAMPLES as f64 {
                    return Err(GeometryError::InvalidParameter);
                }
                let count = count as usize;
                if total_samples.saturating_add(count) > MAX_CURVE_SAMPLES {
                    return Err(GeometryError::InvalidParameter);
                }
                total_samples += count;
                result.push((0..=count).map(|i| i as f64 / count as f64).collect());
            }
            CurveSegment::CubicBezier { .. } => {
                let mut params = vec![0.0];
                cubic_parameters(curve, index, 0.0, 1.0, max_spacing, 0, &mut params)?;
                let added = params.len().saturating_sub(1);
                if total_samples.saturating_add(added) > MAX_CURVE_SAMPLES {
                    return Err(GeometryError::InvalidParameter);
                }
                total_samples += added;
                result.push(params);
            }
        }
    }
    Ok(result)
}

fn normalize_parameters(params: &mut Vec<f64>) {
    params.sort_by(f64::total_cmp);
    params.dedup_by(|a, b| (*a - *b).abs() <= 1.0e-10);
    if params.first().is_none_or(|t| *t > 1.0e-10) {
        params.insert(0, 0.0);
    } else if let Some(first) = params.first_mut() {
        *first = 0.0;
    }
    if params.last().is_none_or(|t| *t < 1.0 - 1.0e-10) {
        params.push(1.0);
    } else if let Some(last) = params.last_mut() {
        *last = 1.0;
    }
}

fn sample_from_parameters(curve: &Curve2D, params: &SegmentParameters) -> SampledCurve {
    let mut points = vec![curve.start];
    let mut ranges = Vec::with_capacity(curve.segments.len());
    for (segment_index, segment_params) in params.iter().enumerate() {
        let start_index = points.len() - 1;
        for &t in segment_params.iter().skip(1) {
            points.push(point_on_segment(curve, segment_index, t));
        }
        ranges.push((start_index, points.len() - 1));
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

fn edge_segment_lengths(
    curve: &Curve2D,
    params: &SegmentParameters,
    edge: &pattern_core::EdgeRef,
) -> Result<Vec<f64>, GeometryError> {
    let start = edge.segment_start as usize;
    let end = edge.segment_end as usize;
    if start >= end || end > curve.segments.len() || params.len() != curve.segments.len() {
        return Err(GeometryError::InvalidSegment);
    }
    Ok((start..end).map(|segment_index| {
        let ts = &params[segment_index];
        ts.windows(2).map(|pair| {
            point_on_segment(curve, segment_index, pair[0])
                .distance(point_on_segment(curve, segment_index, pair[1]))
        }).sum::<f64>()
    }).collect())
}

/// Frações de comprimento de arco (0..1) para todos os nós existentes na borda.
fn edge_sample_fractions(
    curve: &Curve2D,
    params: &SegmentParameters,
    edge: &pattern_core::EdgeRef,
) -> Result<Vec<f64>, GeometryError> {
    let lengths = edge_segment_lengths(curve, params, edge)?;
    let total = lengths.iter().sum::<f64>();
    if !total.is_finite() || total <= EPS {
        return Err(GeometryError::DegenerateEdge);
    }
    let mut fractions = vec![0.0, 1.0];
    let mut preceding = 0.0;
    for (offset, length) in lengths.iter().copied().enumerate() {
        let segment_index = edge.segment_start as usize + offset;
        let ts = &params[segment_index];
        let mut local = 0.0;
        for (sample_index, pair) in ts.windows(2).enumerate() {
            let a = point_on_segment(curve, segment_index, pair[0]);
            let b = point_on_segment(curve, segment_index, pair[1]);
            if sample_index == 0 {
                let fraction = (preceding / total).clamp(0.0, 1.0);
                fractions.push(if edge.reversed { 1.0 - fraction } else { fraction });
            }
            local += a.distance(b);
            let fraction = ((preceding + local) / total).clamp(0.0, 1.0);
            fractions.push(if edge.reversed { 1.0 - fraction } else { fraction });
        }
        preceding += length;
    }
    Ok(fractions)
}

/// Converte fração de arco orientada para um parâmetro de segmento usando a amostragem base.
fn segment_parameter_at_edge_fraction(
    curve: &Curve2D,
    params: &SegmentParameters,
    edge: &pattern_core::EdgeRef,
    oriented_fraction: f64,
) -> Result<(usize, f64), GeometryError> {
    let lengths = edge_segment_lengths(curve, params, edge)?;
    let total = lengths.iter().sum::<f64>();
    if !total.is_finite() || total <= EPS {
        return Err(GeometryError::DegenerateEdge);
    }
    let canonical_fraction = if edge.reversed {
        1.0 - oriented_fraction
    } else {
        oriented_fraction
    }.clamp(0.0, 1.0);
    let target = canonical_fraction * total;
    let mut preceding = 0.0;

    for (offset, length) in lengths.iter().copied().enumerate() {
        let segment_index = edge.segment_start as usize + offset;
        let ts = &params[segment_index];
        if target <= preceding + length + 1.0e-10 || offset + 1 == lengths.len() {
            let local_target = (target - preceding).clamp(0.0, length);
            let mut local = 0.0;
            for pair in ts.windows(2) {
                let a = point_on_segment(curve, segment_index, pair[0]);
                let b = point_on_segment(curve, segment_index, pair[1]);
                let span = a.distance(b);
                if local_target <= local + span + 1.0e-10 {
                    let alpha = if span <= EPS { 0.0 } else {
                        ((local_target - local) / span).clamp(0.0, 1.0)
                    };
                    return Ok((segment_index, pair[0] + (pair[1] - pair[0]) * alpha));
                }
                local += span;
            }
            return Ok((segment_index, 1.0));
        }
        preceding += length;
    }
    Err(GeometryError::InvalidSegment)
}

#[derive(Clone, Copy, Debug)]
struct PathSample { param: CurveParam, point: Point2, fraction: f64 }

#[derive(Clone, Debug)]
struct EdgeSamples { piece_id: u64, samples: Vec<PathSample>, length: f64, prefix: f64 }

#[derive(Clone, Debug)]
struct SampledPathChain { edges: Vec<EdgeSamples>, total_length: f64 }

fn edge_path_samples(
    curve: &Curve2D, params: &SegmentParameters, edge: &EdgePath,
) -> Result<(Vec<PathSample>, f64), GeometryError> {
    if edge.curve != BoundaryCurve::Outline
        || !edge.start.t.is_finite() || !edge.end.t.is_finite()
        || !(0.0..=1.0).contains(&edge.start.t) || !(0.0..=1.0).contains(&edge.end.t)
        || edge.start.segment as usize >= curve.segments.len()
        || edge.end.segment as usize >= curve.segments.len()
        || edge.start_position() >= edge.end_position()
        || params.len() != curve.segments.len()
    { return Err(GeometryError::InvalidSegment); }

    let mut raw: Vec<(CurveParam, Point2)> = Vec::new();
    for segment_index in edge.start.segment as usize..=edge.end.segment as usize {
        let t0 = if segment_index == edge.start.segment as usize { edge.start.t } else { 0.0 };
        let t1 = if segment_index == edge.end.segment as usize { edge.end.t } else { 1.0 };
        let mut ts = vec![t0];
        ts.extend(params[segment_index].iter().copied()
            .filter(|t| *t > t0 + 1.0e-10 && *t < t1 - 1.0e-10));
        if t1 > t0 + 1.0e-10 { ts.push(t1); }
        for t in ts {
            let param = CurveParam { segment: segment_index as u32, t };
            let point = point_on_segment(curve, segment_index, t);
            if raw.last().is_none_or(|(previous, _)| previous.segment != param.segment
                || (previous.t - param.t).abs() > 1.0e-10) {
                raw.push((param, point));
            }
        }
    }
    if edge.reversed { raw.reverse(); }
    if raw.len() < 2 { return Err(GeometryError::DegenerateEdge); }

    let mut cumulative = vec![0.0_f64; raw.len()];
    for index in 1..raw.len() {
        cumulative[index] = cumulative[index - 1] + raw[index - 1].1.distance(raw[index].1);
    }
    let total = *cumulative.last().unwrap_or(&0.0);
    if !total.is_finite() || total <= EPS { return Err(GeometryError::DegenerateEdge); }
    let samples = raw.into_iter().enumerate().map(|(index, (param, point))| PathSample {
        param, point, fraction: (cumulative[index] / total).clamp(0.0, 1.0),
    }).collect();
    Ok((samples, total))
}

fn sample_path_chain(
    pieces_by_id: &HashMap<u64, &PatternPiece>,
    base_params: &HashMap<u64, SegmentParameters>,
    paths: &[EdgePath],
) -> Result<SampledPathChain, GeometryError> {
    let mut edges = Vec::new();
    let mut total_length = 0.0;
    for edge in paths {
        if edge.curve != BoundaryCurve::Outline { return Err(GeometryError::InvalidSegment); }
        let piece = *pieces_by_id.get(&edge.piece_id).ok_or(GeometryError::InvalidSegment)?;
        let params = base_params.get(&piece.id).ok_or(GeometryError::InvalidSegment)?;
        let (samples, length) = edge_path_samples(&piece.outline, params, edge)?;
        edges.push(EdgeSamples { piece_id: piece.id, samples, length, prefix: total_length });
        total_length += length;
    }
    if !total_length.is_finite() || total_length <= EPS { return Err(GeometryError::DegenerateEdge); }
    Ok(SampledPathChain { edges, total_length })
}

impl SampledPathChain {
    fn fractions(&self) -> Vec<f64> {
        let mut fractions = vec![0.0, 1.0];
        for edge in &self.edges {
            fractions.extend(edge.samples.iter().map(|sample|
                ((edge.prefix + sample.fraction * edge.length) / self.total_length).clamp(0.0, 1.0)));
        }
        fractions
    }

    fn parameter_at_fraction(&self, fraction: f64) -> Result<(u64, CurveParam), GeometryError> {
        let target = fraction.clamp(0.0, 1.0) * self.total_length;
        let edge = self.edges.iter().find(|edge| target <= edge.prefix + edge.length + 1.0e-10)
            .or_else(|| self.edges.last()).ok_or(GeometryError::InvalidSegment)?;
        let local = ((target - edge.prefix) / edge.length).clamp(0.0, 1.0);
        let samples = &edge.samples;
        if local <= 1.0e-10 { return Ok((edge.piece_id, samples[0].param)); }
        for pair in samples.windows(2) {
            let span = pair[1].fraction - pair[0].fraction;
            if span <= 1.0e-12 { continue; }
            if local <= pair[1].fraction + 1.0e-10 {
                let alpha = ((local - pair[0].fraction) / span).clamp(0.0, 1.0);
                let param = if pair[0].param.segment == pair[1].param.segment {
                    CurveParam { segment: pair[0].param.segment,
                        t: pair[0].param.t + (pair[1].param.t - pair[0].param.t) * alpha }
                } else if alpha < 0.5 { pair[0].param } else { pair[1].param };
                return Ok((edge.piece_id, param));
            }
        }
        Ok((edge.piece_id, samples.last().ok_or(GeometryError::InvalidSegment)?.param))
    }
}

/// Sincroniza cadeias M:N pelo arco acumulado e força piques/extremidades parciais como vértices.
pub fn sample_pattern_outlines_with_seam_sync(
    pieces: &[PatternPiece], seams: &[SeamDefinition],
) -> Result<HashMap<u64, SampledCurve>, GeometryError> {
    let pieces_by_id: HashMap<_, _> = pieces.iter().map(|piece| (piece.id, piece)).collect();
    let mut base_params = HashMap::with_capacity(pieces.len());
    for piece in pieces {
        let mut params = segment_parameters(&piece.outline, piece.fabric.particle_distance_meters)?;
        for notch in &piece.notches {
            if let Some(segment) = params.get_mut(notch.segment_index as usize) {
                if notch.t.is_finite() && (0.0..=1.0).contains(&notch.t) { segment.push(notch.t); }
            }
        }
        base_params.insert(piece.id, params);
    }

    // Força todos os parâmetros de costura antes de sincronizar; nenhuma extremidade depende de snap.
    for seam in seams {
        for edge in seam.sides_a.iter().chain(seam.sides_b.iter()) {
            if edge.curve != BoundaryCurve::Outline { continue; }
            let params = base_params.get_mut(&edge.piece_id).ok_or(GeometryError::InvalidSegment)?;
            for param in [edge.start, edge.end] {
                if !param.t.is_finite() || !(0.0..=1.0).contains(&param.t) {
                    return Err(GeometryError::InvalidParameter);
                }
                params.get_mut(param.segment as usize).ok_or(GeometryError::InvalidSegment)?.push(param.t);
            }
        }
    }
    for params in base_params.values_mut() {
        for segment in params { normalize_parameters(segment); }
    }

    let mut synchronized_params = base_params.clone();
    for seam in seams {
        if seam.sides_a.is_empty() || seam.sides_b.is_empty() { continue; }
        if seam.sides_a.iter().chain(seam.sides_b.iter()).any(|edge| edge.curve != BoundaryCurve::Outline) {
            continue;
        }
        let chain_a = sample_path_chain(&pieces_by_id, &base_params, &seam.sides_a)?;
        let chain_b = sample_path_chain(&pieces_by_id, &base_params, &seam.sides_b)?;
        let mut fractions = chain_a.fractions();
        fractions.extend(chain_b.fractions());
        fractions.sort_by(f64::total_cmp);
        fractions.dedup_by(|a, b| (*a - *b).abs() <= 1.0e-8);
        if fractions.len() > MAX_CURVE_SAMPLES { return Err(GeometryError::InvalidParameter); }

        for fraction in fractions {
            for (piece_id, param) in [
                chain_a.parameter_at_fraction(fraction)?,
                chain_b.parameter_at_fraction(fraction)?,
            ] {
                synchronized_params.get_mut(&piece_id).ok_or(GeometryError::InvalidSegment)?
                    .get_mut(param.segment as usize).ok_or(GeometryError::InvalidSegment)?.push(param.t);
            }
        }
        for paths in [&seam.sides_a, &seam.sides_b] {
            for edge in paths {
                let params = synchronized_params.get(&edge.piece_id).ok_or(GeometryError::InvalidSegment)?;
                let total = 1usize.saturating_add(
                    params.iter().map(|segment| segment.len().saturating_sub(1)).sum::<usize>()
                );
                if total > MAX_CURVE_SAMPLES { return Err(GeometryError::InvalidParameter); }
            }
        }
    }

    let mut sampled = HashMap::with_capacity(pieces.len());
    for piece in pieces {
        let params = synchronized_params.get_mut(&piece.id).ok_or(GeometryError::InvalidSegment)?;
        for segment in params.iter_mut() { normalize_parameters(segment); }
        sampled.insert(piece.id, sample_from_parameters(&piece.outline, params));
    }
    Ok(sampled)
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
    if polygon.len() > MAX_CURVE_SAMPLES {
        return Err(GeometryError::InvalidParameter);
    }
    if polygon.iter().any(|point| !point.x.is_finite() || !point.y.is_finite()) {
        return Err(GeometryError::InvalidParameter);
    }
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
    let area = signed_area(polygon);
    if !area.is_finite() {
        return Err(GeometryError::InvalidParameter);
    }
    if area.abs() <= EPS { return Err(GeometryError::TooFewPoints); }
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
    fn cut_closed_outline_creates_two_valid_pieces_and_cut_edge_ranges() {
        let outline = rectangle(Point2::new(0.0, 0.0), Point2::new(2.0, 1.0)).unwrap();
        let cut = Curve2D::polyline(&[
            Point2::new(1.0, 0.0), Point2::new(1.0, 1.0)
        ], false).unwrap();
        let result = cut_closed_outline(&outline, &cut, 0.02).unwrap();
        assert!(result.first.closed && result.second.closed);
        assert_eq!(result.first_cut_segment_range.1 - result.first_cut_segment_range.0, 1);
        assert_eq!(result.second_cut_segment_range.1 - result.second_cut_segment_range.0, 1);
        let first_area = signed_area(&sample_curve(&result.first, 16)).abs();
        let second_area = signed_area(&sample_curve(&result.second, 16)).abs();
        assert!((first_area - 1.0).abs() < 1e-6);
        assert!((second_area - 1.0).abs() < 1e-6);
        assert!(result.source_segment_fragments[0].len() == 2);
        assert!(result.source_segment_fragments[2].len() == 2);
    }

    #[test]
    fn cut_closed_outline_rejects_endpoints_far_from_boundary() {
        let outline = rectangle(Point2::new(0.0, 0.0), Point2::new(2.0, 1.0)).unwrap();
        let cut = Curve2D::polyline(&[
            Point2::new(0.5, 0.4), Point2::new(1.5, 0.4)
        ], false).unwrap();
        assert!(cut_closed_outline(&outline, &cut, 0.01).is_err());
    }

    #[test]
    fn cut_closed_outline_rejects_path_that_leaves_the_pattern() {
        let outline = rectangle(Point2::new(0.0, 0.0), Point2::new(2.0, 1.0)).unwrap();
        let cut = Curve2D::polyline(&[
            Point2::new(0.0, 0.0), Point2::new(1.0, 1.2), Point2::new(2.0, 0.0)
        ], false).unwrap();
        assert!(cut_closed_outline(&outline, &cut, 0.02).is_err());
    }

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

    fn edge_points(sampled: &SampledCurve, segment_index: usize) -> Vec<Point2> {
        let (start, end) = sampled.segment_vertex_ranges[segment_index];
        if end >= start {
            (start..=end).map(|index| sampled.points[index]).collect()
        } else {
            (start..sampled.points.len()).chain(0..=end)
                .map(|index| sampled.points[index]).collect()
        }
    }

    fn normalized_arc(points: &[Point2]) -> Vec<f64> {
        let mut cumulative = vec![0.0];
        for pair in points.windows(2) {
            cumulative.push(cumulative.last().copied().unwrap() + pair[0].distance(pair[1]));
        }
        let total = cumulative.last().copied().unwrap_or(0.0).max(EPS);
        cumulative.iter().map(|length| length / total).collect()
    }

    #[test]
    fn sampling_rejects_aggregate_vertex_overflow_across_multiple_segments() {
        let mut piece = PatternPiece::from_polygon(77, "Alta resolução", &[
            Point2::new(0.0, 0.0),
            Point2::new(1.1, 0.0),
            Point2::new(1.1, 1.1),
            Point2::new(0.0, 1.1),
        ]).unwrap();
        piece.fabric.particle_distance_meters = 0.001;
        assert!(sample_pattern_outlines_with_seam_sync(&[piece], &[]).is_err());
    }

    #[test]
    fn spacing_aware_sampling_subdivides_long_linear_boundaries() {
        let mut piece = PatternPiece::from_polygon(1, "Retângulo", &[
            Point2::new(0.0, 0.0),
            Point2::new(2.0, 0.0),
            Point2::new(2.0, 1.0),
            Point2::new(0.0, 1.0),
        ]).unwrap();
        piece.fabric.particle_distance_meters = 0.25;
        let sampled = sample_pattern_outlines_with_seam_sync(&[piece], &[]).unwrap();
        let outline = &sampled[&1];
        assert_eq!(outline.segment_vertex_ranges.len(), 4);
        assert!(outline.points.len() >= 24);
        for index in 0..outline.points.len() {
            let next = outline.points[(index + 1) % outline.points.len()];
            assert!(outline.points[index].distance(next) <= 0.250_001);
        }
    }

    #[test]
    fn asymmetric_seam_edges_get_identical_normalized_node_distribution() {
        let mut longer = PatternPiece::from_polygon(1, "Borda longa", &[
            Point2::new(0.0, 0.0),
            Point2::new(1.0, 0.0),
            Point2::new(1.0, 2.0),
            Point2::new(0.0, 2.0),
        ]).unwrap();
        let mut shorter = PatternPiece::from_polygon(2, "Borda curta", &[
            Point2::new(2.0, 0.0),
            Point2::new(3.0, 0.0),
            Point2::new(3.0, 1.0),
            Point2::new(2.0, 1.0),
        ]).unwrap();
        longer.fabric.particle_distance_meters = 0.5;
        shorter.fabric.particle_distance_meters = 0.4;
        let edge_a = pattern_core::EdgeRef::single_segment(1, 1);
        let mut edge_b = pattern_core::EdgeRef::single_segment(2, 3);
        // A borda esquerda do segundo molde tem orientação oposta à borda direita do primeiro.
        edge_b.reversed = true;
        let seam = SeamDefinition {
            id: 7,
            name: "Lateral".into(),
            sides_a: vec![edge_a.into()],
            sides_b: vec![edge_b.into()],
            ease_meters: 0.0,
        };
        let sampled = sample_pattern_outlines_with_seam_sync(&[longer, shorter], &[seam]).unwrap();
        let a = edge_points(&sampled[&1], 1);
        let mut b = edge_points(&sampled[&2], 3);
        b.reverse();
        assert_eq!(a.len(), b.len(), "ambos os lados da costura devem ter nós correspondentes");
        let arc_a = normalized_arc(&a);
        let arc_b = normalized_arc(&b);
        assert_eq!(arc_a.len(), arc_b.len());
        for (a, b) in arc_a.iter().zip(&arc_b) {
            assert!((a - b).abs() < 1.0e-6, "frações de arco divergentes: {a} vs {b}");
        }
    }

    #[test]
    fn synchronized_sampling_keeps_original_segment_ranges_and_reversed_closure() {
        let piece = PatternPiece::from_polygon(5, "Costura de fechamento", &[
            Point2::new(0.0, 0.0),
            Point2::new(1.0, 0.0),
            Point2::new(1.0, 1.0),
            Point2::new(0.0, 1.0),
        ]).unwrap();
        let sampled = sample_pattern_outlines_with_seam_sync(&[piece], &[]).unwrap();
        let outline = &sampled[&5];
        assert_eq!(outline.segment_vertex_ranges.len(), 4);
        let closing = outline.segment_vertex_ranges[3];
        assert_eq!(closing.1, 0);
        assert!(edge_points(outline, 3).len() >= 2);
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
    fn simple_polygon_rejects_excessive_vertex_count_before_quadratic_checks() {
        let polygon = (0..=MAX_CURVE_SAMPLES).map(|index| {
            let angle = std::f64::consts::TAU * index as f64 / (MAX_CURVE_SAMPLES + 1) as f64;
            Point2::new(angle.cos(), angle.sin())
        }).collect::<Vec<_>>();
        assert_eq!(validate_simple_polygon(&polygon), Err(GeometryError::InvalidParameter));
    }

    #[test]
    fn simple_polygon_rejects_non_finite_coordinates() {
        let polygon = [
            Point2::new(0.0, 0.0), Point2::new(1.0, 0.0),
            Point2::new(f64::NAN, 1.0), Point2::new(0.0, 1.0),
        ];
        assert_eq!(validate_simple_polygon(&polygon), Err(GeometryError::InvalidParameter));
    }

    #[test]
    fn simple_polygon_rejects_overflowing_area_calculation() {
        let polygon = [
            Point2::new(-1.0e200, -1.0e200),
            Point2::new(1.0e200, -1.0e200),
            Point2::new(1.0e200, 1.0e200),
            Point2::new(-1.0e200, 1.0e200),
        ];
        assert_eq!(validate_simple_polygon(&polygon), Err(GeometryError::InvalidParameter));
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
