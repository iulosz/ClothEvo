//! Malhador 2D para moldes. Os vértices de contorno são preservados no início da
//! lista de vértices, na mesma ordem da curva amostrada, para mapear bordas de costura.
use pattern_core::Point2;
use pattern_geom::{point_in_polygon, signed_area, validate_simple_polygon, GeometryError};
use spade::{ConstrainedDelaunayTriangulation, Point2 as SpadePoint, Triangulation};
use std::fmt;

#[derive(Clone, Debug, Default, PartialEq)]
pub struct PatternMesh {
    pub vertices: Vec<Point2>,
    pub triangles: Vec<[u32; 3]>,
    pub boundary_edges: Vec<[u32; 2]>,
    pub boundary_vertex_count: usize,
}

#[derive(Clone, Debug, PartialEq)]
pub enum MeshError {
    Geometry(GeometryError),
    InvalidSpacing,
    Triangulation(String),
    EmptyResult,
}

impl fmt::Display for MeshError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Geometry(e) => write!(f, "{e}"),
            Self::InvalidSpacing => write!(f, "O espaçamento da malha deve ser positivo e finito."),
            Self::Triangulation(e) => write!(f, "Falha na triangulação: {e}"),
            Self::EmptyResult => write!(f, "A triangulação não produziu triângulos dentro do contorno."),
        }
    }
}
impl std::error::Error for MeshError {}
impl From<GeometryError> for MeshError {
    fn from(value: GeometryError) -> Self { Self::Geometry(value) }
}

/// Triangula um polígono fechado por CDT e cria pontos interiores com espaçamento aproximado.
/// Contornos côncavos são preservados; triângulos exteriores são filtrados pelo centróide.
pub fn triangulate_outline(outline: &[Point2], target_spacing: f64) -> Result<PatternMesh, MeshError> {
    if !target_spacing.is_finite() || target_spacing <= 0.0 {
        return Err(MeshError::InvalidSpacing);
    }
    let mut boundary = outline.to_vec();
    if boundary.len() > 3 && boundary[0].distance(*boundary.last().unwrap()) <= 1.0e-9 {
        boundary.pop();
    }
    validate_simple_polygon(&boundary)?;

    let min_x = boundary.iter().map(|p| p.x).fold(f64::INFINITY, f64::min);
    let min_y = boundary.iter().map(|p| p.y).fold(f64::INFINITY, f64::min);
    let max_x = boundary.iter().map(|p| p.x).fold(f64::NEG_INFINITY, f64::max);
    let max_y = boundary.iter().map(|p| p.y).fold(f64::NEG_INFINITY, f64::max);

    let mut points = boundary.clone();
    let mut y = min_y + target_spacing * 0.5;
    while y < max_y {
        let mut x = min_x + target_spacing * 0.5;
        while x < max_x {
            let candidate = Point2::new(x, y);
            if point_in_polygon(candidate, &boundary)
                && distance_to_boundary(candidate, &boundary) >= target_spacing * 0.2
            {
                points.push(candidate);
            }
            x += target_spacing;
        }
        y += target_spacing;
    }

    let boundary_count = boundary.len();
    let constraints = (0..boundary_count)
        .map(|i| [i, (i + 1) % boundary_count])
        .collect::<Vec<_>>();
    let spade_points = points.iter().map(|p| SpadePoint::new(p.x, p.y)).collect();
    let cdt = ConstrainedDelaunayTriangulation::<SpadePoint<f64>>::bulk_load_cdt(
        spade_points,
        constraints,
    ).map_err(|e| MeshError::Triangulation(e.to_string()))?;

    // O bulk_load_cdt preserva a ordem dos vértices inseridos; os índices fixos são usados
    // nos triângulos e no buffer retornado.
    let mut vertices = vec![Point2::default(); cdt.num_vertices()];
    for vertex in cdt.vertices() {
        let index = vertex.fix().index();
        let p = vertex.position();
        vertices[index] = Point2::new(p.x, p.y);
    }

    let mut triangles = Vec::new();
    for face in cdt.inner_faces() {
        let [a, b, c] = face.vertices().map(|vertex| vertex.fix().index() as u32);
        let pa = vertices[a as usize];
        let pb = vertices[b as usize];
        let pc = vertices[c as usize];
        let centroid = Point2::new((pa.x + pb.x + pc.x) / 3.0, (pa.y + pb.y + pc.y) / 3.0);
        if !point_in_polygon(centroid, &boundary) { continue; }

        let cross = (pb.x - pa.x) * (pc.y - pa.y) - (pb.y - pa.y) * (pc.x - pa.x);
        if cross.abs() <= 1.0e-12 { continue; }
        triangles.push(if cross > 0.0 { [a, b, c] } else { [a, c, b] });
    }

    if triangles.is_empty() { return Err(MeshError::EmptyResult); }
    let boundary_edges = (0..boundary_count)
        .map(|i| [i as u32, ((i + 1) % boundary_count) as u32])
        .collect();

    if signed_area(&boundary) == 0.0 {
        return Err(MeshError::Geometry(GeometryError::TooFewPoints));
    }
    Ok(PatternMesh { vertices, triangles, boundary_edges, boundary_vertex_count: boundary_count })
}

fn distance_to_boundary(point: Point2, polygon: &[Point2]) -> f64 {
    let mut min = f64::INFINITY;
    for i in 0..polygon.len() {
        let a = polygon[i];
        let b = polygon[(i + 1) % polygon.len()];
        let dx = b.x - a.x;
        let dy = b.y - a.y;
        let length_squared = dx * dx + dy * dy;
        if length_squared <= 1.0e-18 {
            min = min.min(point.distance(a));
        } else {
            let t = (((point.x - a.x) * dx + (point.y - a.y) * dy) / length_squared).clamp(0.0, 1.0);
            min = min.min(point.distance(Point2::new(a.x + t * dx, a.y + t * dy)));
        }
    }
    min
}

#[cfg(test)]
mod tests {
    use super::*;

    fn rectangle() -> [Point2; 4] {
        [
            Point2::new(0.0, 0.0), Point2::new(1.0, 0.0),
            Point2::new(1.0, 1.0), Point2::new(0.0, 1.0),
        ]
    }

    #[test]
    fn cdt_preserves_rectangle_boundary_and_area() {
        let mesh = triangulate_outline(&rectangle(), 0.25).unwrap();
        assert_eq!(mesh.boundary_vertex_count, 4);
        assert_eq!(mesh.boundary_edges.len(), 4);
        assert!(mesh.vertices.len() > 4);
        let area = mesh.triangles.iter().map(|t| {
            let a = mesh.vertices[t[0] as usize];
            let b = mesh.vertices[t[1] as usize];
            let c = mesh.vertices[t[2] as usize];
            ((b.x-a.x)*(c.y-a.y) - (b.y-a.y)*(c.x-a.x)).abs() * 0.5
        }).sum::<f64>();
        assert!((area - 1.0).abs() < 1e-6);
    }

    #[test]
    fn concave_outline_is_supported() {
        let l_shape = [
            Point2::new(0.0, 0.0), Point2::new(2.0, 0.0),
            Point2::new(2.0, 0.5), Point2::new(0.5, 0.5),
            Point2::new(0.5, 2.0), Point2::new(0.0, 2.0),
        ];
        let mesh = triangulate_outline(&l_shape, 0.2).unwrap();
        let area = mesh.triangles.iter().map(|t| {
            let a = mesh.vertices[t[0] as usize];
            let b = mesh.vertices[t[1] as usize];
            let c = mesh.vertices[t[2] as usize];
            ((b.x-a.x)*(c.y-a.y) - (b.y-a.y)*(c.x-a.x)).abs() * 0.5
        }).sum::<f64>();
        assert!((area - 1.75).abs() < 1e-5);
    }

    #[test]
    fn invalid_spacing_is_rejected() {
        assert_eq!(triangulate_outline(&rectangle(), 0.0), Err(MeshError::InvalidSpacing));
    }
}
