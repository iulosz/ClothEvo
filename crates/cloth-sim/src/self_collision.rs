//! Geometria e broadphase da auto-colisão de superfícies de tecido.

use std::collections::{BTreeMap, BTreeSet, HashMap, HashSet};

use glam::Vec3;

type VertexPair = (u32, u32);
type Cell = (i32, i32, i32);

const MAX_CELLS_PER_PRIMITIVE: usize = 4096;
const MAX_VT_CANDIDATES_PER_VERTEX: usize = 96;
const MAX_EE_CANDIDATES_PER_EDGE: usize = 96;
const MAX_BROADPHASE_SCAN_PER_PRIMITIVE: usize = 768;
const MAX_SELF_COLLISION_CANDIDATES_TOTAL: usize = 2_000_000;
const MAX_SELF_COLLISION_GRID_ENTRIES_PER_KIND: usize = 1_000_000;

fn ordered_pair(a: u32, b: u32) -> VertexPair {
    if a < b { (a, b) } else { (b, a) }
}

#[derive(Clone, Debug)]
pub(crate) struct CollisionEdge {
    pub(crate) vertices: [u32; 2],
    /// Faces que compartilham esta aresta. Um vetor também cobre malhas não-manifold.
    #[allow(dead_code)]
    pub(crate) faces: Vec<u32>,
}

#[derive(Debug)]
pub(crate) struct CollisionTopology {
    triangles: Vec<[u32; 3]>,
    edges: Vec<CollisionEdge>,
    #[allow(dead_code)]
    vertex_neighbors: Vec<Vec<u32>>,
    #[allow(dead_code)]
    vertex_triangles: Vec<Vec<u32>>,
    vertex_pair_exclusions: HashSet<VertexPair>,
    local_vertex_pairs: HashSet<VertexPair>,
}

impl CollisionTopology {
    pub(crate) fn new(
        vertex_count: usize,
        triangles: &[[u32; 3]],
        seam_pairs: impl IntoIterator<Item = (u32, u32)>,
    ) -> Self {
        let mut edge_faces: BTreeMap<VertexPair, Vec<u32>> = BTreeMap::new();
        let mut mesh_neighbors = vec![Vec::new(); vertex_count];
        let mut vertex_triangles = vec![Vec::new(); vertex_count];

        for (face_index, triangle) in triangles.iter().enumerate() {
            let face_index = face_index as u32;
            for &vertex in triangle {
                if let Some(incident_faces) = vertex_triangles.get_mut(vertex as usize) {
                    incident_faces.push(face_index);
                }
            }

            for k in 0..3 {
                let (a, b) = (triangle[k], triangle[(k + 1) % 3]);
                if a == b || a as usize >= vertex_count || b as usize >= vertex_count {
                    continue;
                }
                let edge = ordered_pair(a, b);
                edge_faces.entry(edge).or_default().push(face_index);
                mesh_neighbors[a as usize].push(b);
                mesh_neighbors[b as usize].push(a);
            }
        }

        for neighbors in &mut mesh_neighbors {
            neighbors.sort_unstable();
            neighbors.dedup();
        }
        for incident_faces in &mut vertex_triangles {
            incident_faces.sort_unstable();
            incident_faces.dedup();
        }

        let mut seams = BTreeSet::new();
        for (a, b) in seam_pairs {
            if a != b && (a as usize) < vertex_count && (b as usize) < vertex_count {
                seams.insert(ordered_pair(a, b));
            }
        }

        let mut vertex_pair_exclusions = HashSet::new();
        let mut local_vertex_pairs = HashSet::new();
        for &edge in edge_faces.keys() {
            vertex_pair_exclusions.insert(edge);
            local_vertex_pairs.insert(edge);
        }
        for &seam in &seams {
            vertex_pair_exclusions.insert(seam);
            local_vertex_pairs.insert(seam);
        }

        // Costuras ampliam a exclusão somente por uma vizinhança de cada lado.
        // Isso protege o entorno imediato da união sem excluir painéis remotos conectados
        // por uma cadeia longa de costuras.
        for &(a, b) in &seams {
            let mut side_a = Vec::with_capacity(mesh_neighbors[a as usize].len() + 1);
            let mut side_b = Vec::with_capacity(mesh_neighbors[b as usize].len() + 1);
            side_a.push(a);
            side_a.extend_from_slice(&mesh_neighbors[a as usize]);
            side_b.push(b);
            side_b.extend_from_slice(&mesh_neighbors[b as usize]);
            for &va in &side_a {
                for &vb in &side_b {
                    if va != vb {
                        local_vertex_pairs.insert(ordered_pair(va, vb));
                    }
                }
            }
        }

        let mut vertex_neighbors = mesh_neighbors;
        for &(a, b) in &seams {
            vertex_neighbors[a as usize].push(b);
            vertex_neighbors[b as usize].push(a);
        }
        for neighbors in &mut vertex_neighbors {
            neighbors.sort_unstable();
            neighbors.dedup();
        }

        let edges = edge_faces
            .into_iter()
            .map(|((a, b), mut faces)| {
                faces.sort_unstable();
                faces.dedup();
                CollisionEdge {
                    vertices: [a, b],
                    faces,
                }
            })
            .collect();

        Self {
            triangles: triangles.to_vec(),
            edges,
            vertex_neighbors,
            vertex_triangles,
            vertex_pair_exclusions,
            local_vertex_pairs,
        }
    }

    pub(crate) fn edges(&self) -> &[CollisionEdge] {
        &self.edges
    }

    pub(crate) fn triangles(&self) -> &[[u32; 3]] {
        &self.triangles
    }

    #[allow(dead_code)]
    pub(crate) fn vertex_neighbors(&self, vertex: usize) -> &[u32] {
        self.vertex_neighbors.get(vertex).map(Vec::as_slice).unwrap_or(&[])
    }

    #[allow(dead_code)]
    pub(crate) fn vertex_triangles(&self, vertex: usize) -> &[u32] {
        self.vertex_triangles.get(vertex).map(Vec::as_slice).unwrap_or(&[])
    }

    pub(crate) fn vertex_pair_exclusions(&self) -> &HashSet<VertexPair> {
        &self.vertex_pair_exclusions
    }

    pub(crate) fn is_local_vertex_triangle(&self, vertex: u32, face: u32) -> bool {
        let Some(triangle) = self.triangles.get(face as usize) else {
            return false;
        };
        triangle.iter().any(|&face_vertex| {
            vertex == face_vertex
                || self
                    .local_vertex_pairs
                    .contains(&ordered_pair(vertex, face_vertex))
        })
    }

    pub(crate) fn edges_are_local(&self, first: usize, second: usize) -> bool {
        let (Some(a), Some(b)) = (self.edges.get(first), self.edges.get(second)) else {
            return false;
        };
        a.vertices.iter().any(|&va| {
            b.vertices.iter().any(|&vb| {
                va == vb || self.local_vertex_pairs.contains(&ordered_pair(va, vb))
            })
        })
    }
}

#[derive(Clone, Copy, Debug, Default, PartialEq)]
struct ClosestPoint {
    point: Vec3,
    barycentric: [f32; 3],
}

/// Candidato gerado pelo broadphase. Contagens incluem as colisões omitidas pelo orçamento.
#[derive(Debug, Default)]
pub(crate) struct SelfCollisionCandidates {
    pub(crate) vertex_triangle: Vec<(u32, u32)>,
    pub(crate) edge_edge: Vec<(u32, u32)>,
    pub(crate) vertex_triangle_candidates: usize,
    pub(crate) edge_edge_candidates: usize,
    pub(crate) dropped_candidates: usize,
    pub(crate) overflowed_primitives: usize,
}

#[derive(Clone, Copy, Debug, Default)]
pub(crate) struct SelfCollisionResolution {
    pub(crate) vertex_triangle_verified: usize,
    pub(crate) vertex_triangle_contacts: usize,
    pub(crate) edge_edge_verified: usize,
    pub(crate) edge_edge_contacts: usize,
    pub(crate) max_penetration_before_m: f32,
}

#[derive(Clone, Copy, Debug)]
struct Aabb {
    min: Vec3,
    max: Vec3,
}

impl Aabb {
    fn from_points(points: &[Vec3]) -> Option<Self> {
        let (&first, rest) = points.split_first()?;
        if !first.is_finite() || rest.iter().any(|point| !point.is_finite()) {
            return None;
        }
        let mut bounds = Self { min: first, max: first };
        for &point in rest {
            bounds.min = bounds.min.min(point);
            bounds.max = bounds.max.max(point);
        }
        Some(bounds)
    }

    fn expanded(self, radius: f32) -> Self {
        let amount = Vec3::splat(radius.max(0.0));
        Self { min: self.min - amount, max: self.max + amount }
    }
}

fn cell_of(point: Vec3, cell_size: f32) -> Cell {
    (
        (point.x / cell_size).floor() as i32,
        (point.y / cell_size).floor() as i32,
        (point.z / cell_size).floor() as i32,
    )
}

fn cells_for_aabb(bounds: Aabb, cell_size: f32) -> Option<Vec<Cell>> {
    let min = cell_of(bounds.min, cell_size);
    let max = cell_of(bounds.max, cell_size);
    let dx = max.0 as i64 - min.0 as i64 + 1;
    let dy = max.1 as i64 - min.1 as i64 + 1;
    let dz = max.2 as i64 - min.2 as i64 + 1;
    if dx <= 0 || dy <= 0 || dz <= 0 {
        return Some(Vec::new());
    }
    let cell_count = (dx as u64)
        .checked_mul(dy as u64)
        .and_then(|xy| xy.checked_mul(dz as u64))?;
    if cell_count > MAX_CELLS_PER_PRIMITIVE as u64 {
        return None;
    }

    let mut cells = Vec::with_capacity(cell_count as usize);
    for x in min.0..=max.0 {
        for y in min.1..=max.1 {
            for z in min.2..=max.2 {
                cells.push((x, y, z));
            }
        }
    }
    Some(cells)
}

fn indexed_bounds(indices: &[u32], positions: &[Vec3]) -> Option<Aabb> {
    let points = indices
        .iter()
        .map(|&index| positions.get(index as usize).copied())
        .collect::<Option<Vec<_>>>()?;
    Aabb::from_points(&points)
}

fn effective_thickness(base: f32, scale: f32) -> f32 {
    base.max(0.0) * scale.max(1.0e-4)
}

/// Insere AABBs expandidas de faces e arestas; consulta células com chaves e ordem estáveis.
pub(crate) fn build_candidates(
    positions: &[Vec3],
    thickness_scale: &[f32],
    base_thickness: f32,
    topology: &CollisionTopology,
) -> SelfCollisionCandidates {
    let mut output = SelfCollisionCandidates::default();
    if positions.len() < 2 || positions.len() != thickness_scale.len() {
        return output;
    }
    let max_thickness = thickness_scale
        .iter()
        .map(|&scale| effective_thickness(base_thickness, scale))
        .fold(0.0_f32, f32::max);
    if max_thickness <= 1.0e-6 {
        return output;
    }
    let cell_size = (2.0 * max_thickness).max(0.005);
    let expansion = max_thickness;
    let mut triangle_grid: HashMap<Cell, Vec<u32>> = HashMap::new();
    let mut edge_grid: HashMap<Cell, Vec<u32>> = HashMap::new();
    let mut triangle_grid_entries = 0usize;
    let mut edge_grid_entries = 0usize;

    for (face, triangle) in topology.triangles().iter().enumerate() {
        let Some(bounds) = indexed_bounds(triangle, positions).map(|aabb| aabb.expanded(expansion)) else {
            continue;
        };
        let Some(cells) = cells_for_aabb(bounds, cell_size) else {
            output.overflowed_primitives += 1;
            output.dropped_candidates += 1;
            continue;
        };
        if triangle_grid_entries.saturating_add(cells.len()) > MAX_SELF_COLLISION_GRID_ENTRIES_PER_KIND {
            output.overflowed_primitives += 1;
            output.dropped_candidates += 1;
            continue;
        }
        triangle_grid_entries += cells.len();
        for cell in cells {
            triangle_grid.entry(cell).or_default().push(face as u32);
        }
    }

    for (edge_index, edge) in topology.edges().iter().enumerate() {
        let Some(bounds) = indexed_bounds(&edge.vertices, positions).map(|aabb| aabb.expanded(expansion)) else {
            continue;
        };
        let Some(cells) = cells_for_aabb(bounds, cell_size) else {
            output.overflowed_primitives += 1;
            output.dropped_candidates += 1;
            continue;
        };
        if edge_grid_entries.saturating_add(cells.len()) > MAX_SELF_COLLISION_GRID_ENTRIES_PER_KIND {
            output.overflowed_primitives += 1;
            output.dropped_candidates += 1;
            continue;
        }
        edge_grid_entries += cells.len();
        for cell in cells {
            edge_grid.entry(cell).or_default().push(edge_index as u32);
        }
    }

    for (vertex, &position) in positions.iter().enumerate() {
        if !position.is_finite() {
            continue;
        }
        let Some(face_ids) = triangle_grid.get(&cell_of(position, cell_size)) else {
            continue;
        };
        let mut accepted_for_vertex = 0usize;
        for (candidate_index, face) in face_ids.iter().copied().enumerate() {
            if candidate_index >= MAX_BROADPHASE_SCAN_PER_PRIMITIVE {
                output.dropped_candidates += face_ids.len() - candidate_index;
                break;
            }
            if topology.is_local_vertex_triangle(vertex as u32, face) {
                continue;
            }
            output.vertex_triangle_candidates += 1;
            if accepted_for_vertex >= MAX_VT_CANDIDATES_PER_VERTEX
                || output.vertex_triangle.len() >= MAX_SELF_COLLISION_CANDIDATES_TOTAL
            {
                output.dropped_candidates += face_ids.len() - candidate_index;
                break;
            }
            output.vertex_triangle.push((vertex as u32, face));
            accepted_for_vertex += 1;
        }
    }

    for (edge_index, edge) in topology.edges().iter().enumerate() {
        let Some(bounds) = indexed_bounds(&edge.vertices, positions).map(|aabb| aabb.expanded(expansion)) else {
            continue;
        };
        let Some(cells) = cells_for_aabb(bounds, cell_size) else {
            continue;
        };
        let mut neighbors = BTreeSet::new();
        let mut scan_overflow = false;
        let mut scanned = 0usize;
        'edge_cells: for cell in cells {
            let Some(edge_ids) = edge_grid.get(&cell) else {
                continue;
            };
            for &other in edge_ids {
                if other as usize <= edge_index {
                    continue;
                }
                scanned += 1;
                if scanned > MAX_BROADPHASE_SCAN_PER_PRIMITIVE {
                    output.dropped_candidates += 1;
                    scan_overflow = true;
                    break;
                }
                if topology.edges_are_local(edge_index, other as usize) {
                    continue;
                }
                if !neighbors.insert(other) {
                    continue;
                }
                output.edge_edge_candidates += 1;
                if neighbors.len() > MAX_EE_CANDIDATES_PER_EDGE
                    || output.edge_edge.len() >= MAX_SELF_COLLISION_CANDIDATES_TOTAL
                {
                    neighbors.remove(&other);
                    output.dropped_candidates += 1;
                    scan_overflow = true;
                    break;
                }
                output.edge_edge.push((edge_index as u32, other));
            }
            if scan_overflow {
                break 'edge_cells;
            }
        }
    }

    output
}

fn closest_point_segment(point: Vec3, a: Vec3, b: Vec3) -> (Vec3, f32) {
    let ab = b - a;
    let denominator = ab.length_squared();
    let t = if denominator > 1.0e-20 {
        ((point - a).dot(ab) / denominator).clamp(0.0, 1.0)
    } else {
        0.0
    };
    (a + ab * t, t)
}

/// Ponto mais próximo do triângulo e seus pesos baricêntricos. Faces degeneradas usam suas arestas.
fn closest_point_triangle(point: Vec3, a: Vec3, b: Vec3, c: Vec3) -> Option<ClosestPoint> {
    if !point.is_finite() || !a.is_finite() || !b.is_finite() || !c.is_finite() {
        return None;
    }
    let ab = b - a;
    let ac = c - a;
    let face_normal = ab.cross(ac);
    if face_normal.length_squared() <= 1.0e-16 {
        let candidates = [
            (0, 1, closest_point_segment(point, a, b)),
            (1, 2, closest_point_segment(point, b, c)),
            (2, 0, closest_point_segment(point, c, a)),
        ];
        let mut best: Option<ClosestPoint> = None;
        let mut best_distance = f32::INFINITY;
        for (i, j, (closest, t)) in candidates {
            let distance_squared = (point - closest).length_squared();
            if distance_squared < best_distance {
                let mut barycentric = [0.0; 3];
                barycentric[i] = 1.0 - t;
                barycentric[j] = t;
                best = Some(ClosestPoint { point: closest, barycentric });
                best_distance = distance_squared;
            }
        }
        return best;
    }

    // Regiões de vértice A, B e C.
    let ap = point - a;
    let d1 = ab.dot(ap);
    let d2 = ac.dot(ap);
    if d1 <= 0.0 && d2 <= 0.0 {
        return Some(ClosestPoint { point: a, barycentric: [1.0, 0.0, 0.0] });
    }
    let bp = point - b;
    let d3 = ab.dot(bp);
    let d4 = ac.dot(bp);
    if d3 >= 0.0 && d4 <= d3 {
        return Some(ClosestPoint { point: b, barycentric: [0.0, 1.0, 0.0] });
    }

    // Região da aresta AB.
    let vc = d1 * d4 - d3 * d2;
    if vc <= 0.0 && d1 >= 0.0 && d3 <= 0.0 {
        let v = d1 / (d1 - d3).max(1.0e-20);
        return Some(ClosestPoint { point: a + ab * v, barycentric: [1.0 - v, v, 0.0] });
    }

    // Regiões de vértice C e das arestas AC e BC.
    let cp = point - c;
    let d5 = ab.dot(cp);
    let d6 = ac.dot(cp);
    if d6 >= 0.0 && d5 <= d6 {
        return Some(ClosestPoint { point: c, barycentric: [0.0, 0.0, 1.0] });
    }
    let vb = d5 * d2 - d1 * d6;
    if vb <= 0.0 && d2 >= 0.0 && d6 <= 0.0 {
        let w = d2 / (d2 - d6).max(1.0e-20);
        return Some(ClosestPoint { point: a + ac * w, barycentric: [1.0 - w, 0.0, w] });
    }
    let va = d3 * d6 - d5 * d4;
    if va <= 0.0 && (d4 - d3) >= 0.0 && (d5 - d6) >= 0.0 {
        let edge = c - b;
        let w = (d4 - d3) / ((d4 - d3) + (d5 - d6)).max(1.0e-20);
        return Some(ClosestPoint { point: b + edge * w, barycentric: [0.0, 1.0 - w, w] });
    }

    let denominator = (va + vb + vc).max(1.0e-20);
    let v = vb / denominator;
    let w = vc / denominator;
    let u = 1.0 - v - w;
    Some(ClosestPoint { point: a * u + b * v + c * w, barycentric: [u, v, w] })
}

/// Pontos mais próximos de dois segmentos e seus parâmetros normalizados.
fn closest_segment_segment(p1: Vec3, q1: Vec3, p2: Vec3, q2: Vec3) -> Option<(f32, f32, Vec3, Vec3)> {
    if !p1.is_finite() || !q1.is_finite() || !p2.is_finite() || !q2.is_finite() {
        return None;
    }
    let d1 = q1 - p1;
    let d2 = q2 - p2;
    let r = p1 - p2;
    let a = d1.dot(d1);
    let e = d2.dot(d2);
    let f = d2.dot(r);
    let epsilon = 1.0e-20;
    let (mut s, mut t);

    if a <= epsilon && e <= epsilon {
        s = 0.0;
        t = 0.0;
    } else if a <= epsilon {
        s = 0.0;
        t = (f / e).clamp(0.0, 1.0);
    } else {
        let c = d1.dot(r);
        if e <= epsilon {
            t = 0.0;
            s = (-c / a).clamp(0.0, 1.0);
        } else {
            let b = d1.dot(d2);
            let denominator = a * e - b * b;
            s = if denominator.abs() > epsilon {
                ((b * f - c * e) / denominator).clamp(0.0, 1.0)
            } else {
                0.0
            };
            t = (b * s + f) / e;
            if t < 0.0 {
                t = 0.0;
                s = (-c / a).clamp(0.0, 1.0);
            } else if t > 1.0 {
                t = 1.0;
                s = ((b - c) / a).clamp(0.0, 1.0);
            }
        }
    }
    Some((s, t, p1 + d1 * s, p2 + d2 * t))
}

fn triangle_for(topology: &CollisionTopology, face: u32, positions: &[Vec3]) -> Option<([u32; 3], [Vec3; 3])> {
    let indices = *topology.triangles().get(face as usize)?;
    let points = [
        *positions.get(indices[0] as usize)?,
        *positions.get(indices[1] as usize)?,
        *positions.get(indices[2] as usize)?,
    ];
    Some((indices, points))
}

fn vertex_triangle_gap(
    vertex: u32,
    face: u32,
    positions: &[Vec3],
    thickness_scale: &[f32],
    base_thickness: f32,
    topology: &CollisionTopology,
) -> Option<f32> {
    let (indices, points) = triangle_for(topology, face, positions)?;
    let point = *positions.get(vertex as usize)?;
    let closest = closest_point_triangle(point, points[0], points[1], points[2])?;
    let thickness_at_face = (0..3)
        .map(|i| closest.barycentric[i] * effective_thickness(base_thickness, thickness_scale[indices[i] as usize]))
        .sum::<f32>();
    let minimum = 0.5 * (
        effective_thickness(base_thickness, thickness_scale[vertex as usize]) + thickness_at_face
    );
    Some(minimum - (point - closest.point).length())
}

fn edge_edge_gap(
    edge_a: u32,
    edge_b: u32,
    positions: &[Vec3],
    thickness_scale: &[f32],
    base_thickness: f32,
    topology: &CollisionTopology,
) -> Option<f32> {
    let a = topology.edges().get(edge_a as usize)?.vertices;
    let b = topology.edges().get(edge_b as usize)?.vertices;
    let (s, t, point_a, point_b) = closest_segment_segment(
        *positions.get(a[0] as usize)?,
        *positions.get(a[1] as usize)?,
        *positions.get(b[0] as usize)?,
        *positions.get(b[1] as usize)?,
    )?;
    let thickness_a =
        (1.0 - s) * effective_thickness(base_thickness, thickness_scale[a[0] as usize])
        + s * effective_thickness(base_thickness, thickness_scale[a[1] as usize]);
    let thickness_b =
        (1.0 - t) * effective_thickness(base_thickness, thickness_scale[b[0] as usize])
        + t * effective_thickness(base_thickness, thickness_scale[b[1] as usize]);
    Some(0.5 * (thickness_a + thickness_b) - (point_a - point_b).length())
}

/// Projeta todos os contatos discretos com Gauss-Seidel serial; cada par é unilateral.
pub(crate) fn solve_contacts(
    candidates: &SelfCollisionCandidates,
    topology: &CollisionTopology,
    positions: &mut [Vec3],
    previous: &mut [Vec3],
    inverse_mass: &[f32],
    thickness_scale: &[f32],
    base_thickness: f32,
) -> SelfCollisionResolution {
    let mut result = SelfCollisionResolution::default();

    for &(vertex, face) in &candidates.vertex_triangle {
        result.vertex_triangle_verified += 1;
        let Some((indices, points)) = triangle_for(topology, face, positions) else {
            continue;
        };
        let Some(point) = positions.get(vertex as usize).copied() else {
            continue;
        };
        let Some(closest) = closest_point_triangle(point, points[0], points[1], points[2]) else {
            continue;
        };
        let thickness_at_face = (0..3)
            .map(|i| closest.barycentric[i] * effective_thickness(base_thickness, thickness_scale[indices[i] as usize]))
            .sum::<f32>();
        let minimum = 0.5 * (
            effective_thickness(base_thickness, thickness_scale[vertex as usize]) + thickness_at_face
        );
        let delta = point - closest.point;
        let distance = delta.length();
        if !distance.is_finite() || distance >= minimum {
            continue;
        }
        let penetration = minimum - distance;
        result.vertex_triangle_contacts += 1;
        result.max_penetration_before_m = result.max_penetration_before_m.max(penetration);

        let normal = if distance > 1.0e-7 {
            Some(delta / distance)
        } else {
            let previous_triangle = indices
                .iter()
                .enumerate()
                .map(|(i, &index)| previous[index as usize] * closest.barycentric[i])
                .fold(Vec3::ZERO, |sum, point| sum + point);
            let history = previous[vertex as usize] - previous_triangle;
            let face_normal = (points[1] - points[0]).cross(points[2] - points[0]);
            face_normal.try_normalize().map(|mut normal| {
                if history.is_finite() && history.length_squared() > 1.0e-12 && history.dot(normal) < 0.0 {
                    normal = -normal;
                }
                normal
            }).or_else(|| history.try_normalize())
        };
        let Some(normal) = normal else {
            continue;
        };

        let weights = closest.barycentric;
        let wp = inverse_mass[vertex as usize].max(0.0);
        let wa = inverse_mass[indices[0] as usize].max(0.0);
        let wb = inverse_mass[indices[1] as usize].max(0.0);
        let wc = inverse_mass[indices[2] as usize].max(0.0);
        let denominator = wp + wa * weights[0] * weights[0]
            + wb * weights[1] * weights[1] + wc * weights[2] * weights[2];
        if denominator <= 1.0e-12 || !denominator.is_finite() {
            continue;
        }
        let scale = penetration / denominator;
        apply_correction(vertex as usize, normal * (wp * scale), positions, previous, inverse_mass);
        for i in 0..3 {
            let index = indices[i] as usize;
            let correction = -normal * (inverse_mass[index].max(0.0) * weights[i] * scale);
            apply_correction(index, correction, positions, previous, inverse_mass);
        }
    }

    for &(edge_a, edge_b) in &candidates.edge_edge {
        result.edge_edge_verified += 1;
        let (Some(a), Some(b)) = (
            topology.edges().get(edge_a as usize),
            topology.edges().get(edge_b as usize),
        ) else {
            continue;
        };
        let (a, b) = (a.vertices, b.vertices);
        let Some((s, t, point_a, point_b)) = closest_segment_segment(
            positions[a[0] as usize], positions[a[1] as usize],
            positions[b[0] as usize], positions[b[1] as usize],
        ) else {
            continue;
        };
        let thickness_a =
            (1.0 - s) * effective_thickness(base_thickness, thickness_scale[a[0] as usize])
            + s * effective_thickness(base_thickness, thickness_scale[a[1] as usize]);
        let thickness_b =
            (1.0 - t) * effective_thickness(base_thickness, thickness_scale[b[0] as usize])
            + t * effective_thickness(base_thickness, thickness_scale[b[1] as usize]);
        let minimum = 0.5 * (thickness_a + thickness_b);
        let delta = point_a - point_b;
        let distance = delta.length();
        if !distance.is_finite() || distance >= minimum {
            continue;
        }
        let penetration = minimum - distance;
        result.edge_edge_contacts += 1;
        result.max_penetration_before_m = result.max_penetration_before_m.max(penetration);

        let wa = [1.0 - s, s];
        let wb = [1.0 - t, t];
        let history_a = previous[a[0] as usize] * wa[0] + previous[a[1] as usize] * wa[1];
        let history_b = previous[b[0] as usize] * wb[0] + previous[b[1] as usize] * wb[1];
        let normal = if distance > 1.0e-7 {
            Some(delta / distance)
        } else {
            (history_a - history_b).try_normalize().or_else(|| {
                let tangent_a = positions[a[1] as usize] - positions[a[0] as usize];
                let tangent_b = positions[b[1] as usize] - positions[b[0] as usize];
                tangent_a.cross(tangent_b).try_normalize()
            })
        };
        let Some(normal) = normal else {
            continue;
        };

        let ids_a = [a[0] as usize, a[1] as usize];
        let ids_b = [b[0] as usize, b[1] as usize];
        let weights = [wa[0], wa[1], -wb[0], -wb[1]];
        let ids = [ids_a[0], ids_a[1], ids_b[0], ids_b[1]];
        let denominator = (0..4)
            .map(|i| inverse_mass[ids[i]].max(0.0) * weights[i] * weights[i])
            .sum::<f32>();
        if denominator <= 1.0e-12 || !denominator.is_finite() {
            continue;
        }
        let scale = penetration / denominator;
        for i in 0..4 {
            let correction = normal * (inverse_mass[ids[i]].max(0.0) * weights[i] * scale);
            apply_correction(ids[i], correction, positions, previous, inverse_mass);
        }
    }

    result
}

fn apply_correction(
    index: usize,
    correction: Vec3,
    positions: &mut [Vec3],
    previous: &mut [Vec3],
    inverse_mass: &[f32],
) {
    if inverse_mass[index] > 0.0 && correction.is_finite() {
        positions[index] += correction;
        // A projeção de posição não deve criar uma velocidade normal artificial.
        previous[index] += correction;
    }
}

/// Maior violação de espessura que restou após a última iteração.
pub(crate) fn residual_penetration(
    candidates: &SelfCollisionCandidates,
    topology: &CollisionTopology,
    positions: &[Vec3],
    thickness_scale: &[f32],
    base_thickness: f32,
) -> f32 {
    let mut residual = 0.0_f32;
    for &(vertex, face) in &candidates.vertex_triangle {
        if let Some(gap) = vertex_triangle_gap(vertex, face, positions, thickness_scale, base_thickness, topology) {
            residual = residual.max(gap.max(0.0));
        }
    }
    for &(edge_a, edge_b) in &candidates.edge_edge {
        if let Some(gap) = edge_edge_gap(edge_a, edge_b, positions, thickness_scale, base_thickness, topology) {
            residual = residual.max(gap.max(0.0));
        }
    }
    residual
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn stores_unique_edges_and_their_incident_faces() {
        let topology = CollisionTopology::new(4, &[[0, 1, 2], [2, 1, 3]], std::iter::empty());
        assert_eq!(topology.edges().len(), 5);
        let shared = topology.edges().iter().find(|edge| edge.vertices == [1, 2]).unwrap();
        assert_eq!(shared.faces.as_slice(), &[0_u32, 1][..]);
        assert_eq!(topology.vertex_triangles(1), &[0_u32, 1][..]);
        assert_eq!(topology.vertex_neighbors(1), &[0_u32, 2, 3][..]);
    }

    #[test]
    fn seam_exclusions_cover_only_the_immediate_cross_seam_neighborhood() {
        let topology = CollisionTopology::new(9, &[[0, 1, 2], [3, 4, 5], [6, 7, 8]], [(2, 3)]);
        assert!(topology.is_local_vertex_triangle(0, 1));
        assert!(topology.is_local_vertex_triangle(2, 1));
        assert!(topology.edges_are_local(0, 3));
        assert!(!topology.is_local_vertex_triangle(0, 2));
        assert!(!topology.edges_are_local(0, 6));
    }

    #[test]
    fn closest_point_triangle_covers_face_edge_vertex_and_degenerate_regions() {
        let a = Vec3::ZERO;
        let b = Vec3::X;
        let c = Vec3::Y;
        let face = closest_point_triangle(Vec3::new(0.25, 0.25, 0.2), a, b, c).unwrap();
        assert!((face.point - Vec3::new(0.25, 0.25, 0.0)).length() < 1.0e-6);
        assert!((face.barycentric.iter().sum::<f32>() - 1.0).abs() < 1.0e-6);

        let edge = closest_point_triangle(Vec3::new(0.5, -0.2, 0.0), a, b, c).unwrap();
        assert!((edge.point - Vec3::new(0.5, 0.0, 0.0)).length() < 1.0e-6);

        let vertex = closest_point_triangle(Vec3::new(-0.2, -0.2, 0.0), a, b, c).unwrap();
        assert_eq!(vertex.barycentric, [1.0, 0.0, 0.0]);

        let degenerate = closest_point_triangle(Vec3::new(0.5, 0.1, 0.0), a, b, b).unwrap();
        assert!(degenerate.point.is_finite());
        assert!((degenerate.barycentric.iter().sum::<f32>() - 1.0).abs() < 1.0e-6);
    }

    #[test]
    fn crossing_edges_are_detected_between_their_endpoints() {
        let result = closest_segment_segment(
            Vec3::new(-1.0, 0.0, 0.0),
            Vec3::new(1.0, 0.0, 0.0),
            Vec3::new(0.0, -1.0, 0.0),
            Vec3::new(0.0, 1.0, 0.0),
        ).unwrap();
        assert!((result.0 - 0.5).abs() < 1.0e-6);
        assert!((result.1 - 0.5).abs() < 1.0e-6);
        assert!((result.2 - result.3).length() < 1.0e-6);
    }

    #[test]
    fn broadphase_finds_a_vertex_near_a_large_face_without_nearby_face_vertices() {
        let topology = CollisionTopology::new(4, &[[0, 1, 2]], std::iter::empty());
        let positions = [
            Vec3::new(-1.0, 0.0, -1.0),
            Vec3::new(1.0, 0.0, -1.0),
            Vec3::new(0.0, 0.0, 1.0),
            Vec3::new(0.0, 0.05, 0.0),
        ];
        let candidates = build_candidates(&positions, &[1.0; 4], 0.1, &topology);
        assert!(candidates.vertex_triangle.contains(&(3, 0)));
    }

    #[test]
    fn vertex_triangle_projection_respects_pins_and_preserves_velocity() {
        let topology = CollisionTopology::new(4, &[[0, 1, 2]], std::iter::empty());
        let mut positions = [
            Vec3::new(-1.0, 0.0, -1.0),
            Vec3::new(1.0, 0.0, -1.0),
            Vec3::new(0.0, 0.0, 1.0),
            Vec3::new(0.0, 0.05, 0.0),
        ];
        let mut previous = positions;
        previous[3].y = 0.04;
        let candidates = build_candidates(&positions, &[1.0; 4], 0.1, &topology);
        let before_velocity = positions[3] - previous[3];
        let result = solve_contacts(
            &candidates, &topology, &mut positions, &mut previous,
            &[0.0, 0.0, 0.0, 1.0], &[1.0; 4], 0.1,
        );
        assert_eq!(result.vertex_triangle_contacts, 1);
        assert!(positions[3].y >= 0.099);
        assert_eq!(positions[0], Vec3::new(-1.0, 0.0, -1.0));
        assert!(((positions[3] - previous[3]) - before_velocity).length() < 1.0e-6);
    }

    #[test]
    fn crossing_edge_broadphase_yields_a_deduplicated_pair() {
        let topology = CollisionTopology::new(
            6,
            &[[0, 1, 2], [3, 4, 5]],
            std::iter::empty(),
        );
        let positions = [
            Vec3::new(-1.0, 0.0, 0.0),
            Vec3::new(1.0, 0.0, 0.0),
            Vec3::new(-1.0, -1.0, 0.0),
            Vec3::new(0.0, -1.0, 0.0),
            Vec3::new(0.0, 1.0, 0.0),
            Vec3::new(1.0, 1.0, 0.0),
        ];
        let candidates = build_candidates(&positions, &[1.0; 6], 0.1, &topology);
        let crossing = candidates.edge_edge.iter().filter(|&&(a, b)| {
            topology.edges()[a as usize].vertices == [0, 1]
                && topology.edges()[b as usize].vertices == [3, 4]
        }).count();
        assert_eq!(crossing, 1);
    }
    #[test]
    fn crossing_edge_projection_respects_thickness_and_uses_segment_weights() {
        let topology = CollisionTopology::new(
            6,
            &[[0, 1, 2], [3, 4, 5]],
            std::iter::empty(),
        );
        let mut positions = [
            Vec3::new(-1.0, 0.0, 0.0),
            Vec3::new(1.0, 0.0, 0.0),
            Vec3::new(-1.0, -1.0, 0.0),
            Vec3::new(0.0, -1.0, 0.0),
            Vec3::new(0.0, 1.0, 0.0),
            Vec3::new(1.0, 1.0, 0.0),
        ];
        let mut previous = positions;
        previous[0].z = 0.05;
        previous[1].z = 0.05;
        previous[3].z = -0.05;
        previous[4].z = -0.05;
        let candidates = SelfCollisionCandidates {
            edge_edge: vec![(0, 3)],
            ..SelfCollisionCandidates::default()
        };
        let result = solve_contacts(
            &candidates,
            &topology,
            &mut positions,
            &mut previous,
            &[1.0; 6],
            &[1.0; 6],
            0.1,
        );
        assert_eq!(result.edge_edge_contacts, 1);
        let (_, _, point_a, point_b) = closest_segment_segment(
            positions[0], positions[1], positions[3], positions[4],
        ).unwrap();
        assert!((point_a - point_b).length() >= 0.099);
    }

    #[test]
    fn broadphase_reports_candidates_omitted_by_its_budgets() {
        let face_count = 256usize;
        let positions = vec![Vec3::ZERO; face_count * 3];
        let triangles = (0..face_count)
            .map(|face| {
                let first = (face * 3) as u32;
                [first, first + 1, first + 2]
            })
            .collect::<Vec<_>>();
        let topology = CollisionTopology::new(positions.len(), &triangles, std::iter::empty());
        let candidates = build_candidates(&positions, &vec![1.0; positions.len()], 0.01, &topology);
        assert!(candidates.vertex_triangle_candidates > candidates.vertex_triangle.len());
        assert!(candidates.dropped_candidates > 0);
        assert!(candidates.edge_edge.len() <= MAX_SELF_COLLISION_CANDIDATES_TOTAL);
    }

}
