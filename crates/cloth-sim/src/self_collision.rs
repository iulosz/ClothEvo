//! Estruturas topológicas estáticas usadas pela auto-colisão do tecido.
//!
//! A geometria dos contatos muda durante a simulação; a conectividade da superfície e as
//! relações locais entre vértices não. Esta preparação é feita uma vez em Simulation::from_parts.

use std::collections::{BTreeMap, BTreeSet, HashSet};

type VertexPair = (u32, u32);

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
    #[allow(dead_code)]
    triangles: Vec<[u32; 3]>,
    edges: Vec<CollisionEdge>,
    #[allow(dead_code)]
    vertex_neighbors: Vec<Vec<u32>>,
    #[allow(dead_code)]
    vertex_triangles: Vec<Vec<u32>>,
    vertex_pair_exclusions: HashSet<VertexPair>,
    #[allow(dead_code)]
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

    pub(crate) fn vertex_pair_exclusions(&self) -> &HashSet<VertexPair> {
        &self.vertex_pair_exclusions
    }

    #[cfg(test)]
    fn vertex_neighbors(&self, vertex: usize) -> &[u32] {
        self.vertex_neighbors.get(vertex).map(Vec::as_slice).unwrap_or(&[])
    }

    #[cfg(test)]
    fn vertex_triangles(&self, vertex: usize) -> &[u32] {
        self.vertex_triangles.get(vertex).map(Vec::as_slice).unwrap_or(&[])
    }

    #[cfg(test)]
    fn is_local_vertex_triangle(&self, vertex: u32, face: u32) -> bool {
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

    #[cfg(test)]
    fn edges_are_local(&self, first: usize, second: usize) -> bool {
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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn stores_unique_edges_and_their_incident_faces() {
        let topology = CollisionTopology::new(
            4,
            &[[0, 1, 2], [2, 1, 3]],
            std::iter::empty(),
        );

        assert_eq!(topology.edges().len(), 5);
        let shared = topology
            .edges()
            .iter()
            .find(|edge| edge.vertices == [1, 2])
            .expect("shared edge");
        assert_eq!(shared.faces.as_slice(), &[0_u32, 1][..]);
        assert_eq!(topology.vertex_triangles(1), &[0_u32, 1][..]);
        assert_eq!(topology.vertex_neighbors(1), &[0_u32, 2, 3][..]);
    }

    #[test]
    fn seam_exclusions_cover_only_the_immediate_cross_seam_neighborhood() {
        let topology = CollisionTopology::new(
            9,
            &[[0, 1, 2], [3, 4, 5], [6, 7, 8]],
            [(2, 3)],
        );

        // O vértice e a face do outro lado da costura, incluindo o entorno de um anel,
        // são locais e não devem gerar auto-contato.
        assert!(topology.is_local_vertex_triangle(0, 1));
        assert!(topology.is_local_vertex_triangle(2, 1));
        assert!(topology.edges_are_local(0, 3));

        // A costura não propaga a exclusão até uma terceira região distante.
        assert!(!topology.is_local_vertex_triangle(0, 2));
        assert!(!topology.edges_are_local(0, 6));
    }

    #[test]
    fn seam_links_are_present_in_the_static_vertex_graph() {
        let topology = CollisionTopology::new(4, &[[0, 1, 2]], [(2, 3)]);
        assert_eq!(topology.vertex_neighbors(2), &[0_u32, 1, 3][..]);
        assert!(topology.vertex_pair_exclusions().contains(&(2, 3)));
    }
}
