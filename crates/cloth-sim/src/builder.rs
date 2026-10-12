use std::collections::BTreeMap;

use glam::{Vec2, Vec3};

use crate::collider::Collider;
use crate::coloring::{greedy_coloring, group_by_color};
use crate::mesh::{PanelMesh, Placement};
use crate::params::{ConstraintKind, PanelMaterial, SimParams};
use crate::solver::{BendConstraint, DistanceConstraint, Simulation};

/// Referência a um painel já adicionado: converte índices locais do painel em globais.
#[derive(Clone, Copy, Debug)]
pub struct PanelHandle {
    pub first: u32,
    pub count: u32,
}

impl PanelHandle {
    pub fn global(&self, local: u32) -> u32 {
        debug_assert!(local < self.count);
        self.first + local
    }

    pub fn globals(&self, locals: &[u32]) -> Vec<u32> {
        locals.iter().map(|&l| self.global(l)).collect()
    }
}

/// Monta uma [`Simulation`] a partir de painéis 2D, costuras, pinos e colisores.
#[derive(Default)]
pub struct ClothBuilder {
    pos: Vec<Vec3>,
    uv: Vec<Vec2>,
    panel_of: Vec<u32>,
    grains: Vec<Vec2>,
    triangles: Vec<[u32; 3]>,
    pinned: Vec<u32>,
    seams: Vec<(u32, u32)>,
    colliders: Vec<Collider>,
    /// Uma entrada por painel adicionado. None preserva os parâmetros globais para cenas antigas.
    panel_materials: Vec<Option<PanelMaterial>>,
}

impl ClothBuilder {
    pub fn new() -> Self {
        Self::default()
    }

    /// Adiciona um painel. A direção do fio (urdume) é `+v` do painel.
    pub fn add_panel(&mut self, mesh: &PanelMesh, placement: Placement) -> PanelHandle {
        self.add_panel_with_grain(mesh, placement, Vec2::Y)
    }

    /// Adiciona um painel com direção do fio personalizada, mantendo os parâmetros globais.
    pub fn add_panel_with_grain(
        &mut self,
        mesh: &PanelMesh,
        placement: Placement,
        grain: Vec2,
    ) -> PanelHandle {
        self.add_panel_inner(mesh, placement, grain, None)
    }

    /// Adiciona um painel cujas propriedades físicas são definidas pelo molde 2D.
    pub fn add_panel_with_material(
        &mut self,
        mesh: &PanelMesh,
        placement: Placement,
        material: PanelMaterial,
    ) -> Result<PanelHandle, &'static str> {
        self.add_panel_with_grain_and_material(mesh, placement, Vec2::Y, material)
    }

    /// Variante com linha de fio e propriedades físicas específicas do molde.
    pub fn add_panel_with_grain_and_material(
        &mut self,
        mesh: &PanelMesh,
        placement: Placement,
        grain: Vec2,
        material: PanelMaterial,
    ) -> Result<PanelHandle, &'static str> {
        let material = material.validate()?;
        Ok(self.add_panel_inner(mesh, placement, grain, Some(material)))
    }

    fn add_panel_inner(
        &mut self,
        mesh: &PanelMesh,
        placement: Placement,
        grain: Vec2,
        material: Option<PanelMaterial>,
    ) -> PanelHandle {
        let first = self.pos.len() as u32;
        let panel_id = self.grains.len() as u32;
        self.grains.push(grain.try_normalize().unwrap_or(Vec2::Y));
        self.panel_materials.push(material);

        for &uv in &mesh.uv {
            self.pos.push(placement.map(uv));
            self.uv.push(uv);
            self.panel_of.push(panel_id);
        }
        for t in &mesh.triangles {
            self.triangles.push([first + t[0], first + t[1], first + t[2]]);
        }
        PanelHandle { first, count: mesh.uv.len() as u32 }
    }

    /// Fixa uma partícula no lugar (massa infinita).
    pub fn pin(&mut self, global_index: u32) {
        self.pinned.push(global_index);
    }

    pub fn add_collider(&mut self, collider: Collider) {
        self.colliders.push(collider);
    }

    /// Costura duas bordas (listas ordenadas de partículas globais) mapeando por comprimento de arco.
    /// Bordas de tamanhos diferentes são aceitas: a diferença vira folga (ease) distribuída.
    pub fn sew(&mut self, edge_a: &[u32], edge_b: &[u32]) {
        let pairs = pair_by_arc_length(edge_a, edge_b, &self.uv);
        self.seams.extend(pairs);
    }

    pub fn particle_count(&self) -> usize {
        self.pos.len()
    }

    /// Ruído determinístico (±`amplitude` metros) nas posições 3D iniciais, sem alterar o molde 2D.
    /// Quebra a simetria de painéis perfeitamente planos, que de outra forma nunca flambariam.
    pub fn perturb(&mut self, amplitude: f32) {
        fn hash(mut x: u32) -> f32 {
            x ^= x >> 16;
            x = x.wrapping_mul(0x7feb_352d);
            x ^= x >> 15;
            x = x.wrapping_mul(0x846c_a68b);
            x ^= x >> 16;
            (x as f32 / u32::MAX as f32) - 0.5
        }
        for (i, p) in self.pos.iter_mut().enumerate() {
            let i = i as u32;
            *p += Vec3::new(hash(i * 3), hash(i * 3 + 1), hash(i * 3 + 2)) * (2.0 * amplitude);
        }
    }

    pub fn build(self, params: SimParams) -> Simulation {
        let n = self.pos.len();
        let global_stiffness = params.stiffness_table();
        let global_bend = params.bend.max(1.0e-9);

        // Área de repouso (2D) por partícula e por aresta.
        let tri_area = |t: &[u32; 3]| -> f32 {
            let a = self.uv[t[0] as usize];
            let b = self.uv[t[1] as usize];
            let c = self.uv[t[2] as usize];
            0.5 * (b - a).perp_dot(c - a).abs()
        };

        struct Edge {
            area: f32,
            // vértices opostos, um por triângulo adjacente (máx. 2)
            opposite: [Option<u32>; 2],
        }
        let mut edges: BTreeMap<(u32, u32), Edge> = BTreeMap::new();
        let mut area = vec![0.0f32; n];

        for t in &self.triangles {
            let a = tri_area(t);
            for &v in t {
                area[v as usize] += a / 3.0;
            }
            for k in 0..3 {
                let (i, j, opp) = (t[k], t[(k + 1) % 3], t[(k + 2) % 3]);
                let key = if i < j { (i, j) } else { (j, i) };
                let e = edges.entry(key).or_insert(Edge { area: 0.0, opposite: [None, None] });
                e.area += a;
                if e.opposite[0].is_none() {
                    e.opposite[0] = Some(opp);
                } else {
                    e.opposite[1] = Some(opp);
                }
            }
        }

        // Restrições de distância (stretch / shear) e de flexão.
        let mut distance: Vec<DistanceConstraint> = Vec::with_capacity(edges.len() + self.seams.len());
        let mut bending: Vec<BendConstraint> = Vec::new();

        let cos_warp = (22.5f32).to_radians().cos();
        let cos_weft = (67.5f32).to_radians().cos();

        for (&(i, j), e) in &edges {
            let rest = (self.uv[i as usize] - self.uv[j as usize]).length();
            if rest < 1e-9 {
                continue;
            }
            let grain = self.grains[self.panel_of[i as usize] as usize];
            let dir = (self.uv[j as usize] - self.uv[i as usize]) / rest;
            let c = dir.dot(grain).abs();
            let kind = if c >= cos_warp {
                ConstraintKind::Warp
            } else if c <= cos_weft {
                ConstraintKind::Weft
            } else {
                ConstraintKind::Shear
            };
            // k_aresta = Y · A_adjacente / l²  →  α = (l² / A) / Y
            let geometric_scale = (rest * rest / e.area).max(1e-6);
            let material_scale = self.panel_materials[self.panel_of[i as usize] as usize]
                .map(|material| {
                    global_stiffness[kind as usize] / material.stiffness(kind).max(1.0e-3)
                })
                .unwrap_or(1.0);
            distance.push(DistanceConstraint {
                a: i,
                b: j,
                rest,
                // XPBD: scale/global_k is equivalent to geometric_scale/panel_k.
                scale: geometric_scale * material_scale,
                kind,
            });

            if let [Some(p3), Some(p4)] = e.opposite {
                // k_flexão = 3 B l² / (A1 + A2)  →  α = (A / (3 l²)) / B
                let geometric_scale = (e.area / (3.0 * rest * rest)).max(1e-6);
                let material_scale = self.panel_materials[self.panel_of[i as usize] as usize]
                    .map(|material| global_bend / material.bend_nm.max(1.0e-9))
                    .unwrap_or(1.0);
                bending.push(BendConstraint {
                    p: [i, j, p3, p4],
                    rest_angle: 0.0, // painéis são planos no repouso
                    scale: geometric_scale * material_scale,
                });
            }
        }

        for &(a, b) in &self.seams {
            if a != b {
                distance.push(DistanceConstraint {
                    a,
                    b,
                    rest: 0.0,
                    scale: 1.0,
                    kind: ConstraintKind::Seam,
                });
            }
        }

        // Coloração para solução paralela.
        let dist_ids: Vec<[u32; 2]> = distance.iter().map(|c| [c.a, c.b]).collect();
        let dist_colors = greedy_coloring(&dist_ids, n);
        let (distance, dist_ranges) = group_by_color(&distance, &dist_colors);

        let bend_ids: Vec<[u32; 4]> = bending.iter().map(|c| c.p).collect();
        let bend_colors = greedy_coloring(&bend_ids, n);
        let (bending, bend_ranges) = group_by_color(&bending, &bend_colors);

        let mut pinned = vec![false; n];
        for &p in &self.pinned {
            pinned[p as usize] = true;
        }
        // Scale factors preserve each panel's material mass/thickness while global controls
        // remain useful as a master multiplier for the whole simulation.
        let mut mass_scale = vec![1.0f32; n];
        let mut thickness_scale = vec![1.0f32; n];
        let global_density = params.density.max(1.0e-4);
        let global_thickness = params.thickness.max(1.0e-4);
        for i in 0..n {
            if let Some(material) = self.panel_materials[self.panel_of[i] as usize] {
                mass_scale[i] = (material.density_kg_m2 / global_density).max(1.0e-4);
                thickness_scale[i] = (material.thickness_meters / global_thickness).max(1.0e-4);
            }
        }

        Simulation::from_parts(
            self.pos,
            self.triangles,
            area,
            pinned,
            mass_scale,
            thickness_scale,
            distance,
            dist_ranges,
            bending,
            bend_ranges,
            self.colliders,
            params,
        )
    }
}

/// Pareia partículas de duas bordas pelo parâmetro de comprimento de arco normalizado.
/// Cada partícula de cada borda recebe a mais próxima da outra; duplicatas são removidas.
fn pair_by_arc_length(a: &[u32], b: &[u32], uv: &[Vec2]) -> Vec<(u32, u32)> {
    if a.len() < 2 || b.len() < 2 {
        return Vec::new();
    }
    let params = |edge: &[u32]| -> Vec<f32> {
        let mut t = vec![0.0f32; edge.len()];
        for k in 1..edge.len() {
            t[k] = t[k - 1] + (uv[edge[k] as usize] - uv[edge[k - 1] as usize]).length();
        }
        let total = t[edge.len() - 1].max(1e-9);
        t.iter().map(|v| v / total).collect()
    };
    let ta = params(a);
    let tb = params(b);

    let nearest = |t: f32, ts: &[f32]| -> usize {
        let mut best = 0;
        let mut best_d = f32::MAX;
        for (k, &v) in ts.iter().enumerate() {
            let d = (v - t).abs();
            if d < best_d {
                best_d = d;
                best = k;
            }
        }
        best
    };

    let mut pairs: Vec<(u32, u32)> = Vec::with_capacity(a.len() + b.len());
    for (i, &t) in ta.iter().enumerate() {
        pairs.push((a[i], b[nearest(t, &tb)]));
    }
    for (j, &t) in tb.iter().enumerate() {
        pairs.push((a[nearest(t, &ta)], b[j]));
    }
    pairs.sort_unstable();
    pairs.dedup();
    pairs
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn arc_length_pairing_matches_equal_edges_one_to_one() {
        let uv: Vec<Vec2> = (0..4)
            .map(|k| Vec2::new(0.0, k as f32))
            .chain((0..4).map(|k| Vec2::new(1.0, k as f32)))
            .collect();
        let a = [0u32, 1, 2, 3];
        let b = [4u32, 5, 6, 7];
        let pairs = pair_by_arc_length(&a, &b, &uv);
        assert_eq!(pairs, vec![(0, 4), (1, 5), (2, 6), (3, 7)]);
    }

    #[test]
    fn arc_length_pairing_handles_different_lengths() {
        let uv: Vec<Vec2> = (0..3)
            .map(|k| Vec2::new(0.0, k as f32))
            .chain((0..5).map(|k| Vec2::new(1.0, k as f32 * 0.5)))
            .collect();
        let a = [0u32, 1, 2];
        let b = [3u32, 4, 5, 6, 7];
        let pairs = pair_by_arc_length(&a, &b, &uv);
        // todo ponto de ambas as bordas deve aparecer em ao menos um par
        for p in a.iter().chain(b.iter()) {
            assert!(pairs.iter().any(|&(x, y)| x == *p || y == *p));
        }
    }
}
