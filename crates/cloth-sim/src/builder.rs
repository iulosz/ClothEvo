use std::collections::{BTreeMap, HashSet};

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

#[derive(Clone, Debug, Default)]
struct SeamSpec {
    pairs: Vec<(u32, u32)>,
    edge_a: Vec<u32>,
    edge_b: Vec<u32>,
    /// Diferença-alvo de comprimento A − B, em metros. Positivo significa A mais longa.
    ease_meters: f32,
}

#[derive(Clone, Copy, Debug)]
struct FoldEdgeSpec {
    /// Fração da rigidez de flexão local, normalizada em [0.02, 1].
    stiffness: f32,
    /// Ângulo diedral de repouso, já orientado pelo sentido da aresta.
    rest_angle: f32,
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
    seams: Vec<SeamSpec>,
    /// Dobradiças físicas por aresta de malha (índices globais ordenados).
    fold_edges: BTreeMap<(u32, u32), FoldEdgeSpec>,
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

    /// Costura duas bordas por comprimento de arco, assumindo diferença de comprimento zero.
    pub fn sew(&mut self, edge_a: &[u32], edge_b: &[u32]) {
        self.sew_with_ease(edge_a, edge_b, 0.0);
    }

    /// Costura duas bordas com a diferença-alvo assinada A − B (metros).
    ///
    /// Ex.: +0,04 m indica que a borda A deve conter 4 cm a mais que B; -0,04 m
    /// indica o inverso. O comprimento excedente é distribuído pelas arestas de contorno
    /// para gerar franzido/acomodação em vez de apenas guardar o valor no projeto.
    pub fn sew_with_ease(&mut self, edge_a: &[u32], edge_b: &[u32], ease_meters: f32) {
        self.sew_with_markers(edge_a, edge_b, ease_meters, &[]);
    }

    /// Versão com piques pareados por posição normalizada ao longo de cada borda.
    /// Os pares de piques são reforçados explicitamente mesmo quando as malhas têm resoluções
    /// diferentes; os vínculos regulares continuam cobrindo toda a extensão da costura.
    pub fn sew_with_markers(
        &mut self,
        edge_a: &[u32],
        edge_b: &[u32],
        ease_meters: f32,
        notch_markers: &[(f32, f32)],
    ) {
        self.sew_internal(edge_a, edge_b, ease_meters, notch_markers, false);
    }

    /// Variante para contornos reamostrados por uma lista comum de frações de comprimento
    /// de arco. Quando os lados possuem a mesma quantidade de nós, o vínculo é índice a índice:
    /// não se usa vizinho mais próximo, evitando muitos-para-um e pares cruzados.
    pub fn sew_synchronized_with_markers(
        &mut self,
        edge_a: &[u32],
        edge_b: &[u32],
        ease_meters: f32,
        notch_markers: &[(f32, f32)],
    ) {
        self.sew_internal(edge_a, edge_b, ease_meters, notch_markers, true);
    }

    fn sew_internal(
        &mut self,
        edge_a: &[u32],
        edge_b: &[u32],
        ease_meters: f32,
        notch_markers: &[(f32, f32)],
        synchronized: bool,
    ) {
        if edge_a.len() < 2 || edge_b.len() < 2 || !ease_meters.is_finite() {
            return;
        }
        let mut pairs = if synchronized && edge_a.len() == edge_b.len() {
            edge_a.iter().copied().zip(edge_b.iter().copied()).collect()
        } else {
            // Fallback compatível com callers externos que não passaram pelo sincronizador
            // do Pattern Studio.
            pair_by_arc_length(edge_a, edge_b, &self.uv)
        };
        let ta = edge_arc_params(edge_a, &self.uv);
        let tb = edge_arc_params(edge_b, &self.uv);
        for &(marker_a, marker_b) in notch_markers {
            let Some(&a) = edge_a.get(nearest_arc_index(marker_a, &ta)) else { continue; };
            let Some(&b) = edge_b.get(nearest_arc_index(marker_b, &tb)) else { continue; };
            if a != b {
                pairs.push((a, b));
            }
        }
        pairs.sort_unstable();
        pairs.dedup();
        self.seams.push(SeamSpec {
            pairs,
            edge_a: edge_a.to_vec(),
            edge_b: edge_b.to_vec(),
            ease_meters,
        });
    }

    /// Associa uma aresta interna à dobra física. A linha deve ter sido mapeada para
    /// arestas existentes da malha; arestas de contorno sem duas faces não geram dobradiça.
    /// stiffness é a fração da rigidez de flexão base; rest_angle é em radianos.
    pub fn add_fold_edge(
        &mut self,
        handle: PanelHandle,
        local_a: u32,
        local_b: u32,
        stiffness: f32,
        rest_angle: f32,
    ) {
        if local_a >= handle.count || local_b >= handle.count || local_a == local_b
            || !stiffness.is_finite() || !rest_angle.is_finite()
        {
            return;
        }
        let global_a = handle.global(local_a);
        let global_b = handle.global(local_b);
        let key = if global_a < global_b { (global_a, global_b) } else { (global_b, global_a) };
        let spec = FoldEdgeSpec {
            stiffness: stiffness.clamp(0.02, 1.0),
            rest_angle: rest_angle.clamp(-std::f32::consts::PI, std::f32::consts::PI),
        };
        self.fold_edges.entry(key)
            .and_modify(|existing| {
                // Linhas de dobra coincidentes combinam seus alvos; preserva a menor rigidez.
                existing.stiffness = existing.stiffness.min(spec.stiffness);
                existing.rest_angle =
                    (existing.rest_angle + spec.rest_angle).clamp(-std::f32::consts::PI, std::f32::consts::PI);
            })
            .or_insert(spec);
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
        let pinned = self.pinned.iter().copied().collect::<HashSet<_>>();
        for (i, p) in self.pos.iter_mut().enumerate() {
            let i = i as u32;
            // A mounting pin defines an exact initial anchor; don't perturb it away from its
            // configured placement before the first simulation step.
            if pinned.contains(&i) { continue; }
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
        let mut distance: Vec<DistanceConstraint> = Vec::with_capacity(edges.len() + self.seams.iter().map(|s| s.pairs.len()).sum::<usize>());
        let mut edge_constraint_indices: BTreeMap<(u32, u32), usize> = BTreeMap::new();
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
            edge_constraint_indices.insert((i, j), distance.len() - 1);

            if let [Some(mut p3), Some(mut p4)] = e.opposite {
                // Canonicaliza os vértices opostos: p3 sempre fica à esquerda da aresta
                // orientada i→j e p4 à direita. Isso torna o sinal de rest_angle estável
                // mesmo que a ordem dos triângulos na CDT varie.
                let edge = self.uv[j as usize] - self.uv[i as usize];
                let side3 = edge.perp_dot(self.uv[p3 as usize] - self.uv[i as usize]);
                let side4 = edge.perp_dot(self.uv[p4 as usize] - self.uv[i as usize]);
                if side3 < side4 { std::mem::swap(&mut p3, &mut p4); }

                // k_flexão = 3 B l² / (A1 + A2) → α = (A / (3 l²)) / B.
                // Uma dobra com rigidez fracionária s usa compliance 1/s e seu próprio
                // ângulo diedral de repouso; as demais arestas mantêm exatamente o alvo plano.
                let geometric_scale = (e.area / (3.0 * rest * rest)).max(1e-6);
                let material_scale = self.panel_materials[self.panel_of[i as usize] as usize]
                    .map(|material| global_bend / material.bend_nm.max(1.0e-9))
                    .unwrap_or(1.0);
                let fold = self.fold_edges.get(&(i, j));
                let fold_compliance = fold.map(|spec| 1.0 / spec.stiffness.clamp(0.02, 1.0)).unwrap_or(1.0);
                bending.push(BendConstraint {
                    p: [i, j, p3, p4],
                    rest_angle: fold.map(|spec| spec.rest_angle).unwrap_or(0.0),
                    scale: geometric_scale * material_scale * fold_compliance,
                });
            }
        }

        // Aplica a folga/franzido como diferença-alvo de comprimento A − B.
        // As restrições de borda são escaladas proporcionalmente; as costuras continuam
        // fazendo a correspondência por comprimento de arco entre vértices de ambas as malhas.
        let mut adjusted_boundary_edges = HashSet::new();
        for seam in &self.seams {
            adjust_seam_ease(
                &mut distance,
                &edge_constraint_indices,
                &self.uv,
                seam,
                &mut adjusted_boundary_edges,
            );
        }

        for seam in &self.seams {
            for &(a, b) in &seam.pairs {
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
        // 0.4 is the default global friction. Per-panel friction acts as a
        // scale factor, so the global slider remains a useful master control.
        let mut friction_scale = vec![1.0f32; n];
        let global_density = params.density.max(1.0e-4);
        let global_thickness = params.thickness.max(1.0e-4);
        for i in 0..n {
            if let Some(material) = self.panel_materials[self.panel_of[i] as usize] {
                mass_scale[i] = (material.density_kg_m2 / global_density).max(1.0e-4);
                thickness_scale[i] = (material.thickness_meters / global_thickness).max(1.0e-4);
                friction_scale[i] = (material.friction_coefficient / 0.4).clamp(0.0, 5.0);
            }
        }

        Simulation::from_parts(
            self.pos,
            self.triangles,
            area,
            pinned,
            mass_scale,
            thickness_scale,
            friction_scale,
            distance,
            dist_ranges,
            bending,
            bend_ranges,
            self.colliders,
            params,
        )
    }
}

/// Parâmetros de arco normalizados para uma cadeia de vértices da borda.
fn edge_arc_params(edge: &[u32], uv: &[Vec2]) -> Vec<f32> {
    let mut t = vec![0.0f32; edge.len()];
    for k in 1..edge.len() {
        t[k] = t[k - 1] + (uv[edge[k] as usize] - uv[edge[k - 1] as usize]).length();
    }
    let total = t.last().copied().unwrap_or(0.0).max(1e-9);
    t.iter().map(|v| v / total).collect()
}

fn nearest_arc_index(t: f32, ts: &[f32]) -> usize {
    let mut best = 0;
    let mut best_d = f32::MAX;
    for (k, &v) in ts.iter().enumerate() {
        let d = (v - t.clamp(0.0, 1.0)).abs();
        if d < best_d {
            best_d = d;
            best = k;
        }
    }
    best
}

/// Pareia vértices das duas bordas pelo parâmetro de comprimento de arco normalizado.
/// Os dois lados são cobertos mesmo quando possuem contagens diferentes.
fn pair_by_arc_length(a: &[u32], b: &[u32], uv: &[Vec2]) -> Vec<(u32, u32)> {
    if a.len() < 2 || b.len() < 2 {
        return Vec::new();
    }
    let ta = edge_arc_params(a, uv);
    let tb = edge_arc_params(b, uv);
    let mut pairs: Vec<(u32, u32)> = Vec::with_capacity(a.len() + b.len());
    for (i, &t) in ta.iter().enumerate() {
        pairs.push((a[i], b[nearest_arc_index(t, &tb)]));
    }
    for (j, &t) in tb.iter().enumerate() {
        pairs.push((a[nearest_arc_index(t, &ta)], b[j]));
    }
    pairs.sort_unstable();
    pairs.dedup();
    pairs
}

fn chain_length(edge: &[u32], uv: &[Vec2]) -> f32 {
    edge.windows(2)
        .map(|pair| (uv[pair[1] as usize] - uv[pair[0] as usize]).length())
        .sum()
}

/// Ajusta os comprimentos de repouso da borda excedente para obter a diferença assinada desejada.
fn adjust_seam_ease(
    distance: &mut [DistanceConstraint],
    edge_constraint_indices: &BTreeMap<(u32, u32), usize>,
    uv: &[Vec2],
    seam: &SeamSpec,
    adjusted: &mut HashSet<(u32, u32)>,
) {
    let len_a = chain_length(&seam.edge_a, uv);
    let len_b = chain_length(&seam.edge_b, uv);
    if len_a <= 1e-8 || len_b <= 1e-8 {
        return;
    }

    let ease = seam.ease_meters;
    let (target_a, target_b) = if ease >= 0.0 {
        (len_b + ease, len_b)
    } else {
        (len_a, len_a - ease)
    };
    let scale_a = (target_a / len_a).clamp(0.05, 20.0);
    let scale_b = (target_b / len_b).clamp(0.05, 20.0);

    for (edge, factor) in [(&seam.edge_a, scale_a), (&seam.edge_b, scale_b)] {
        if (factor - 1.0).abs() < 1e-5 { continue; }
        for pair in edge.windows(2) {
            let key = if pair[0] < pair[1] { (pair[0], pair[1]) } else { (pair[1], pair[0]) };
            if !adjusted.insert(key) { continue; }
            if let Some(&index) = edge_constraint_indices.get(&key) {
                distance[index].rest = (distance[index].rest * factor).max(1e-6);
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn physical_fold_edge_sets_rest_angle_and_local_bending_compliance() {
        let mesh = PanelMesh {
            uv: vec![
                Vec2::new(0.0, 0.0), Vec2::new(1.0, 0.0),
                Vec2::new(1.0, 1.0), Vec2::new(0.0, 1.0),
            ],
            triangles: vec![[0, 1, 2], [0, 2, 3]],
        };
        let mut flat_builder = ClothBuilder::new();
        flat_builder.add_panel(&mesh, Placement::at(Vec3::ZERO));
        let flat = flat_builder.build(SimParams::default());

        let mut folded_builder = ClothBuilder::new();
        let handle = folded_builder.add_panel(&mesh, Placement::at(Vec3::ZERO));
        folded_builder.add_fold_edge(handle, 0, 2, 0.5, 0.4);
        let folded = folded_builder.build(SimParams::default());

        let find_hinge = |sim: &Simulation| sim.bending.iter().find(|bend| {
            bend.p.contains(&0) && bend.p.contains(&2)
        }).unwrap();
        let flat_hinge = find_hinge(&flat);
        let fold_hinge = find_hinge(&folded);
        assert_eq!(flat_hinge.rest_angle, 0.0);
        assert!((fold_hinge.rest_angle - 0.4).abs() < 1e-6);
        assert!((fold_hinge.scale / flat_hinge.scale - 2.0).abs() < 1e-5);
    }

    #[test]
    fn seam_ease_adjusts_rest_length_to_requested_signed_difference() {
        let uv = vec![
            Vec2::new(0.0, 0.0), Vec2::new(0.0, 1.0),
            Vec2::new(1.0, 0.0), Vec2::new(1.0, 0.5),
        ];
        let mut distance = vec![
            DistanceConstraint { a: 0, b: 1, rest: 1.0, scale: 1.0, kind: ConstraintKind::Warp },
            DistanceConstraint { a: 2, b: 3, rest: 0.5, scale: 1.0, kind: ConstraintKind::Warp },
        ];
        let indices = BTreeMap::from([((0, 1), 0usize), ((2, 3), 1usize)]);
        let seam = SeamSpec {
            pairs: vec![(0, 2), (1, 3)],
            edge_a: vec![0, 1],
            edge_b: vec![2, 3],
            ease_meters: 0.5,
        };
        adjust_seam_ease(&mut distance, &indices, &uv, &seam, &mut HashSet::new());
        assert!((distance[0].rest - 1.0).abs() < 1e-6);
        assert!((distance[1].rest - 0.5).abs() < 1e-6);

        let zero_ease = SeamSpec { ease_meters: 0.0, ..seam };
        adjust_seam_ease(&mut distance, &indices, &uv, &zero_ease, &mut HashSet::new());
        assert!((distance[0].rest - 0.5).abs() < 1e-6);
    }

    #[test]
    fn notch_marker_pair_is_added_even_when_arc_sampling_would_choose_neighbours() {
        let uv = vec![
            Vec2::new(0.0, 0.0), Vec2::new(0.0, 0.3), Vec2::new(0.0, 1.0),
            Vec2::new(1.0, 0.0), Vec2::new(1.0, 0.5), Vec2::new(1.0, 1.0),
        ];
        let mut builder = ClothBuilder::new();
        builder.uv = uv;
        let a = [0u32, 1, 2];
        let b = [3u32, 4, 5];
        builder.sew_with_markers(&a, &b, 0.0, &[(1.0, 0.0)]);
        assert!(builder.seams[0].pairs.contains(&(2, 3)));
    }

    #[test]
    fn synchronized_sewing_uses_paired_indices_not_nearest_arc_neighbours() {
        let uv = vec![
            Vec2::new(0.0, 0.0), Vec2::new(0.0, 0.25), Vec2::new(0.0, 1.0),
            Vec2::new(1.0, 0.0), Vec2::new(1.0, 0.75), Vec2::new(1.0, 1.0),
        ];
        let mut builder = ClothBuilder::new();
        builder.uv = uv;
        let a = [0u32, 1, 2];
        let b = [3u32, 4, 5];
        builder.sew_synchronized_with_markers(&a, &b, 0.0, &[]);
        assert_eq!(builder.seams[0].pairs, vec![(0, 3), (1, 4), (2, 5)]);
    }

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
