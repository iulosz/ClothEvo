use std::collections::{BTreeMap, HashMap, HashSet};

use glam::{Vec2, Vec3};
use pattern_core::{polyline_length_meters, Point2};

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
    edge_a_paths: Vec<Vec<u32>>,
    edge_b_paths: Vec<Vec<u32>>,
    /// Diferença-alvo de comprimento A − B, em metros. Positivo significa A mais longa.
    ease_meters: f32,
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
        self.sew_with_path_chains(&[edge_a.to_vec()], &[edge_b.to_vec()], ease_meters, notch_markers);
    }

    /// Costura cadeias M:N. O arco acumulado reinicia a cada borda de entrada, sem inventar
    /// um segmento entre peças/trechos desconectados da mesma cadeia.
    pub fn sew_with_path_chains(
        &mut self,
        paths_a: &[Vec<u32>],
        paths_b: &[Vec<u32>],
        ease_meters: f32,
        notch_markers: &[(f32, f32)],
    ) {
        if !ease_meters.is_finite() { return; }
        let (edge_a, ta) = flatten_path_chain(paths_a, &self.uv);
        let (edge_b, tb) = flatten_path_chain(paths_b, &self.uv);
        if edge_a.len() < 2 || edge_b.len() < 2 { return; }

        let pairs = pair_by_arc_between_anchors(&edge_a, &ta, &edge_b, &tb, notch_markers);
        self.seams.push(SeamSpec {
            pairs,
            edge_a,
            edge_b,
            edge_a_paths: paths_a.to_vec(),
            edge_b_paths: paths_b.to_vec(),
            ease_meters,
        });
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

        // Aplica a folga/franzido como diferença-alvo de comprimento A − B.
        // As restrições de borda são escaladas proporcionalmente; as costuras continuam
        // fazendo a correspondência por comprimento de arco entre vértices de ambas as malhas.
        // Cada costura contribui com um fator para cada segmento de contorno. Fatores que
        // incidem na mesma aresta são combinados geometricamente e aplicados uma única vez,
        // sem depender de qual costura aparece primeiro na lista.
        let mut ease_votes: HashMap<(u32, u32), Vec<f32>> = HashMap::new();
        for seam in &self.seams {
            for (key, factor) in seam_ease_adjustment_factors(seam, &self.uv) {
                ease_votes.entry(key).or_default().push(factor);
            }
        }
        for (key, votes) in ease_votes {
            let Some(&index) = edge_constraint_indices.get(&key) else { continue; };
            if votes.is_empty() { continue; }
            let mean_log = votes.iter().map(|factor| factor.ln()).sum::<f32>() / votes.len() as f32;
            let factor = mean_log.exp();
            distance[index].rest = (distance[index].rest * factor).max(1.0e-6);
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
fn flatten_path_chain(paths: &[Vec<u32>], uv: &[Vec2]) -> (Vec<u32>, Vec<f32>) {
    let total = paths.iter().map(|path| chain_length(path, uv)).sum::<f32>();
    if total <= 1.0e-8 { return (Vec::new(), Vec::new()); }
    let mut edge = Vec::new();
    let mut progress = Vec::new();
    let mut prefix = 0.0_f32;
    for path in paths {
        let length = chain_length(path, uv);
        if path.len() < 2 || length <= 1.0e-8 { continue; }
        let local = edge_arc_params(path, uv);
        for (&index, &t) in path.iter().zip(local.iter()) {
            let global_t = ((prefix + t * length) / total).clamp(0.0, 1.0);
            if edge.last().copied() == Some(index) {
                if let Some(last) = progress.last_mut() { *last = global_t; }
            } else {
                edge.push(index);
                progress.push(global_t);
            }
        }
        prefix += length;
    }
    (edge, progress)
}

fn pair_by_arc_between_anchors(
    a: &[u32],
    ta: &[f32],
    b: &[u32],
    tb: &[f32],
    notch_markers: &[(f32, f32)],
) -> Vec<(u32, u32)> {
    if a.len() < 2 || b.len() < 2 || ta.len() != a.len() || tb.len() != b.len() {
        return Vec::new();
    }
    let mut anchors = vec![(0.0_f32, 0.0_f32)];
    anchors.extend(notch_markers.iter().copied()
        .filter(|(pa, pb)| pa.is_finite() && pb.is_finite()
            && *pa > 0.0 && *pa < 1.0 && *pb > 0.0 && *pb < 1.0));
    anchors.push((1.0, 1.0));
    anchors.sort_by(|left, right| left.0.partial_cmp(&right.0)
        .unwrap_or(std::cmp::Ordering::Equal));
    anchors.dedup_by(|left, right| (left.0-right.0).abs() <= 1.0e-6
        && (left.1-right.1).abs() <= 1.0e-6);

    let mut pairs = Vec::new();
    for &(pa, pb) in &anchors {
        pairs.push((a[nearest_arc_index(pa, ta)], b[nearest_arc_index(pb, tb)]));
    }
    for window in anchors.windows(2) {
        let (a0, b0) = window[0];
        let (a1, b1) = window[1];
        if a1 - a0 <= 1.0e-7 || b1 - b0 <= 1.0e-7 { continue; }
        for (i, &progress_a) in ta.iter().enumerate() {
            if progress_a < a0 - 1.0e-6 || progress_a > a1 + 1.0e-6 { continue; }
            let t = ((progress_a - a0) / (a1 - a0)).clamp(0.0, 1.0);
            let progress_b = b0 + (b1 - b0) * t;
            pairs.push((a[i], b[nearest_arc_index(progress_b, tb)]));
        }
        for (i, &progress_b) in tb.iter().enumerate() {
            if progress_b < b0 - 1.0e-6 || progress_b > b1 + 1.0e-6 { continue; }
            let t = ((progress_b - b0) / (b1 - b0)).clamp(0.0, 1.0);
            let progress_a = a0 + (a1 - a0) * t;
            pairs.push((a[nearest_arc_index(progress_a, ta)], b[i]));
        }
    }
    pairs.sort_unstable();
    pairs.dedup();
    pairs
}

fn pair_by_arc_progress(a: &[u32], ta: &[f32], b: &[u32], tb: &[f32]) -> Vec<(u32, u32)> {
    if a.len() < 2 || b.len() < 2 || ta.len() != a.len() || tb.len() != b.len() {
        return Vec::new();
    }
    let mut pairs = Vec::with_capacity(a.len() + b.len());
    for (index, &t) in ta.iter().enumerate() {
        pairs.push((a[index], b[nearest_arc_index(t, tb)]));
    }
    for (index, &t) in tb.iter().enumerate() {
        pairs.push((a[nearest_arc_index(t, ta)], b[index]));
    }
    pairs.sort_unstable();
    pairs.dedup();
    pairs
}

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
    let points = edge.iter().map(|&index| {
        let point = uv[index as usize];
        Point2::new(point.x as f64, point.y as f64)
    }).collect::<Vec<_>>();
    polyline_length_meters(&points) as f32
}

fn seam_paths_length(paths: &[Vec<u32>], fallback: &[u32], uv: &[Vec2]) -> f32 {
    if paths.is_empty() { chain_length(fallback, uv) }
    else { paths.iter().map(|path| chain_length(path, uv)).sum() }
}

fn seam_ease_adjustment_factors(seam: &SeamSpec, uv: &[Vec2]) -> Vec<((u32, u32), f32)> {
    let len_a = seam_paths_length(&seam.edge_a_paths, &seam.edge_a, uv);
    let len_b = seam_paths_length(&seam.edge_b_paths, &seam.edge_b, uv);
    if len_a <= 1.0e-8 || len_b <= 1.0e-8 { return Vec::new(); }
    let ease = seam.ease_meters;
    let (target_a, target_b) = if len_a - len_b > ease {
        ((len_b + ease).max(1.0e-6), len_b)
    } else {
        (len_a, (len_a - ease).max(1.0e-6))
    };
    let factors = [
        (if seam.edge_a_paths.is_empty() { vec![seam.edge_a.clone()] } else { seam.edge_a_paths.clone() },
            (target_a / len_a).clamp(0.05, 20.0)),
        (if seam.edge_b_paths.is_empty() { vec![seam.edge_b.clone()] } else { seam.edge_b_paths.clone() },
            (target_b / len_b).clamp(0.05, 20.0)),
    ];
    let mut out = Vec::new();
    for (paths, factor) in factors {
        if (factor - 1.0).abs() < 1.0e-5 { continue; }
        for edge in paths {
            for pair in edge.windows(2) {
                let key = if pair[0] < pair[1] { (pair[0], pair[1]) } else { (pair[1], pair[0]) };
                out.push((key, factor));
            }
        }
    }
    out
}

/// Ajusta os comprimentos de repouso da borda excedente para obter a diferença assinada desejada.
fn adjust_seam_ease(
    distance: &mut [DistanceConstraint],
    edge_constraint_indices: &BTreeMap<(u32, u32), usize>,
    uv: &[Vec2],
    seam: &SeamSpec,
    adjusted: &mut HashSet<(u32, u32)>,
) {
    for (key, factor) in seam_ease_adjustment_factors(seam, uv) {
        if !adjusted.insert(key) { continue; }
        if let Some(&index) = edge_constraint_indices.get(&key) {
            distance[index].rest = (distance[index].rest * factor).max(1.0e-6);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

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

        let zero_ease = SeamSpec { ease_meters: 0.0, ..seam.clone() };
        adjust_seam_ease(&mut distance, &indices, &uv, &zero_ease, &mut HashSet::new());
        assert!((distance[0].rest - 0.5).abs() < 1e-6);
        assert!((distance[1].rest - 0.5).abs() < 1e-6);

        // Folga negativa também reduz A quando A é a borda excedente.
        distance[0].rest = 1.0;
        distance[1].rest = 0.5;
        let negative_ease = SeamSpec { ease_meters: -0.25, ..seam.clone() };
        adjust_seam_ease(&mut distance, &indices, &uv, &negative_ease, &mut HashSet::new());
        assert!((distance[0].rest - 0.25).abs() < 1e-6);
        assert!((distance[1].rest - 0.5).abs() < 1e-6);

        // Quando B excede a relação desejada, B é reduzida mesmo com folga positiva.
        distance[0].rest = 0.5;
        distance[1].rest = 1.0;
        let positive_ease = SeamSpec { ease_meters: 0.25, ..seam };
        adjust_seam_ease(&mut distance, &indices, &uv, &positive_ease, &mut HashSet::new());
        assert!((distance[0].rest - 0.5).abs() < 1e-6);
        assert!((distance[1].rest - 0.25).abs() < 1e-6);
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
