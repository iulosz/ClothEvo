use std::f32::consts::PI;
use std::ops::Range;
use std::time::Instant;

use glam::Vec3;
use rayon::prelude::*;

use crate::collider::Collider;
use crate::params::{ConstraintKind, SimParams, KIND_COUNT};

/// Grupos de cor menores que isso são resolvidos em série (overhead do rayon não compensa).
const PAR_MIN: usize = 256;

/// Restrição de distância entre duas partículas (stretch, shear ou costura).
#[derive(Clone, Copy, Debug)]
pub struct DistanceConstraint {
    pub a: u32,
    pub b: u32,
    /// Comprimento de repouso, vindo do molde 2D.
    pub rest: f32,
    /// Fator geométrico: a compliance XPBD é `scale / rigidez(tipo)`.
    pub scale: f32,
    pub kind: ConstraintKind,
}

/// Restrição de flexão (ângulo diedral) entre dois triângulos que compartilham uma aresta.
/// `p[0]`/`p[1]` são a aresta compartilhada; `p[2]`/`p[3]` são os vértices opostos.
#[derive(Clone, Copy, Debug)]
pub struct BendConstraint {
    pub p: [u32; 4],
    pub rest_angle: f32,
    /// A compliance XPBD é `scale / B`, com B a rigidez de flexão em N·m.
    pub scale: f32,
}

/// Métricas de tempo (diagnóstico para a UI).
#[derive(Clone, Copy, Debug, Default)]
pub struct Stats {
    /// Média móvel do tempo gasto por `step`, em ms.
    pub step_ms: f32,
    pub steps: u64,
}

/// Ponteiro compartilhável entre threads. Só é seguro porque a coloração garante que
/// restrições do mesmo grupo nunca tocam a mesma partícula.
struct SharedPtr<T>(*mut T);
impl<T> Clone for SharedPtr<T> {
    fn clone(&self) -> Self {
        *self
    }
}
impl<T> Copy for SharedPtr<T> {}
unsafe impl<T> Send for SharedPtr<T> {}
unsafe impl<T> Sync for SharedPtr<T> {}
impl<T> SharedPtr<T> {
    #[inline(always)]
    fn get(self) -> *mut T {
        self.0
    }
}

pub struct Simulation {
    pub pos: Vec<Vec3>,
    pub prev: Vec<Vec3>,
    pub vel: Vec<Vec3>,
    pub triangles: Vec<[u32; 3]>,
    pub colliders: Vec<Collider>,
    pub params: SimParams,
    pub stats: Stats,

    rest_pos: Vec<Vec3>,
    area: Vec<f32>,
    pinned: Vec<bool>,
    inv_mass: Vec<f32>,
    last_density: f32,

    distance: Vec<DistanceConstraint>,
    dist_lambda: Vec<f32>,
    dist_colors: Vec<Range<usize>>,

    bending: Vec<BendConstraint>,
    bend_lambda: Vec<f32>,
    bend_colors: Vec<Range<usize>>,
}

impl Simulation {
    #[allow(clippy::too_many_arguments)]
    pub(crate) fn from_parts(
        pos: Vec<Vec3>,
        triangles: Vec<[u32; 3]>,
        area: Vec<f32>,
        pinned: Vec<bool>,
        distance: Vec<DistanceConstraint>,
        dist_colors: Vec<Range<usize>>,
        bending: Vec<BendConstraint>,
        bend_colors: Vec<Range<usize>>,
        colliders: Vec<Collider>,
        params: SimParams,
    ) -> Self {
        let n = pos.len();
        let mut sim = Self {
            prev: pos.clone(),
            rest_pos: pos.clone(),
            vel: vec![Vec3::ZERO; n],
            pos,
            triangles,
            colliders,
            last_density: f32::NAN,
            params,
            stats: Stats::default(),
            area,
            pinned,
            inv_mass: vec![0.0; n],
            dist_lambda: vec![0.0; distance.len()],
            distance,
            dist_colors,
            bend_lambda: vec![0.0; bending.len()],
            bending,
            bend_colors,
        };
        sim.refresh_masses();
        sim
    }

    pub fn particle_count(&self) -> usize {
        self.pos.len()
    }

    pub fn distance_constraint_count(&self) -> usize {
        self.distance.len()
    }

    pub fn bend_constraint_count(&self) -> usize {
        self.bending.len()
    }

    pub fn is_pinned(&self, i: usize) -> bool {
        self.pinned[i]
    }

    /// Volta ao estado inicial (posições de colocação dos painéis, velocidades zeradas).
    pub fn reset(&mut self) {
        self.pos.copy_from_slice(&self.rest_pos);
        self.prev.copy_from_slice(&self.rest_pos);
        self.vel.fill(Vec3::ZERO);
        self.stats = Stats::default();
    }

    fn refresh_masses(&mut self) {
        let density = self.params.density.max(1e-4);
        for i in 0..self.pos.len() {
            self.inv_mass[i] = if self.pinned[i] {
                0.0
            } else {
                1.0 / (density * self.area[i]).max(1e-9)
            };
        }
        self.last_density = self.params.density;
    }

    /// Avança a simulação por `dt` segundos.
    pub fn step(&mut self, dt: f32) {
        let start = Instant::now();
        if self.params.density != self.last_density {
            self.refresh_masses();
        }

        let substeps = self.params.substeps.max(1);
        let iterations = self.params.iterations.max(1);
        let h = dt / substeps as f32;
        let inv_h2 = 1.0 / (h * h);
        let stiff = self.params.stiffness_table();
        let bend_k = self.params.bend.max(1e-9);
        let seam_step = self.params.seam_pull_speed.max(0.0) * h;
        let damp = (-self.params.damping.max(0.0) * h).exp();
        let g = self.params.gravity;
        let parallel = self.params.parallel;

        for _ in 0..substeps {
            // 1. Predição de posições.
            for i in 0..self.pos.len() {
                self.prev[i] = self.pos[i];
                if self.inv_mass[i] > 0.0 {
                    self.vel[i] = (self.vel[i] + g * h) * damp;
                    self.pos[i] += self.vel[i] * h;
                }
            }

            // 2. Restrições (XPBD, λ zerado a cada substep).
            self.dist_lambda.fill(0.0);
            self.bend_lambda.fill(0.0);
            for _ in 0..iterations {
                for r in &self.dist_colors {
                    solve_distance(
                        &self.distance[r.clone()],
                        &mut self.dist_lambda[r.clone()],
                        &mut self.pos,
                        &self.inv_mass,
                        inv_h2,
                        &stiff,
                        seam_step,
                        parallel,
                    );
                }
                for r in &self.bend_colors {
                    solve_bending(
                        &self.bending[r.clone()],
                        &mut self.bend_lambda[r.clone()],
                        &mut self.pos,
                        &self.inv_mass,
                        inv_h2,
                        bend_k,
                        parallel,
                    );
                }
            }

            // 3. Colisão com atrito.
            self.collide();

            // 4. Velocidades a partir das posições.
            for i in 0..self.pos.len() {
                self.vel[i] = if self.inv_mass[i] > 0.0 {
                    (self.pos[i] - self.prev[i]) / h
                } else {
                    Vec3::ZERO
                };
            }
        }

        let ms = start.elapsed().as_secs_f32() * 1000.0;
        self.stats.step_ms = if self.stats.steps == 0 {
            ms
        } else {
            0.9 * self.stats.step_ms + 0.1 * ms
        };
        self.stats.steps += 1;
    }

    fn collide(&mut self) {
        if self.colliders.is_empty() {
            return;
        }
        let thickness = self.params.thickness;
        let mu = self.params.friction.max(0.0);
        for i in 0..self.pos.len() {
            if self.inv_mass[i] <= 0.0 {
                continue;
            }
            let mut p = self.pos[i];
            let q = self.prev[i];
            for c in &self.colliders {
                let d = c.sdf(p) - thickness;
                if d < 0.0 {
                    let n = c.normal(p);
                    p -= n * d; // empurra para fora
                    // Atrito de Coulomb em posição: o deslocamento tangencial é limitado por μ·penetração.
                    let dx = p - q;
                    let t = dx - n * dx.dot(n);
                    let tl = t.length();
                    if tl > 1e-9 {
                        let max_friction = mu * (-d);
                        let s = if tl <= max_friction { 1.0 } else { max_friction / tl };
                        p -= t * s;
                    }
                }
            }
            self.pos[i] = p;
        }
    }

    /// Maior deformação relativa (|comprimento/repouso − 1|) entre as arestas de stretch/shear.
    pub fn max_strain(&self) -> f32 {
        let mut worst = 0.0f32;
        for c in &self.distance {
            if c.kind == ConstraintKind::Seam {
                continue;
            }
            let len = (self.pos[c.a as usize] - self.pos[c.b as usize]).length();
            let s = (len / c.rest - 1.0).abs();
            // NaN também entra aqui, para que uma simulação divergida apareça na UI.
            if s.is_nan() || s > worst {
                worst = s;
            }
        }
        worst
    }

    /// Aresta de stretch/shear mais deformada: (a, b, tipo, deformação). Para diagnóstico.
    pub fn worst_strain_edge(&self) -> Option<(u32, u32, ConstraintKind, f32)> {
        let mut worst: Option<(u32, u32, ConstraintKind, f32)> = None;
        for c in &self.distance {
            if c.kind == ConstraintKind::Seam {
                continue;
            }
            let len = (self.pos[c.a as usize] - self.pos[c.b as usize]).length();
            let s = (len / c.rest - 1.0).abs();
            if worst.is_none_or(|w| s > w.3) {
                worst = Some((c.a, c.b, c.kind, s));
            }
        }
        worst
    }

    /// Percentil `q` (0..1) da deformação das arestas de stretch/shear. Para diagnóstico.
    pub fn strain_percentile(&self, q: f32) -> f32 {
        let mut v: Vec<f32> = self
            .distance
            .iter()
            .filter(|c| c.kind != ConstraintKind::Seam)
            .map(|c| ((self.pos[c.a as usize] - self.pos[c.b as usize]).length() / c.rest - 1.0).abs())
            .collect();
        if v.is_empty() {
            return 0.0;
        }
        v.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
        v[((v.len() - 1) as f32 * q.clamp(0.0, 1.0)) as usize]
    }

    pub fn kinetic_energy(&self) -> f32 {
        let mut e = 0.0;
        for i in 0..self.pos.len() {
            if self.inv_mass[i] > 0.0 {
                e += 0.5 * self.vel[i].length_squared() / self.inv_mass[i];
            }
        }
        e
    }

    pub fn is_finite(&self) -> bool {
        self.pos.iter().all(|p| p.is_finite())
    }

    /// Normais por vértice (média ponderada pela área), para o render.
    pub fn compute_vertex_normals(&self, out: &mut Vec<Vec3>) {
        out.clear();
        out.resize(self.pos.len(), Vec3::ZERO);
        for t in &self.triangles {
            let a = self.pos[t[0] as usize];
            let b = self.pos[t[1] as usize];
            let c = self.pos[t[2] as usize];
            let n = (b - a).cross(c - a);
            for &v in t {
                out[v as usize] += n;
            }
        }
        for n in out.iter_mut() {
            *n = n.try_normalize().unwrap_or(Vec3::Y);
        }
    }

    /// Pares (a, b) de todas as restrições de costura; útil para testes e visualização.
    pub fn seam_pairs(&self) -> Vec<(u32, u32)> {
        self.distance
            .iter()
            .filter(|c| c.kind == ConstraintKind::Seam)
            .map(|c| (c.a, c.b))
            .collect()
    }

    /// Arestas únicas da malha (para o overlay de wireframe).
    pub fn mesh_edges(&self) -> Vec<[u32; 2]> {
        let mut edges: Vec<[u32; 2]> = Vec::with_capacity(self.triangles.len() * 3);
        for t in &self.triangles {
            for k in 0..3 {
                let (a, b) = (t[k], t[(k + 1) % 3]);
                edges.push(if a < b { [a, b] } else { [b, a] });
            }
        }
        edges.sort_unstable();
        edges.dedup();
        edges
    }
}

#[allow(clippy::too_many_arguments)]
fn solve_distance(
    cs: &[DistanceConstraint],
    lambda: &mut [f32],
    pos: &mut [Vec3],
    inv_mass: &[f32],
    inv_h2: f32,
    stiff: &[f32; KIND_COUNT],
    seam_step: f32,
    parallel: bool,
) {
    let n = pos.len();
    let ptr = SharedPtr(pos.as_mut_ptr());

    let kernel = move |c: &DistanceConstraint, lam: &mut f32| {
        let (a, b) = (c.a as usize, c.b as usize);
        debug_assert!(a < n && b < n && a != b);
        let (wa, wb) = (inv_mass[a], inv_mass[b]);
        let w = wa + wb;
        if w == 0.0 {
            return;
        }
        // SAFETY: a coloração garante que nenhuma outra restrição deste grupo usa `a` ou `b`.
        unsafe {
            let pa = *ptr.get().add(a);
            let pb = *ptr.get().add(b);
            let d = pa - pb;
            let len = d.length();
            if len < 1e-9 {
                return;
            }
            let mut cval = len - c.rest;
            if c.kind == ConstraintKind::Seam {
                // A costura "puxa" devagar, para não lançar o tecido.
                cval = cval.min(seam_step);
            }
            let alpha = c.scale / stiff[c.kind as usize] * inv_h2;
            let dl = (-cval - alpha * *lam) / (w + alpha);
            *lam += dl;
            let corr = d * (dl / len);
            *ptr.get().add(a) = pa + corr * wa;
            *ptr.get().add(b) = pb - corr * wb;
        }
    };

    if parallel && cs.len() >= PAR_MIN {
        cs.par_iter()
            .zip(lambda.par_iter_mut())
            .with_min_len(PAR_MIN)
            .for_each(|(c, l)| kernel(c, l));
    } else {
        for (c, l) in cs.iter().zip(lambda.iter_mut()) {
            kernel(c, l);
        }
    }
}

#[allow(clippy::too_many_arguments)]
fn solve_bending(
    cs: &[BendConstraint],
    lambda: &mut [f32],
    pos: &mut [Vec3],
    inv_mass: &[f32],
    inv_h2: f32,
    bend_k: f32,
    parallel: bool,
) {
    let ptr = SharedPtr(pos.as_mut_ptr());

    let kernel = move |c: &BendConstraint, lam: &mut f32| {
        let idx = [c.p[0] as usize, c.p[1] as usize, c.p[2] as usize, c.p[3] as usize];
        let w = [inv_mass[idx[0]], inv_mass[idx[1]], inv_mass[idx[2]], inv_mass[idx[3]]];
        if w.iter().all(|&x| x == 0.0) {
            return;
        }
        // SAFETY: a coloração garante acesso exclusivo às 4 partículas dentro do grupo.
        unsafe {
            let p = [
                *ptr.get().add(idx[0]),
                *ptr.get().add(idx[1]),
                *ptr.get().add(idx[2]),
                *ptr.get().add(idx[3]),
            ];
            let Some((theta, g)) = bend_angle_and_gradients(p[0], p[1], p[2], p[3]) else {
                return;
            };
            let cval = wrap_angle(theta - c.rest_angle);
            let denom: f32 = (0..4).map(|k| w[k] * g[k].length_squared()).sum();
            let alpha = c.scale / bend_k * inv_h2;
            if denom + alpha <= 0.0 {
                return;
            }
            let dl = (-cval - alpha * *lam) / (denom + alpha);
            *lam += dl;
            for k in 0..4 {
                *ptr.get().add(idx[k]) = p[k] + g[k] * (w[k] * dl);
            }
        }
    };

    if parallel && cs.len() >= PAR_MIN {
        cs.par_iter()
            .zip(lambda.par_iter_mut())
            .with_min_len(PAR_MIN)
            .for_each(|(c, l)| kernel(c, l));
    } else {
        for (c, l) in cs.iter().zip(lambda.iter_mut()) {
            kernel(c, l);
        }
    }
}

fn wrap_angle(a: f32) -> f32 {
    if a > PI {
        a - 2.0 * PI
    } else if a < -PI {
        a + 2.0 * PI
    } else {
        a
    }
}

/// Ângulo diedral assinado entre os triângulos (p1,p2,p3) e (p1,p2,p4) e seus gradientes.
///
/// `θ = 0` quando os dois triângulos estão no mesmo plano (p3 e p4 em lados opostos da aresta).
/// Usa `atan2`, então é suave em torno de `θ = 0` (o estado de repouso de um painel plano),
/// ao contrário da formulação com `acos` do PBD original.
///
/// Gradientes (derivados por invariância a translação e rotação):
/// `g3 = -n_A/h3`, `g4 = -n_B/h4`, `g2 = -(a3·g3 + a4·g4)`, `g1 = -(g2+g3+g4)`,
/// com `h` a altura do vértice sobre a aresta e `a` a projeção ao longo dela.
pub(crate) fn bend_angle_and_gradients(
    p1: Vec3,
    p2: Vec3,
    p3: Vec3,
    p4: Vec3,
) -> Option<(f32, [Vec3; 4])> {
    let e = p2 - p1;
    let el = e.length();
    if el < 1e-9 {
        return None;
    }
    let ehat = e / el;
    let na_v = e.cross(p3 - p1);
    let nb_v = (p4 - p1).cross(e);
    let la = na_v.length();
    let lb = nb_v.length();
    if la < 1e-12 || lb < 1e-12 {
        return None;
    }
    let na = na_v / la;
    let nb = nb_v / lb;
    let theta = na.cross(nb).dot(ehat).atan2(na.dot(nb));

    let h3 = la / el;
    let h4 = lb / el;
    let g3 = -na / h3;
    let g4 = -nb / h4;
    let a3 = (p3 - p1).dot(ehat) / el;
    let a4 = (p4 - p1).dot(ehat) / el;
    let g2 = -(g3 * a3 + g4 * a4);
    let g1 = -(g2 + g3 + g4);
    Some((theta, [g1, g2, g3, g4]))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::builder::ClothBuilder;
    use crate::coloring::greedy_coloring;
    use crate::mesh::{GridPanel, PanelMesh, Placement};
    use crate::scenes::{build_scene, SceneKind};
    use glam::Vec2;

    #[test]
    fn bend_gradients_match_finite_differences() {
        let p = [
            Vec3::new(0.0, 0.0, 0.0),
            Vec3::new(1.0, 0.1, -0.05),
            Vec3::new(0.4, 0.9, 0.5),
            Vec3::new(0.6, -0.8, 0.3),
        ];
        let (theta, g) = bend_angle_and_gradients(p[0], p[1], p[2], p[3]).unwrap();
        assert!(theta.is_finite());
        let eps = 1e-3;
        for k in 0..4 {
            for axis in 0..3 {
                let mut hi = p;
                let mut lo = p;
                hi[k][axis] += eps;
                lo[k][axis] -= eps;
                let th = bend_angle_and_gradients(hi[0], hi[1], hi[2], hi[3]).unwrap().0;
                let tl = bend_angle_and_gradients(lo[0], lo[1], lo[2], lo[3]).unwrap().0;
                let numeric = (th - tl) / (2.0 * eps);
                assert!(
                    (numeric - g[k][axis]).abs() < 2e-2,
                    "k={k} axis={axis}: numérico {numeric} vs analítico {}",
                    g[k][axis]
                );
            }
        }
        // invariância a translação
        let sum = g[0] + g[1] + g[2] + g[3];
        assert!(sum.length() < 1e-4);
    }

    #[test]
    fn flat_hinge_has_zero_angle() {
        let (theta, _) = bend_angle_and_gradients(
            Vec3::new(0.0, 0.0, 0.0),
            Vec3::new(1.0, 0.0, 0.0),
            Vec3::new(0.5, 1.0, 0.0),
            Vec3::new(0.5, -1.0, 0.0),
        )
        .unwrap();
        assert!(theta.abs() < 1e-6);
    }

    #[test]
    fn constraint_groups_are_conflict_free() {
        let sim = build_scene(SceneKind::SewnTube, 30, SimParams::default());
        let mut start = 0;
        for r in &sim.dist_colors {
            assert_eq!(r.start, start);
            start = r.end;
            let mut seen = std::collections::HashSet::new();
            for c in &sim.distance[r.clone()] {
                assert!(seen.insert(c.a) && seen.insert(c.b), "partícula repetida na cor");
            }
        }
        for r in &sim.bend_colors {
            let mut seen = std::collections::HashSet::new();
            for c in &sim.bending[r.clone()] {
                for &q in &c.p {
                    assert!(seen.insert(q), "partícula repetida na cor (flexão)");
                }
            }
        }
        // sanidade do coloring isolado
        let colors = greedy_coloring(&[[0u32, 1], [1, 2]], 3);
        assert_ne!(colors[0], colors[1]);
    }

    #[test]
    fn folded_hinge_flattens_under_bending() {
        // Dois triângulos compartilhando a aresta 0-1; o triângulo (0,1,2) é dobrado 60° em torno da aresta.
        let mesh = PanelMesh {
            uv: vec![
                Vec2::new(0.0, 0.0),
                Vec2::new(1.0, 0.0),
                Vec2::new(0.5, 1.0),
                Vec2::new(0.5, -1.0),
            ],
            triangles: vec![[0, 1, 2], [0, 3, 1]],
        };
        let mut b = ClothBuilder::new();
        let h = b.add_panel(&mesh, Placement::at(Vec3::ZERO));
        b.pin(h.global(0));
        b.pin(h.global(1));
        // Rigidez alta e amortecimento forte: o teste verifica o mecanismo (a flexão
        // devolve o painel ao plano), não o tempo de relaxamento de um tecido real.
        let params = SimParams {
            gravity: Vec3::ZERO,
            bend: 0.5,
            damping: 5.0,
            ..SimParams::default()
        };
        let mut sim = b.build(params);
        // rotação rígida de p3 em torno da aresta: preserva todos os comprimentos
        sim.pos[2] = Vec3::new(0.5, 0.5, 0.866_025_4);
        sim.prev.copy_from_slice(&sim.pos);
        let angle = |s: &Simulation| {
            bend_angle_and_gradients(s.pos[0], s.pos[1], s.pos[2], s.pos[3]).unwrap().0
        };
        let before = angle(&sim);
        assert!(before.abs() > 0.9, "ângulo inicial deveria ser ~60°: {before}");
        for _ in 0..600 {
            sim.step(1.0 / 60.0);
        }
        assert!(sim.is_finite());
        // O painel pode terminar plano mas girado em torno da aresta presa, por isso
        // medimos o ângulo diedral (e não a coordenada z).
        let after = angle(&sim);
        assert!(after.abs() < 0.05, "não achatou: diedro = {after} rad");
    }

    #[test]
    fn hanging_cloth_is_stable_and_barely_stretches() {
        let mut sim = build_scene(SceneKind::HangingCloth, 24, SimParams::default());
        for _ in 0..600 {
            sim.step(1.0 / 60.0);
        }
        assert!(sim.is_finite(), "simulação divergiu");
        // O tecido todo fica quase inextensível; só há concentração perto dos pinos
        // (limitação conhecida do XPBD com poucas iterações, ver README).
        let median = sim.strain_percentile(0.5);
        let p99 = sim.strain_percentile(0.99);
        let max = sim.max_strain();
        assert!(median < 0.01, "deformação mediana alta: {median}");
        assert!(p99 < 0.05, "deformação p99 alta: {p99}");
        assert!(max < 0.25, "deformação máxima alta: {max}");
        let lowest = sim.pos.iter().map(|p| p.y).fold(f32::MAX, f32::min);
        assert!(lowest > -0.01, "atravessou o chão: {lowest}");
    }

    #[test]
    fn cloth_does_not_penetrate_sphere() {
        let mut sim = build_scene(SceneKind::SphereDrop, 24, SimParams::default());
        for _ in 0..600 {
            sim.step(1.0 / 60.0);
        }
        assert!(sim.is_finite());
        for c in &sim.colliders {
            for p in &sim.pos {
                assert!(c.sdf(*p) > -0.02, "partícula dentro do colisor: {:?}", c);
            }
        }
        // o tecido deve ter descido (e parado em cima da esfera/chão)
        let avg_y = sim.pos.iter().map(|p| p.y).sum::<f32>() / sim.pos.len() as f32;
        assert!(avg_y < 1.0, "o tecido não caiu: {avg_y}");
    }

    #[test]
    fn seams_close_the_gap() {
        let mut sim = build_scene(SceneKind::SewnTube, 24, SimParams::default());
        let before: f32 = sim
            .seam_pairs()
            .iter()
            .map(|&(a, b)| (sim.pos[a as usize] - sim.pos[b as usize]).length())
            .sum::<f32>();
        for _ in 0..600 {
            sim.step(1.0 / 60.0);
        }
        assert!(sim.is_finite());
        let pairs = sim.seam_pairs();
        assert!(!pairs.is_empty());
        let after: f32 = pairs
            .iter()
            .map(|&(a, b)| (sim.pos[a as usize] - sim.pos[b as usize]).length())
            .sum::<f32>();
        let mean_after = after / pairs.len() as f32;
        assert!(after < before * 0.2, "costuras não fecharam: {before} -> {after}");
        assert!(mean_after < 0.03, "folga média nas costuras: {mean_after} m");
    }

    #[test]
    fn parallel_and_serial_give_identical_results() {
        let run = |parallel: bool| {
            let params = SimParams { parallel, ..SimParams::default() };
            let mut sim = build_scene(SceneKind::SphereDrop, 64, params);
            for _ in 0..20 {
                sim.step(1.0 / 60.0);
            }
            sim.pos.clone()
        };
        let a = run(false);
        let b = run(true);
        let max_diff = a.iter().zip(&b).map(|(x, y)| (*x - *y).length()).fold(0.0, f32::max);
        assert!(max_diff < 1e-6, "diferença entre serial e paralelo: {max_diff}");
    }

    #[test]
    fn grid_panel_has_expected_counts() {
        let g = GridPanel::new(1.0, 1.0, 4, 4);
        assert_eq!(g.mesh.uv.len(), 25);
        assert_eq!(g.mesh.triangles.len(), 32);
    }
}
