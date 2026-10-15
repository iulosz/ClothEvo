use std::collections::{HashMap, HashSet};
use std::f32::consts::PI;
use std::ops::Range;
use std::time::Instant;

use glam::Vec3;
use rayon::prelude::*;

use crate::collider::Collider;
use crate::params::{ConstraintKind, SimParams, KIND_COUNT};
use crate::self_collision::{
    begin_contact_iteration, body_contact_distance, build_candidates as build_self_collision_candidates,
    combine_friction, distribute_friction_correction, effective_thickness, finish_contact_iteration,
    pair_contact_distance, residual_penetration, solve_contacts as solve_self_collision_contacts,
    update_contact_friction, CollisionTopology, ContactFrictionMetrics, ContactFrictionState,
    ContactKey,
};

/// Grupos de cor menores que isso são resolvidos em série (overhead do rayon não compensa).
const PAR_MIN: usize = 256;
/// Limita o custo da auto-colisão quando muitos pontos colapsam na mesma célula.
const MAX_SELF_COLLISION_CANDIDATES_PER_PARTICLE: usize = 96;
/// Orçamento agregado para que a lista de pares nunca cresça com N² nem com 96·N sem limite.
const MAX_SELF_COLLISION_CANDIDATE_PAIRS_TOTAL: usize = 2_000_000;

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

/// Métricas de desempenho e contato do último passo (diagnóstico para a UI e benchmarks).
#[derive(Clone, Copy, Debug, Default)]
pub struct Stats {
    /// Média móvel do tempo gasto por `step`, em ms.
    pub step_ms: f32,
    /// Média móvel do tempo gasto resolvendo restrições XPBD, em ms por passo.
    pub constraint_ms: f32,
    /// Média móvel do tempo gasto consultando e projetando colisores externos, em ms por passo.
    pub body_collision_ms: f32,
    /// Projeções discretas de contato com colisores externos durante o último passo.
    pub body_contact_projections: usize,
    /// Trajetórias longas examinadas pelo detector contínuo aproximado no último passo.
    pub body_sweep_tests: usize,
    /// Contatos detectados ao longo da trajetória, mesmo quando a posição final sairia do corpo.
    pub body_swept_contacts: usize,
    /// Maior penetração relativa à espessura configurada, medida antes da correção, em metros.
    pub body_max_penetration_m: f32,
    /// Contatos em que a resposta tangencial foi calculada no último passo.
    pub contact_friction_contacts: usize,
    /// Contatos que permaneceram no regime estático de Coulomb.
    pub contact_friction_static_contacts: usize,
    /// Contatos que excederam o limite estático e deslizaram.
    pub contact_friction_sliding_contacts: usize,
    /// Maior velocidade tangencial relativa observada nos contatos, em m/s.
    pub contact_max_tangential_speed_m_s: f32,
    /// Maior multiplicador normal posicional acumulado, não uma força calibrada.
    pub contact_max_normal_correction_m: f32,
    /// Candidatos vértice-vértice encontrados pela hash grid no último passo.
    pub self_collision_vv_candidates: usize,
    /// Contatos vértice-vértice projetados no último passo.
    pub self_collision_vv_contacts: usize,
    /// Candidatos vértice-triângulo gerados antes do filtro geométrico.
    pub self_collision_vt_candidates: usize,
    /// Candidatos vértice-triângulo que passaram pela fase geométrica.
    pub self_collision_vt_verified: usize,
    /// Contatos vértice-triângulo projetados nas iterações.
    pub self_collision_vt_contacts: usize,
    /// Trajetórias vértice-triângulo examinadas pelo CCD aproximado.
    pub self_collision_vt_sweep_tests: usize,
    /// Cruzamentos vértice-triângulo interceptados entre os extremos do substep.
    pub self_collision_vt_swept_contacts: usize,
    /// Candidatos aresta-aresta gerados antes do filtro geométrico.
    pub self_collision_ee_candidates: usize,
    /// Candidatos aresta-aresta que passaram pela fase geométrica.
    pub self_collision_ee_verified: usize,
    /// Contatos aresta-aresta projetados nas iterações.
    pub self_collision_ee_contacts: usize,
    /// Trajetórias aresta-aresta examinadas pelo CCD aproximado.
    pub self_collision_ee_sweep_tests: usize,
    /// Cruzamentos aresta-aresta interceptados entre os extremos do substep.
    pub self_collision_ee_swept_contacts: usize,
    /// Pares potencialmente válidos omitidos pelos limites do broadphase/geradores.
    pub self_collision_dropped_candidates: usize,
    /// Primitivas ignoradas porque sua AABB excedeu o orçamento de células.
    pub self_collision_overflowed_primitives: usize,
    /// Maior penetração interna antes das correções, em metros.
    pub self_collision_penetration_before_m: f32,
    /// Maior penetração de superfície que restou após a última iteração, em metros.
    pub self_collision_penetration_after_m: f32,
    /// Média móvel do custo de atualizar as grades e gerar candidatos, em ms.
    pub self_collision_broadphase_ms: f32,
    /// Média móvel do custo de projetar os contatos, em ms.
    pub self_collision_solve_ms: f32,
    /// Média móvel total da auto-colisão, em ms.
    pub self_collision_ms: f32,
    pub steps: u64,
}

fn record_contact_friction(stats: &mut Stats, metrics: ContactFrictionMetrics) {
    stats.contact_friction_contacts += metrics.contacts;
    stats.contact_friction_static_contacts += metrics.static_contacts;
    stats.contact_friction_sliding_contacts += metrics.sliding_contacts;
    stats.contact_max_tangential_speed_m_s = stats
        .contact_max_tangential_speed_m_s
        .max(metrics.max_tangential_speed_m_s);
    stats.contact_max_normal_correction_m = stats
        .contact_max_normal_correction_m
        .max(metrics.max_normal_correction_m);
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
    /// Multiplicador de densidade por partícula, definido pelo material do painel.
    mass_scale: Vec<f32>,
    /// Multiplicador da espessura de colisão por partícula.
    thickness_scale: Vec<f32>,
    /// Multiplicadores de atrito estático e dinâmico por partícula.
    friction_scale: Vec<f32>,
    friction_dynamic_scale: Vec<f32>,
    /// Duração do último substep, usada para estimar carga das costuras pela lambda XPBD.
    last_substep_dt: f32,
    /// Relações estáticas da superfície usadas para filtrar contatos locais.
    collision_topology: CollisionTopology,
    /// Contatos de auto-colisão acumulados no passo de simulação atual.
    last_self_collision_contacts: usize,
    /// Maior penetração interna detectada no passo atual, em metros.
    last_self_collision_penetration: f32,
    /// Recuperações/clamps de estabilidade ocorridos no passo atual.
    last_stability_corrections: usize,
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
        mass_scale: Vec<f32>,
        thickness_scale: Vec<f32>,
        friction_scale: Vec<f32>,
        friction_dynamic_scale: Vec<f32>,
        distance: Vec<DistanceConstraint>,
        dist_colors: Vec<Range<usize>>,
        bending: Vec<BendConstraint>,
        bend_colors: Vec<Range<usize>>,
        colliders: Vec<Collider>,
        params: SimParams,
    ) -> Self {
        let n = pos.len();
        let seam_pairs = distance
            .iter()
            .filter(|constraint| constraint.kind == ConstraintKind::Seam)
            .map(|constraint| (constraint.a, constraint.b));
        let collision_topology = CollisionTopology::new(n, &triangles, seam_pairs);
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
            mass_scale,
            thickness_scale,
            friction_scale,
            friction_dynamic_scale,
            last_substep_dt: 0.0,
            collision_topology,
            last_self_collision_contacts: 0,
            last_self_collision_penetration: 0.0,
            last_stability_corrections: 0,
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

    /// Acesso somente de crate para validar os alvos das dobradiças físicas sem expor
    /// os buffers internos do solver à API pública.
    #[cfg(test)]
    pub(crate) fn bending_constraints(&self) -> &[BendConstraint] {
        &self.bending
    }

    pub fn is_pinned(&self, i: usize) -> bool {
        self.pinned[i]
    }

    /// Volta ao estado inicial (posições de colocação dos painéis, velocidades zeradas).
    pub fn reset(&mut self) {
        self.pos.copy_from_slice(&self.rest_pos);
        self.prev.copy_from_slice(&self.rest_pos);
        self.vel.fill(Vec3::ZERO);
        self.last_self_collision_contacts = 0;
        self.last_self_collision_penetration = 0.0;
        self.last_stability_corrections = 0;
        self.last_substep_dt = 0.0;
        self.stats = Stats::default();
    }

    /// Translada um intervalo de partículas, incluindo o estado de repouso usado por Reiniciar.
    ///
    /// É usado pelo modo de posicionamento pré-simulação: mover um painel na viewport deve
    /// sobreviver ao reset e preservar as velocidades em zero antes de iniciar a física.
    pub fn translate_particles(&mut self, start: usize, end: usize, delta: Vec3) -> bool {
        if start >= end || end > self.pos.len() || !delta.is_finite() {
            return false;
        }
        for i in start..end {
            self.pos[i] += delta;
            self.prev[i] += delta;
            self.rest_pos[i] += delta;
            self.vel[i] = Vec3::ZERO;
        }
        self.last_substep_dt = 0.0;
        true
    }

    /// Gira um intervalo de partículas em torno de um centro, incluindo o estado de repouso.
    /// Assim a transformação interativa continua correta depois de Reiniciar.
    pub fn rotate_particles(&mut self, start: usize, end: usize, center: Vec3, rotation: glam::Quat) -> bool {
        if start >= end || end > self.pos.len() || !center.is_finite()
            || !rotation.is_finite() || rotation.length_squared() <= 1.0e-12
        {
            return false;
        }
        let rotation = rotation.normalize();
        for i in start..end {
            self.pos[i] = center + rotation * (self.pos[i] - center);
            self.prev[i] = center + rotation * (self.prev[i] - center);
            self.rest_pos[i] = center + rotation * (self.rest_pos[i] - center);
            self.vel[i] = rotation * self.vel[i];
        }
        self.last_substep_dt = 0.0;
        true
    }

    /// Escala um painel ao longo de um eixo normalizado em torno do centro indicado.
    /// É usado apenas durante o posicionamento; a prévia é reconstruída ao soltar o gizmo
    /// para atualizar áreas, massas e comprimentos de repouso com a nova dimensão.
    pub fn scale_particles_axis(
        &mut self,
        start: usize,
        end: usize,
        center: Vec3,
        axis: Vec3,
        factor: f32,
    ) -> bool {
        if start >= end || end > self.pos.len() || !center.is_finite() || !axis.is_finite()
            || axis.length_squared() <= 1.0e-12 || !factor.is_finite()
            || !(0.01..=100.0).contains(&factor)
        {
            return false;
        }
        let axis = axis.normalize();
        let scale_point = |point: Vec3| {
            let offset = point - center;
            center + offset + axis * (offset.dot(axis) * (factor - 1.0))
        };
        for i in start..end {
            self.pos[i] = scale_point(self.pos[i]);
            self.prev[i] = scale_point(self.prev[i]);
            self.rest_pos[i] = scale_point(self.rest_pos[i]);
            let velocity = self.vel[i];
            self.vel[i] = velocity + axis * (velocity.dot(axis) * (factor - 1.0));
        }
        self.last_substep_dt = 0.0;
        true
    }

    fn refresh_masses(&mut self) {
        let global_density = self.params.density.max(1e-4);
        for i in 0..self.pos.len() {
            let density = global_density * self.mass_scale[i].max(1.0e-4);
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
        if !dt.is_finite() || dt <= 0.0 {
            return;
        }
        let start = Instant::now();
        if self.params.density != self.last_density {
            self.refresh_masses();
        }

        // Protege contra valores inválidos em arquivos/configurações futuras.
        let substeps = self.params.substeps.clamp(1, 120);
        let iterations = self.params.iterations.clamp(1, 20);
        let h = dt / substeps as f32;
        self.last_substep_dt = h;
        let inv_h2 = 1.0 / (h * h);
        let stiff = self.params.stiffness_table();
        let bend_k = self.params.bend.max(1e-9);
        let seam_step = self.params.seam_pull_speed.max(0.0) * h;
        let damp = (-self.params.damping.max(0.0) * h).exp();
        let g = self.params.gravity;
        let parallel = self.params.parallel;
        let mut constraint_elapsed_ms = 0.0_f32;
        let mut body_collision_elapsed_ms = 0.0_f32;
        let mut self_collision_broadphase_elapsed_ms = 0.0_f32;
        let mut self_collision_solve_elapsed_ms = 0.0_f32;
        self.stats.body_contact_projections = 0;
        self.stats.body_sweep_tests = 0;
        self.stats.body_swept_contacts = 0;
        self.stats.body_max_penetration_m = 0.0;
        self.stats.contact_friction_contacts = 0;
        self.stats.contact_friction_static_contacts = 0;
        self.stats.contact_friction_sliding_contacts = 0;
        self.stats.contact_max_tangential_speed_m_s = 0.0;
        self.stats.contact_max_normal_correction_m = 0.0;
        self.stats.self_collision_vv_candidates = 0;
        self.stats.self_collision_vv_contacts = 0;
        self.stats.self_collision_vt_candidates = 0;
        self.stats.self_collision_vt_verified = 0;
        self.stats.self_collision_vt_contacts = 0;
        self.stats.self_collision_vt_sweep_tests = 0;
        self.stats.self_collision_vt_swept_contacts = 0;
        self.stats.self_collision_ee_candidates = 0;
        self.stats.self_collision_ee_verified = 0;
        self.stats.self_collision_ee_contacts = 0;
        self.stats.self_collision_ee_sweep_tests = 0;
        self.stats.self_collision_ee_swept_contacts = 0;
        self.stats.self_collision_dropped_candidates = 0;
        self.stats.self_collision_overflowed_primitives = 0;
        self.stats.self_collision_penetration_before_m = 0.0;
        self.stats.self_collision_penetration_after_m = 0.0;
        self.last_self_collision_contacts = 0;
        self.last_self_collision_penetration = 0.0;
        self.last_stability_corrections = 0;
        let mut friction_states = HashMap::new();

        for _ in 0..substeps {
            friction_states.clear();
            // 1. Predição de posições.
            for i in 0..self.pos.len() {
                self.prev[i] = self.pos[i];
                if self.inv_mass[i] > 0.0 {
                    self.vel[i] = (self.vel[i] + g * h) * damp;
                    self.pos[i] += self.vel[i] * h;
                }
            }

            // 2. Restrições XPBD alternadas com contato corporal.
            self.dist_lambda.fill(0.0);
            self.bend_lambda.fill(0.0);
            for _ in 0..iterations {
                begin_contact_iteration(&mut friction_states);
                let constraint_start = Instant::now();
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
                constraint_elapsed_ms += constraint_start.elapsed().as_secs_f32() * 1000.0;

                // A lista é reconstruída em cada iteração porque as restrições anteriores
                // podem aproximar superfícies que ainda não eram candidatas.
                if self.params.self_collision {
                    let vv_start = Instant::now();
                    let (vv_candidates, vv_dropped) =
                        self.collide_self(&mut friction_states, h);
                    self_collision_solve_elapsed_ms += vv_start.elapsed().as_secs_f32() * 1000.0;
                    self.stats.self_collision_vv_candidates += vv_candidates;
                    self.stats.self_collision_dropped_candidates += vv_dropped;

                    let broadphase_start = Instant::now();
                    let candidates = build_self_collision_candidates(
                        &self.pos,
                        &self.prev,
                        &self.thickness_scale,
                        self.params.thickness,
                        &self.collision_topology,
                    );
                    self_collision_broadphase_elapsed_ms += broadphase_start.elapsed().as_secs_f32() * 1000.0;
                    self.stats.self_collision_vt_candidates += candidates.vertex_triangle_candidates;
                    self.stats.self_collision_ee_candidates += candidates.edge_edge_candidates;
                    self.stats.self_collision_dropped_candidates += candidates.dropped_candidates;
                    self.stats.self_collision_overflowed_primitives += candidates.overflowed_primitives;

                    let solve_start = Instant::now();
                    let resolved = solve_self_collision_contacts(
                        &candidates,
                        &self.collision_topology,
                        &mut self.pos,
                        &mut self.prev,
                        &self.inv_mass,
                        &self.thickness_scale,
                        &self.friction_scale,
                        &self.friction_dynamic_scale,
                        self.params.thickness,
                        self.params.friction,
                        h,
                        &mut friction_states,
                    );
                    self_collision_solve_elapsed_ms += solve_start.elapsed().as_secs_f32() * 1000.0;
                    self.stats.self_collision_vt_verified += resolved.vertex_triangle_verified;
                    self.stats.self_collision_vt_contacts += resolved.vertex_triangle_contacts;
                    self.stats.self_collision_vt_sweep_tests += resolved.vertex_triangle_sweep_tests;
                    self.stats.self_collision_vt_swept_contacts += resolved.vertex_triangle_swept_contacts;
                    self.stats.self_collision_ee_verified += resolved.edge_edge_verified;
                    self.stats.self_collision_ee_contacts += resolved.edge_edge_contacts;
                    self.stats.self_collision_ee_sweep_tests += resolved.edge_edge_sweep_tests;
                    self.stats.self_collision_ee_swept_contacts += resolved.edge_edge_swept_contacts;
                    record_contact_friction(&mut self.stats, resolved.friction);
                    self.stats.self_collision_penetration_before_m = self.stats
                        .self_collision_penetration_before_m
                        .max(resolved.max_penetration_before_m);
                    self.last_self_collision_contacts +=
                        resolved.vertex_triangle_contacts + resolved.edge_edge_contacts;
                    self.last_self_collision_penetration = self.last_self_collision_penetration
                        .max(resolved.max_penetration_before_m);
                }

                let body_collision_start = Instant::now();
                self.collide(&mut friction_states, h);
                body_collision_elapsed_ms += body_collision_start.elapsed().as_secs_f32() * 1000.0;
                finish_contact_iteration(&mut friction_states);
            }

            if self.params.self_collision {
                let broadphase_start = Instant::now();
                let residual_candidates = build_self_collision_candidates(
                    &self.pos,
                    &self.prev,
                    &self.thickness_scale,
                    self.params.thickness,
                    &self.collision_topology,
                );
                self_collision_broadphase_elapsed_ms += broadphase_start.elapsed().as_secs_f32() * 1000.0;
                self.stats.self_collision_vt_candidates += residual_candidates.vertex_triangle_candidates;
                self.stats.self_collision_ee_candidates += residual_candidates.edge_edge_candidates;
                self.stats.self_collision_dropped_candidates += residual_candidates.dropped_candidates;
                self.stats.self_collision_overflowed_primitives += residual_candidates.overflowed_primitives;
                self.stats.self_collision_penetration_after_m = residual_penetration(
                    &residual_candidates,
                    &self.collision_topology,
                    &self.pos,
                    &self.thickness_scale,
                    self.params.thickness,
                );
            }

            // 4. Reconstrói velocidades, recuperando estados não finitos e limitando picos.
            let max_speed = self.params.max_particle_speed.clamp(1.0, 100.0);
            let max_displacement = max_speed * h;
            for i in 0..self.pos.len() {
                if !self.pos[i].is_finite() || !self.prev[i].is_finite() {
                    self.pos[i] = if self.prev[i].is_finite() {
                        self.prev[i]
                    } else {
                        self.rest_pos[i]
                    };
                    self.prev[i] = self.pos[i];
                    self.vel[i] = Vec3::ZERO;
                    self.last_stability_corrections += 1;
                    continue;
                }
                if self.inv_mass[i] <= 0.0 {
                    self.vel[i] = Vec3::ZERO;
                    continue;
                }
                let mut delta = self.pos[i] - self.prev[i];
                let displacement = delta.length();
                if !delta.is_finite() {
                    self.pos[i] = self.prev[i];
                    self.vel[i] = Vec3::ZERO;
                    self.last_stability_corrections += 1;
                    continue;
                }
                if displacement > max_displacement {
                    delta *= max_displacement / displacement.max(1e-9);
                    self.pos[i] = self.prev[i] + delta;
                    self.last_stability_corrections += 1;
                }
                self.vel[i] = delta / h;
            }
        }

        let ms = start.elapsed().as_secs_f32() * 1000.0;
        let first_sample = self.stats.steps == 0;
        let blend = |previous: f32, sample: f32| {
            if first_sample { sample } else { 0.9 * previous + 0.1 * sample }
        };
        self.stats.step_ms = blend(self.stats.step_ms, ms);
        self.stats.constraint_ms = blend(self.stats.constraint_ms, constraint_elapsed_ms);
        self.stats.body_collision_ms = blend(self.stats.body_collision_ms, body_collision_elapsed_ms);
        self.stats.self_collision_broadphase_ms =
            blend(self.stats.self_collision_broadphase_ms, self_collision_broadphase_elapsed_ms);
        self.stats.self_collision_solve_ms =
            blend(self.stats.self_collision_solve_ms, self_collision_solve_elapsed_ms);
        self.stats.self_collision_ms = blend(
            self.stats.self_collision_ms,
            self_collision_broadphase_elapsed_ms + self_collision_solve_elapsed_ms,
        );
        self.stats.steps += 1;
    }

    /// Auto-colisão aproximada vértice-vértice acelerada por uma grade espacial uniforme.
    /// Pares ligados por arestas/triângulos ou por costura são excluídos para evitar jitter.
    fn collide_self(
        &mut self,
        friction_states: &mut HashMap<ContactKey, ContactFrictionState>,
        substep_dt: f32,
    ) -> (usize, usize) {
        let count = self.pos.len();
        if count < 2 {
            return (0, 0);
        }
        let max_thickness = (0..count)
            .map(|i| effective_thickness(self.params.thickness, self.thickness_scale[i]))
            .fold(0.0_f32, f32::max);
        if max_thickness <= 1.0e-6 {
            return (0, 0);
        }
        let cell_size = (2.0 * max_thickness).max(0.005);
        let batch = bounded_self_collision_candidate_pairs(
            &self.pos,
            cell_size,
            self.collision_topology.vertex_pair_exclusions(),
        );

        for (ia, ib) in batch.pairs.iter().copied() {
            let (i, j) = (ia as usize, ib as usize);
            let (wi, wj) = (self.inv_mass[i], self.inv_mass[j]);
            let weight = wi + wj;
            if weight <= 0.0 {
                continue;
            }
            let ti = effective_thickness(self.params.thickness, self.thickness_scale[i]);
            let tj = effective_thickness(self.params.thickness, self.thickness_scale[j]);
            let min_distance = pair_contact_distance(ti, tj);
            if min_distance <= 1.0e-6 {
                continue;
            }
            let delta = self.pos[i] - self.pos[j];
            let distance = delta.length();
            if !distance.is_finite() || distance >= min_distance {
                continue;
            }
            let normal = if distance > 1.0e-7 {
                delta / distance
            } else {
                match (i.wrapping_mul(31) ^ j) % 3 {
                    0 => Vec3::X,
                    1 => Vec3::Y,
                    _ => Vec3::Z,
                }
            };
            let penetration = min_distance - distance;
            self.last_self_collision_contacts += 1;
            self.stats.self_collision_vv_contacts += 1;
            self.last_self_collision_penetration =
                self.last_self_collision_penetration.max(penetration);
            self.stats.self_collision_penetration_before_m = self.stats
                .self_collision_penetration_before_m
                .max(penetration);

            let correction = normal * penetration;
            self.pos[i] += correction * (wi / weight);
            self.pos[j] -= correction * (wj / weight);
            // A separação normal não cria velocidade artificial.
            self.prev[i] += correction * (wi / weight);
            self.prev[j] -= correction * (wj / weight);

            let relative_motion =
                (self.pos[i] - self.prev[i]) - (self.pos[j] - self.prev[j]);
            let friction = update_contact_friction(
                ContactKey::VertexVertex(ia.min(ib), ia.max(ib)),
                friction_states,
                normal,
                relative_motion,
                penetration,
                combine_friction(
                    self.params.friction,
                    self.friction_scale[i],
                    self.friction_scale[j],
                ),
                combine_friction(
                    self.params.friction,
                    self.friction_dynamic_scale[i],
                    self.friction_dynamic_scale[j],
                ),
                substep_dt,
            );
            if let Some(friction) = friction {
                record_contact_friction(&mut self.stats, ContactFrictionMetrics {
                    contacts: 1,
                    static_contacts: if friction.sticking { 1 } else { 0 },
                    sliding_contacts: if !friction.sticking { 1 } else { 0 },
                    max_tangential_speed_m_s: friction.tangential_speed_m_s,
                    max_normal_correction_m: friction.normal_lambda_m,
                });
                distribute_friction_correction(
                    friction.delta_lambda_m,
                    &[ia, ib],
                    &[1.0, -1.0],
                    &self.inv_mass,
                    &mut self.pos,
                );
            }
        }
        (batch.candidate_count, batch.dropped_candidates)
    }

    fn collide(
        &mut self,
        friction_states: &mut HashMap<ContactKey, ContactFrictionState>,
        substep_dt: f32,
    ) {
        if self.colliders.is_empty() {
            return;
        }
        for i in 0..self.pos.len() {
            if self.inv_mass[i] <= 0.0 {
                continue;
            }
            let static_mu =
                self.params.friction.max(0.0) * self.friction_scale[i].max(0.0);
            let dynamic_mu = self.params.friction.max(0.0)
                * self.friction_dynamic_scale[i].max(0.0)
                .min(self.friction_scale[i].max(0.0));
            let clearance = body_contact_distance(effective_thickness(
                self.params.thickness,
                self.thickness_scale[i],
            ));
            let mut p = self.pos[i];
            let mut q = self.prev[i];

            // Captura cruzamentos mesmo quando o início e o fim estão fora do corpo.
            if body_sweep_is_needed(q, p, clearance) {
                self.stats.body_sweep_tests += 1;
                if let Some((swept_position, normal, penetration)) =
                    swept_body_contact(q, p, clearance, &self.colliders)
                {
                    p = swept_position;
                    self.stats.body_swept_contacts += 1;
                    self.stats.body_max_penetration_m =
                        self.stats.body_max_penetration_m.max(penetration);
                    let normal_delta = (p - q).dot(normal);
                    q += normal * normal_delta;
                }
            }

            // Costuras e restrições internas são resolvidas antes desta passada; o
            // contato do corpo corrige a geometria em cada iteração do mesmo substep.
            for _pass in 0..3 {
                let mut touched = false;
                for (collider_index, collider) in self.colliders.iter().enumerate() {
                    let d = collider.sdf(p) - clearance;
                    if d < 0.0 {
                        touched = true;
                        self.stats.body_contact_projections += 1;
                        self.stats.body_max_penetration_m =
                            self.stats.body_max_penetration_m.max(-d);
                        let normal = collider.normal(p);
                        p -= normal * d;

                        // A correção normal preserva a velocidade de entrada. A resposta
                        // tangencial tem um acumulador separado e só vive enquanto há contato.
                        let normal_delta = (p - q).dot(normal);
                        q += normal * normal_delta;
                        if let Some(friction) = update_contact_friction(
                            ContactKey::BodyCollider(i as u32, collider_index as u32),
                            friction_states,
                            normal,
                            p - q,
                            -d,
                            static_mu,
                            dynamic_mu,
                            substep_dt,
                        ) {
                            record_contact_friction(&mut self.stats, ContactFrictionMetrics {
                                contacts: 1,
                                static_contacts: if friction.sticking { 1 } else { 0 },
                                sliding_contacts: if friction.sticking { 0 } else { 1 },
                                max_tangential_speed_m_s: friction.tangential_speed_m_s,
                                max_normal_correction_m: friction.normal_lambda_m,
                            });
                            p += friction.delta_lambda_m;
                        }
                    }
                }
                if !touched {
                    break;
                }
            }
            self.pos[i] = p;
            self.prev[i] = q;
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
            && self.prev.iter().all(|p| p.is_finite())
            && self.vel.iter().all(|v| v.is_finite())
    }

    /// Velocidade máxima instantânea de uma partícula móvel (m/s).
    pub fn max_particle_speed(&self) -> f32 {
        self.vel.iter().enumerate()
            .filter(|(i, _)| self.inv_mass[*i] > 0.0)
            .map(|(_, v)| v.length())
            .filter(|speed| speed.is_finite())
            .fold(0.0, f32::max)
    }

    /// Quantidade de partículas com velocidade acima do limiar ou dados não finitos.
    pub fn unstable_particle_count(&self, speed_limit_m_s: f32) -> usize {
        let limit = speed_limit_m_s.max(0.0);
        (0..self.pos.len()).filter(|&i| {
            !self.pos[i].is_finite() || !self.vel[i].is_finite()
                || (self.inv_mass[i] > 0.0 && self.vel[i].length() > limit)
        }).count()
    }

    /// Maior deslocamento ocorrido no último substep, em metros.
    pub fn max_particle_displacement(&self) -> f32 {
        self.pos.iter().zip(&self.prev)
            .map(|(p, previous)| (*p - *previous).length())
            .filter(|d| d.is_finite())
            .fold(0.0, f32::max)
    }

    /// Contatos de auto-colisão encontrados em todos os substeps do último passo.
    pub fn self_collision_contacts(&self) -> usize {
        self.last_self_collision_contacts
    }

    /// Maior penetração entre partículas não adjacentes no último passo, em metros.
    pub fn self_collision_max_penetration(&self) -> f32 {
        self.last_self_collision_penetration
    }

    /// Quantidade de partículas recuperadas ou limitadas pelo mecanismo de estabilidade.
    pub fn stability_corrections(&self) -> usize {
        self.last_stability_corrections
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

    /// Pares de costura com abertura superior ao limite em metros.
    /// É um indicador visual de abertura/tensão, não uma medição de força em newtons.
    pub fn seam_warning_pairs(&self, gap_limit_meters: f32) -> Vec<(u32, u32)> {
        let limit = gap_limit_meters.max(0.0);
        self.distance.iter()
            .filter(|c| c.kind == ConstraintKind::Seam)
            .filter_map(|c| {
                let gap = (self.pos[c.a as usize] - self.pos[c.b as usize]).length();
                (gap > limit).then_some((c.a, c.b))
            })
            .collect()
    }

    /// Maior abertura instantânea entre as partículas emparelhadas de costura.
    pub fn seam_gap_max(&self) -> f32 {
        self.distance.iter()
            .filter(|c| c.kind == ConstraintKind::Seam)
            .map(|c| (self.pos[c.a as usize] - self.pos[c.b as usize]).length())
            .fold(0.0, f32::max)
    }

    /// Maior força equivalente estimada a partir dos multiplicadores XPBD do último substep.
    /// É um indicador relativo do solver, não substitui uma calibração experimental do tecido.
    pub fn seam_tension_estimate_max_n(&self) -> f32 {
        if self.last_substep_dt <= 1.0e-9 { return 0.0; }
        let inv_dt2 = 1.0 / (self.last_substep_dt * self.last_substep_dt);
        self.distance.iter().zip(&self.dist_lambda)
            .filter(|(constraint, _)| constraint.kind == ConstraintKind::Seam)
            .map(|(_, lambda)| lambda.abs() * inv_dt2)
            .filter(|force| force.is_finite())
            .fold(0.0, f32::max)
    }

    /// Arestas únicas da malha (para o overlay de wireframe e a broadphase futura).
    pub fn mesh_edges(&self) -> Vec<[u32; 2]> {
        self.collision_topology
            .edges()
            .iter()
            .map(|edge| edge.vertices)
            .collect()
    }
}


#[derive(Default)]
struct BoundedSelfCollisionCandidatePairs {
    pairs: Vec<(u32, u32)>,
    candidate_count: usize,
    dropped_candidates: usize,
}

fn bounded_self_collision_candidate_pairs(
    positions: &[Vec3],
    cell_size: f32,
    exclusions: &HashSet<(u32, u32)>,
) -> BoundedSelfCollisionCandidatePairs {
    let mut output = BoundedSelfCollisionCandidatePairs::default();
    if positions.len() < 2 { return output; }
    let cell_size = cell_size.max(1.0e-5);
    let cell_of = |p: Vec3| -> (i32, i32, i32) {
        ((p.x / cell_size).floor() as i32,
         (p.y / cell_size).floor() as i32,
         (p.z / cell_size).floor() as i32)
    };
    let mut grid: HashMap<(i32, i32, i32), Vec<u32>> = HashMap::new();
    for (i, position) in positions.iter().enumerate() {
        if position.is_finite() {
            grid.entry(cell_of(*position)).or_default().push(i as u32);
        }
    }
    let per_particle_budget = (MAX_SELF_COLLISION_CANDIDATE_PAIRS_TOTAL / positions.len())
        .min(MAX_SELF_COLLISION_CANDIDATES_PER_PARTICLE);
    let mut total_inspected = 0usize;

    for (i, position) in positions.iter().enumerate() {
        if !position.is_finite() { continue; }
        let (cx, cy, cz) = cell_of(*position);
        let mut total_candidates = 0usize;
        for dx in -1..=1 {
            for dy in -1..=1 {
                for dz in -1..=1 {
                    if let Some(neighbours) = grid.get(&(cx + dx, cy + dy, cz + dz)) {
                        let first_after_i = neighbours.partition_point(|&j| (j as usize) <= i);
                        total_candidates = total_candidates.saturating_add(neighbours.len() - first_after_i);
                    }
                }
            }
        }
        output.candidate_count = output.candidate_count.saturating_add(total_candidates);
        let remaining_global = MAX_SELF_COLLISION_CANDIDATE_PAIRS_TOTAL - total_inspected;
        let inspect_limit = total_candidates.min(per_particle_budget).min(remaining_global);
        output.dropped_candidates = output
            .dropped_candidates
            .saturating_add(total_candidates - inspect_limit);
        if inspect_limit == 0 { continue; }

        let mut inspected = 0usize;
        'neighbor_cells: for dx in -1..=1 {
            for dy in -1..=1 {
                for dz in -1..=1 {
                    let Some(neighbours) = grid.get(&(cx + dx, cy + dy, cz + dz)) else { continue; };
                    let first_after_i = neighbours.partition_point(|&j| (j as usize) <= i);
                    for &j in &neighbours[first_after_i..] {
                        if inspected >= inspect_limit {
                            break 'neighbor_cells;
                        }
                        inspected += 1;
                        let pair = (i as u32, j);
                        if !exclusions.contains(&pair) {
                            output.pairs.push(pair);
                        }
                    }
                }
            }
        }
        total_inspected += inspected;
    }
    output
}

#[cfg(test)]
mod self_collision_candidate_tests {
    use super::*;

    #[test]
    fn candidate_pairs_are_bounded_when_all_particles_collapse_into_one_cell() {
        let positions = vec![Vec3::ZERO; 800];
        let batch = bounded_self_collision_candidate_pairs(&positions, 0.01, &HashSet::new());
        assert!(!batch.pairs.is_empty());
        assert!(batch.pairs.len() <= positions.len() * MAX_SELF_COLLISION_CANDIDATES_PER_PARTICLE);
        assert!(batch.pairs.iter().all(|&(a, b)| a < b));
        assert!(batch.dropped_candidates > 0);
        let unique = batch.pairs.iter().copied().collect::<HashSet<_>>();
        assert_eq!(unique.len(), batch.pairs.len());
    }

    #[test]
    fn candidate_pairs_obey_global_memory_budget_on_large_collapsed_mesh() {
        let positions = vec![Vec3::ZERO; 25_000];
        let batch = bounded_self_collision_candidate_pairs(&positions, 0.01, &HashSet::new());
        assert!(!batch.pairs.is_empty());
        assert!(batch.pairs.len() <= MAX_SELF_COLLISION_CANDIDATE_PAIRS_TOTAL);
        assert!(batch.pairs.len() < positions.len() * MAX_SELF_COLLISION_CANDIDATES_PER_PARTICLE);
        assert!(batch.dropped_candidates > 0);
    }

    #[test]
    fn candidate_pairs_skip_excluded_and_non_finite_particles() {
        let positions = vec![Vec3::ZERO, Vec3::ZERO, Vec3::splat(f32::NAN), Vec3::ZERO];
        let exclusions = HashSet::from([(0_u32, 1_u32)]);
        let batch = bounded_self_collision_candidate_pairs(&positions, 0.01, &exclusions);
        assert!(batch.pairs.iter().all(|&(a, b)| a != 2 && b != 2));
        assert!(!batch.pairs.contains(&(0, 1)));
        assert!(batch.pairs.contains(&(0, 3)));
    }
}

#[allow(clippy::too_many_arguments)]
/// Limite conservador para detectar cruzamentos entre o estado anterior e o previsto.
fn body_sweep_is_needed(start: Vec3, end: Vec3, clearance: f32) -> bool {
    let threshold = (clearance.max(0.0) * 1.5).max(0.004);
    (end - start).length_squared() > threshold * threshold
}

/// Menor distância assinada ao conjunto de colisores (união aproximada dos volumes).
fn closest_body_gap(point: Vec3, clearance: f32, colliders: &[Collider]) -> Option<(f32, usize)> {
    let mut best: Option<(f32, usize)> = None;
    for (index, collider) in colliders.iter().enumerate() {
        let gap = collider.sdf(point) - clearance;
        if !gap.is_finite() { continue; }
        if best.is_none_or(|(current, _)| gap < current) {
            best = Some((gap, index));
        }
    }
    best
}

/// CCD aproximado por SDF: avança pela trajetória com passos limitados pela distância à superfície.
/// Remove a componente de movimento para dentro do corpo e preserva o deslizamento tangencial.
fn swept_body_contact(
    start: Vec3, end: Vec3, clearance: f32, colliders: &[Collider],
) -> Option<(Vec3, Vec3, f32)> {
    const MAX_STEPS: usize = 128;
    const MAX_ADVANCE_M: f32 = 0.025;
    const MIN_ADVANCE_M: f32 = 0.0005;
    const START_PENETRATION_TOLERANCE_M: f32 = 0.002;

    if colliders.is_empty() || !start.is_finite() || !end.is_finite() || !clearance.is_finite() {
        return None;
    }
    let clearance = clearance.max(0.0);
    let motion = end - start;
    let travel = motion.length();
    if !travel.is_finite() || !body_sweep_is_needed(start, end, clearance) { return None; }
    let (initial_gap, _) = closest_body_gap(start, clearance, colliders)?;
    // Sobreposições profundas preexistentes são tratadas pela projeção discreta.
    if initial_gap < -START_PENETRATION_TOLERANCE_M { return None; }

    let epsilon = (clearance * 0.2).clamp(0.0005, 0.0015);
    let mut t = 0.0_f32;
    for _ in 0..MAX_STEPS {
        let sample = start + motion * t;
        let (gap, collider_index) = closest_body_gap(sample, clearance, colliders)?;
        let normal = colliders[collider_index].normal(sample).try_normalize()?;
        let moving_inward = motion.dot(normal) < -1.0e-5;
        if gap <= epsilon && (moving_inward || gap < -1.0e-5) {
            let penetration = (-gap).max(0.0);
            let boundary_position = sample + normal * penetration;
            let remaining = end - sample;
            let inward_remainder = remaining.dot(normal);
            let slide = if inward_remainder < 0.0 {
                remaining - normal * inward_remainder
            } else { remaining };
            return Some((boundary_position + slide, normal, penetration));
        }
        let remaining_distance = travel * (1.0 - t);
        if remaining_distance <= MIN_ADVANCE_M || t >= 1.0 { break; }
        let advance = (gap.max(MIN_ADVANCE_M) * 0.75).min(MAX_ADVANCE_M).max(MIN_ADVANCE_M);
        let next_t = (t + advance / travel).min(1.0);
        if next_t <= t + 1.0e-7 { break; }
        t = next_t;
    }
    None
}

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
            let cval = len - c.rest;
            let alpha = c.scale / stiff[c.kind as usize] * inv_h2;
            let mut dl = (-cval - alpha * *lam) / (w + alpha);
            if c.kind == ConstraintKind::Seam {
                // Limita a correção acumulada de cada costura por substep.
                // Assim, mais iterações melhoram a convergência, mas não aumentam
                // indefinidamente a distância que a costura pode puxar neste substep.
                let max_lambda = seam_step / w;
                let next_lambda = (*lam + dl).clamp(-max_lambda, max_lambda);
                dl = next_lambda - *lam;
                *lam = next_lambda;
            } else {
                *lam += dl;
            }
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
    use crate::params::{PanelMaterial, SimParams};
    use crate::scenes::{build_scene, SceneKind};
    use glam::Vec2;

    #[test]
    fn per_panel_material_changes_mass_stretch_bend_and_collision_thickness() {
        let mesh = PanelMesh {
            uv: vec![
                Vec2::new(0.0, 0.0), Vec2::new(1.0, 0.0),
                Vec2::new(1.0, 1.0), Vec2::new(0.0, 1.0),
            ],
            triangles: vec![[0, 1, 2], [0, 2, 3]],
        };
        let material_a = PanelMaterial::default();
        let material_b = PanelMaterial {
            density_kg_m2: 0.8,
            stretch_warp_n_m: 1_000.0,
            stretch_weft_n_m: 1_000.0,
            shear_n_m: 100.0,
            bend_nm: 2.0e-4,
            thickness_meters: 0.002,
            friction_coefficient: 0.8,
            friction_dynamic_coefficient: 0.3,
        };
        let mut builder = ClothBuilder::new();
        let a = builder.add_panel_with_grain_and_material(
            &mesh, Placement::at(Vec3::ZERO), Vec2::Y, material_a,
        ).unwrap();
        let b = builder.add_panel_with_grain_and_material(
            &mesh, Placement::at(Vec3::new(3.0, 0.0, 0.0)), Vec2::Y, material_b,
        ).unwrap();
        assert_eq!(a.count, 4);
        assert_eq!(b.first, 4);

        let sim = builder.build(SimParams::default());
        assert!((sim.mass_scale[0] - 1.0).abs() < 1.0e-6);
        assert!((sim.mass_scale[4] - 4.0).abs() < 1.0e-6);
        assert!((sim.inv_mass[0] / sim.inv_mass[4] - 4.0).abs() < 1.0e-4);
        assert!((sim.thickness_scale[0] - 1.0).abs() < 1.0e-6);
        assert!((sim.thickness_scale[4] - 0.4).abs() < 1.0e-5);
        assert!((sim.friction_scale[4] - 2.0).abs() < 1.0e-5);
        assert!((sim.friction_dynamic_scale[4] - 0.75).abs() < 1.0e-5);

        let edge_a = sim.distance.iter()
            .find(|constraint| {
                constraint.kind == ConstraintKind::Weft
                    && ((constraint.a == 0 && constraint.b == 1)
                        || (constraint.a == 1 && constraint.b == 0))
            }).unwrap();
        let edge_b = sim.distance.iter()
            .find(|constraint| {
                constraint.kind == ConstraintKind::Weft
                    && ((constraint.a == 4 && constraint.b == 5)
                        || (constraint.a == 5 && constraint.b == 4))
            }).unwrap();
        assert!((edge_b.scale / edge_a.scale - 5.0).abs() < 1.0e-4);

        let bend_a = sim.bending.iter().find(|constraint| {
            constraint.p.contains(&0) && constraint.p.contains(&2)
        }).unwrap();
        let bend_b = sim.bending.iter().find(|constraint| {
            constraint.p.contains(&4) && constraint.p.contains(&6)
        }).unwrap();
        assert!((bend_b.scale / bend_a.scale - 0.25).abs() < 1.0e-4);
    }

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
    fn collider_projection_does_not_create_launch_velocity_from_initial_overlap() {
        let panel = GridPanel::new(1.0, 1.0, 12, 12);
        let mut builder = ClothBuilder::new();
        // Simula o caso antigo: painel vertical coplanar com o eixo do cilindro corporal.
        builder.add_panel(&panel.mesh, Placement::at(Vec3::new(-0.5, 0.4, 0.0)));
        builder.add_collider(Collider::Cylinder {
            a: Vec3::ZERO,
            b: Vec3::new(0.0, 1.5, 0.0),
            radius: 0.14,
        });
        let params = SimParams {
            gravity: Vec3::ZERO,
            substeps: 1,
            damping: 0.0,
            parallel: false,
            ..SimParams::default()
        };
        let mut sim = builder.build(params);
        sim.step(1.0 / 60.0);
        assert!(sim.is_finite());
        let max_speed = sim.vel.iter().map(|v| v.length()).fold(0.0, f32::max);
        assert!(
            max_speed < 1e-3,
            "a correção do colisor criou velocidade artificial: {max_speed} m/s"
        );
    }

    #[test]
    fn cloth_does_not_penetrate_sphere() {
        let mut sim = build_scene(SceneKind::SphereDrop, 24, SimParams::default());
        let mut contact_projections = 0usize;
        let mut max_penetration = 0.0_f32;
        for _ in 0..600 {
            sim.step(1.0 / 60.0);
            contact_projections += sim.stats.body_contact_projections;
            max_penetration = max_penetration.max(sim.stats.body_max_penetration_m);
        }
        assert!(sim.is_finite());
        assert!(contact_projections > 0, "a cena deveria registrar contatos com o corpo");
        assert!(max_penetration.is_finite() && max_penetration > 0.0);
        assert!(sim.stats.constraint_ms.is_finite() && sim.stats.body_collision_ms.is_finite());
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
    fn sdf_avatar_scene_records_reference_contact_metrics() {
        let mut sim = build_scene(SceneKind::SdfAvatarDrop, 16, SimParams::default());
        let mut contact_projections = 0usize;
        let mut max_penetration = 0.0_f32;
        for _ in 0..240 {
            sim.step(1.0 / 60.0);
            contact_projections += sim.stats.body_contact_projections;
            max_penetration = max_penetration.max(sim.stats.body_max_penetration_m);
        }

        let sdf = sim.colliders.iter().find_map(|collider| match collider {
            Collider::TriangleMeshSdf { field } => Some(field),
            _ => None,
        }).expect("a cena deve usar o SDF do torso");
        let minimum_margin = sim.pos.iter()
            .map(|position| sdf.sample(*position) - sim.params.thickness)
            .fold(f32::INFINITY, f32::min);

        assert!(sim.is_finite(), "a cena com SDF divergiu");
        assert!(contact_projections > 0, "o SDF deveria registrar contatos");
        assert!(max_penetration.is_finite() && max_penetration > 0.0);
        assert!(minimum_margin > -0.04, "penetração final excessiva no SDF: {minimum_margin} m");
        assert!(sim.stats.constraint_ms.is_finite() && sim.stats.body_collision_ms.is_finite());
    }

    #[test]
    fn fast_sdf_crossing_is_caught_by_continuous_collision() {
        let mut sim = build_scene(SceneKind::FastSdfCrossing, 12, SimParams::default());
        sim.step(1.0 / 60.0);
        let sdf = sim.colliders.iter().find_map(|collider| match collider {
            Collider::TriangleMeshSdf { field } => Some(field),
            _ => None,
        }).expect("a cena deve usar o SDF do torso");
        let minimum_margin = sim.pos.iter()
            .map(|position| sdf.sample(*position) - sim.params.thickness)
            .fold(f32::INFINITY, f32::min);
        assert!(sim.is_finite());
        assert!(sim.stats.body_sweep_tests > 0, "deveria examinar trajetórias longas");
        assert!(sim.stats.body_swept_contacts > 0, "deveria interceptar a travessia");
        assert!(minimum_margin > -0.015, "o tecido atravessou o torso: margem {minimum_margin} m");
    }

    #[test]
    fn swept_contact_stops_an_outside_to_outside_sphere_crossing() {
        let sphere = Collider::Sphere { center: Vec3::ZERO, radius: 1.0 };
        let (position, normal, penetration) = swept_body_contact(
            Vec3::new(-2.0, 0.0, 0.0), Vec3::new(2.0, 0.0, 0.0), 0.05,
            std::slice::from_ref(&sphere),
        ).expect("o movimento deveria atingir a esfera");
        assert!(normal.dot(Vec3::NEG_X) > 0.99, "normal incorreta: {normal:?}");
        assert!(position.x < -1.0, "deveria parar na entrada: {position:?}");
        assert!(sphere.sdf(position) >= 0.05 - 1.0e-4, "ficou dentro da margem: {position:?}");
        assert!(penetration <= 1.0e-3);
    }

    #[test]
    fn swept_contact_does_not_block_motion_away_from_surface() {
        let sphere = Collider::Sphere { center: Vec3::ZERO, radius: 1.0 };
        assert!(swept_body_contact(
            Vec3::new(-1.1, 0.0, 0.0), Vec3::new(-2.0, 0.0, 0.0), 0.05,
            std::slice::from_ref(&sphere),
        ).is_none());
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


#[cfg(test)]
mod stability_tests {
    use super::*;

    fn minimal_simulation(self_collision: bool, z_offset: f32) -> Simulation {
        let tri = [0u32, 1, 2];
        let positions = vec![
            Vec3::new(0.0, 0.0, 0.0),
            Vec3::new(0.2, 0.0, 0.0),
            Vec3::new(0.0, 0.2, 0.0),
            Vec3::new(0.0, 0.0, z_offset),
            Vec3::new(0.2, 0.0, z_offset),
            Vec3::new(0.0, 0.2, z_offset),
        ];
        let mut params = SimParams::default();
        params.gravity = Vec3::ZERO;
        params.damping = 0.0;
        params.substeps = 1;
        params.iterations = 1;
        params.thickness = 0.02;
        params.friction = 0.0;
        params.self_collision = self_collision;
        Simulation::from_parts(
            positions,
            vec![tri, [3, 4, 5]],
            vec![0.01; 6],
            vec![false; 6],
            vec![1.0; 6],
            vec![1.0; 6],
            vec![1.0; 6],
            vec![0.8; 6],
            Vec::new(),
            Vec::new(),
            Vec::new(),
            Vec::new(),
            Vec::new(),
            params,
        )
    }

    #[test]
    fn imported_mesh_sdf_projects_tissue_and_reports_collision_metrics() {
        let positions = vec![
            [-0.5, 0.0, -0.5], [0.5, 0.0, -0.5], [0.5, 1.0, -0.5], [-0.5, 1.0, -0.5],
            [-0.5, 0.0, 0.5], [0.5, 0.0, 0.5], [0.5, 1.0, 0.5], [-0.5, 1.0, 0.5],
        ];
        let indices = vec![
            0, 2, 1, 0, 3, 2, 4, 5, 6, 4, 6, 7,
            0, 1, 5, 0, 5, 4, 3, 7, 6, 3, 6, 2,
            0, 4, 7, 0, 7, 3, 1, 2, 6, 1, 6, 5,
        ];
        let field = crate::sdf::SignedDistanceField::from_triangle_mesh(
            &positions, &indices, 0.025,
        ).unwrap();
        assert!(field.has_reliable_inside_outside());

        let mut sim = minimal_simulation(false, 0.0);
        sim.colliders = vec![Collider::TriangleMeshSdf {
            field: std::sync::Arc::new(field.clone()),
        }];
        for (index, position) in sim.pos.iter_mut().enumerate() {
            *position = Vec3::splat(3.0 + index as f32);
        }
        sim.pos[0] = Vec3::new(0.49, 0.5, 0.0);
        sim.prev.clone_from(&sim.pos);
        sim.params.gravity = Vec3::ZERO;
        sim.params.thickness = 0.02;
        sim.params.substeps = 1;
        sim.params.iterations = 2;

        sim.step(1.0 / 60.0);

        assert!(sim.is_finite());
        assert!(sim.stats.body_contact_projections > 0);
        assert!(
            field.sample(sim.pos[0]) >= sim.params.thickness - 0.002,
            "o tecido deveria terminar fora da superfície: pos={:?}, distância={}",
            sim.pos[0],
            field.sample(sim.pos[0]),
        );
        assert!(sim.stats.body_collision_ms.is_finite());
    }

    #[test]
    fn seam_pull_budget_is_independent_of_solver_iteration_count() {
        let run = |iterations: usize| {
            let mut positions = vec![Vec3::ZERO, Vec3::X];
            let inv_mass = [1.0_f32, 1.0];
            let constraint = DistanceConstraint {
                a: 0, b: 1, rest: 0.0, scale: 1.0e-6, kind: ConstraintKind::Seam,
            };
            let mut lambda = [0.0_f32];
            let stiffness = [1.0, 1.0, 1.0, 5_000.0];
            for _ in 0..iterations {
                solve_distance(
                    &[constraint], &mut lambda, &mut positions, &inv_mass,
                    1.0, &stiffness, 0.01, false,
                );
            }
            1.0 - (positions[0] - positions[1]).length()
        };
        let one_iteration = run(1);
        let many_iterations = run(8);
        assert!(one_iteration <= 0.010_001, "single iteration exceeded seam budget: {one_iteration}");
        assert!(many_iterations <= 0.010_001, "multiple iterations exceeded seam budget: {many_iterations}");
        assert!((one_iteration - many_iterations).abs() < 1.0e-5);
    }

    #[test]
    fn self_collision_separates_non_adjacent_overlapping_panels() {
        let mut sim = minimal_simulation(true, 0.001);
        let before = (sim.pos[0] - sim.pos[3]).length();
        sim.step(1.0 / 60.0);
        let after = (sim.pos[0] - sim.pos[3]).length();
        assert!(after > before, "self collision should increase layer separation");
        assert!(sim.is_finite());
        assert!(sim.self_collision_contacts() > 0);
        assert!(sim.self_collision_max_penetration() > 0.0);
    }

    #[test]
    fn self_collision_can_be_disabled_for_performance() {
        let mut sim = minimal_simulation(false, 0.001);
        let before = (sim.pos[0] - sim.pos[3]).length();
        sim.step(1.0 / 60.0);
        let after = (sim.pos[0] - sim.pos[3]).length();
        assert!((after - before).abs() < 1e-5);
    }

    #[test]
    fn velocity_is_clamped_to_configured_safety_limit() {
        let mut sim = minimal_simulation(false, 0.0);
        sim.params.gravity = Vec3::new(0.0, -100_000.0, 0.0);
        sim.params.max_particle_speed = 2.0;
        sim.step(1.0 / 60.0);
        assert!(sim.max_particle_speed() <= 2.01);
        assert!(sim.stability_corrections() > 0);
        assert_eq!(sim.unstable_particle_count(10.0), 0);
    }

    #[test]
    fn non_finite_particle_state_is_recovered_and_reported() {
        let mut sim = minimal_simulation(false, 0.0);
        sim.pos[0] = Vec3::splat(f32::NAN);
        sim.step(1.0 / 60.0);
        assert!(sim.is_finite());
        assert!(sim.stability_corrections() > 0);
        assert_eq!(sim.vel[0], Vec3::ZERO);
    }
    #[test]
    fn body_friction_uses_material_dynamic_limit_and_drops_after_separation() {
        let run = |dynamic_scale: f32| {
            let mut sim = minimal_simulation(false, 0.0);
            sim.params.gravity = Vec3::ZERO;
            sim.params.damping = 0.0;
            sim.params.substeps = 1;
            sim.params.iterations = 1;
            sim.params.thickness = 0.02;
            sim.params.friction = 1.0;
            sim.colliders.push(Collider::ground());
            sim.friction_scale.fill(0.0);
            sim.friction_dynamic_scale.fill(0.0);
            sim.friction_scale[0] = 1.0;
            sim.friction_dynamic_scale[0] = dynamic_scale;
            sim.vel[0] = Vec3::X * 6.0;
            sim.step(1.0 / 60.0);
            let tangential_speed = sim.vel[0].x;
            (sim, tangential_speed)
        };

        let (_, no_dynamic_friction_speed) = run(0.0);
        let (mut sim, dynamic_friction_speed) = run(0.5);
        assert!(dynamic_friction_speed < no_dynamic_friction_speed - 0.5);
        assert!(sim.stats.contact_friction_sliding_contacts > 0);
        assert!(sim.stats.contact_max_tangential_speed_m_s >= 5.9);

        sim.pos[0].y = 0.2;
        sim.vel[0] = Vec3::X;
        sim.step(1.0 / 60.0);
        assert!((sim.vel[0].x - 1.0).abs() < 1.0e-5, "atrito persistiu sem contato");
    }
}

#[cfg(test)]
mod placement_transform_tests {
    use super::*;

    fn tiny_simulation() -> Simulation {
        Simulation::from_parts(
            vec![Vec3::ZERO, Vec3::X, Vec3::Y],
            Vec::new(),
            vec![1.0; 3],
            vec![false; 3],
            vec![1.0; 3],
            vec![1.0; 3],
            vec![1.0; 3],
            vec![0.8; 3],
            Vec::new(),
            Vec::new(),
            Vec::new(),
            Vec::new(),
            Vec::new(),
            SimParams::default(),
        )
    }

    #[test]
    fn translating_panel_range_persists_after_reset() {
        let mut sim = tiny_simulation();
        let delta = Vec3::new(0.25, 0.5, -0.75);
        assert!(sim.translate_particles(1, 3, delta));
        assert_eq!(sim.pos[0], Vec3::ZERO);
        assert_eq!(sim.pos[1], Vec3::X + delta);
        assert_eq!(sim.prev[2], Vec3::Y + delta);
        sim.reset();
        assert_eq!(sim.pos[1], Vec3::X + delta);
        assert_eq!(sim.pos[2], Vec3::Y + delta);
        assert_eq!(sim.vel[1], Vec3::ZERO);
    }

    #[test]
    fn rotating_panel_range_persists_after_reset() {
        let mut sim = tiny_simulation();
        let rotation = glam::Quat::from_rotation_z(std::f32::consts::FRAC_PI_2);
        assert!(sim.rotate_particles(0, 3, Vec3::ZERO, rotation));
        assert!((sim.pos[1] - Vec3::Y).length() < 1e-5);
        sim.reset();
        assert!((sim.pos[1] - Vec3::Y).length() < 1e-5);
        assert_eq!(sim.vel[1], Vec3::ZERO);
    }

    #[test]
    fn scaling_panel_range_persists_after_reset_and_rejects_invalid_factors() {
        let mut sim = tiny_simulation();
        assert!(sim.scale_particles_axis(0, 3, Vec3::ZERO, Vec3::X, 2.0));
        assert!((sim.pos[1] - Vec3::X * 2.0).length() < 1e-5);
        assert!((sim.pos[2] - Vec3::Y).length() < 1e-5);
        sim.reset();
        assert!((sim.pos[1] - Vec3::X * 2.0).length() < 1e-5);
        assert!(!sim.scale_particles_axis(0, 3, Vec3::ZERO, Vec3::ZERO, 2.0));
        assert!(!sim.scale_particles_axis(0, 3, Vec3::ZERO, Vec3::X, 0.0));
    }

    #[test]
    fn translating_panel_range_rejects_invalid_bounds_or_delta() {
        let mut sim = tiny_simulation();
        assert!(!sim.translate_particles(2, 2, Vec3::X));
        assert!(!sim.translate_particles(0, 4, Vec3::X));
        assert!(!sim.translate_particles(0, 1, Vec3::splat(f32::NAN)));
        assert_eq!(sim.pos[0], Vec3::ZERO);
    }
}
