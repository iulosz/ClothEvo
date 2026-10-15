//! Cenas de teste da Fase 0: validam estabilidade, colisão e costura do solver.

use glam::{Quat, Vec3};

use crate::builder::ClothBuilder;
use crate::collider::Collider;
use crate::mesh::{GridPanel, Placement};
use crate::params::SimParams;
use crate::sdf::SignedDistanceField;
use crate::solver::Simulation;
use std::sync::Arc;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum SceneKind {
    /// Pano horizontal preso por dois cantos: stretch, shear e flexão sob grande movimento.
    HangingCloth,
    /// Pano caindo sobre uma esfera: colisão e atrito.
    SphereDrop,
    /// Dois painéis costurados nas laterais ao redor de um cilindro: costuras + colisão.
    SewnTube,
    /// Tecido sobre um torso fechado com colisão por SDF de malha importada.
    SdfAvatarDrop,
    /// Travessia rápida do torso em um único substep para validar a detecção contínua aproximada.
    FastSdfCrossing,
    /// Dois painéis sobrepostos para medir contatos internos vértice-face e aresta-aresta.
    SelfCollisionLayers,
    /// Um painel atravessa outro durante um único substep para exercitar o CCD aproximado.
    FastSelfCollisionCrossing,
}

impl SceneKind {
    /// Cenas disponibilizadas no seletor interativo do app.
    pub const ALL: [SceneKind; 3] = [Self::HangingCloth, Self::SphereDrop, Self::SewnTube];

    /// Cenários fixos do diagnóstico/benchmark de colisão, SDF, movimento rápido e auto-colisão.
    pub const DIAGNOSTIC: [SceneKind; 7] = [
        Self::HangingCloth,
        Self::SphereDrop,
        Self::SewnTube,
        Self::SdfAvatarDrop,
        Self::FastSdfCrossing,
        Self::SelfCollisionLayers,
        Self::FastSelfCollisionCrossing,
    ];

    pub fn name(self) -> &'static str {
        match self {
            Self::HangingCloth => "1. Pano pendurado",
            Self::SphereDrop => "2. Pano sobre esfera",
            Self::SewnTube => "3. Tubo costurado",
            Self::SdfAvatarDrop => "4. Tecido sobre torso (SDF)",
            Self::FastSdfCrossing => "5. Travessia rápida do torso (SDF)",
            Self::SelfCollisionLayers => "6. Painéis sobrepostos (auto-colisão)",
            Self::FastSelfCollisionCrossing => "7. Travessia rápida entre painéis",
        }
    }
}

/// Constrói uma cena. `res` é o número de células por metro (≈ resolução da malha):
/// 40 → ~1,7k partículas, 100 → ~10k, 141 → ~20k (para o pano de 1 m).
pub fn build_scene(kind: SceneKind, res: u32, params: SimParams) -> Simulation {
    let res = res.max(8);
    let mut b = ClothBuilder::new();
    let mut self_collision_velocity_range = None;
    if !matches!(kind, SceneKind::SdfAvatarDrop | SceneKind::FastSdfCrossing) {
        b.add_collider(Collider::ground());
    }

    match kind {
        SceneKind::HangingCloth => {
            let panel = GridPanel::new(1.0, 1.0, res, res);
            // Deitado no plano XZ, face da frente para cima; presa pelos dois cantos do lado -Z.
            let placement = Placement::new(
                Vec3::new(-0.5, 1.4, 0.5),
                Quat::from_rotation_x(-std::f32::consts::FRAC_PI_2),
            );
            let h = b.add_panel(&panel.mesh, placement);
            b.pin(h.global(panel.index(0, panel.ny)));
            b.pin(h.global(panel.index(panel.nx, panel.ny)));
            b.perturb(0.002);
        }
        SceneKind::SphereDrop => {
            let cells = (1.2 * res as f32).round() as u32;
            let panel = GridPanel::new(1.2, 1.2, cells, cells);
            let placement = Placement::new(
                Vec3::new(-0.6, 1.2, 0.6),
                Quat::from_rotation_x(-std::f32::consts::FRAC_PI_2),
            );
            b.add_panel(&panel.mesh, placement);
            b.add_collider(Collider::Sphere { center: Vec3::new(0.0, 0.4, 0.0), radius: 0.4 });
            b.perturb(0.002);
        }
        SceneKind::SdfAvatarDrop | SceneKind::FastSdfCrossing => {
            let panel = GridPanel::new(0.6, 0.6, res, res);
            let placement = Placement::new(
                Vec3::new(-0.3, 1.3, 0.3),
                Quat::from_rotation_x(-std::f32::consts::FRAC_PI_2),
            );
            b.add_panel(&panel.mesh, placement);
            b.add_collider(Collider::TriangleMeshSdf { field: avatar_torso_sdf() });
            b.perturb(0.002);
        }
        SceneKind::SelfCollisionLayers => {
            let cells = ((0.6 * res as f32).round() as u32).max(4);
            let panel = GridPanel::new(0.6, 0.6, cells, cells);
            let placement = Placement::at(Vec3::new(-0.3, 0.8, 0.0));
            // Superfícies desconectadas na topologia, coincidentes no espaço.
            b.add_panel(&panel.mesh, placement);
            b.add_panel(&panel.mesh, placement);
        }
        SceneKind::FastSelfCollisionCrossing => {
            let cells = ((0.6 * res as f32).round() as u32).max(4);
            let panel = GridPanel::new(0.6, 0.6, cells, cells);
            let rotation = Quat::from_rotation_x(-std::f32::consts::FRAC_PI_2);
            let fixed = Placement::new(Vec3::new(-0.3, 0.78, 0.3), rotation);
            let moving = Placement::new(Vec3::new(-0.3, 0.72, 0.3), rotation);
            b.add_panel(&panel.mesh, fixed);
            let moving_panel = b.add_panel(&panel.mesh, moving);
            self_collision_velocity_range = Some((
                moving_panel.first as usize,
                (moving_panel.first + moving_panel.count) as usize,
            ));
        }
        SceneKind::SewnTube => {
            let nx = ((0.6 * res as f32).round() as u32).max(4);
            let ny = ((0.9 * res as f32).round() as u32).max(4);
            let panel = GridPanel::new(0.6, 0.9, nx, ny);

            // Painel da frente: face para +Z.
            let front = b.add_panel(&panel.mesh, Placement::at(Vec3::new(-0.3, 0.9, 0.2)));
            // Painel de trás: girado 180° em Y (face para -Z, `u` aponta para -X).
            let back = b.add_panel(
                &panel.mesh,
                Placement::new(Vec3::new(0.3, 0.9, -0.2), Quat::from_rotation_y(std::f32::consts::PI)),
            );

            // x = +0,3: borda direita da frente com a esquerda de trás; x = -0,3: o contrário.
            b.sew(&front.globals(&panel.right_edge()), &back.globals(&panel.left_edge()));
            b.sew(&front.globals(&panel.left_edge()), &back.globals(&panel.right_edge()));

            b.add_collider(Collider::Cylinder {
                a: Vec3::ZERO,
                b: Vec3::new(0.0, 1.5, 0.0),
                radius: 0.14,
            });
        }
    }

    let mut simulation = b.build(params);
    if kind == SceneKind::FastSdfCrossing {
        simulation.params.gravity = Vec3::ZERO;
        simulation.params.substeps = 1;
        simulation.params.max_particle_speed = 100.0;
        simulation.vel.fill(Vec3::NEG_Y * 100.0);
    }
    if kind == SceneKind::SelfCollisionLayers {
        simulation.params.self_collision = true;
        simulation.params.gravity = Vec3::ZERO;
    }
    if let Some((start, end)) = self_collision_velocity_range {
        simulation.params.self_collision = true;
        simulation.params.gravity = Vec3::ZERO;
        simulation.params.substeps = 1;
        simulation.params.iterations = 1;
        simulation.params.max_particle_speed = 10.0;
        simulation.vel[start..end].fill(Vec3::Y * 10.0);
    }
    simulation
}

/// SDF de referência fechado para exercitar o caminho de colisão de avatares importados.
/// O torso é uma caixa triangularizada de 1,4 × 0,8 × 0,7 m; não depende de assets externos.
fn avatar_torso_sdf() -> Arc<SignedDistanceField> {
    let positions = [
        [-0.7, 0.2, -0.35], [0.7, 0.2, -0.35], [0.7, 1.0, -0.35], [-0.7, 1.0, -0.35],
        [-0.7, 0.2, 0.35], [0.7, 0.2, 0.35], [0.7, 1.0, 0.35], [-0.7, 1.0, 0.35],
    ];
    let indices = [
        0, 2, 1, 0, 3, 2,
        4, 5, 6, 4, 6, 7,
        0, 1, 5, 0, 5, 4,
        3, 7, 6, 3, 6, 2,
        0, 4, 7, 0, 7, 3,
        1, 2, 6, 1, 6, 5,
    ];
    Arc::new(SignedDistanceField::from_triangle_mesh(&positions, &indices, 0.015)
        .expect("o torso interno deve ser uma malha fechada válida"))
}


#[cfg(test)]
mod self_collision_scene_tests {
    use super::*;

    #[test]
    fn layered_panels_exercise_surface_contacts_without_non_finite_positions() {
        let mut params = SimParams::default();
        params.substeps = 1;
        params.iterations = 4;
        let mut simulation = build_scene(SceneKind::SelfCollisionLayers, 12, params);
        simulation.step(1.0 / 60.0);
        assert!(simulation.is_finite());
        assert!(simulation.stats.self_collision_vt_candidates > 0);
        assert!(simulation.stats.self_collision_ee_candidates > 0);
    }
    #[test]
    fn sewn_panels_remain_finite_when_surface_self_collision_is_enabled() {
        let mut params = SimParams::default();
        params.substeps = 1;
        params.iterations = 2;
        let mut simulation = build_scene(SceneKind::SewnTube, 10, params);
        simulation.params.self_collision = true;
        simulation.step(1.0 / 60.0);
        assert!(simulation.is_finite());
        assert!(simulation.seam_gap_max().is_finite());
    }

    #[test]
    fn fast_layer_crossing_is_intercepted_and_remains_finite() {
        let mut params = SimParams::default();
        params.substeps = 1;
        params.iterations = 1;
        let mut simulation = build_scene(SceneKind::FastSelfCollisionCrossing, 12, params);
        simulation.step(1.0 / 60.0);
        assert!(simulation.is_finite());
        assert!(
            simulation.stats.self_collision_vt_swept_contacts
                + simulation.stats.self_collision_ee_swept_contacts > 0
        );
    }

}
