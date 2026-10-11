//! Cenas de teste da Fase 0: validam estabilidade, colisão e costura do solver.

use glam::{Quat, Vec3};

use crate::builder::ClothBuilder;
use crate::collider::Collider;
use crate::mesh::{GridPanel, Placement};
use crate::params::SimParams;
use crate::solver::Simulation;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum SceneKind {
    /// Pano horizontal preso por dois cantos: stretch, shear e flexão sob grande movimento.
    HangingCloth,
    /// Pano caindo sobre uma esfera: colisão e atrito.
    SphereDrop,
    /// Dois painéis costurados nas laterais ao redor de um cilindro: costuras + colisão.
    SewnTube,
}

impl SceneKind {
    pub const ALL: [SceneKind; 3] = [Self::HangingCloth, Self::SphereDrop, Self::SewnTube];

    pub fn name(self) -> &'static str {
        match self {
            Self::HangingCloth => "1. Pano pendurado",
            Self::SphereDrop => "2. Pano sobre esfera",
            Self::SewnTube => "3. Tubo costurado",
        }
    }
}

/// Constrói uma cena. `res` é o número de células por metro (≈ resolução da malha):
/// 40 → ~1,7k partículas, 100 → ~10k, 141 → ~20k (para o pano de 1 m).
pub fn build_scene(kind: SceneKind, res: u32, params: SimParams) -> Simulation {
    let res = res.max(8);
    let mut b = ClothBuilder::new();
    b.add_collider(Collider::ground());

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

    b.build(params)
}
