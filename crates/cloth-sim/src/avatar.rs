//! Manequim procedural composto por colisores analíticos para a simulação de roupas.
use glam::Vec3;

use crate::collider::Collider;

/// Medidas básicas do manequim, em metros.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct AvatarConfig {
    pub height_m: f32,
    pub shoulder_width_m: f32,
    pub waist_width_m: f32,
    pub hip_width_m: f32,
}

impl Default for AvatarConfig {
    fn default() -> Self {
        Self { height_m: 1.72, shoulder_width_m: 0.42, waist_width_m: 0.29, hip_width_m: 0.36 }
    }
}

impl AvatarConfig {
    pub fn validate(&self) -> Result<(), String> {
        let values = [self.height_m, self.shoulder_width_m, self.waist_width_m, self.hip_width_m];
        if values.iter().any(|v| !v.is_finite()) {
            return Err("As medidas do avatar precisam ser números finitos.".into());
        }
        if !(1.0..=2.5).contains(&self.height_m) {
            return Err("A altura do avatar deve ficar entre 1,00 m e 2,50 m.".into());
        }
        if !(0.20..=0.90).contains(&self.shoulder_width_m) {
            return Err("A largura dos ombros deve ficar entre 0,20 m e 0,90 m.".into());
        }
        if !(0.15..=0.70).contains(&self.waist_width_m) {
            return Err("A largura da cintura deve ficar entre 0,15 m e 0,70 m.".into());
        }
        if !(0.18..=0.80).contains(&self.hip_width_m) {
            return Err("A largura do quadril deve ficar entre 0,18 m e 0,80 m.".into());
        }
        Ok(())
    }
}

fn ellipsoid(center: Vec3, radii: Vec3) -> Collider {
    Collider::Ellipsoid { center, radii }
}

fn capsule(a: Vec3, b: Vec3, radius: f32) -> Collider {
    Collider::Capsule { a, b, radius }
}

/// Cria 18 volumes de colisão: cabeça, pescoço, tronco, braços, mãos,
/// coxas, canelas e pés. A mesma lista alimenta a física e a visualização.
pub fn build_avatar_colliders(config: &AvatarConfig) -> Result<Vec<Collider>, String> {
    config.validate()?;
    let h = config.height_m;
    let scale = h / 1.72;
    let shoulders = config.shoulder_width_m;
    let waist = config.waist_width_m;
    let hips = config.hip_width_m;
    let mut out = Vec::with_capacity(18);

    out.push(ellipsoid(Vec3::new(0.0, h * 0.535, 0.0),
        Vec3::new(hips * 0.50, h * 0.095, hips * 0.31))); // pelve
    out.push(ellipsoid(Vec3::new(0.0, h * 0.625, 0.0),
        Vec3::new(waist * 0.46, h * 0.105, waist * 0.31))); // cintura/abdômen
    out.push(ellipsoid(Vec3::new(0.0, h * 0.725, 0.0),
        Vec3::new(shoulders * 0.47, h * 0.12, shoulders * 0.30))); // tórax
    out.push(ellipsoid(Vec3::new(0.0, h * 0.795, 0.0),
        Vec3::new(shoulders * 0.50, h * 0.045, shoulders * 0.29))); // ombros
    out.push(ellipsoid(Vec3::new(0.0, h * 0.855, 0.0),
        Vec3::new(0.055 * scale, h * 0.055, 0.055 * scale))); // pescoço
    out.push(ellipsoid(Vec3::new(0.0, h * 0.94, 0.0),
        Vec3::new(0.087 * scale, h * 0.06, 0.094 * scale))); // cabeça

    // Braços e antebraços inclinados para baixo, sem tampas rígidas nas juntas.
    for sign in [-1.0_f32, 1.0_f32] {
        let shoulder = Vec3::new(sign * shoulders * 0.43, h * 0.785, 0.0);
        let elbow = Vec3::new(sign * shoulders * 0.66, h * 0.625, 0.0);
        let wrist = Vec3::new(sign * shoulders * 0.76, h * 0.445, 0.015 * scale);
        out.push(capsule(shoulder, elbow, 0.052 * scale));
        out.push(capsule(elbow, wrist, 0.038 * scale));
        out.push(ellipsoid(wrist, Vec3::new(0.035 * scale, 0.043 * scale, 0.032 * scale)));
    }

    // Pernas separadas na pelve, com coxas e canelas de raio independente.
    for sign in [-1.0_f32, 1.0_f32] {
        let x = sign * hips * 0.245;
        let hip = Vec3::new(x, h * 0.535, 0.0);
        let knee = Vec3::new(x * 1.02, h * 0.305, 0.005 * scale);
        let ankle = Vec3::new(x * 1.04, h * 0.075, 0.0);
        out.push(capsule(hip, knee, 0.074 * scale));
        out.push(capsule(knee, ankle, 0.047 * scale));
        out.push(ellipsoid(Vec3::new(x * 1.04, 0.022 * scale, 0.066 * scale),
            Vec3::new(0.054 * scale, 0.022 * scale, 0.12 * scale)));
    }
    Ok(out)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn default_avatar_has_body_colliders_for_all_major_regions() {
        let colliders = build_avatar_colliders(&AvatarConfig::default()).unwrap();
        assert_eq!(colliders.len(), 18);
        assert!(colliders.iter().all(|c| c.sdf(Vec3::new(0.0, 0.9, 0.0)).is_finite()));
        assert!(colliders.iter().any(|c| c.sdf(Vec3::new(0.0, 1.55, 0.0)) < 0.0));
        // Topo da cabeça e sola do pé coincidem com as medidas nominais e o chão.
        assert!(colliders[5].sdf(Vec3::new(0.0, AvatarConfig::default().height_m, 0.0)).abs() < 1e-5);
        assert!(colliders[17].sdf(Vec3::new(AvatarConfig::default().hip_width_m * 0.245 * 1.04, 0.0, 0.066)).abs() < 0.01);
        assert!(colliders.iter().any(|c| c.sdf(Vec3::new(0.0, 0.9, 0.0)) < 0.0));
        assert!(colliders.iter().any(|c| c.sdf(Vec3::new(0.09, 0.9, 0.0)) > 0.0));
    }

    #[test]
    fn shoulder_and_hip_measurements_change_collision_shape() {
        let narrow = build_avatar_colliders(&AvatarConfig { shoulder_width_m: 0.34, ..AvatarConfig::default() }).unwrap();
        let wide = build_avatar_colliders(&AvatarConfig { shoulder_width_m: 0.60, ..AvatarConfig::default() }).unwrap();
        assert!(narrow[2].sdf(Vec3::new(0.20, 1.72 * 0.725, 0.0)) > 0.0);
        assert!(wide[2].sdf(Vec3::new(0.20, 1.72 * 0.725, 0.0)) < 0.0);
        let narrow_hips = build_avatar_colliders(&AvatarConfig { hip_width_m: 0.28, ..AvatarConfig::default() }).unwrap();
        let wide_hips = build_avatar_colliders(&AvatarConfig { hip_width_m: 0.52, ..AvatarConfig::default() }).unwrap();
        assert!(narrow_hips[0].sdf(Vec3::new(0.22, 1.72 * 0.535, 0.0)) > 0.0);
        assert!(wide_hips[0].sdf(Vec3::new(0.22, 1.72 * 0.535, 0.0)) < 0.0);
    }

    #[test]
    fn avatar_rejects_invalid_dimensions() {
        let bad = AvatarConfig { height_m: f32::NAN, ..AvatarConfig::default() };
        assert!(build_avatar_colliders(&bad).is_err());
        let too_wide = AvatarConfig { shoulder_width_m: 1.5, ..AvatarConfig::default() };
        assert!(build_avatar_colliders(&too_wide).is_err());
    }
}
