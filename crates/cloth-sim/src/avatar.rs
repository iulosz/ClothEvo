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

/// Cria 18 volumes de colisão: cabeça, pescoço, tronco, braços em T-pose,
/// mãos, coxas, canelas e pés. A mesma lista alimenta a física e a visualização.
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

    // Braços estendidos horizontalmente em T-pose, sem tampas rígidas nas juntas.
    for sign in [-1.0_f32, 1.0_f32] {
        let shoulder = Vec3::new(sign * shoulders * 0.43, h * 0.785, 0.0);
        let elbow = Vec3::new(sign * shoulders * 1.08, h * 0.785, 0.0);
        let wrist = Vec3::new(sign * shoulders * 1.72, h * 0.780, 0.015 * scale);
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

/// Ponto de posicionamento no corpo, para arrumar moldes em volta do avatar.
/// `anchor` fica dentro do volume do corpo (centro da região) e `normal` aponta para fora:
/// +Z é a frente do avatar. Convenção: o avatar olha para +Z e "esquerdo" é o lado +X.
#[derive(Clone, Debug, PartialEq)]
pub struct ArrangementPoint {
    pub id: String,
    pub name: String,
    pub anchor: Vec3,
    pub normal: Vec3,
}

fn push_front_back(out: &mut Vec<ArrangementPoint>, id: &str, name: &str, anchor: Vec3) {
    out.push(ArrangementPoint {
        id: format!("{id}_frente"),
        name: format!("{name} · frente"),
        anchor,
        normal: Vec3::Z,
    });
    out.push(ArrangementPoint {
        id: format!("{id}_costas"),
        name: format!("{name} · costas"),
        anchor,
        normal: Vec3::NEG_Z,
    });
}

/// Pontos de posicionamento derivados das mesmas medidas que geram os colisores analíticos:
/// pescoço, peito, cintura e quadril (frente/costas) e braços, antebraços, coxas e canelas
/// (esquerdo/direito, frente/costas).
pub fn arrangement_points(config: &AvatarConfig) -> Vec<ArrangementPoint> {
    let h = config.height_m;
    let scale = h / 1.72;
    let shoulders = config.shoulder_width_m;
    let hips = config.hip_width_m;
    let mut out = Vec::new();

    push_front_back(&mut out, "pescoco", "Pescoço", Vec3::new(0.0, h * 0.855, 0.0));
    push_front_back(&mut out, "tronco_peito", "Peito", Vec3::new(0.0, h * 0.725, 0.0));
    push_front_back(&mut out, "tronco_cintura", "Cintura", Vec3::new(0.0, h * 0.625, 0.0));
    push_front_back(&mut out, "quadril", "Quadril", Vec3::new(0.0, h * 0.535, 0.0));

    for (sign, side_id, side_name) in [(1.0_f32, "esq", "esquerdo"), (-1.0_f32, "dir", "direito")] {
        let shoulder = Vec3::new(sign * shoulders * 0.43, h * 0.785, 0.0);
        let elbow = Vec3::new(sign * shoulders * 1.08, h * 0.785, 0.0);
        let wrist = Vec3::new(sign * shoulders * 1.72, h * 0.780, 0.015 * scale);
        push_front_back(&mut out, &format!("braco_{side_id}"), &format!("Braço {side_name}"),
            (shoulder + elbow) * 0.5);
        push_front_back(&mut out, &format!("antebraco_{side_id}"), &format!("Antebraço {side_name}"),
            (elbow + wrist) * 0.5);

        let x = sign * hips * 0.245;
        let hip = Vec3::new(x, h * 0.535, 0.0);
        let knee = Vec3::new(x * 1.02, h * 0.305, 0.005 * scale);
        let ankle = Vec3::new(x * 1.04, h * 0.075, 0.0);
        push_front_back(&mut out, &format!("coxa_{side_id}"), &format!("Coxa {side_name}"),
            (hip + knee) * 0.5);
        push_front_back(&mut out, &format!("canela_{side_id}"), &format!("Canela {side_name}"),
            (knee + ankle) * 0.5);
    }
    out
}

/// Posição na superfície do corpo para o centro de um molde: caminha do `anchor` ao longo da
/// `normal` em passos de 5 mm até sair de todos os colisores com folga `clearance` (teto de 1 m).
/// Funciona com colisores analíticos e com o SDF de uma malha importada.
pub fn arrangement_surface_position(
    colliders: &[Collider],
    point: &ArrangementPoint,
    clearance: f32,
) -> Vec3 {
    const STEP: f32 = 0.005;
    const MAX_DISTANCE: f32 = 1.0;
    let direction = point.normal.normalize_or_zero();
    let mut distance = 0.0_f32;
    while distance < MAX_DISTANCE {
        let position = point.anchor + direction * distance;
        if colliders.iter().all(|collider| collider.sdf(position) >= clearance) {
            return position;
        }
        distance += STEP;
    }
    point.anchor + direction * MAX_DISTANCE
}

/// Profundidade frontal do corpo num ponto (x, y), para um molde plano voltado para +Z.
/// Caminha de z = 1 m para trás, em passos de 5 mm, até encontrar a superfície com a folga pedida.
/// Retorna `None` se não encontrar corpo naquela posição, por exemplo fora da silhueta.
pub fn front_surface_z(colliders: &[Collider], x: f32, y: f32, clearance: f32) -> Option<f32> {
    const STEP: f32 = 0.005;
    const START_Z: f32 = 1.0;
    const END_Z: f32 = -0.6;
    let mut z = START_Z;
    while z > END_Z {
        let position = Vec3::new(x, y, z);
        if colliders.iter().any(|collider| collider.sdf(position) < clearance) {
            return Some(z + STEP);
        }
        z -= STEP;
    }
    None
}

/// Menor raio r >= min_radius para que o centro de uma faixa do molde fique fora
/// dos colisores em alturas amostradas. É uma estimativa por linha central: não inclui
/// a largura tangencial do molde e limita a busca a 1,5 m.
pub fn clear_shell_radius(
    colliders: &[Collider],
    angle: f32,
    y_min: f32,
    y_max: f32,
    clearance: f32,
    min_radius: f32,
) -> f32 {
    const SAMPLES: usize = 9;
    const STEP: f32 = 0.005;
    const MAX_RADIUS: f32 = 1.5;
    let (sin_a, cos_a) = angle.sin_cos();
    let (lo, hi) = if y_min <= y_max { (y_min, y_max) } else { (y_max, y_min) };
    let start = min_radius.max(0.0);
    let mut radius = start;

    for k in 0..SAMPLES {
        let t = k as f32 / (SAMPLES - 1) as f32;
        let y = lo + (hi - lo) * t;
        let mut r = start;
        while r < MAX_RADIUS {
            let point = Vec3::new(sin_a * r, y, cos_a * r);
            if colliders.iter().all(|collider| collider.sdf(point) >= clearance) {
                break;
            }
            r += STEP;
        }
        radius = radius.max(r);
    }
    radius
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

    #[test]
    fn clear_shell_radius_leaves_sphere_and_keeps_minimum_when_clear() {
        let sphere = vec![Collider::Sphere { center: Vec3::new(0.0, 1.0, 0.0), radius: 0.2 }];
        let radius = clear_shell_radius(&sphere, 0.0, 0.9, 1.1, 0.0, 0.14);
        assert!(radius >= 0.2 && radius < 0.21, "raio fora do esperado: {radius}");
        let far = clear_shell_radius(&sphere, 0.0, 2.0, 2.2, 0.0, 0.14);
        assert!((far - 0.14).abs() < 1e-6);
    }

    #[test]
    fn clear_shell_radius_is_finite_and_not_below_minimum_for_default_avatar() {
        let colliders = build_avatar_colliders(&AvatarConfig::default()).unwrap();
        for angle in [-1.5_f32, 0.0, 1.5] {
            let radius = clear_shell_radius(&colliders, angle, 0.9, 1.6, 0.0, 0.14);
            assert!(radius.is_finite() && radius >= 0.14);
        }
    }
}

#[cfg(test)]
mod arrangement_tests {
    use super::*;

    #[test]
    fn points_cover_front_and_back_with_unique_ids() {
        let points = arrangement_points(&AvatarConfig::default());
        assert!(points.len() >= 24, "pontos: {}", points.len());
        let mut ids: Vec<_> = points.iter().map(|point| point.id.clone()).collect();
        ids.sort();
        ids.dedup();
        assert_eq!(ids.len(), points.len(), "ids devem ser únicos");
        assert!(points.iter().any(|point| point.id == "tronco_peito_costas" && point.normal == Vec3::NEG_Z));
        assert!(points.iter().any(|point| point.id == "tronco_peito_frente" && point.normal == Vec3::Z));
    }

    #[test]
    fn surface_position_leaves_the_body_along_the_normal_at_the_anchor_height() {
        let config = AvatarConfig::default();
        let colliders = build_avatar_colliders(&config).unwrap();
        for point in arrangement_points(&config) {
            let position = arrangement_surface_position(&colliders, &point, 0.002);
            assert!(
                colliders.iter().all(|collider| collider.sdf(position) >= 0.002 - 1.0e-4),
                "{} ficou dentro do corpo", point.id
            );
            assert!((position - point.anchor).dot(point.normal) > 0.0, "{} não saiu do corpo", point.id);
            assert!((position.y - point.anchor.y).abs() < 1.0e-5 && (position.x - point.anchor.x).abs() < 1.0e-5);
        }
    }
}

#[cfg(test)]
mod front_projection_tests {
    use super::*;

    #[test]
    fn front_surface_z_finds_the_chest_front_and_ignores_empty_space() {
        let config = AvatarConfig::default();
        let colliders = build_avatar_colliders(&config).unwrap();
        let chest_y = config.height_m * 0.725;
        let z = front_surface_z(&colliders, 0.0, chest_y, 0.0).expect("deve haver corpo no peito");
        assert!(z > 0.10 && z < 0.20, "z do peito fora do esperado: {z}");
        let with_clearance = front_surface_z(&colliders, 0.0, chest_y, 0.02).unwrap();
        assert!(with_clearance > z, "a folga deve afastar o painel do corpo");
        assert!(front_surface_z(&colliders, 2.0, chest_y, 0.0).is_none());
        assert!(front_surface_z(&colliders, 0.0, config.height_m + 0.8, 0.0).is_none());
    }

    #[test]
    fn front_surface_z_does_not_depend_on_other_positions() {
        let config = AvatarConfig::default();
        let colliders = build_avatar_colliders(&config).unwrap();
        let first = front_surface_z(&colliders, 0.1, 1.2, 0.002);
        let _other = front_surface_z(&colliders, -0.1, 1.0, 0.002);
        assert_eq!(first, front_surface_z(&colliders, 0.1, 1.2, 0.002));
    }
}
