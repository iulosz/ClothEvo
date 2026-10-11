use glam::{Vec2, Vec3};

/// Colisores analíticos descritos por campo de distância assinada (SDF).
/// Positivo = fora, negativo = dentro.
#[derive(Clone, Debug, PartialEq)]
pub enum Collider {
    /// Semiespaço: `dot(normal, p) - offset`. A `normal` deve estar normalizada.
    Plane { normal: Vec3, offset: f32 },
    Sphere { center: Vec3, radius: f32 },
    /// Cilindro com tampas planas, do ponto `a` ao ponto `b`.
    Cylinder { a: Vec3, b: Vec3, radius: f32 },
}

impl Collider {
    /// Chão em y = 0.
    pub fn ground() -> Self {
        Collider::Plane { normal: Vec3::Y, offset: 0.0 }
    }

    pub fn sdf(&self, p: Vec3) -> f32 {
        match *self {
            Collider::Plane { normal, offset } => normal.dot(p) - offset,
            Collider::Sphere { center, radius } => (p - center).length() - radius,
            Collider::Cylinder { a, b, radius } => {
                let axis = b - a;
                let h = axis.length();
                if h < 1e-9 {
                    return (p - a).length() - radius;
                }
                let u = axis / h;
                let pa = p - a;
                let t = pa.dot(u);
                let radial = (pa - u * t).length();
                let dx = radial - radius;
                let dy = (t - 0.5 * h).abs() - 0.5 * h;
                let outside = Vec2::new(dx.max(0.0), dy.max(0.0)).length();
                outside + dx.max(dy).min(0.0)
            }
        }
    }

    /// Normal da superfície (gradiente do SDF), apontando para fora.
    pub fn normal(&self, p: Vec3) -> Vec3 {
        match *self {
            Collider::Plane { normal, .. } => normal,
            Collider::Sphere { center, .. } => (p - center).try_normalize().unwrap_or(Vec3::Y),
            Collider::Cylinder { .. } => {
                const E: f32 = 1e-3;
                let g = Vec3::new(
                    self.sdf(p + Vec3::X * E) - self.sdf(p - Vec3::X * E),
                    self.sdf(p + Vec3::Y * E) - self.sdf(p - Vec3::Y * E),
                    self.sdf(p + Vec3::Z * E) - self.sdf(p - Vec3::Z * E),
                );
                g.try_normalize().unwrap_or(Vec3::Y)
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn sphere_sdf_and_normal() {
        let s = Collider::Sphere { center: Vec3::ZERO, radius: 1.0 };
        assert!((s.sdf(Vec3::new(2.0, 0.0, 0.0)) - 1.0).abs() < 1e-6);
        assert!(s.sdf(Vec3::new(0.5, 0.0, 0.0)) < 0.0);
        assert!((s.normal(Vec3::new(0.0, 3.0, 0.0)) - Vec3::Y).length() < 1e-6);
    }

    #[test]
    fn cylinder_sdf_regions() {
        let c = Collider::Cylinder { a: Vec3::ZERO, b: Vec3::new(0.0, 2.0, 0.0), radius: 0.5 };
        // lado, tampa e interior
        assert!((c.sdf(Vec3::new(1.0, 1.0, 0.0)) - 0.5).abs() < 1e-5);
        assert!((c.sdf(Vec3::new(0.0, 3.0, 0.0)) - 1.0).abs() < 1e-5);
        assert!(c.sdf(Vec3::new(0.0, 1.0, 0.0)) < 0.0);
        // normal lateral aponta para fora
        let n = c.normal(Vec3::new(0.6, 1.0, 0.0));
        assert!(n.dot(Vec3::X) > 0.99);
    }
}
