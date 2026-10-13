use glam::{Vec2, Vec3};
use std::sync::Arc;

use crate::sdf::SignedDistanceField;

/// Colisores analíticos descritos por campo de distância assinada (SDF).
/// Positivo = fora, negativo = dentro.
#[derive(Clone, Debug, PartialEq)]
pub enum Collider {
    /// Semiespaço: `dot(normal, p) - offset`. A `normal` deve estar normalizada.
    Plane { normal: Vec3, offset: f32 },
    Sphere { center: Vec3, radius: f32 },
    /// Elipsoide alinhado aos eixos; `radii` contém os semieixos X/Y/Z.
    Ellipsoid { center: Vec3, radii: Vec3 },
    /// Cápsula com extremidades arredondadas, ao longo do segmento a–b.
    Capsule { a: Vec3, b: Vec3, radius: f32 },
    /// Cilindro com tampas planas, do ponto `a` ao ponto `b`.
    Cylinder { a: Vec3, b: Vec3, radius: f32 },
    /// Colisão pela superfície de um avatar, amostrada num campo de distância em metros.
    TriangleMeshSdf { field: Arc<SignedDistanceField> },
}

impl Collider {
    /// Chão em y = 0.
    pub fn ground() -> Self {
        Collider::Plane { normal: Vec3::Y, offset: 0.0 }
    }

    pub fn sdf(&self, p: Vec3) -> f32 {
        match self {
            Collider::Plane { normal, offset } => normal.dot(p) - *offset,
            Collider::Sphere { center, radius } => (p - *center).length() - *radius,
            Collider::Ellipsoid { center, radii } => {
                // Aproximação de distância assinada de um elipsoide (positiva fora).
                let q = p - *center;
                let scaled = Vec3::new(q.x / radii.x, q.y / radii.y, q.z / radii.z);
                let k0 = scaled.length();
                let k1 = Vec3::new(
                    q.x / (radii.x * radii.x),
                    q.y / (radii.y * radii.y),
                    q.z / (radii.z * radii.z),
                ).length();
                if k1 < 1e-9 { -radii.min_element() } else { k0 * (k0 - 1.0) / k1 }
            }
            Collider::Capsule { a, b, radius } => {
                let ab = *b - *a;
                let t = if ab.length_squared() > 1e-12 {
                    ((p - *a).dot(ab) / ab.length_squared()).clamp(0.0, 1.0)
                } else { 0.0 };
                (p - (*a + ab * t)).length() - *radius
            }
            Collider::Cylinder { a, b, radius } => {
                let axis = *b - *a;
                let h = axis.length();
                if h < 1e-9 {
                    return (p - *a).length() - *radius;
                }
                let u = axis / h;
                let pa = p - *a;
                let t = pa.dot(u);
                let radial = (pa - u * t).length();
                let dx = radial - *radius;
                let dy = (t - 0.5 * h).abs() - 0.5 * h;
                let outside = Vec2::new(dx.max(0.0), dy.max(0.0)).length();
                outside + dx.max(dy).min(0.0)
            }
            Collider::TriangleMeshSdf { field } => field.sample(p),
        }
    }

    /// Normal da superfície (gradiente do SDF), apontando para fora.
    pub fn normal(&self, p: Vec3) -> Vec3 {
        match self {
            Collider::Plane { normal, .. } => *normal,
            Collider::Sphere { center, .. } => (p - *center).try_normalize().unwrap_or(Vec3::Y),
            Collider::Ellipsoid { center, radii } => {
                let q = p - *center;
                Vec3::new(q.x / (radii.x * radii.x), q.y / (radii.y * radii.y), q.z / (radii.z * radii.z))
                    .try_normalize().unwrap_or(Vec3::Y)
            }
            Collider::Capsule { a, b, .. } => {
                let ab = *b - *a;
                let t = if ab.length_squared() > 1e-12 {
                    ((p - *a).dot(ab) / ab.length_squared()).clamp(0.0, 1.0)
                } else { 0.0 };
                (p - (*a + ab * t)).try_normalize().unwrap_or(Vec3::X)
            }
            Collider::Cylinder { a, b, .. } => {
                const E: f32 = 1e-3;
                let g = Vec3::new(
                    self.sdf(p + Vec3::X * E) - self.sdf(p - Vec3::X * E),
                    self.sdf(p + Vec3::Y * E) - self.sdf(p - Vec3::Y * E),
                    self.sdf(p + Vec3::Z * E) - self.sdf(p - Vec3::Z * E),
                );
                if let Some(normal) = g.try_normalize() {
                    return normal;
                }

                // No eixo do cilindro, a derivada radial numérica é zero por simetria.
                // Usar +Y como fallback empurrava o tecido para cima, injetando energia.
                let axis = *b - *a;
                let length = axis.length();
                if length < 1e-9 {
                    return (p - *a).try_normalize().unwrap_or(Vec3::X);
                }
                let axis_dir = axis / length;
                let axial_position = (p - *a).dot(axis_dir);
                if axial_position < 0.0 {
                    return -axis_dir;
                }
                if axial_position > length {
                    return axis_dir;
                }

                let helper = if axis_dir.x.abs() < 0.9 { Vec3::X } else { Vec3::Z };
                axis_dir.cross(helper).try_normalize().unwrap_or(Vec3::Z)
            }
            Collider::TriangleMeshSdf { field } => field.gradient(p),
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
    fn cylinder_normal_on_axis_is_radial_not_axial() {
        let cylinder = Collider::Cylinder {
            a: Vec3::ZERO,
            b: Vec3::new(0.0, 1.5, 0.0),
            radius: 0.14,
        };
        // A derivada do SDF é zero no eixo, então o fallback deve apontar radialmente,
        // nunca para cima: a normal +Y introduzia velocidade vertical enorme em uma colisão.
        let normal = cylinder.normal(Vec3::new(0.0, 0.75, 0.0));
        assert!(normal.is_finite());
        assert!((normal.length() - 1.0).abs() < 1e-5);
        assert!(normal.dot(Vec3::Y).abs() < 1e-5, "normal inesperadamente axial: {normal:?}");
    }

    #[test]
    fn ellipsoid_and_capsule_sdf_and_normals() {
        let e = Collider::Ellipsoid { center: Vec3::ZERO, radii: Vec3::new(0.2, 0.4, 0.1) };
        assert!(e.sdf(Vec3::ZERO) < 0.0);
        assert!(e.sdf(Vec3::new(0.4, 0.0, 0.0)) > 0.0);
        assert!(e.sdf(Vec3::new(0.2, 0.0, 0.0)).abs() < 1e-5);
        assert!(e.normal(Vec3::new(0.3, 0.0, 0.0)).dot(Vec3::X) > 0.99);
        let c = Collider::Capsule { a: Vec3::ZERO, b: Vec3::new(0.0, 1.0, 0.0), radius: 0.1 };
        assert!((c.sdf(Vec3::new(0.0, 0.5, 0.0)) + 0.1).abs() < 1e-6);
        assert!((c.sdf(Vec3::new(0.0, 1.2, 0.0)) - 0.1).abs() < 1e-6);
        assert!(c.normal(Vec3::new(0.1, 0.5, 0.0)).dot(Vec3::X) > 0.99);
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
