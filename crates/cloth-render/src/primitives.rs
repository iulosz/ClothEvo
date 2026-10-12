//! Geradores de malhas simples para os colisores e o chão.

use glam::Vec3;

use crate::{MeshKind, StaticMeshData, Vertex};

fn v(p: Vec3, n: Vec3) -> Vertex {
    Vertex { pos: p.to_array(), normal: n.to_array() }
}

/// Esfera UV com normais suaves.
pub fn uv_sphere(center: Vec3, radius: f32, segments: u32, rings: u32, color: [f32; 3]) -> StaticMeshData {
    let (segments, rings) = (segments.max(3), rings.max(2));
    let mut vertices = Vec::new();
    let mut indices = Vec::new();
    for r in 0..=rings {
        let phi = std::f32::consts::PI * r as f32 / rings as f32;
        for s in 0..=segments {
            let theta = std::f32::consts::TAU * s as f32 / segments as f32;
            let n = Vec3::new(phi.sin() * theta.cos(), phi.cos(), phi.sin() * theta.sin());
            vertices.push(v(center + n * radius, n));
        }
    }
    let stride = segments + 1;
    for r in 0..rings {
        for s in 0..segments {
            let a = r * stride + s;
            let b = a + stride;
            // sentido anti-horário visto de fora
            indices.extend_from_slice(&[a, a + 1, b, a + 1, b + 1, b]);
        }
    }
    StaticMeshData { vertices, indices, color, kind: MeshKind::Shaded }
}

/// Elipsoide suave alinhado aos eixos, usado para exibir o mesmo volume do colisor SDF.
pub fn ellipsoid(center: Vec3, radii: Vec3, segments: u32, rings: u32, color: [f32; 3]) -> StaticMeshData {
    let mut mesh = uv_sphere(Vec3::ZERO, 1.0, segments, rings, color);
    for vertex in &mut mesh.vertices {
        let unit = Vec3::from(vertex.pos);
        let n = Vec3::from(vertex.normal);
        let normal = Vec3::new(n.x / radii.x.max(1e-6), n.y / radii.y.max(1e-6), n.z / radii.z.max(1e-6)).normalize_or_zero();
        vertex.pos = (center + Vec3::new(unit.x * radii.x, unit.y * radii.y, unit.z * radii.z)).to_array();
        vertex.normal = if normal.length_squared() > 1e-12 { normal.to_array() } else { Vec3::Y.to_array() };
    }
    mesh
}

/// Cilindro com tampas planas, do ponto `a` ao ponto `b`.
pub fn cylinder(a: Vec3, b: Vec3, radius: f32, segments: u32, color: [f32; 3]) -> StaticMeshData {
    let segments = segments.max(3);
    let axis = (b - a).try_normalize().unwrap_or(Vec3::Y);
    let helper = if axis.y.abs() < 0.9 { Vec3::Y } else { Vec3::X };
    let u = axis.cross(helper).normalize();
    let w = axis.cross(u);

    let mut vertices = Vec::new();
    let mut indices = Vec::new();

    // Lateral
    for s in 0..=segments {
        let t = std::f32::consts::TAU * s as f32 / segments as f32;
        let n = u * t.cos() + w * t.sin();
        vertices.push(v(a + n * radius, n));
        vertices.push(v(b + n * radius, n));
    }
    for s in 0..segments {
        let i = s * 2;
        indices.extend_from_slice(&[i, i + 2, i + 1, i + 1, i + 2, i + 3]);
    }

    // Tampas
    for (center, normal, flip) in [(a, -axis, true), (b, axis, false)] {
        let base = vertices.len() as u32;
        vertices.push(v(center, normal));
        for s in 0..=segments {
            let t = std::f32::consts::TAU * s as f32 / segments as f32;
            vertices.push(v(center + (u * t.cos() + w * t.sin()) * radius, normal));
        }
        for s in 0..segments {
            let (p, q) = (base + 1 + s, base + 2 + s);
            if flip {
                indices.extend_from_slice(&[base, q, p]);
            } else {
                indices.extend_from_slice(&[base, p, q]);
            }
        }
    }
    StaticMeshData { vertices, indices, color, kind: MeshKind::Shaded }
}

/// Quadrado horizontal em `y`, de lado `2 * half`, com a grade desenhada pelo shader.
pub fn ground_quad(y: f32, half: f32, color: [f32; 3]) -> StaticMeshData {
    let n = Vec3::Y;
    let vertices = vec![
        v(Vec3::new(-half, y, -half), n),
        v(Vec3::new(half, y, -half), n),
        v(Vec3::new(half, y, half), n),
        v(Vec3::new(-half, y, half), n),
    ];
    // anti-horário visto de cima (+Y)
    let indices = vec![0, 2, 1, 0, 3, 2];
    StaticMeshData { vertices, indices, color, kind: MeshKind::Ground }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn outward_triangles(m: &StaticMeshData, inside: Vec3) {
        for t in m.indices.chunks(3) {
            let p = |i: u32| Vec3::from(m.vertices[i as usize].pos);
            let (a, b, c) = (p(t[0]), p(t[1]), p(t[2]));
            let n = (b - a).cross(c - a);
            if n.length_squared() < 1e-12 {
                continue; // triângulo degenerado nos polos
            }
            assert!(n.dot((a + b + c) / 3.0 - inside) > 0.0, "triângulo voltado para dentro");
        }
    }

    #[test]
    fn sphere_triangles_face_outward() {
        let m = uv_sphere(Vec3::new(0.0, 0.4, 0.0), 0.4, 24, 12, [1.0; 3]);
        outward_triangles(&m, Vec3::new(0.0, 0.4, 0.0));
    }

    #[test]
    fn cylinder_triangles_face_outward() {
        let m = cylinder(Vec3::ZERO, Vec3::new(0.0, 1.5, 0.0), 0.14, 24, [1.0; 3]);
        outward_triangles(&m, Vec3::new(0.0, 0.75, 0.0));
    }

    #[test]
    fn ground_faces_up() {
        let m = ground_quad(0.0, 4.0, [1.0; 3]);
        let p = |i: u32| Vec3::from(m.vertices[i as usize].pos);
        let n = (p(m.indices[1]) - p(m.indices[0])).cross(p(m.indices[2]) - p(m.indices[0]));
        assert!(n.y > 0.0);
    }
}
