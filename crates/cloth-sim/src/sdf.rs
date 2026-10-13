//! Campo de distância assinado (SDF) estático gerado de uma malha triangular.
//! O BVH acelera a distância ao triângulo mais próximo e a classificação de interior/exterior.

use glam::Vec3;

const LEAF_TRIANGLES: usize = 8;
const MAX_GRID_CELLS: usize = 2_500_000;

#[derive(Clone, Debug, PartialEq)]
pub struct SignedDistanceField {
    origin: Vec3,
    spacing: f32,
    dims: [usize; 3],
    values: Vec<f32>,
}

#[derive(Clone, Copy, Debug)]
struct Aabb {
    min: Vec3,
    max: Vec3,
}

impl Aabb {
    fn empty() -> Self {
        Self { min: Vec3::splat(f32::INFINITY), max: Vec3::splat(f32::NEG_INFINITY) }
    }

    fn include(&mut self, p: Vec3) {
        self.min = self.min.min(p);
        self.max = self.max.max(p);
    }

    fn distance_squared(self, p: Vec3) -> f32 {
        let q = p.max(self.min).min(self.max);
        (p - q).length_squared()
    }

    fn extent(self) -> Vec3 { self.max - self.min }

    fn longest_axis(self) -> usize {
        let e = self.extent();
        if e.x >= e.y && e.x >= e.z { 0 }
        else if e.y >= e.z { 1 } else { 2 }
    }

    fn intersects_ray(self, origin: Vec3, direction: Vec3) -> bool {
        let mut t_min = 0.0_f32;
        let mut t_max = f32::INFINITY;
        for axis in 0..3 {
            let o = origin[axis];
            let d = direction[axis];
            if d.abs() < 1.0e-9 {
                if o < self.min[axis] || o > self.max[axis] { return false; }
                continue;
            }
            let inv = 1.0 / d;
            let mut a = (self.min[axis] - o) * inv;
            let mut b = (self.max[axis] - o) * inv;
            if a > b { std::mem::swap(&mut a, &mut b); }
            t_min = t_min.max(a);
            t_max = t_max.min(b);
            if t_max < t_min { return false; }
        }
        t_max > 1.0e-7
    }
}

#[derive(Clone, Copy, Debug)]
struct Triangle {
    a: Vec3,
    b: Vec3,
    c: Vec3,
    bounds: Aabb,
    centroid: Vec3,
}

impl Triangle {
    fn new(a: Vec3, b: Vec3, c: Vec3) -> Self {
        let mut bounds = Aabb::empty();
        bounds.include(a);
        bounds.include(b);
        bounds.include(c);
        Self { a, b, c, bounds, centroid: (a + b + c) / 3.0 }
    }

    fn distance_squared(self, p: Vec3) -> f32 {
        point_triangle_distance_squared(p, self.a, self.b, self.c)
    }

    fn ray_hit(self, origin: Vec3, direction: Vec3) -> bool {
        let e1 = self.b - self.a;
        let e2 = self.c - self.a;
        let h = direction.cross(e2);
        let determinant = e1.dot(h);
        if determinant.abs() < 1.0e-9 { return false; }
        let inv_det = 1.0 / determinant;
        let s = origin - self.a;
        let u = s.dot(h) * inv_det;
        if u < -1.0e-7 || u > 1.0 + 1.0e-7 { return false; }
        let q = s.cross(e1);
        let v = direction.dot(q) * inv_det;
        if v < -1.0e-7 || u + v > 1.0 + 1.0e-7 { return false; }
        let t = e2.dot(q) * inv_det;
        t > 1.0e-6
    }
}

#[derive(Clone, Copy, Debug)]
struct BvhNode {
    bounds: Aabb,
    left: usize,
    right: usize,
    start: usize,
    count: usize,
}

impl BvhNode {
    fn placeholder() -> Self {
        Self {
            bounds: Aabb::empty(),
            left: usize::MAX,
            right: usize::MAX,
            start: 0,
            count: 0,
        }
    }

    fn is_leaf(self) -> bool { self.count > 0 }
}

struct TriangleBvh {
    triangles: Vec<Triangle>,
    nodes: Vec<BvhNode>,
    root: usize,
}

impl TriangleBvh {
    fn new(triangles: Vec<Triangle>) -> Result<Self, String> {
        if triangles.is_empty() { return Err("A malha de colisão não contém triângulos válidos.".into()); }
        let mut this = Self { triangles, nodes: Vec::new(), root: 0 };
        let len = this.triangles.len();
        this.root = this.build_node(0, len);
        Ok(this)
    }

    fn build_node(&mut self, start: usize, end: usize) -> usize {
        let mut bounds = Aabb::empty();
        let mut centroid_bounds = Aabb::empty();
        for tri in &self.triangles[start..end] {
            bounds.include(tri.bounds.min);
            bounds.include(tri.bounds.max);
            centroid_bounds.include(tri.centroid);
        }

        let index = self.nodes.len();
        self.nodes.push(BvhNode::placeholder());
        let count = end - start;
        if count <= LEAF_TRIANGLES {
            self.nodes[index] = BvhNode { bounds, left: usize::MAX, right: usize::MAX, start, count };
            return index;
        }

        let axis = centroid_bounds.longest_axis();
        self.triangles[start..end].sort_unstable_by(|a, b| {
            a.centroid[axis].total_cmp(&b.centroid[axis])
        });
        let middle = start + count / 2;
        let left = self.build_node(start, middle);
        let right = self.build_node(middle, end);
        self.nodes[index] = BvhNode { bounds, left, right, start: 0, count: 0 };
        index
    }

    fn nearest_distance_squared(&self, node_index: usize, p: Vec3, best: &mut f32) {
        let node = self.nodes[node_index];
        if node.bounds.distance_squared(p) >= *best { return; }
        if node.is_leaf() {
            for triangle in &self.triangles[node.start..node.start + node.count] {
                *best = best.min(triangle.distance_squared(p));
            }
            return;
        }
        let left_dist = self.nodes[node.left].bounds.distance_squared(p);
        let right_dist = self.nodes[node.right].bounds.distance_squared(p);
        if left_dist <= right_dist {
            if left_dist < *best { self.nearest_distance_squared(node.left, p, best); }
            if right_dist < *best { self.nearest_distance_squared(node.right, p, best); }
        } else {
            if right_dist < *best { self.nearest_distance_squared(node.right, p, best); }
            if left_dist < *best { self.nearest_distance_squared(node.left, p, best); }
        }
    }

    fn ray_hit_count(&self, node_index: usize, origin: Vec3, direction: Vec3) -> usize {
        let node = self.nodes[node_index];
        if !node.bounds.intersects_ray(origin, direction) { return 0; }
        if node.is_leaf() {
            return self.triangles[node.start..node.start + node.count]
                .iter().filter(|triangle| triangle.ray_hit(origin, direction)).count();
        }
        self.ray_hit_count(node.left, origin, direction)
            + self.ray_hit_count(node.right, origin, direction)
    }
}

impl SignedDistanceField {
    /// Gera o SDF em metros. O espaçamento pode ser aumentado automaticamente para limitar
    /// o volume a 2,5 milhões de células e proteger o tempo/memória de importação.
    pub fn from_triangle_mesh(
        positions: &[[f32; 3]],
        indices: &[u32],
        requested_spacing: f32,
    ) -> Result<Self, String> {
        if positions.len() < 4 || indices.len() < 12 || indices.len() % 3 != 0 {
            return Err("A colisão por malha requer uma superfície triangular fechada.".into());
        }
        if positions.iter().flatten().any(|v| !v.is_finite()) {
            return Err("A malha de colisão contém coordenadas não finitas.".into());
        }
        if indices.iter().any(|index| *index as usize >= positions.len()) {
            return Err("A malha de colisão contém índices de triângulo inválidos.".into());
        }

        let mut triangles = Vec::with_capacity(indices.len() / 3);
        for tri in indices.chunks_exact(3) {
            let a = Vec3::from_array(positions[tri[0] as usize]);
            let b = Vec3::from_array(positions[tri[1] as usize]);
            let c = Vec3::from_array(positions[tri[2] as usize]);
            if (b - a).cross(c - a).length_squared() > 1.0e-16 {
                triangles.push(Triangle::new(a, b, c));
            }
        }
        let bvh = TriangleBvh::new(triangles)?;

        let mut mesh_bounds = Aabb::empty();
        for position in positions { mesh_bounds.include(Vec3::from_array(*position)); }

        let mut spacing = if requested_spacing.is_finite() {
            requested_spacing.clamp(0.004, 0.03)
        } else { 0.01 };
        let (origin, dims) = loop {
            let padding = Vec3::splat(spacing * 3.0);
            let origin = mesh_bounds.min - padding;
            let maximum = mesh_bounds.max + padding;
            let extent = maximum - origin;
            let dims = [
                (extent.x / spacing).ceil() as usize + 1,
                (extent.y / spacing).ceil() as usize + 1,
                (extent.z / spacing).ceil() as usize + 1,
            ];
            let cells = (dims[0] as u128) * (dims[1] as u128) * (dims[2] as u128);
            if cells <= MAX_GRID_CELLS as u128 {
                break (origin, dims);
            }
            let factor = (cells as f64 / MAX_GRID_CELLS as f64).cbrt().max(1.08) as f32;
            spacing *= factor * 1.01;
            if spacing > 0.08 {
                return Err("O SDF exigiria uma grade grande demais; simplifique a malha de colisão.".into());
            }
        };

        let cell_count = dims[0] * dims[1] * dims[2];
        let mut values = vec![0.0_f32; cell_count];
        // Raio levemente inclinado evita passar repetidamente pelos mesmos vértices/arestas.
        let ray = Vec3::new(1.0, 0.000_371, 0.000_173).normalize();
        let mut point_index = 0usize;
        for z in 0..dims[2] {
            for y in 0..dims[1] {
                for x in 0..dims[0] {
                    let point = origin + Vec3::new(x as f32, y as f32, z as f32) * spacing;
                    let mut distance_squared = f32::INFINITY;
                    bvh.nearest_distance_squared(bvh.root, point, &mut distance_squared);
                    let distance = distance_squared.max(0.0).sqrt();
                    let intersections = bvh.ray_hit_count(bvh.root, point, ray);
                    values[point_index] = if intersections % 2 == 1 { -distance } else { distance };
                    point_index += 1;
                }
            }
        }

        Ok(Self { origin, spacing, dims, values })
    }

    pub fn voxel_spacing(&self) -> f32 { self.spacing }

    pub fn dimensions(&self) -> [usize; 3] { self.dims }

    /// Amostragem trilinear. Fora da grade o campo é estendido como distância positiva
    /// à borda, evitando que partículas que saiam do volume escapem da colisão.
    pub fn sample(&self, p: Vec3) -> f32 {
        if !p.is_finite() { return 1.0e6; }
        let grid = (p - self.origin) / self.spacing;
        let maximum = Vec3::new(
            (self.dims[0] - 1) as f32,
            (self.dims[1] - 1) as f32,
            (self.dims[2] - 1) as f32,
        );
        let clamped_grid = grid.max(Vec3::ZERO).min(maximum);
        let clamped_world = self.origin + clamped_grid * self.spacing;
        let border_value = self.sample_grid(clamped_grid);
        let outside = (p - clamped_world).length();
        if outside > 0.0 { border_value.max(0.0) + outside } else { border_value }
    }

    pub fn gradient(&self, p: Vec3) -> Vec3 {
        let e = (self.spacing * 0.5).max(0.001);
        Vec3::new(
            self.sample(p + Vec3::X * e) - self.sample(p - Vec3::X * e),
            self.sample(p + Vec3::Y * e) - self.sample(p - Vec3::Y * e),
            self.sample(p + Vec3::Z * e) - self.sample(p - Vec3::Z * e),
        ).try_normalize().unwrap_or(Vec3::Y)
    }

    fn sample_grid(&self, grid: Vec3) -> f32 {
        let x0 = (grid.x.floor() as usize).min(self.dims[0] - 2);
        let y0 = (grid.y.floor() as usize).min(self.dims[1] - 2);
        let z0 = (grid.z.floor() as usize).min(self.dims[2] - 2);
        let fx = (grid.x - x0 as f32).clamp(0.0, 1.0);
        let fy = (grid.y - y0 as f32).clamp(0.0, 1.0);
        let fz = (grid.z - z0 as f32).clamp(0.0, 1.0);
        let c000 = self.value(x0, y0, z0);
        let c100 = self.value(x0 + 1, y0, z0);
        let c010 = self.value(x0, y0 + 1, z0);
        let c110 = self.value(x0 + 1, y0 + 1, z0);
        let c001 = self.value(x0, y0, z0 + 1);
        let c101 = self.value(x0 + 1, y0, z0 + 1);
        let c011 = self.value(x0, y0 + 1, z0 + 1);
        let c111 = self.value(x0 + 1, y0 + 1, z0 + 1);
        let lerp = |a: f32, b: f32, t: f32| a + (b - a) * t;
        let x00 = lerp(c000, c100, fx);
        let x10 = lerp(c010, c110, fx);
        let x01 = lerp(c001, c101, fx);
        let x11 = lerp(c011, c111, fx);
        lerp(lerp(x00, x10, fy), lerp(x01, x11, fy), fz)
    }

    fn value(&self, x: usize, y: usize, z: usize) -> f32 {
        self.values[(z * self.dims[1] + y) * self.dims[0] + x]
    }
}

fn point_triangle_distance_squared(p: Vec3, a: Vec3, b: Vec3, c: Vec3) -> f32 {
    let ab = b - a;
    let ac = c - a;
    let ap = p - a;
    let d1 = ab.dot(ap);
    let d2 = ac.dot(ap);
    if d1 <= 0.0 && d2 <= 0.0 { return ap.length_squared(); }

    let bp = p - b;
    let d3 = ab.dot(bp);
    let d4 = ac.dot(bp);
    if d3 >= 0.0 && d4 <= d3 { return bp.length_squared(); }

    let vc = d1 * d4 - d3 * d2;
    if vc <= 0.0 && d1 >= 0.0 && d3 <= 0.0 {
        let v = d1 / (d1 - d3);
        return (p - (a + ab * v)).length_squared();
    }

    let cp = p - c;
    let d5 = ab.dot(cp);
    let d6 = ac.dot(cp);
    if d6 >= 0.0 && d5 <= d6 { return cp.length_squared(); }

    let vb = d5 * d2 - d1 * d6;
    if vb <= 0.0 && d2 >= 0.0 && d6 <= 0.0 {
        let w = d2 / (d2 - d6);
        return (p - (a + ac * w)).length_squared();
    }

    let va = d3 * d6 - d5 * d4;
    if va <= 0.0 && (d4 - d3) >= 0.0 && (d5 - d6) >= 0.0 {
        let w = (d4 - d3) / ((d4 - d3) + (d5 - d6));
        return (p - (b + (c - b) * w)).length_squared();
    }

    let denominator = 1.0 / (va + vb + vc).max(1.0e-20);
    let v = vb * denominator;
    let w = vc * denominator;
    (p - (a + ab * v + ac * w)).length_squared()
}

#[cfg(test)]
mod tests {
    use super::*;

    fn cube() -> (Vec<[f32; 3]>, Vec<u32>) {
        let vertices = vec![
            [-0.5, 0.0, -0.5], [0.5, 0.0, -0.5], [0.5, 1.0, -0.5], [-0.5, 1.0, -0.5],
            [-0.5, 0.0, 0.5], [0.5, 0.0, 0.5], [0.5, 1.0, 0.5], [-0.5, 1.0, 0.5],
        ];
        let indices = vec![
            0, 2, 1, 0, 3, 2, // back
            4, 5, 6, 4, 6, 7, // front
            0, 1, 5, 0, 5, 4, // bottom
            3, 7, 6, 3, 6, 2, // top
            0, 4, 7, 0, 7, 3, // left
            1, 2, 6, 1, 6, 5, // right
        ];
        (vertices, indices)
    }

    #[test]
    fn mesh_sdf_has_negative_inside_positive_outside_and_outward_gradient() {
        let (positions, indices) = cube();
        let sdf = SignedDistanceField::from_triangle_mesh(&positions, &indices, 0.025).unwrap();
        assert!(sdf.sample(Vec3::new(0.0, 0.5, 0.0)) < 0.0);
        assert!(sdf.sample(Vec3::new(0.8, 0.5, 0.0)) > 0.0);
        let normal = sdf.gradient(Vec3::new(0.55, 0.5, 0.0));
        assert!(normal.x > 0.7, "gradiente deveria apontar para fora: {normal:?}");
        assert!(sdf.voxel_spacing() >= 0.004);
    }

    #[test]
    fn mesh_sdf_caps_grid_size_for_large_models() {
        let (positions, indices) = cube();
        let sdf = SignedDistanceField::from_triangle_mesh(&positions, &indices, 0.004).unwrap();
        let dims = sdf.dimensions();
        assert!(dims[0] * dims[1] * dims[2] <= MAX_GRID_CELLS);
    }
}
