use glam::{Quat, Vec2, Vec3};

/// Malha triangular de um painel de molde, em coordenadas 2D de repouso (metros).
#[derive(Clone, Debug, Default)]
pub struct PanelMesh {
    pub uv: Vec<Vec2>,
    /// Triângulos em sentido anti-horário (visto pela face da frente do painel).
    pub triangles: Vec<[u32; 3]>,
}

impl PanelMesh {
    pub fn triangle_area(&self, t: [u32; 3]) -> f32 {
        let a = self.uv[t[0] as usize];
        let b = self.uv[t[1] as usize];
        let c = self.uv[t[2] as usize];
        0.5 * (b - a).perp_dot(c - a).abs()
    }
}

/// Painel retangular em grade, útil para as cenas de teste.
/// `u` cresce ao longo da largura e `v` ao longo da altura (a direção do fio é `v`).
#[derive(Clone, Debug)]
pub struct GridPanel {
    pub mesh: PanelMesh,
    pub width: f32,
    pub height: f32,
    pub nx: u32,
    pub ny: u32,
}

impl GridPanel {
    pub fn new(width: f32, height: f32, nx: u32, ny: u32) -> Self {
        assert!(nx >= 1 && ny >= 1, "a grade precisa de ao menos 1x1 células");
        let idx = |i: u32, j: u32| j * (nx + 1) + i;

        let mut uv = Vec::with_capacity(((nx + 1) * (ny + 1)) as usize);
        for j in 0..=ny {
            for i in 0..=nx {
                uv.push(Vec2::new(
                    width * i as f32 / nx as f32,
                    height * j as f32 / ny as f32,
                ));
            }
        }

        // Diagonais alternadas ("criss-cross") evitam viés direcional no cisalhamento.
        let mut triangles = Vec::with_capacity((nx * ny * 2) as usize);
        for j in 0..ny {
            for i in 0..nx {
                let a = idx(i, j);
                let b = idx(i + 1, j);
                let c = idx(i + 1, j + 1);
                let d = idx(i, j + 1);
                if (i + j) % 2 == 0 {
                    triangles.push([a, b, c]);
                    triangles.push([a, c, d]);
                } else {
                    triangles.push([a, b, d]);
                    triangles.push([b, c, d]);
                }
            }
        }

        Self { mesh: PanelMesh { uv, triangles }, width, height, nx, ny }
    }

    pub fn index(&self, i: u32, j: u32) -> u32 {
        j * (self.nx + 1) + i
    }

    /// Borda esquerda (u = 0), de baixo para cima.
    pub fn left_edge(&self) -> Vec<u32> {
        (0..=self.ny).map(|j| self.index(0, j)).collect()
    }

    /// Borda direita (u = largura), de baixo para cima.
    pub fn right_edge(&self) -> Vec<u32> {
        (0..=self.ny).map(|j| self.index(self.nx, j)).collect()
    }

    /// Borda superior (v = altura), da esquerda para a direita.
    pub fn top_edge(&self) -> Vec<u32> {
        (0..=self.nx).map(|i| self.index(i, self.ny)).collect()
    }

    /// Borda inferior (v = 0), da esquerda para a direita.
    pub fn bottom_edge(&self) -> Vec<u32> {
        (0..=self.nx).map(|i| self.index(i, 0)).collect()
    }
}

/// Posição inicial do painel no mundo 3D: `p = origin + rotation * (u, v, 0)`.
/// A normal da face da frente é `rotation * +Z`.
#[derive(Clone, Copy, Debug)]
pub struct Placement {
    pub origin: Vec3,
    pub rotation: Quat,
}

impl Placement {
    pub fn new(origin: Vec3, rotation: Quat) -> Self {
        Self { origin, rotation }
    }

    pub fn at(origin: Vec3) -> Self {
        Self { origin, rotation: Quat::IDENTITY }
    }

    pub fn map(&self, uv: Vec2) -> Vec3 {
        self.origin + self.rotation * Vec3::new(uv.x, uv.y, 0.0)
    }
}
