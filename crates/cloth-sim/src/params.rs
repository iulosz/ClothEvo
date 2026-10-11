use glam::Vec3;

/// Tipo de restrição de distância. Cada tipo tem a sua própria rigidez em [`SimParams`].
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
#[repr(u8)]
pub enum ConstraintKind {
    /// Aresta alinhada ao fio (urdume / warp).
    Warp = 0,
    /// Aresta perpendicular ao fio (trama / weft).
    Weft = 1,
    /// Aresta diagonal: resiste ao cisalhamento.
    Shear = 2,
    /// Costura entre duas bordas (comprimento de repouso zero).
    Seam = 3,
}

pub const KIND_COUNT: usize = 4;

/// Parâmetros do solver e do material. Todos podem ser alterados com a simulação rodando.
#[derive(Clone, Debug)]
pub struct SimParams {
    pub gravity: Vec3,
    /// Substeps por passo de tempo ("small steps"): mais estável do que mais iterações.
    pub substeps: u32,
    /// Iterações de restrições por substep.
    pub iterations: u32,
    /// Amortecimento de velocidade global (1/s).
    pub damping: f32,
    /// Densidade superficial do tecido (kg/m²). Algodão ≈ 0,2.
    pub density: f32,
    /// Rigidez de stretch ao longo do fio (N/m).
    pub stretch_warp: f32,
    /// Rigidez de stretch perpendicular ao fio (N/m).
    pub stretch_weft: f32,
    /// Rigidez de cisalhamento (N/m).
    pub shear: f32,
    /// Rigidez de flexão (N·m). Algodão ≈ 5e-5, seda ≈ 5e-6, jeans ≈ 3e-4.
    pub bend: f32,
    /// Rigidez das costuras (N/m).
    pub seam: f32,
    /// Velocidade máxima com que uma costura puxa as bordas (m/s). Evita "estalos".
    pub seam_pull_speed: f32,
    /// Espessura de colisão (m): distância mínima entre o tecido e o colisor.
    pub thickness: f32,
    /// Coeficiente de atrito de Coulomb com os colisores.
    pub friction: f32,
    /// Resolver restrições em paralelo (rayon).
    pub parallel: bool,
}

impl Default for SimParams {
    fn default() -> Self {
        let mut p = Self {
            gravity: Vec3::new(0.0, -9.81, 0.0),
            // Medido na cena "pano pendurado" (res 24): 10 substeps → strain máx 33%,
            // 20 → 11%, 40 → 5%. Substeps rendem mais que iterações pelo mesmo custo.
            substeps: 20,
            iterations: 1,
            damping: 0.3,
            density: 0.2,
            stretch_warp: 5000.0,
            stretch_weft: 5000.0,
            shear: 300.0,
            bend: 5e-5,
            seam: 5000.0,
            seam_pull_speed: 0.5,
            thickness: 0.005,
            friction: 0.4,
            parallel: true,
        };
        FabricPreset::Cotton.apply(&mut p);
        p
    }
}

impl SimParams {
    /// Rigidez por tipo de restrição, protegida contra zero.
    pub fn stiffness_table(&self) -> [f32; KIND_COUNT] {
        let mut t = [0.0; KIND_COUNT];
        t[ConstraintKind::Warp as usize] = self.stretch_warp;
        t[ConstraintKind::Weft as usize] = self.stretch_weft;
        t[ConstraintKind::Shear as usize] = self.shear;
        t[ConstraintKind::Seam as usize] = self.seam;
        for v in &mut t {
            *v = v.max(1e-3);
        }
        t
    }
}

/// Presets de tecido (valores de partida; ajuste ao gosto).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum FabricPreset {
    Cotton,
    Silk,
    Denim,
    Jersey,
}

impl FabricPreset {
    pub const ALL: [FabricPreset; 4] = [Self::Cotton, Self::Silk, Self::Denim, Self::Jersey];

    pub fn name(self) -> &'static str {
        match self {
            Self::Cotton => "Algodão",
            Self::Silk => "Seda",
            Self::Denim => "Jeans",
            Self::Jersey => "Malha (jersey)",
        }
    }

    /// Sobrescreve apenas as propriedades do material, preservando o resto dos parâmetros.
    pub fn apply(self, p: &mut SimParams) {
        let (density, warp, weft, shear, bend) = match self {
            Self::Cotton => (0.20, 5000.0, 5000.0, 300.0, 5e-5),
            Self::Silk => (0.08, 2500.0, 2500.0, 120.0, 5e-6),
            Self::Denim => (0.40, 20000.0, 20000.0, 2000.0, 3e-4),
            Self::Jersey => (0.18, 900.0, 350.0, 80.0, 1e-5),
        };
        p.density = density;
        p.stretch_warp = warp;
        p.stretch_weft = weft;
        p.shear = shear;
        p.bend = bend;
    }
}
