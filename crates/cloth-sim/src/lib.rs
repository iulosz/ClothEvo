//! # cloth-sim
//!
//! Solver de tecido baseado em XPBD (Macklin et al., 2016) do ClothEvo.
//!
//! Princípios de projeto:
//! * **Comprimentos de repouso vêm do molde 2D**, não da posição 3D inicial. Assim um
//!   painel posicionado "torto" ao redor do avatar ainda tem o tamanho real do tecido.
//! * **Rigidez física e independente de resolução**: stretch/shear em N/m e flexão em N·m.
//! * **Gauss-Seidel paralelo por coloração de grafo**: restrições da mesma cor não
//!   compartilham partículas e podem ser resolvidas em paralelo com `rayon`.
//! * Sem dependência de render, para poder migrar o solver para GPU depois.

pub mod avatar;
pub mod builder;
pub mod collider;
pub mod sdf;
mod coloring;
pub mod mesh;
pub mod params;
pub mod scenes;
pub mod solver;

pub use avatar::{build_avatar_colliders, clear_shell_radius, AvatarConfig};
pub use builder::{ClothBuilder, PanelHandle};
pub use collider::Collider;
pub use sdf::SignedDistanceField;
pub use mesh::{GridPanel, PanelMesh, Placement};
pub use params::{ConstraintKind, FabricPreset, PanelMaterial, SimParams};
pub use scenes::{build_scene, SceneKind};
pub use solver::{Simulation, Stats};
