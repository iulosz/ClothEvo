use glam::{Mat4, Vec3};

/// Câmera orbital em torno de um alvo (eixo Y para cima).
#[derive(Clone, Debug)]
pub struct OrbitCamera {
    pub target: Vec3,
    /// Rotação horizontal (rad). 0 = olhando de +Z para o alvo.
    pub yaw: f32,
    /// Elevação (rad), limitada para evitar o gimbal lock nos polos.
    pub pitch: f32,
    pub distance: f32,
    pub fov_y: f32,
}

impl Default for OrbitCamera {
    fn default() -> Self {
        Self {
            target: Vec3::new(0.0, 0.7, 0.0),
            // Vista frontal por padrão: frente do avatar/moldes aponta para +Z.
            yaw: 0.0,
            pitch: 0.10,
            distance: 3.2,
            fov_y: 45f32.to_radians(),
        }
    }
}

/// Vistas pré-definidas (atalhos 1/2/3 no viewport 3D).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum CameraPreset {
    Front,
    Side,
    Top,
}
impl CameraPreset {
    pub fn yaw_pitch(self) -> (f32, f32) {
        match self {
            CameraPreset::Front => (0.0, 0.0),
            CameraPreset::Side => (std::f32::consts::FRAC_PI_2, 0.0),
            CameraPreset::Top => (0.0, 89f32.to_radians()),
        }
    }
}
#[derive(Clone, Copy, Debug)]
pub struct CameraGoal {
    pub target: Vec3,
    pub yaw: f32,
    pub pitch: f32,
    pub distance: f32,
}
fn wrap_angle(angle: f32) -> f32 {
    (angle + std::f32::consts::PI).rem_euclid(std::f32::consts::TAU) - std::f32::consts::PI
}
impl OrbitCamera {
    pub fn goal_for(&self, preset: CameraPreset, target: Vec3, radius: f32, aspect: f32) -> CameraGoal {
        let (yaw, pitch) = preset.yaw_pitch();
        let half_v = (self.fov_y * 0.5).tan();
        let half_h = half_v * aspect.max(0.1);
        let fit = half_v.min(half_h).max(1.0e-3);
        let distance = (radius.max(0.1) / fit * 1.15).clamp(0.3, 40.0);
        CameraGoal { target, yaw, pitch, distance }
    }
    pub fn approach(&mut self, goal: &CameraGoal, dt: f32) -> bool {
        let k = 1.0 - (-12.0 * dt.clamp(0.0, 0.1)).exp();
        let delta_yaw = wrap_angle(goal.yaw - self.yaw);
        self.yaw += delta_yaw * k;
        self.pitch += (goal.pitch - self.pitch) * k;
        self.distance += (goal.distance - self.distance) * k;
        self.target += (goal.target - self.target) * k;
        let done = delta_yaw.abs() < 1.0e-3
            && (goal.pitch - self.pitch).abs() < 1.0e-3
            && (goal.distance - self.distance).abs() < 1.0e-3
            && (goal.target - self.target).length() < 1.0e-3;
        if done {
            self.yaw = goal.yaw;
            self.pitch = goal.pitch;
            self.distance = goal.distance;
            self.target = goal.target;
        }
        done
    }
    pub fn eye(&self) -> Vec3 {
        let (sy, cy) = self.yaw.sin_cos();
        let (sp, cp) = self.pitch.sin_cos();
        self.target + self.distance * Vec3::new(cp * sy, sp, cp * cy)
    }

    pub fn view(&self) -> Mat4 {
        glam::camera::rh::view::look_at_mat4(self.eye(), self.target, Vec3::Y)
    }

    /// Projeção com profundidade [0, 1] (convenção do wgpu/DirectX).
    pub fn view_proj(&self, aspect: f32) -> Mat4 {
        let proj = glam::camera::rh::proj::directx::perspective(self.fov_y, aspect.max(1e-3), 0.05, 100.0);
        proj * self.view()
    }

    /// Gira a câmera. `dx`/`dy` em pixels.
    pub fn orbit(&mut self, dx: f32, dy: f32) {
        self.yaw -= dx * 0.008;
        let limit = 89f32.to_radians();
        self.pitch = (self.pitch + dy * 0.008).clamp(-limit, limit);
    }

    /// Desloca o alvo no plano da tela. `viewport_height` em pixels.
    pub fn pan(&mut self, dx: f32, dy: f32, viewport_height: f32) {
        let view = self.view();
        let right = view.row(0).truncate();
        let up = view.row(1).truncate();
        let world_per_px = 2.0 * self.distance * (self.fov_y * 0.5).tan() / viewport_height.max(1.0);
        self.target += (-right * dx + up * dy) * world_per_px;
    }

    /// Aproxima/afasta de forma exponencial (`scroll` > 0 aproxima).
    pub fn zoom(&mut self, scroll: f32) {
        self.distance = (self.distance * (-scroll * 0.0015).exp()).clamp(0.3, 40.0);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn default_camera_looks_at_the_front_from_plus_z() {
        let cam = OrbitCamera::default();
        let offset = cam.eye() - cam.target;
        assert!(offset.x.abs() < 1.0e-5 && offset.z > 0.0);
    }
    #[test]
    fn presets_place_the_eye_in_front_side_and_above() {
        let cam = OrbitCamera::default();
        for (preset, check) in [(CameraPreset::Front, Vec3::Z),(CameraPreset::Side, Vec3::X),(CameraPreset::Top, Vec3::Y)] {
            let goal = cam.goal_for(preset, Vec3::new(0.0, 0.9, 0.0), 1.0, 1.5);
            let mut moved = cam.clone();
            moved.yaw = goal.yaw; moved.pitch = goal.pitch; moved.distance = goal.distance; moved.target = goal.target;
            let direction = (moved.eye() - moved.target).normalize();
            assert!(direction.dot(check) > 0.99, "{preset:?}: {direction:?}");
        }
    }
    #[test]
    fn approach_converges_to_goal_by_the_shortest_yaw() {
        let mut cam = OrbitCamera::default();
        cam.yaw = 6.0;
        let goal = CameraGoal { target: Vec3::new(0.0, 1.0, 0.0), yaw: 0.0, pitch: 0.0, distance: 2.5 };
        let mut done = false;
        for _ in 0..600 { if cam.approach(&goal, 1.0 / 60.0) { done = true; break; } }
        assert!(done);
        assert_eq!(cam.yaw, 0.0);
        assert!((cam.distance - 2.5).abs() < 1.0e-6);
    }
    #[test]
    fn eye_is_at_orbit_distance() {
        let cam = OrbitCamera::default();
        assert!(((cam.eye() - cam.target).length() - cam.distance).abs() < 1e-5);
    }

    #[test]
    fn target_projects_to_screen_center() {
        let cam = OrbitCamera::default();
        let clip = cam.view_proj(1.5) * cam.target.extend(1.0);
        let ndc = clip.truncate() / clip.w;
        assert!(ndc.x.abs() < 1e-4 && ndc.y.abs() < 1e-4);
        assert!((0.0..=1.0).contains(&ndc.z), "profundidade fora de [0,1]: {}", ndc.z);
    }

    #[test]
    fn pitch_is_clamped() {
        let mut cam = OrbitCamera::default();
        cam.orbit(0.0, 1.0e6);
        assert!(cam.pitch < std::f32::consts::FRAC_PI_2);
    }
}
