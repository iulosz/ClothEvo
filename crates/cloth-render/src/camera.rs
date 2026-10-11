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
            yaw: 0.6,
            pitch: 0.35,
            distance: 3.2,
            fov_y: 45f32.to_radians(),
        }
    }
}

impl OrbitCamera {
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
