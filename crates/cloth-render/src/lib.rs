//! # cloth-render
//!
//! Renderer wgpu próprio do ClothEvo, integrado ao egui por *paint callback*:
//! o egui continua dono da janela e da UI, enquanto o viewport 3D usa o nosso
//! pipeline (shader WGSL, buffers, câmera) dentro do mesmo render pass.
//!
//! Fluxo:
//! 1. Na criação do app: `SceneRenderer::new(..)` e guardar nos `callback_resources`.
//! 2. Ao trocar de cena: `set_cloth_topology` (índices, wireframe, costuras) e `set_statics`.
//! 3. A cada frame: criar um [`ViewportCallback`] com as posições/normais atuais e
//!    entregá-lo ao egui com `egui_wgpu::Callback::new_paint_callback`.

pub mod camera;
pub mod primitives;

use std::borrow::Cow;

use egui_wgpu::wgpu::{self, util::DeviceExt};
use glam::{Mat4, Vec3};

pub use camera::OrbitCamera;

/// Vértice enviado à GPU.
#[repr(C)]
#[derive(Clone, Copy, Debug, bytemuck::Pod, bytemuck::Zeroable)]
pub struct Vertex {
    pub pos: [f32; 3],
    pub normal: [f32; 3],
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum MeshKind {
    /// Superfície sombreada comum (colisores).
    Shaded,
    /// Chão: sombreado com grade procedural.
    Ground,
}

/// Malha estática (colisores, chão) pronta para virar buffers.
#[derive(Clone, Debug)]
pub struct StaticMeshData {
    pub vertices: Vec<Vertex>,
    pub indices: Vec<u32>,
    pub color: [f32; 3],
    pub kind: MeshKind,
}

/// Topologia do tecido (muda só quando a cena ou a resolução muda).
pub struct ClothTopology<'a> {
    pub vertex_count: usize,
    pub triangles: &'a [[u32; 3]],
    /// Arestas únicas para o overlay de wireframe.
    pub wire_edges: &'a [[u32; 2]],
    /// Pares de vértices costurados.
    pub seam_pairs: &'a [(u32, u32)],
}

/// Aparência do tecido e dos overlays.
#[derive(Clone, Debug)]
pub struct ViewStyle {
    pub cloth_front: [f32; 3],
    pub cloth_back: [f32; 3],
    pub wireframe: bool,
    pub seams: bool,
}

impl Default for ViewStyle {
    fn default() -> Self {
        Self {
            cloth_front: [0.80, 0.30, 0.34],
            cloth_back: [0.34, 0.45, 0.78],
            wireframe: false,
            seams: true,
        }
    }
}

// ---------------------------------------------------------------------------
// GPU
// ---------------------------------------------------------------------------

/// Deve espelhar `Uniforms` em `shader.wgsl`.
#[repr(C)]
#[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
struct Uniforms {
    view_proj: [[f32; 4]; 4],
    color_front: [f32; 4],
    color_back: [f32; 4],
    light_dir: [f32; 4],
    eye: [f32; 4],
    params: [f32; 4],
}

const KIND_SHADED: f32 = 0.0;
const KIND_GROUND: f32 = 1.0;
const KIND_LINE: f32 = 2.0;

/// Um uniform buffer + bind group por draw call (cada um tem cor/tipo próprios).
struct DrawSlot {
    buffer: wgpu::Buffer,
    bind_group: wgpu::BindGroup,
}

impl DrawSlot {
    fn new(device: &wgpu::Device, layout: &wgpu::BindGroupLayout, label: &str) -> Self {
        let buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some(label),
            size: std::mem::size_of::<Uniforms>() as u64,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some(label),
            layout,
            entries: &[wgpu::BindGroupEntry {
                binding: 0,
                resource: buffer.as_entire_binding(),
            }],
        });
        Self { buffer, bind_group }
    }
}

struct StaticMesh {
    vertex_buffer: wgpu::Buffer,
    index_buffer: wgpu::Buffer,
    index_count: u32,
    slot: DrawSlot,
    color: [f32; 3],
    kind: MeshKind,
}

struct ClothGpu {
    vertex_count: usize,
    vertex_buffer: wgpu::Buffer,
    tri_indices: wgpu::Buffer,
    tri_count: u32,
    wire_indices: wgpu::Buffer,
    wire_count: u32,
    seam_indices: wgpu::Buffer,
    seam_count: u32,
    surface: DrawSlot,
    wire: DrawSlot,
    seams: DrawSlot,
}

/// Recursos de GPU do viewport. Fica guardado em `callback_resources` do egui.
pub struct SceneRenderer {
    tri_pipeline: wgpu::RenderPipeline,
    line_pipeline: wgpu::RenderPipeline,
    layout: wgpu::BindGroupLayout,
    /// O shader aplica gamma quando o alvo não é sRGB.
    apply_gamma: bool,
    statics: Vec<StaticMesh>,
    cloth: Option<ClothGpu>,
}

fn index_buffer<T: bytemuck::Pod>(device: &wgpu::Device, label: &str, data: &[T]) -> wgpu::Buffer {
    // Buffers vazios não são válidos para todos os backends: garante ao menos 1 elemento.
    let fallback = [T::zeroed()];
    let data = if data.is_empty() { &fallback[..] } else { data };
    device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some(label),
        contents: bytemuck::cast_slice(data),
        usage: wgpu::BufferUsages::INDEX | wgpu::BufferUsages::COPY_DST,
    })
}

impl SceneRenderer {
    /// `depth_format` precisa coincidir com o depth buffer do render pass do egui
    /// (`NativeOptions::depth_buffer = 32` → `Depth32Float`).
    pub fn new(
        device: &wgpu::Device,
        target_format: wgpu::TextureFormat,
        depth_format: wgpu::TextureFormat,
    ) -> Self {
        let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("clothevo shader"),
            source: wgpu::ShaderSource::Wgsl(Cow::Borrowed(include_str!("shader.wgsl"))),
        });

        let layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("clothevo uniforms"),
            entries: &[wgpu::BindGroupLayoutEntry {
                binding: 0,
                visibility: wgpu::ShaderStages::VERTEX | wgpu::ShaderStages::FRAGMENT,
                ty: wgpu::BindingType::Buffer {
                    ty: wgpu::BufferBindingType::Uniform,
                    has_dynamic_offset: false,
                    min_binding_size: None,
                },
                count: None,
            }],
        });
        let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("clothevo pipeline layout"),
            bind_group_layouts: &[Some(&layout)],
            immediate_size: 0,
        });

        let attributes = [
            wgpu::VertexAttribute {
                format: wgpu::VertexFormat::Float32x3,
                offset: 0,
                shader_location: 0,
            },
            wgpu::VertexAttribute {
                format: wgpu::VertexFormat::Float32x3,
                offset: 12,
                shader_location: 1,
            },
        ];
        let vertex_layout = wgpu::VertexBufferLayout {
            array_stride: std::mem::size_of::<Vertex>() as u64,
            step_mode: wgpu::VertexStepMode::Vertex,
            attributes: &attributes,
        };

        let make_pipeline = |label: &str, topology: wgpu::PrimitiveTopology| {
            device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
                label: Some(label),
                layout: Some(&pipeline_layout),
                vertex: wgpu::VertexState {
                    module: &shader,
                    entry_point: Some("vs_main"),
                    compilation_options: Default::default(),
                    buffers: &[Some(vertex_layout.clone())],
                },
                primitive: wgpu::PrimitiveState {
                    topology,
                    // Tecido tem dois lados: sem culling.
                    cull_mode: None,
                    ..Default::default()
                },
                depth_stencil: Some(wgpu::DepthStencilState {
                    format: depth_format,
                    depth_write_enabled: Some(true),
                    depth_compare: Some(wgpu::CompareFunction::Less),
                    stencil: wgpu::StencilState::default(),
                    bias: wgpu::DepthBiasState::default(),
                }),
                multisample: wgpu::MultisampleState::default(),
                fragment: Some(wgpu::FragmentState {
                    module: &shader,
                    entry_point: Some("fs_main"),
                    compilation_options: Default::default(),
                    targets: &[Some(wgpu::ColorTargetState {
                        format: target_format,
                        blend: Some(wgpu::BlendState::REPLACE),
                        write_mask: wgpu::ColorWrites::ALL,
                    })],
                }),
                multiview_mask: None,
                cache: None,
            })
        };

        Self {
            tri_pipeline: make_pipeline("clothevo triangles", wgpu::PrimitiveTopology::TriangleList),
            line_pipeline: make_pipeline("clothevo lines", wgpu::PrimitiveTopology::LineList),
            layout,
            apply_gamma: !target_format.is_srgb(),
            statics: Vec::new(),
            cloth: None,
        }
    }

    /// Substitui os colisores/chão desenhados.
    pub fn set_statics(&mut self, device: &wgpu::Device, meshes: &[StaticMeshData]) {
        self.statics = meshes
            .iter()
            .map(|m| StaticMesh {
                vertex_buffer: device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
                    label: Some("static vertices"),
                    contents: bytemuck::cast_slice(&m.vertices),
                    usage: wgpu::BufferUsages::VERTEX,
                }),
                index_buffer: index_buffer(device, "static indices", &m.indices),
                index_count: m.indices.len() as u32,
                slot: DrawSlot::new(device, &self.layout, "static uniforms"),
                color: m.color,
                kind: m.kind,
            })
            .collect();
    }

    /// (Re)cria os buffers do tecido para uma nova malha.
    pub fn set_cloth_topology(&mut self, device: &wgpu::Device, topo: &ClothTopology) {
        let tris: Vec<u32> = topo.triangles.iter().flatten().copied().collect();
        let wire: Vec<u32> = topo.wire_edges.iter().flatten().copied().collect();
        let seams: Vec<u32> = topo.seam_pairs.iter().flat_map(|&(a, b)| [a, b]).collect();

        let vertex_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("cloth vertices"),
            size: (topo.vertex_count.max(1) * std::mem::size_of::<Vertex>()) as u64,
            usage: wgpu::BufferUsages::VERTEX | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        self.cloth = Some(ClothGpu {
            vertex_count: topo.vertex_count,
            vertex_buffer,
            tri_indices: index_buffer(device, "cloth triangles", &tris),
            tri_count: tris.len() as u32,
            wire_indices: index_buffer(device, "cloth wireframe", &wire),
            wire_count: wire.len() as u32,
            seam_indices: index_buffer(device, "cloth seams", &seams),
            seam_count: seams.len() as u32,
            surface: DrawSlot::new(device, &self.layout, "cloth surface uniforms"),
            wire: DrawSlot::new(device, &self.layout, "cloth wire uniforms"),
            seams: DrawSlot::new(device, &self.layout, "cloth seam uniforms"),
        });
    }
}

// ---------------------------------------------------------------------------
// Callback do egui
// ---------------------------------------------------------------------------

/// Tudo o que o viewport precisa para desenhar um frame.
pub struct ViewportCallback {
    pub view_proj: Mat4,
    pub eye: Vec3,
    /// Posições e normais atuais do tecido (mesmo tamanho de `ClothTopology::vertex_count`).
    pub cloth_vertices: Vec<Vertex>,
    pub style: ViewStyle,
}

impl ViewportCallback {
    fn uniforms(&self, front: [f32; 3], back: [f32; 3], kind: f32, gamma: bool) -> Uniforms {
        let light = Vec3::new(0.45, 0.85, 0.35).normalize();
        Uniforms {
            view_proj: self.view_proj.to_cols_array_2d(),
            color_front: [front[0], front[1], front[2], 1.0],
            color_back: [back[0], back[1], back[2], 1.0],
            light_dir: [light.x, light.y, light.z, 0.0],
            eye: [self.eye.x, self.eye.y, self.eye.z, 1.0],
            params: [kind, if gamma { 1.0 } else { 0.0 }, 0.0, 0.0],
        }
    }
}

impl egui_wgpu::CallbackTrait for ViewportCallback {
    fn prepare(
        &self,
        _device: &wgpu::Device,
        queue: &wgpu::Queue,
        _screen: &egui_wgpu::ScreenDescriptor,
        _encoder: &mut wgpu::CommandEncoder,
        resources: &mut egui_wgpu::CallbackResources,
    ) -> Vec<wgpu::CommandBuffer> {
        let Some(r) = resources.get::<SceneRenderer>() else {
            return Vec::new();
        };
        let gamma = r.apply_gamma;

        for m in &r.statics {
            let kind = match m.kind {
                MeshKind::Shaded => KIND_SHADED,
                MeshKind::Ground => KIND_GROUND,
            };
            let u = self.uniforms(m.color, m.color, kind, gamma);
            queue.write_buffer(&m.slot.buffer, 0, bytemuck::bytes_of(&u));
        }

        if let Some(c) = &r.cloth {
            // Se a malha mudou mas o frame ainda traz dados antigos, não escreve fora do buffer.
            if self.cloth_vertices.len() == c.vertex_count && c.vertex_count > 0 {
                queue.write_buffer(&c.vertex_buffer, 0, bytemuck::cast_slice(&self.cloth_vertices));
            }
            let s = &self.style;
            let surface = self.uniforms(s.cloth_front, s.cloth_back, KIND_SHADED, gamma);
            queue.write_buffer(&c.surface.buffer, 0, bytemuck::bytes_of(&surface));
            let wire = self.uniforms([0.08, 0.08, 0.10], [0.08, 0.08, 0.10], KIND_LINE, gamma);
            queue.write_buffer(&c.wire.buffer, 0, bytemuck::bytes_of(&wire));
            let seam = self.uniforms([1.0, 0.82, 0.10], [1.0, 0.82, 0.10], KIND_LINE, gamma);
            queue.write_buffer(&c.seams.buffer, 0, bytemuck::bytes_of(&seam));
        }
        Vec::new()
    }

    fn paint(
        &self,
        _info: egui::PaintCallbackInfo,
        pass: &mut wgpu::RenderPass<'static>,
        resources: &egui_wgpu::CallbackResources,
    ) {
        let Some(r) = resources.get::<SceneRenderer>() else {
            return;
        };

        pass.set_pipeline(&r.tri_pipeline);
        for m in &r.statics {
            if m.index_count == 0 {
                continue;
            }
            pass.set_bind_group(0, &m.slot.bind_group, &[]);
            pass.set_vertex_buffer(0, m.vertex_buffer.slice(..));
            pass.set_index_buffer(m.index_buffer.slice(..), wgpu::IndexFormat::Uint32);
            pass.draw_indexed(0..m.index_count, 0, 0..1);
        }

        let Some(c) = &r.cloth else {
            return;
        };
        if self.cloth_vertices.len() != c.vertex_count {
            return;
        }
        pass.set_vertex_buffer(0, c.vertex_buffer.slice(..));

        if c.tri_count > 0 {
            pass.set_bind_group(0, &c.surface.bind_group, &[]);
            pass.set_index_buffer(c.tri_indices.slice(..), wgpu::IndexFormat::Uint32);
            pass.draw_indexed(0..c.tri_count, 0, 0..1);
        }

        pass.set_pipeline(&r.line_pipeline);
        if self.style.wireframe && c.wire_count > 0 {
            pass.set_bind_group(0, &c.wire.bind_group, &[]);
            pass.set_index_buffer(c.wire_indices.slice(..), wgpu::IndexFormat::Uint32);
            pass.draw_indexed(0..c.wire_count, 0, 0..1);
        }
        if self.style.seams && c.seam_count > 0 {
            pass.set_bind_group(0, &c.seams.bind_group, &[]);
            pass.set_index_buffer(c.seam_indices.slice(..), wgpu::IndexFormat::Uint32);
            pass.draw_indexed(0..c.seam_count, 0, 0..1);
        }
    }
}
