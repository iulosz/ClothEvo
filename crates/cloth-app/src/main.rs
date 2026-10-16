//! ClothEvo — Fase 0: protótipo do simulador de tecido XPBD (egui + wgpu).
//!
//! Modo de captura (para validar o render sem interação), via variáveis de ambiente:
//! `CLOTHEVO_SCENE=1|2|3`, `CLOTHEVO_FRAMES=<n passos simulados>` e
//! `CLOTHEVO_SCREENSHOT=<arquivo.ppm>` — salva o viewport e fecha o app.

mod avatar_import;
mod ui_customization;

use ui_customization::{CustomToolbarButton, ThemeSettings, UiCommand, UiCustomization};

use cloth_render::{
    primitives, CameraGoal, CameraPreset, ClothTopology, MeshKind, OrbitCamera, SceneRenderer,
    StaticMeshData, Vertex, ViewStyle, ViewportCallback,
};
use cloth_sim::{
    arrangement_points, arrangement_surface_position, build_avatar_colliders, build_scene,
    front_surface_z, AvatarConfig, ClothBuilder, Collider, FabricPreset,
    PanelMaterial, PanelMesh, Placement, SceneKind, SignedDistanceField, Simulation, SimParams,
};
use eframe::{egui, egui_wgpu};
use glam::{EulerRot, Quat, Vec2, Vec3};
use std::collections::{BTreeMap, HashMap, VecDeque};
use std::sync::mpsc::{self, Receiver, TryRecvError};
use std::sync::Arc;
use pattern_core::{
    BoundaryCurve, Curve2D, CurveParam, CurveSegment, EdgePath, EdgeRef, FoldKind, FoldLine, GrainLine,
    suggest_seams, Notch, PatternPiece, PatternProject, Pleat, SeamDefinition, SeamSuggestion,
    SeamSuggestionParams, Point2 as PatternPoint2, SymmetryLink,
};
use pattern_geom::{
    cut_closed_outline, mirror_curve_x, offset_polygon, point_in_polygon, rectangle,
    regular_polygon, resample_curve, sample_curve, sample_curve_with_ranges,
    sample_pattern_outlines_with_seam_sync, validate_simple_polygon, CutAndSewResult,
    SourceSegmentFragment,
};
use pattern_mesh::{triangulate_outline, triangulate_outline_with_forced_vertices};

const FIXED_DT: f32 = 1.0 / 60.0;
const DEFAULT_RES: u32 = 40;
const PATTERN_PIXELS_PER_METER: f32 = 360.0;
const PATTERN_SIM_BODY_RADIUS: f32 = 0.14;
const PATTERN_SIM_PERTURBATION: f32 = 0.001;
/// Diferença de comprimento A/B a partir da qual a costura merece revisão (1 polegada).
const SEAM_LENGTH_WARNING_METERS: f64 = 0.0254;
const PATTERN_SIM_LAYER_SPACING: f32 = 0.003;
const GIZMO_MOVE_LENGTH: f32 = 0.42;
const GIZMO_SCALE_LENGTH: f32 = 0.70;
const GIZMO_ROTATE_RADIUS: f32 = 0.34;



type LocalFoldEdge = (u32, u32, f32, f32);

fn bounded_placement_translation(
    current: [f64; 3],
    requested_delta: Vec3,
) -> Option<([f64; 3], Vec3)> {
    if current.iter().any(|value| !value.is_finite()) || !requested_delta.is_finite() {
        return None;
    }
    let target = [
        (current[0] + requested_delta.x as f64).clamp(-100.0, 100.0),
        (current[1] + requested_delta.y as f64).clamp(-100.0, 100.0),
        (current[2] + requested_delta.z as f64).clamp(-100.0, 100.0),
    ];
    let applied = Vec3::new(
        (target[0] - current[0]) as f32,
        (target[1] - current[1]) as f32,
        (target[2] - current[2]) as f32,
    );
    if !applied.is_finite() { return None; }
    Some((target, applied))
}

fn curve_points_at_spacing(curve: &Curve2D, spacing: f64) -> Option<Vec<PatternPoint2>> {
    let spacing = if spacing.is_finite() { spacing.clamp(0.003, 0.10) } else { 0.02 };
    let sampled = resample_curve(curve, spacing).ok()?;
    let points = sample_curve(&sampled, 1);
    (points.len() >= 2).then_some(points)
}

fn offset_open_curve(
    curve: &Curve2D,
    offset_meters: f64,
    sampling_spacing: f64,
) -> Option<Curve2D> {
    if !offset_meters.is_finite() { return None; }
    let points = curve_points_at_spacing(curve, sampling_spacing)?;
    let mut shifted = Vec::with_capacity(points.len());
    for index in 0..points.len() {
        let (a, b) = if index == 0 {
            (points[0], points[1])
        } else if index + 1 == points.len() {
            (points[index - 1], points[index])
        } else {
            (points[index - 1], points[index + 1])
        };
        let tangent = PatternPoint2::new(b.x - a.x, b.y - a.y);
        let length = (tangent.x * tangent.x + tangent.y * tangent.y).sqrt();
        if length <= 1.0e-9 { return None; }
        let normal = PatternPoint2::new(-tangent.y / length, tangent.x / length);
        shifted.push(PatternPoint2::new(
            points[index].x + normal.x * offset_meters,
            points[index].y + normal.y * offset_meters,
        ));
    }
    Curve2D::polyline(&shifted, false).ok()
}

/// Mapeia uma curva interna para uma cadeia de arestas reais da malha. Cada trecho amostrado
/// encontra os vértices mais próximos e liga-os por caminho na adjacência triangular; só arestas
/// internas compartilhadas por duas faces podem gerar restrições de flexão.
fn trace_curve_to_mesh_edges(
    vertices: &[PatternPoint2],
    triangles: &[[u32; 3]],
    curve: &Curve2D,
    sampling_spacing: f64,
    stiffness: f32,
    base_angle: f32,
) -> Vec<LocalFoldEdge> {
    if vertices.len() < 3 || triangles.is_empty() { return Vec::new(); }
    let Some(path_points) = curve_points_at_spacing(curve, sampling_spacing * 0.5) else {
        return Vec::new();
    };

    let mut adjacency = vec![Vec::<usize>::new(); vertices.len()];
    let mut face_count = HashMap::<(usize, usize), usize>::new();
    for triangle in triangles {
        for k in 0..3 {
            let a = triangle[k] as usize;
            let b = triangle[(k + 1) % 3] as usize;
            if a >= vertices.len() || b >= vertices.len() || a == b { continue; }
            let key = if a < b { (a, b) } else { (b, a) };
            *face_count.entry(key).or_insert(0) += 1;
            adjacency[a].push(b);
            adjacency[b].push(a);
        }
    }
    for neighbors in &mut adjacency {
        neighbors.sort_unstable();
        neighbors.dedup();
    }

    let nearest_vertex = |point: PatternPoint2| -> (usize, f64) {
        vertices.iter().enumerate().min_by(|(_, a), (_, b)| {
            let da = (a.x - point.x).powi(2) + (a.y - point.y).powi(2);
            let db = (b.x - point.x).powi(2) + (b.y - point.y).powi(2);
            da.total_cmp(&db)
        }).map(|(index, vertex)| {
            (index, ((vertex.x - point.x).powi(2) + (vertex.y - point.y).powi(2)).sqrt())
        }).unwrap_or((0, f64::INFINITY))
    };

    // Evita projetar uma dobra em arestas remotas quando a curva sai do molde ou a resolução
    // da malha não consegue representar o traçado com proximidade razoável.
    let max_snap_distance = sampling_spacing.max(0.005) * 0.9;
    let stiffness = stiffness.clamp(0.02, 1.0);
    let mut mapped = BTreeMap::<(usize, usize), (f32, f32)>::new();
    for pair in path_points.windows(2) {
        let (source, source_distance) = nearest_vertex(pair[0]);
        let (target, target_distance) = nearest_vertex(pair[1]);
        if !source_distance.is_finite() || !target_distance.is_finite()
            || source_distance > max_snap_distance || target_distance > max_snap_distance
            || source == target
        {
            continue;
        }

        // BFS sobre o grafo da malha; o caminho curto mantém a linha física na região mais
        // próxima do traçado sem inventar vértices nem modificar a triangulação CDT.
        let mut parent = vec![usize::MAX; vertices.len()];
        let mut queue = VecDeque::new();
        parent[source] = source;
        queue.push_back(source);
        while let Some(current) = queue.pop_front() {
            if current == target { break; }
            for &neighbor in &adjacency[current] {
                if parent[neighbor] == usize::MAX {
                    parent[neighbor] = current;
                    queue.push_back(neighbor);
                }
            }
        }
        if parent[target] == usize::MAX { continue; }
        let mut path = vec![target];
        let mut current = target;
        while current != source {
            current = parent[current];
            if current == usize::MAX { break; }
            path.push(current);
        }
        if path.last().copied() != Some(source) { continue; }
        path.reverse();

        let tangent = Vec2::new(
            (pair[1].x - pair[0].x) as f32,
            (pair[1].y - pair[0].y) as f32,
        );
        if tangent.length_squared() <= 1.0e-12 { continue; }
        for edge in path.windows(2) {
            let (a, b) = (edge[0].min(edge[1]), edge[0].max(edge[1]));
            if face_count.get(&(a, b)).copied().unwrap_or(0) < 2 { continue; }
            let canonical_direction = Vec2::new(
                (vertices[b].x - vertices[a].x) as f32,
                (vertices[b].y - vertices[a].y) as f32,
            );
            let orientation = if canonical_direction.dot(tangent) < 0.0 { -1.0 } else { 1.0 };
            let angle = base_angle * orientation;
            mapped.entry((a, b))
                .and_modify(|spec| {
                    spec.0 = spec.0.min(stiffness);
                    spec.1 = (spec.1 + angle) * 0.5;
                })
                .or_insert((stiffness, angle));
        }
    }
    mapped.into_iter().map(|((a, b), (s, angle))| {
        (a as u32, b as u32, s, angle)
    }).collect()
}

/// Cria dobradiças físicas para linhas de dobra e os dois vincos opostos de uma prega.
/// Linhas Decorativas continuam apenas visuais. O mapeamento segue arestas já existentes,
/// sem retopologizar nem inserir vértices na CDT.
fn merge_fold_curve_edges(
    combined: &mut BTreeMap<(u32, u32), (f32, f32)>,
    vertices: &[PatternPoint2],
    triangles: &[[u32; 3]],
    curve: &Curve2D,
    stiffness: f32,
    angle: f32,
    spacing: f64,
) {
    for (a, b, local_stiffness, rest_angle) in trace_curve_to_mesh_edges(
        vertices, triangles, curve, spacing, stiffness, angle,
    ) {
        combined.entry((a, b))
            .and_modify(|spec| {
                spec.0 = spec.0.min(local_stiffness);
                spec.1 = (spec.1 + rest_angle) * 0.5;
            })
            .or_insert((local_stiffness, rest_angle));
    }
}

/// Cria dobradiças físicas para linhas de dobra e os dois vincos opostos de uma prega.
/// Linhas Decorativas continuam apenas visuais. O mapeamento segue arestas já existentes,
/// sem retopologizar nem inserir vértices na CDT.
fn physical_fold_edges(
    vertices: &[PatternPoint2],
    triangles: &[[u32; 3]],
    piece: &PatternPiece,
) -> Vec<LocalFoldEdge> {
    let mut combined = BTreeMap::<(u32, u32), (f32, f32)>::new();
    for fold in &piece.fold_lines {
        let base_angle = match fold.kind {
            FoldKind::Valley => fold.angle_radians as f32,
            FoldKind::Mountain => -(fold.angle_radians as f32),
            FoldKind::Decorative => continue,
        };
        merge_fold_curve_edges(
            &mut combined, vertices, triangles, &fold.curve,
            fold.stiffness as f32, base_angle, piece.fabric.particle_distance_meters,
        );
    }

    for pleat in &piece.pleats {
        // A profundidade representa a distância entre os dois vincos; cada vinco fica a
        // metade desse valor do eixo central e recebe um ângulo oposto ao outro.
        let half_depth = pleat.depth_meters * 0.5;
        let Some(first_crease) = offset_open_curve(
            &pleat.axis, -half_depth, piece.fabric.particle_distance_meters * 0.5,
        ) else { continue; };
        let Some(second_crease) = offset_open_curve(
            &pleat.axis, half_depth, piece.fabric.particle_distance_meters * 0.5,
        ) else { continue; };
        let outline_polygon = sample_curve(&piece.outline, 32);
        if !curve_samples_inside_polygon(&first_crease, &outline_polygon, 12)
            || !curve_samples_inside_polygon(&second_crease, &outline_polygon, 12)
        {
            continue;
        }
        let angle = pleat.angle_radians as f32;
        merge_fold_curve_edges(
            &mut combined, vertices, triangles, &first_crease,
            0.22, angle, piece.fabric.particle_distance_meters,
        );
        merge_fold_curve_edges(
            &mut combined, vertices, triangles, &second_crease,
            0.22, -angle, piece.fabric.particle_distance_meters,
        );
    }

    combined.into_iter().map(|((a, b), (stiffness, angle))| {
        (a, b, stiffness, angle)
    }).collect()
}

/// Profundidade em que um molde plano, voltado para +Z, fica em frente à superfície corporal sob sua área.
/// Retorna `None` quando não há corpo sob o molde (por exemplo, fora da silhueta).
fn front_projection_z(
    colliders: &[Collider],
    mesh_vertices: &[PatternPoint2],
    origin_y: f32,
    clearance: f32,
) -> Option<f32> {
    let stride = (mesh_vertices.len() / 160).max(1);
    mesh_vertices
        .iter()
        .step_by(stride)
        .filter_map(|p| front_surface_z(colliders, p.x as f32, origin_y + p.y as f32, clearance))
        .fold(None, |best: Option<f32>, z| Some(best.map_or(z, |b| b.max(z))))
}

/// Raio de montagem de fallback quando não há superfície corporal sob o molde.
fn pattern_sim_shell_radius(piece_index: usize, thickness: f32) -> f32 {
    PATTERN_SIM_BODY_RADIUS + thickness + PATTERN_SIM_PERTURBATION
        + piece_index as f32 * PATTERN_SIM_LAYER_SPACING
}

fn main() -> eframe::Result {
    let options = eframe::NativeOptions {
        viewport: egui::ViewportBuilder::default()
            .with_title("ClothEvo")
            .with_inner_size([1360.0, 840.0])
            .with_min_inner_size([900.0, 600.0]),
        renderer: eframe::Renderer::Wgpu,
        // O viewport 3D precisa de depth buffer no render pass do egui.
        depth_buffer: 32,
        ..Default::default()
    };
    eframe::run_native(
        "ClothEvo",
        options,
        Box::new(|cc| Ok(Box::new(ClothEvoApp::new(cc)))),
    )
}

/// Parâmetros do modo de captura.
struct Capture {
    path: String,
    frames: u32,
    requested: bool,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum WorkspaceMode {
    Simulation,
    Pattern,
    /// Posiciona os painéis em 3D antes de liberar a simulação.
    Placement,
    /// Viewport 3D e editor 2D visíveis ao mesmo tempo.
    Split,
}

impl WorkspaceMode {
    /// A simulação avança sempre que a viewport 3D está visível.
    fn runs_simulation(self) -> bool {
        matches!(self, Self::Simulation | Self::Split)
    }
}

#[derive(Clone, Copy, Debug)]
struct PanelPlacementRange {
    piece_id: u64,
    first: usize,
    end: usize,
    boundary_count: usize,
    base_rotation: Quat,
    pivot: Vec3,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum GizmoMode {
    Move,
    Rotate,
    Scale,
    Transform,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum GizmoAction {
    Move,
    Rotate,
    Scale,
}

#[derive(Clone, Copy, Debug)]
struct PlacementGizmoDrag {
    piece_id: u64,
    axis: usize,
    action: GizmoAction,
    axis_world: Vec3,
    screen_axis: egui::Vec2,
    screen_center: egui::Pos2,
    pixels_per_meter: f32,
    last_pointer: egui::Pos2,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum PatternTool {
    Draw,
    Select,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum PieceTransformMode { Move, Rotate, Scale }
#[derive(Clone, Copy)]
struct PieceTransformDrag { piece_index: usize, pivot: PatternPoint2, last_pointer: PatternPoint2 }
#[derive(Clone, Copy, Debug, PartialEq)]
enum PatternEditTarget {
    Piece { piece: usize },
    Anchor { piece: usize, index: usize },
    Control1 { piece: usize, segment: usize },
    Control2 { piece: usize, segment: usize },
    Edge { piece: usize, segment: usize, t: f64 },
}

#[derive(Clone, Debug, PartialEq)]
struct PatternEditorSnapshot {
    project: PatternProject,
    draft_points: Vec<PatternPoint2>,
    selected_piece: usize,
    selected_target: Option<PatternEditTarget>,
}

#[derive(Clone, Debug)]
struct PieceSampleMapping {
    ranges: Vec<(usize, usize)>,
    /// (segment, t, índice de vértice na borda amostrada).
    forced_vertices: Vec<(u32, f64, usize)>,
}

impl PatternEditTarget {
    fn piece(self) -> usize {
        match self {
            Self::Piece { piece } | Self::Anchor { piece, .. } | Self::Control1 { piece, .. }
            | Self::Control2 { piece, .. } | Self::Edge { piece, .. } => piece,
        }
    }

    fn is_movable(self) -> bool {
        !matches!(self, Self::Edge { .. })
    }
}

struct AvatarImportJobResult {
    path: std::path::PathBuf,
    refresh_viewport: bool,
    mark_project_dirty: bool,
    target_height_m: f32,
    collision_spacing_m: f32,
    heading_degrees: f32,
    result: Result<avatar_import::ImportedAvatar, String>,
}

struct ClothEvoApp {
    sim: Simulation,
    kind: SceneKind,
    res: u32,
    preset: FabricPreset,
    playing: bool,
    accumulator: f32,
    camera: OrbitCamera,
    style: ViewStyle,
    seam_pairs: Vec<(u32, u32)>,
    normals: Vec<Vec3>,
    render_state: Option<egui_wgpu::RenderState>,
    diverged: bool,
    last_step_ms: f32,
    capture: Option<Capture>,
    /// Cenas de validação só são usadas quando pedidas explicitamente por CLOTHEVO_SCENE/captura.
    validation_scene: bool,
    /// Painel de métricas é aberto pelo menu View.
    show_metrics: bool,
    /// Preferências da interface, salvas fora dos arquivos de projeto de vestuário.
    ui_customization: UiCustomization,
    show_ui_customizer: bool,
    ui_customizer_new_command: UiCommand,
    ui_customizer_new_label: String,
    ui_customization_message: String,
    ui_theme_dirty: bool,
    workspace: WorkspaceMode,
    /// No modo dividido: o editor 2D fica no primeiro painel (esquerda ou topo).
    split_swapped: bool,
    /// No modo dividido: painéis empilhados (topo/base) em vez de lado a lado.
    split_vertical: bool,
    pattern_project: PatternProject,
    draft_points: Vec<PatternPoint2>,
    pattern_spacing: f64,
    pattern_polygon_sides: usize,
    pattern_symmetry_axis_x: f64,
    pattern_show_mesh: bool,
    selected_piece: usize,
    seam_a_piece: usize,
    seam_b_piece: usize,
    seam_sides_a: Vec<EdgePath>,
    seam_sides_b: Vec<EdgePath>,
    seam_a_active_path: usize,
    seam_b_active_path: usize,
    seam_active_side_a: bool,
    seam_a_segment: usize,
    seam_b_segment: usize,
    /// Fim exclusivo do intervalo selecionado (na UI, o último segmento é inclusivo).
    seam_a_segment_end: usize,
    seam_b_segment_end: usize,
    seam_a_reversed: bool,
    seam_b_reversed: bool,
    seam_a_start_t: f64,
    seam_a_end_t: f64,
    seam_b_start_t: f64,
    seam_b_end_t: f64,
    seam_free_mode: bool,
    /// Captura direta de arestas no canvas para construir uma costura sem navegar por combos.
    seam_pick_mode: bool,
    /// No modo rápido, o primeiro clique seleciona A e o segundo conecta B automaticamente.
    seam_pick_pair_mode: bool,
    /// Estado transitório do atalho Shift+LMB para costura direta entre duas arestas.
    direct_sewing_start: Option<(usize, usize)>,
    /// Aresta selecionada por clique normal, caso o próximo gesto use Shift+clique na borda B.
    direct_sewing_preselected: Option<(usize, usize)>,
    direct_sewing_press_edge: Option<(usize, usize)>,
    direct_sewing_preview: Option<(usize, usize)>,
    direct_sewing_dragging: bool,
    /// Mantém cadeias capturadas como a fonte da costura, mesmo depois de sair do modo de seleção.
    seam_canvas_pick_used: bool,
    seam_snap_tolerance_px: f32,
    dragging_seam_handle: Option<(bool, bool)>,
    seam_drag_before: Option<PatternEditorSnapshot>,
    seam_ease_m: f64,
    seam_name: String,
    selected_seam_id: Option<u64>,
    auto_seam_suggestions: Vec<SeamSuggestion>,
    auto_seam_selected: Vec<bool>,
    hovered_auto_suggestion: Option<usize>,
    auto_seam_minimum_score: f64,
    notch_segment: usize,
    notch_t: f64,
    notch_depth_m: f64,
    notch_width_m: f64,
    selected_notch_index: usize,
    pattern_message: String,
    using_pattern_sim: bool,
    placement_ranges: Vec<PanelPlacementRange>,
    selected_placement_piece: Option<u64>,
    selected_arrangement_point: Option<String>,
    placement_drag_piece: Option<u64>,
    placement_drag_before: Option<PatternEditorSnapshot>,
    placement_gizmo_drag: Option<PlacementGizmoDrag>,
    placement_drag_selected_body: bool,
    gizmo_mode: GizmoMode,
    avatar_guide_texture: Option<egui::TextureHandle>,
    show_avatar_guide: bool,
    avatar_guide_backdrop: f32,
    /// Malha visual importada; inclui arestas limitadas para inspeção em wireframe.
    avatar_mesh: Option<StaticMeshData>,
    /// Campo de colisão estático gerado da mesma geometria e transformação visual.
    avatar_sdf: Option<Arc<SignedDistanceField>>,
    avatar_mesh_name: Option<String>,
    avatar_mesh_path: Option<std::path::PathBuf>,
    avatar_diagnostics: Option<avatar_import::AvatarDiagnostics>,
    avatar_visible: bool,
    avatar_wireframe: bool,
    avatar_collision_enabled: bool,
    avatar_collision_spacing_m: f32,
    avatar_heading_degrees: f32,
    avatar_imported_height_m: Option<f32>,
    avatar_imported_heading_degrees: Option<f32>,
    avatar_imported_collision_spacing_m: Option<f32>,
    avatar_import_receiver: Option<Receiver<AvatarImportJobResult>>,
    avatar_import_pending_path: Option<std::path::PathBuf>,
    pattern_tool: PatternTool,
    pattern_zoom: f32,
    pattern_pan: egui::Vec2,
    pattern_snap: bool,
    selected_target: Option<PatternEditTarget>,
    dragging_target: Option<PatternEditTarget>,
    piece_transform_mode: PieceTransformMode,
    piece_transform_drag: Option<PieceTransformDrag>,
    camera_goal: Option<CameraGoal>,
    dragging_before: Option<PatternEditorSnapshot>,
    drag_last_world: Option<PatternPoint2>,
    pattern_undo: Vec<PatternEditorSnapshot>,
    pattern_redo: Vec<PatternEditorSnapshot>,
    pattern_saved_path: Option<std::path::PathBuf>,
    pattern_dirty: bool,
    pattern_fit_pending: bool,
}

fn statics_from(colliders: &[Collider]) -> Vec<StaticMeshData> {
    let mut out = Vec::new();
    for collider in colliders {
        match collider {
            Collider::Plane { normal, offset } => {
                if normal.dot(Vec3::Y) > 0.99 {
                    out.push(primitives::ground_quad(*offset, 8.0, [0.62, 0.64, 0.66]));
                }
            }
            Collider::Sphere { center, radius } => {
                out.push(primitives::uv_sphere(*center, *radius, 48, 24, [0.78, 0.78, 0.80]));
            }
            Collider::Ellipsoid { center, radii } => {
                out.push(primitives::ellipsoid(*center, *radii, 40, 20, [0.69, 0.73, 0.77]));
            }
            Collider::Capsule { a, b, radius } => {
                out.push(primitives::cylinder(*a, *b, *radius, 32, [0.69, 0.73, 0.77]));
                out.push(primitives::uv_sphere(*a, *radius, 24, 12, [0.69, 0.73, 0.77]));
                out.push(primitives::uv_sphere(*b, *radius, 24, 12, [0.69, 0.73, 0.77]));
            }
            Collider::Cylinder { a, b, radius } => {
                out.push(primitives::cylinder(*a, *b, *radius, 48, [0.70, 0.76, 0.70]));
            }
            // O avatar importado é renderizado pela malha visual separada, não pelo colisor.
            Collider::TriangleMeshSdf { .. } => {}
        }
    }
    out
}

impl ClothEvoApp {
    fn new(cc: &eframe::CreationContext<'_>) -> Self {
        let (ui_customization, ui_load_message) = UiCustomization::load();
        let startup_split_vertical = ui_customization.split_vertical;
        let startup_split_swapped = ui_customization.split_swapped;
        let mut visuals = egui::Visuals::dark();
        visuals.panel_fill = egui::Color32::from_rgb(15, 19, 35);
        visuals.window_fill = egui::Color32::from_rgb(23, 27, 45);
        visuals.extreme_bg_color = egui::Color32::from_rgb(8, 12, 25);
        visuals.faint_bg_color = egui::Color32::from_rgb(28, 33, 54);
        visuals.code_bg_color = egui::Color32::from_rgb(11, 16, 31);
        visuals.hyperlink_color = egui::Color32::from_rgb(255, 180, 65);
        visuals.selection.bg_fill = egui::Color32::from_rgb(205, 57, 135);
        visuals.selection.stroke = egui::Stroke::new(1.0, egui::Color32::from_rgb(255, 193, 80));
        visuals.widgets.inactive.bg_fill = egui::Color32::from_rgb(31, 39, 65);
        visuals.widgets.hovered.bg_fill = egui::Color32::from_rgb(53, 61, 98);
        visuals.widgets.active.bg_fill = egui::Color32::from_rgb(205, 57, 135);
        visuals.widgets.noninteractive.fg_stroke.color = egui::Color32::from_rgb(236, 232, 219);
        cc.egui_ctx.set_visuals(visuals);

        let render_state = cc.wgpu_render_state.clone();
        if let Some(rs) = &render_state {
            let depth = egui_wgpu::depth_format_from_bits(32, 0)
                .expect("formato de profundidade Depth32Float");
            let renderer = SceneRenderer::new(&rs.device, rs.target_format, depth);
            rs.renderer.write().callback_resources.insert(renderer);
        }

        // Modo de captura (opcional)
        let capture = std::env::var("CLOTHEVO_SCREENSHOT").ok().map(|path| Capture {
            path,
            frames: std::env::var("CLOTHEVO_FRAMES").ok().and_then(|s| s.parse().ok()).unwrap_or(120),
            requested: false,
        });
        let validation_scene = capture.is_some() || std::env::var("CLOTHEVO_SCENE").is_ok();
        let kind = std::env::var("CLOTHEVO_SCENE")
            .ok()
            .and_then(|s| s.parse::<usize>().ok())
            .and_then(|i| SceneKind::ALL.get(i.wrapping_sub(1)).copied())
            .unwrap_or(SceneKind::HangingCloth);

        let params = SimParams::default();
        let sim = if validation_scene {
            // Cenas de benchmark/captura continuam disponíveis para CI, mas não são a experiência padrão.
            build_scene(kind, DEFAULT_RES, params.clone())
        } else {
            // Uma sessão nova começa sem pano de demonstração: chão e avatar prontos para receber moldes.
            let mut builder = ClothBuilder::new();
            builder.add_collider(Collider::ground());
            if let Ok(colliders) = build_avatar_colliders(&AvatarConfig::default()) {
                for collider in colliders {
                    builder.add_collider(collider);
                }
            }
            builder.build(params.clone())
        };
        let mut app = Self {
            seam_pairs: sim.seam_pairs(),
            sim,
            kind,
            res: DEFAULT_RES,
            preset: FabricPreset::Cotton,
            playing: validation_scene,
            accumulator: 0.0,
            camera: OrbitCamera::default(),
            style: ViewStyle::default(),
            normals: Vec::new(),
            render_state,
            diverged: false,
            last_step_ms: 0.0,
            capture,
            validation_scene,
            show_metrics: false,
            ui_customization,
            show_ui_customizer: false,
            ui_customizer_new_command: UiCommand::ToggleSimulation,
            ui_customizer_new_label: String::new(),
            ui_customization_message: ui_load_message.unwrap_or_default(),
            ui_theme_dirty: false,
            workspace: WorkspaceMode::Simulation,
            split_swapped: startup_split_swapped,
            split_vertical: startup_split_vertical,
            pattern_project: PatternProject::new("Novo projeto"),
            draft_points: Vec::new(),
            pattern_spacing: 0.08,
            pattern_polygon_sides: 6,
            pattern_symmetry_axis_x: 0.0,
            pattern_show_mesh: true,
            selected_piece: 0,
            seam_a_piece: 0,
            seam_b_piece: 1,
            seam_sides_a: Vec::new(),
            seam_sides_b: Vec::new(),
            seam_a_active_path: 0,
            seam_b_active_path: 0,
            seam_active_side_a: true,
            seam_a_segment: 0,
            seam_b_segment: 0,
            seam_a_segment_end: 1,
            seam_b_segment_end: 1,
            seam_a_reversed: false,
            seam_b_reversed: false,
            seam_a_start_t: 0.0,
            seam_a_end_t: 1.0,
            seam_b_start_t: 0.0,
            seam_b_end_t: 1.0,
            seam_free_mode: false,
            seam_pick_mode: false,
            seam_pick_pair_mode: false,
            direct_sewing_start: None,
            direct_sewing_preselected: None,
            direct_sewing_press_edge: None,
            direct_sewing_preview: None,
            direct_sewing_dragging: false,
            seam_canvas_pick_used: false,
            seam_snap_tolerance_px: 10.0,
            dragging_seam_handle: None,
            seam_drag_before: None,
            seam_ease_m: 0.0,
            seam_name: String::new(),
            selected_seam_id: None,
            auto_seam_suggestions: Vec::new(),
            auto_seam_selected: Vec::new(),
            hovered_auto_suggestion: None,
            auto_seam_minimum_score: 0.35,
            notch_segment: 0,
            notch_t: 0.5,
            notch_depth_m: 0.008,
            notch_width_m: 0.012,
            selected_notch_index: 0,
            pattern_message: String::new(),
            using_pattern_sim: false,
            placement_ranges: Vec::new(),
            selected_placement_piece: None,
            selected_arrangement_point: None,
            placement_drag_piece: None,
            placement_drag_before: None,
            placement_gizmo_drag: None,
            placement_drag_selected_body: false,
            gizmo_mode: GizmoMode::Move,
            avatar_guide_texture: None,
            show_avatar_guide: true,
            avatar_guide_backdrop: 0.25,
            avatar_mesh: None,
            avatar_sdf: None,
            avatar_mesh_name: None,
            avatar_mesh_path: None,
            avatar_diagnostics: None,
            avatar_visible: true,
            avatar_wireframe: false,
            avatar_collision_enabled: true,
            avatar_collision_spacing_m: 0.01,
            avatar_heading_degrees: 0.0,
            avatar_imported_height_m: None,
            avatar_imported_heading_degrees: None,
            avatar_imported_collision_spacing_m: None,
            avatar_import_receiver: None,
            avatar_import_pending_path: None,
            pattern_tool: PatternTool::Draw,
            pattern_zoom: PATTERN_PIXELS_PER_METER,
            pattern_pan: egui::Vec2::ZERO,
            pattern_snap: false,
            selected_target: None,
            dragging_target: None,
            piece_transform_mode: PieceTransformMode::Move,
            piece_transform_drag: None,
            camera_goal: None,
            dragging_before: None,
            drag_last_world: None,
            pattern_undo: Vec::new(),
            pattern_redo: Vec::new(),
            pattern_saved_path: None,
            pattern_dirty: false,
            pattern_fit_pending: false,
        };
        app.try_load_default_avatar();
        if !app.validation_scene {
            app.focus_camera_preset(CameraPreset::Front, 1360.0 / 840.0);
        }
        app.apply_ui_customization(&cc.egui_ctx);
        app.upload_scene();
        app
    }

    fn schedule_avatar_import(
        &mut self,
        path: &std::path::Path,
        refresh_viewport: bool,
        mark_project_dirty: bool,
    ) {
        let path = path.to_path_buf();
        let target_height_m = self.pattern_project.avatar.height_meters as f32;
        let collision_spacing_m = self.avatar_collision_spacing_m;
        let heading_degrees = self.avatar_heading_degrees;
        let worker_path = path.clone();
        let (sender, receiver) = mpsc::channel();
        let worker = std::thread::Builder::new()
            .name("clothevo-avatar-import".into())
            .spawn(move || {
                let result = avatar_import::load_model(
                    &worker_path,
                    target_height_m,
                    collision_spacing_m,
                    heading_degrees,
                );
                let _ = sender.send(AvatarImportJobResult {
                    path: worker_path,
                    refresh_viewport,
                    mark_project_dirty,
                    target_height_m,
                    collision_spacing_m,
                    heading_degrees,
                    result,
                });
            });
        if let Err(error) = worker {
            self.pattern_message = format!("Não foi possível iniciar a importação do avatar: {error}");
            return;
        }
        self.avatar_import_receiver = Some(receiver);
        self.avatar_import_pending_path = Some(path.clone());
        self.pattern_message = format!(
            "Importando {} e preparando a colisão em segundo plano…",
            path.file_name().and_then(|name| name.to_str()).unwrap_or("avatar")
        );
    }

    fn poll_avatar_import(&mut self) {
        let result = match self.avatar_import_receiver.as_ref() {
            Some(receiver) => receiver.try_recv(),
            None => return,
        };
        match result {
            Ok(job) => {
                self.avatar_import_receiver = None;
                self.avatar_import_pending_path = None;
                match job.result {
                    Ok(imported) => {
                        let display_name = imported.display_name.clone();
                        let has_sdf = imported.collision_sdf.is_some();
                        let warning = imported.collision_warning.clone();
                        self.avatar_mesh = Some(imported.mesh);
                        self.avatar_sdf = imported.collision_sdf;
                        if job.mark_project_dirty {
                            self.pattern_dirty = true;
                        }
                        self.avatar_mesh_name = Some(display_name.clone());
                        self.avatar_mesh_path = Some(job.path);
                        self.avatar_diagnostics = Some(imported.diagnostics);
                        self.avatar_imported_height_m = Some(job.target_height_m);
                        self.avatar_imported_heading_degrees = Some(job.heading_degrees);
                        self.avatar_imported_collision_spacing_m = Some(job.collision_spacing_m);
                        self.pattern_message = if has_sdf {
                            format!("Avatar '{display_name}' importado e preparado para colisão por SDF.")
                        } else {
                            format!("Avatar '{display_name}' importado para visualização; a colisão usará o manequim procedural.")
                        };
                        if let Some(warning) = warning {
                            self.pattern_message.push_str(&format!(" Aviso: {warning}"));
                        }
                        if let Err(error) = self.sync_pattern_avatar_colliders() {
                            self.pattern_message.push_str(&format!(" Falha ao atualizar o solver: {error}"));
                        }
                        if job.refresh_viewport {
                            self.upload_scene();
                            if !self.validation_scene && !self.using_pattern_sim {
                                self.focus_camera_preset(CameraPreset::Front, 1360.0 / 840.0);
                            }
                        }
                    }
                    Err(error) => {
                        self.pattern_message = error;
                    }
                }
            }
            Err(TryRecvError::Empty) => {}
            Err(TryRecvError::Disconnected) => {
                self.avatar_import_receiver = None;
                self.avatar_import_pending_path = None;
                self.pattern_message = "A tarefa de importação do avatar foi encerrada sem resultado.".into();
            }
        }
    }

    fn avatar_body_colliders(&self) -> Result<Vec<Collider>, String> {
        if !self.avatar_collision_enabled {
            return Ok(Vec::new());
        }
        if let Some(field) = self.avatar_sdf.as_ref() {
            return Ok(vec![Collider::TriangleMeshSdf {
                field: Arc::clone(field),
            }]);
        }
        let avatar = AvatarConfig {
            height_m: self.pattern_project.avatar.height_meters as f32,
            shoulder_width_m: self.pattern_project.avatar.shoulder_width_meters as f32,
            waist_width_m: self.pattern_project.avatar.waist_width_meters as f32,
            hip_width_m: self.pattern_project.avatar.hip_width_meters as f32,
        };
        build_avatar_colliders(&avatar)
    }

    fn avatar_settings_are_stale(&self) -> bool {
        if self.avatar_mesh_path.is_none() {
            return false;
        }
        let height = self.pattern_project.avatar.height_meters as f32;
        self.avatar_imported_height_m
            .is_none_or(|applied| (applied - height).abs() > 1.0e-4)
            || self.avatar_imported_heading_degrees
                .is_none_or(|applied| (applied - self.avatar_heading_degrees).abs() > 1.0e-3)
            || self.avatar_imported_collision_spacing_m
                .is_none_or(|applied| (applied - self.avatar_collision_spacing_m).abs() > 1.0e-6)
    }

    fn sync_pattern_avatar_colliders(&mut self) -> Result<(), String> {
        // Preserve explicit validation scenes. In the normal editor and custom-pattern simulation,
        // the exact imported SDF (or the procedural mannequin fallback) is the physical avatar.
        if self.validation_scene && !self.using_pattern_sim {
            return Ok(());
        }
        let mut colliders = vec![Collider::ground()];
        colliders.extend(self.avatar_body_colliders()?);
        self.sim.colliders = colliders;
        Ok(())
    }

    fn try_load_default_avatar(&mut self) {
        let mut candidates = Vec::new();
        if let Ok(executable) = std::env::current_exe() {
            if let Some(directory) = executable.parent() {
                candidates.push(directory.join("assets").join("Sarah.fbx"));
                candidates.push(directory.join("Sarah.fbx"));
            }
        }
        if let Ok(directory) = std::env::current_dir() {
            candidates.push(directory.join("assets").join("Sarah.fbx"));
            candidates.push(directory.join("Sarah.fbx"));
        }
        for candidate in candidates {
            if candidate.is_file() {
                self.schedule_avatar_import(&candidate, true, false);
                break;
            }
        }
    }

    fn import_avatar_dialog(&mut self) {
        let selected = rfd::FileDialog::new()
            .set_title("Importar avatar 3D")
            .add_filter("Modelos 3D (OBJ/FBX)", &["obj", "fbx"])
            .add_filter("Wavefront OBJ", &["obj"])
            .add_filter("Autodesk FBX", &["fbx"])
            .pick_file();
        let Some(path) = selected else { return; };
        self.schedule_avatar_import(&path, true, true);
    }

    /// Reconstrói a cena preservando os parâmetros atuais.
    fn load_scene(&mut self, kind: SceneKind, res: u32) {
        let params = self.sim.params.clone();
        self.kind = kind;
        self.using_pattern_sim = false;
        self.res = res;
        self.sim = build_scene(kind, res, params);
        self.seam_pairs = self.sim.seam_pairs();
        self.diverged = false;
        self.accumulator = 0.0;
        self.upload_scene();
    }

    /// Envia topologia do tecido e colisores para a GPU.
    fn upload_scene(&mut self) {
        let Some(rs) = &self.render_state else {
            return;
        };
        let edges = self.sim.mesh_edges();
        let mut statics = statics_from(&self.sim.colliders);
        if (!self.avatar_visible && !self.validation_scene)
            || (self.using_pattern_sim && self.avatar_sdf.is_some())
        {
            // A representação física pode continuar ativa sem duplicar o desenho do corpo.
            statics.retain(|mesh| mesh.kind == MeshKind::Ground);
        }
        if self.avatar_visible {
            if let Some(avatar) = self.avatar_mesh.as_ref() {
                let mut avatar = avatar.clone();
                if !self.avatar_wireframe {
                    avatar.wire_indices.clear();
                }
                statics.push(avatar);
            } else if !self.validation_scene && !self.avatar_collision_enabled {
                // O manequim procedural também precisa continuar visível quando sua função
                // de colisão é desligada. Nesse caso desenhamos seus volumes sem enviá-los ao solver.
                let avatar = AvatarConfig {
                    height_m: self.pattern_project.avatar.height_meters as f32,
                    shoulder_width_m: self.pattern_project.avatar.shoulder_width_meters as f32,
                    waist_width_m: self.pattern_project.avatar.waist_width_meters as f32,
                    hip_width_m: self.pattern_project.avatar.hip_width_meters as f32,
                };
                if let Ok(visual_colliders) = build_avatar_colliders(&avatar) {
                    statics.extend(statics_from(&visual_colliders));
                }
            }
        }
        let mut renderer = rs.renderer.write();
        if let Some(r) = renderer.callback_resources.get_mut::<SceneRenderer>() {
            r.set_statics(&rs.device, &statics);
            r.set_cloth_topology(
                &rs.device,
                &ClothTopology {
                    vertex_count: self.sim.particle_count(),
                    triangles: &self.sim.triangles,
                    wire_edges: &edges,
                    seam_pairs: &self.seam_pairs,
                },
            );
        }
    }

    fn advance(&mut self, frame_dt: f32) {
        if !self.playing {
            return;
        }
        // Passo fixo; se a simulação não acompanha o tempo real, ela simplesmente desacelera.
        self.accumulator = (self.accumulator + frame_dt).min(2.0 * FIXED_DT);
        while self.accumulator >= FIXED_DT {
            self.sim.step(FIXED_DT);
            self.accumulator -= FIXED_DT;
            if !self.sim.is_finite() {
                self.playing = false;
                self.diverged = true;
                self.accumulator = 0.0;
                self.sim.reset();
                return;
            }
        }
    }

    fn seam_gap_max(&self) -> f32 {
        self.seam_pairs
            .iter()
            .map(|&(a, b)| (self.sim.pos[a as usize] - self.sim.pos[b as usize]).length())
            .fold(0.0, f32::max)
    }

    fn top_navigation(&mut self, ui: &mut egui::Ui) {
        const ORANGE: egui::Color32 = egui::Color32::from_rgb(255, 170, 53);
        const PAPER: egui::Color32 = egui::Color32::from_rgb(247, 239, 215);

        ui.set_min_height(58.0);
        let mut render_changed = false;
        let mut open_ui_customizer = false;
        ui.horizontal(|ui| {
            ui.add_space(2.0);
            ui.vertical(|ui| {
                ui.label(egui::RichText::new("CLOTHEVO").size(23.0).strong().color(PAPER));
            });
            ui.add_space(16.0);
            ui.separator();
            ui.add_space(8.0);

            if ui.selectable_label(
                self.workspace == WorkspaceMode::Simulation,
                egui::RichText::new("3D Viewport").strong(),
            ).clicked() {
                self.workspace = WorkspaceMode::Simulation;
            }
            if ui.selectable_label(
                self.workspace == WorkspaceMode::Pattern,
                egui::RichText::new("2D Viewport").strong(),
            ).clicked() {
                self.playing = false;
                self.workspace = WorkspaceMode::Pattern;
            }
            if ui.selectable_label(
                self.workspace == WorkspaceMode::Placement,
                egui::RichText::new("Posicionamento 3D").strong(),
            ).clicked() {
                self.playing = false;
                let already_prepared = self.workspace == WorkspaceMode::Placement
                    && self.using_pattern_sim
                    && !self.placement_ranges.is_empty();
                if !already_prepared {
                    if self.pattern_project.pieces.is_empty() {
                        self.workspace = WorkspaceMode::Pattern;
                        self.pattern_message = "Crie pelo menos um molde antes de entrar no posicionamento 3D.".into();
                    } else if let Err(error) = self.start_pattern_simulation() {
                        self.pattern_message = error;
                        self.workspace = WorkspaceMode::Pattern;
                    }
                } else {
                    self.workspace = WorkspaceMode::Placement;
                }
            }

            ui.menu_button(
                egui::RichText::new("View  ↘").strong().color(ORANGE),
                |ui| {
                    if ui.selectable_label(self.workspace == WorkspaceMode::Split, "3D + 2D").clicked() {
                        self.workspace = WorkspaceMode::Split;
                        self.pattern_fit_pending = true;
                    }
                    ui.menu_button("Visualização de malhas e costura", |ui| {
                        render_changed |= ui.checkbox(&mut self.style.wireframe, "Mostrar malha / wireframe").changed();
                        render_changed |= ui.checkbox(&mut self.style.seams, "Mostrar linhas de costura").changed();
                    });
                    ui.separator();
                    ui.checkbox(&mut self.show_metrics, "Métricas");
                },
            );

            ui.menu_button(
                egui::RichText::new("Preferences  ↘").strong().color(ORANGE),
                |ui| {
                    if ui.button("Personalizar interface…").clicked() {
                        open_ui_customizer = true;
                        ui.close();
                    }
                    if ui.button("Salvar preferências da interface").clicked() {
                        self.save_ui_customization();
                        ui.close();
                    }
                },
            );
        });
        if open_ui_customizer {
            self.show_ui_customizer = true;
        }
        if render_changed {
            self.upload_scene();
        }
    }

    fn apply_ui_customization(&self, ctx: &egui::Context) {
        let theme = &self.ui_customization.theme;
        let mut style = (*ctx.style()).clone();
        let mut visuals = style.visuals.clone();
        visuals.panel_fill = ui_rgb(theme.panel_bg);
        visuals.window_fill = ui_rgb(theme.window_bg);
        visuals.extreme_bg_color = ui_rgb(theme.deep_bg);
        visuals.faint_bg_color = ui_rgb(theme.muted_bg);
        visuals.code_bg_color = ui_rgb(theme.deep_bg);
        visuals.hyperlink_color = ui_rgb(theme.accent);
        visuals.selection.bg_fill = ui_rgb(theme.selection);
        visuals.selection.stroke = egui::Stroke::new(1.0, ui_rgb(theme.accent));
        visuals.widgets.noninteractive.fg_stroke.color = ui_rgb(theme.text);
        visuals.widgets.inactive.fg_stroke.color = ui_rgb(theme.text);
        visuals.widgets.hovered.fg_stroke.color = ui_rgb(theme.text);
        visuals.widgets.active.fg_stroke.color = ui_rgb(theme.text);
        visuals.widgets.inactive.bg_fill = ui_rgb(theme.button_bg);
        visuals.widgets.hovered.bg_fill = ui_rgb(theme.button_hover);
        visuals.widgets.active.bg_fill = ui_rgb(theme.selection);
        visuals.widgets.noninteractive.bg_stroke.color = ui_rgb(theme.border);
        visuals.widgets.inactive.bg_stroke.color = ui_rgb(theme.border);
        visuals.widgets.hovered.bg_stroke.color = ui_rgb(theme.accent);
        visuals.widgets.active.bg_stroke.color = ui_rgb(theme.accent);
        let radius = egui::CornerRadius::same(theme.corner_radius);
        visuals.window_corner_radius = radius;
        visuals.widgets.noninteractive.corner_radius = radius;
        visuals.widgets.inactive.corner_radius = radius;
        visuals.widgets.hovered.corner_radius = radius;
        visuals.widgets.active.corner_radius = radius;
        style.visuals = visuals;
        style.spacing.item_spacing = egui::vec2(theme.spacing, theme.spacing);
        style.spacing.button_padding = egui::vec2(theme.spacing * 0.8, theme.spacing * 0.5);
        ctx.set_style(style);
        ctx.set_zoom_factor(theme.ui_scale.clamp(0.75, 1.75));
    }

    fn save_ui_customization(&mut self) {
        self.ui_customization.split_vertical = self.split_vertical;
        self.ui_customization.split_swapped = self.split_swapped;
        match self.ui_customization.save() {
            Ok(()) => {
                let path = UiCustomization::configuration_path()
                    .map(|path| path.display().to_string())
                    .unwrap_or_else(|_| "pasta de configurações do usuário".into());
                self.ui_customization_message = format!("Preferências salvas em: {path}");
            }
            Err(error) => self.ui_customization_message = error,
        }
    }

    fn custom_toolbar(&mut self, ui: &mut egui::Ui) {
        let buttons: Vec<(u64, String, UiCommand)> = self.ui_customization.toolbar_buttons
            .iter()
            .filter(|button| button.visible)
            .map(|button| (button.id, button.label.clone(), button.command))
            .collect();
        if buttons.is_empty() {
            return;
        }

        let mut actions = Vec::new();
        ui.horizontal_wrapped(|ui| {
            ui.label(egui::RichText::new("MINHA BARRA").size(9.0).strong());
            for (id, label, command) in buttons {
                if ui.push_id(("custom_ui_command", id), |ui| ui.button(label)).inner.clicked() {
                    actions.push(command);
                }
            }
            if ui.small_button("＋ Personalizar").clicked() {
                self.show_ui_customizer = true;
            }
        });
        for command in actions {
            self.execute_ui_command(command);
        }
    }

    fn execute_ui_command(&mut self, command: UiCommand) {
        match command {
            UiCommand::ToggleSimulation => {
                if self.diverged {
                    self.sim.reset();
                    self.accumulator = 0.0;
                    self.diverged = false;
                    self.playing = true;
                } else {
                    self.playing = !self.playing;
                }
            }
            UiCommand::SingleStep => {
                self.playing = false;
                self.accumulator = 0.0;
                self.sim.step(FIXED_DT);
                self.last_step_ms = self.sim.stats.step_ms;
                if !self.sim.is_finite() {
                    self.sim.reset();
                    self.last_step_ms = 0.0;
                    self.diverged = true;
                }
            }
            UiCommand::ResetSimulation => {
                self.sim.reset();
                self.last_step_ms = 0.0;
                self.playing = false;
                self.diverged = false;
                self.accumulator = 0.0;
            }
            UiCommand::NewProject => self.new_pattern_project(),
            UiCommand::OpenProject => self.open_pattern_project_dialog(),
            UiCommand::SaveProject => self.save_pattern_project(),
            UiCommand::ImportAvatar => self.import_avatar_dialog(),
            UiCommand::ToggleAvatar => {
                self.avatar_visible = !self.avatar_visible;
                self.pattern_dirty = true;
                self.upload_scene();
            }
            UiCommand::ToggleAvatarWireframe => {
                self.avatar_wireframe = !self.avatar_wireframe;
                self.pattern_dirty = true;
                self.upload_scene();
            }
            UiCommand::ToggleAvatarCollision => {
                self.avatar_collision_enabled = !self.avatar_collision_enabled;
                self.pattern_dirty = true;
                if let Err(error) = self.sync_pattern_avatar_colliders() {
                    self.pattern_message = format!("Não foi possível atualizar a colisão: {error}");
                }
                self.upload_scene();
            }
            UiCommand::Show3dViewport => self.workspace = WorkspaceMode::Simulation,
            UiCommand::Show2dViewport => {
                self.playing = false;
                self.workspace = WorkspaceMode::Pattern;
            }
            UiCommand::ShowSplitViewport => {
                self.workspace = WorkspaceMode::Split;
                self.pattern_fit_pending = true;
            }
            UiCommand::ToggleMetrics => self.show_metrics = !self.show_metrics,
        }
    }

    fn ui_customizer_window(&mut self, ctx: &egui::Context) {
        if !self.show_ui_customizer {
            return;
        }

        let mut save_clicked = false;
        let mut reset_theme_clicked = false;
        let mut reset_all_clicked = false;
        let mut theme_changed = false;
        let mut layout_changed = false;
        let mut remove_button_id = None;
        let mut add_button: Option<(UiCommand, String)> = None;
        let mut close_clicked = false;

        egui::Window::new("Personalizar interface")
            .open(&mut self.show_ui_customizer)
            .resizable(true)
            .default_size(egui::vec2(620.0, 700.0))
            .show(ctx, |ui| {
                ui.label("Personalize cores, formas, organização dos viewports e comandos. As preferências ficam separadas dos arquivos de vestuário.");
                ui.separator();

                egui::ScrollArea::vertical().show(ui, |ui| {
                    egui::CollapsingHeader::new("Tema e cores")
                        .default_open(true)
                        .show(ui, |ui| {
                            ui.horizontal_wrapped(|ui| {
                                if ui.button("ClothEvo original").clicked() {
                                    reset_theme_clicked = true;
                                }
                                if ui.button("Graphite").clicked() {
                                    self.ui_customization.theme = ThemeSettings::graphite();
                                    theme_changed = true;
                                }
                                if ui.button("Amethyst").clicked() {
                                    self.ui_customization.theme = ThemeSettings::amethyst();
                                    theme_changed = true;
                                }
                            });
                            ui.label("Cada cor pode ser alterada individualmente.");
                            theme_changed |= edit_ui_rgb(ui, "Fundo dos painéis", &mut self.ui_customization.theme.panel_bg);
                            theme_changed |= edit_ui_rgb(ui, "Fundo das janelas", &mut self.ui_customization.theme.window_bg);
                            theme_changed |= edit_ui_rgb(ui, "Fundo profundo", &mut self.ui_customization.theme.deep_bg);
                            theme_changed |= edit_ui_rgb(ui, "Fundo secundário", &mut self.ui_customization.theme.muted_bg);
                            theme_changed |= edit_ui_rgb(ui, "Texto", &mut self.ui_customization.theme.text);
                            theme_changed |= edit_ui_rgb(ui, "Cor de destaque", &mut self.ui_customization.theme.accent);
                            theme_changed |= edit_ui_rgb(ui, "Seleção / estado ativo", &mut self.ui_customization.theme.selection);
                            theme_changed |= edit_ui_rgb(ui, "Botão", &mut self.ui_customization.theme.button_bg);
                            theme_changed |= edit_ui_rgb(ui, "Botão ao passar o mouse", &mut self.ui_customization.theme.button_hover);
                            theme_changed |= edit_ui_rgb(ui, "Contornos", &mut self.ui_customization.theme.border);
                        });

                    egui::CollapsingHeader::new("Formas e densidade")
                        .default_open(true)
                        .show(ui, |ui| {
                            theme_changed |= ui.add(
                                egui::Slider::new(&mut self.ui_customization.theme.corner_radius, 0..=16)
                                    .text("Arredondamento dos cantos"),
                            ).changed();
                            theme_changed |= ui.add(
                                egui::Slider::new(&mut self.ui_customization.theme.spacing, 2.0..=20.0)
                                    .text("Espaçamento dos componentes"),
                            ).changed();
                            theme_changed |= ui.add(
                                egui::Slider::new(&mut self.ui_customization.theme.ui_scale, 0.75..=1.75)
                                    .text("Escala global da interface"),
                            ).changed();
                        });

                    egui::CollapsingHeader::new("Organização dos viewports")
                        .default_open(true)
                        .show(ui, |ui| {
                            ui.label("Estas opções controlam a área dividida 3D + 2D.");
                            layout_changed |= ui.checkbox(
                                &mut self.ui_customization.split_vertical,
                                "Empilhar viewports verticalmente",
                            ).changed();
                            layout_changed |= ui.checkbox(
                                &mut self.ui_customization.split_swapped,
                                "Colocar 2D Pattern primeiro",
                            ).changed();
                            if ui.button("Abrir layout 3D + 2D").clicked() {
                                self.workspace = WorkspaceMode::Split;
                                self.pattern_fit_pending = true;
                            }
                        });

                    egui::CollapsingHeader::new("Minha barra de comandos")
                        .default_open(true)
                        .show(ui, |ui| {
                            ui.label("Adicione comandos existentes à sua própria barra. Cada botão pode ser renomeado, ocultado, reordenado ou removido.");
                            let mut previous_order = None;
                            let button_count = self.ui_customization.toolbar_buttons.len();
                            for index in 0..button_count {
                                let button = &mut self.ui_customization.toolbar_buttons[index];
                                ui.horizontal(|ui| {
                                    ui.checkbox(&mut button.visible, "");
                                    ui.text_edit_singleline(&mut button.label);
                                    egui::ComboBox::from_id_salt(("toolbar_command", button.id))
                                        .selected_text(button.command.label())
                                        .width(170.0)
                                        .show_ui(ui, |ui| {
                                            for (command, label) in UiCommand::ALL {
                                                ui.selectable_value(&mut button.command, *command, *label);
                                            }
                                        });
                                    if ui.small_button("↑").clicked() {
                                        previous_order = Some((index, index.saturating_sub(1)));
                                    }
                                    if ui.small_button("↓").clicked() && index + 1 < button_count {
                                        previous_order = Some((index, index + 1));
                                    }
                                    if ui.small_button("×").on_hover_text("Remover botão").clicked() {
                                        remove_button_id = Some(button.id);
                                    }
                                });
                            }
                            if let Some((from, to)) = previous_order {
                                self.ui_customization.toolbar_buttons.swap(from, to);
                            }
                            ui.separator();
                            ui.label("Adicionar um comando");
                            ui.horizontal(|ui| {
                                egui::ComboBox::from_id_salt("new_custom_toolbar_command")
                                    .selected_text(self.ui_customizer_new_command.label())
                                    .width(220.0)
                                    .show_ui(ui, |ui| {
                                        for (command, label) in UiCommand::ALL {
                                            ui.selectable_value(&mut self.ui_customizer_new_command, *command, *label);
                                        }
                                    });
                                ui.text_edit_singleline(&mut self.ui_customizer_new_label);
                            });
                            if ui.button("＋ Adicionar botão").clicked() {
                                add_button = Some((
                                    self.ui_customizer_new_command,
                                    self.ui_customizer_new_label.trim().to_owned(),
                                ));
                            }
                        });
                });

                if !self.ui_customization_message.is_empty() {
                    ui.separator();
                    ui.label(egui::RichText::new(&self.ui_customization_message).size(11.0));
                }
                ui.separator();
                ui.horizontal_wrapped(|ui| {
                    if ui.button("Salvar preferências").clicked() {
                        save_clicked = true;
                    }
                    if ui.button("Restaurar todas as preferências").clicked() {
                        reset_all_clicked = true;
                    }
                    if ui.button("Fechar").clicked() {
                        close_clicked = true;
                    }
                });
            });

        if close_clicked {
            self.show_ui_customizer = false;
        }
        if reset_theme_clicked {
            self.ui_customization.theme = ThemeSettings::default();
            theme_changed = true;
        }
        if reset_all_clicked {
            self.ui_customization = UiCustomization::default();
            self.split_vertical = false;
            self.split_swapped = false;
            self.workspace = WorkspaceMode::Split;
            theme_changed = true;
            layout_changed = false;
        }
        if layout_changed {
            self.split_vertical = self.ui_customization.split_vertical;
            self.split_swapped = self.ui_customization.split_swapped;
        }
        if let Some(id) = remove_button_id {
            self.ui_customization.toolbar_buttons.retain(|button| button.id != id);
        }
        if let Some((command, label)) = add_button {
            let id = self.ui_customization.toolbar_buttons.iter().map(|button| button.id).max().unwrap_or(0) + 1;
            let label = if label.is_empty() { command.label().to_owned() } else { label };
            self.ui_customization.toolbar_buttons.push(CustomToolbarButton {
                id,
                label,
                command,
                visible: true,
            });
            self.ui_customization_message = "Botão adicionado. Salve as preferências para mantê-lo.".into();
        }
        if theme_changed {
            self.ui_theme_dirty = true;
        }
        if save_clicked {
            self.save_ui_customization();
        }
    }

    fn global_toolbar(&mut self, ui: &mut egui::Ui) {
        let mut new_project = false;
        let mut open_project = false;
        let mut save_project = false;
        let mut import_avatar = false;
        let mut avatar_visibility_changed = false;
        let mut wireframe_changed = false;
        let mut collision_changed = false;
        let import_pending = self.avatar_import_pending_path.is_some();
        let imported_avatar_available = self.avatar_mesh.is_some();

        ui.set_min_height(42.0);
        ui.horizontal_wrapped(|ui| {
            ui.label(egui::RichText::new("PROJETO").size(10.0).strong());
            new_project = ui.button("Novo Projeto").clicked();
            open_project = ui.button("Abrir Projeto").clicked();
            save_project = ui.button("Salvar Projeto").clicked();

            ui.separator();
            ui.label(egui::RichText::new("AVATAR").size(10.0).strong());
            avatar_visibility_changed |= ui.checkbox(&mut self.avatar_visible, "Exibir Avatar").changed();
            wireframe_changed |= ui.add_enabled(
                imported_avatar_available,
                egui::Checkbox::new(&mut self.avatar_wireframe, "Wireframe"),
            ).on_hover_text("Disponível após importar uma malha OBJ ou FBX.").changed();
            collision_changed |= ui.checkbox(
                &mut self.avatar_collision_enabled,
                "Usar Avatar como Superfície de Colisão",
            ).changed();
            import_avatar = ui.add_enabled(
                !import_pending,
                egui::Button::new("Importar Avatar (.OBJ / .FBX)…"),
            ).clicked();

            if import_pending {
                ui.spinner();
                ui.label("Importando avatar…");
            }
        });

        if new_project {
            self.new_pattern_project();
        }
        if open_project {
            self.open_pattern_project_dialog();
        }
        if save_project {
            self.save_pattern_project();
        }

        if avatar_visibility_changed || wireframe_changed || collision_changed {
            self.pattern_dirty = true;
        }
        if collision_changed {
            if let Err(error) = self.sync_pattern_avatar_colliders() {
                self.pattern_message = format!("Não foi possível atualizar a colisão do avatar: {error}");
            }
        }
        if avatar_visibility_changed || wireframe_changed || collision_changed {
            self.upload_scene();
        }
        if import_avatar {
            self.import_avatar_dialog();
        }
    }

    fn environment_controls(&mut self, ui: &mut egui::Ui) {
        ui.heading("3D VIEWPORT");
        ui.label("Espaço de trabalho pronto para receber os moldes e a simulação.");
        ui.separator();

        let mut reprocess = false;
        let mut remove = false;
        let import_pending = self.avatar_import_pending_path.is_some();

        egui::CollapsingHeader::new("Avatar e colisão corporal")
            .default_open(true)
            .show(ui, |ui| {
                ui.label("O modelo padrão é procurado automaticamente. Também é possível importar OBJ ou FBX.");
                if self.avatar_mesh_path.is_some() {
                    ui.horizontal(|ui| {
                        reprocess = ui.add_enabled(!import_pending, egui::Button::new("Aplicar escala / colisão")).clicked();
                        remove = ui.add_enabled(!import_pending, egui::Button::new("Remover modelo")).clicked();
                    });
                }
                if import_pending {
                    ui.horizontal(|ui| {
                        ui.spinner();
                        ui.label("Preparando geometria e colisão em segundo plano…");
                    });
                }
                if let Some(name) = self.avatar_mesh_name.as_ref() {
                    ui.label(format!("Arquivo: {name}"));
                    if let Some(diag) = self.avatar_diagnostics.as_ref() {
                        ui.label(format!("{} vértices · {} triângulos", diag.vertex_count, diag.triangle_count));
                        ui.label(format!("Dimensões: {:.2} × {:.2} × {:.2} m",
                            diag.dimensions_meters[0], diag.dimensions_meters[1], diag.dimensions_meters[2]));
                        if let Some(warning) = diag.topology {
                            if !warning.has_reliable_inside_outside() {
                                ui.colored_label(egui::Color32::from_rgb(255, 170, 53), format!(
                                    "Aviso de topologia: {} arestas abertas; a colisão por volume pode ser imprecisa.",
                                    warning.boundary_edges
                                ));
                            }
                        }
                    }
                } else {
                    ui.colored_label(
                        egui::Color32::from_rgb(255, 170, 53),
                        "Nenhum Sarah.fbx/OBJ padrão foi encontrado. A sessão continua com o avatar procedural; importe um modelo para usar sua malha visual e colisão.",
                    );
                }
                if self.pattern_project.pieces.is_empty() {
                    ui.weak("Próximo passo: criar os moldes no 2D Viewport.");
                } else {
                    ui.weak("Use Posicionamento 3D para organizar os painéis antes de iniciar a simulação.");
                }
            });

        if reprocess {
            self.pattern_dirty = true;
            if let Some(path) = self.avatar_mesh_path.clone() {
                self.schedule_avatar_import(&path, true, true);
            }
        }
        if remove {
            self.pattern_dirty = true;
            self.avatar_import_receiver = None;
            self.avatar_import_pending_path = None;
            self.avatar_mesh = None;
            self.avatar_sdf = None;
            self.avatar_mesh_name = None;
            self.avatar_mesh_path = None;
            self.avatar_diagnostics = None;
            self.avatar_imported_height_m = None;
            self.avatar_imported_heading_degrees = None;
            self.avatar_imported_collision_spacing_m = None;
            if let Err(error) = self.sync_pattern_avatar_colliders() {
                self.pattern_message = format!("Não foi possível restaurar a colisão procedural: {error}");
            }
            self.upload_scene();
        }
        if !self.pattern_message.is_empty() {
            ui.separator();
            ui.label(&self.pattern_message);
        }
    }

    fn controls(&mut self, ui: &mut egui::Ui) {
        if self.workspace == WorkspaceMode::Pattern {
            self.pattern_controls(ui);
            return;
        }
        if self.workspace == WorkspaceMode::Placement {
            self.placement_controls(ui);
            return;
        }
        if self.workspace == WorkspaceMode::Split {
            ui.weak("VIEW  /  3D + 2D");
            ui.horizontal(|ui| {
                ui.checkbox(&mut self.split_swapped, "Trocar lados");
                ui.checkbox(&mut self.split_vertical, "Empilhar");
            });
            egui::CollapsingHeader::new("2D Viewport")
                .default_open(true)
                .show(ui, |ui| self.pattern_controls(ui));
            ui.separator();
        }

        // Uma sessão normal começa como ambiente de trabalho vazio, com avatar e chão,
        // nunca com um pano de demonstração. A cena de benchmark só aparece quando solicitada
        // explicitamente por variável de ambiente/captura.
        if !self.validation_scene && !self.using_pattern_sim {
            self.environment_controls(ui);
            return;
        }

        if self.using_pattern_sim {
            ui.heading("3D VIEWPORT");
            ui.label("Roupa personalizada convertida para o solver XPBD.");
        } else {
            ui.heading("3D VIEWPORT");
        }
        ui.separator();
        ui.horizontal(|ui| {
            let label = if self.playing { "⏸ Pausar" } else { "▶ Continuar" };
            if ui.button(label).clicked() {
                if self.diverged {
                    self.sim.reset();
                    self.accumulator = 0.0;
                    self.diverged = false;
                    self.playing = true;
                } else {
                    self.playing = !self.playing;
                }
            }
            if ui.button("⏭ Passo único").clicked() {
                self.playing = false;
                self.accumulator = 0.0;
                self.sim.step(FIXED_DT);
                self.last_step_ms = self.sim.stats.step_ms;
                if !self.sim.is_finite() {
                    self.sim.reset();
                    self.last_step_ms = 0.0;
                    self.diverged = true;
                }
            }
            if ui.button("↺ Reiniciar").clicked() {
                self.sim.reset();
                self.last_step_ms = 0.0;
                self.playing = false;
                self.diverged = false;
                self.accumulator = 0.0;
            }
        });
        if self.diverged {
            ui.colored_label(
                egui::Color32::LIGHT_RED,
                "Instabilidade numérica detectada: simulação pausada e restaurada ao estado inicial. Aumente substeps/iterações ou reduza rigidez.",
            );
        }

        // ----- Tecido
        ui.separator();
        egui::CollapsingHeader::new(egui::RichText::new("Tecido").strong())
            .default_open(true)
            .show(ui, |ui| {
        egui::ComboBox::from_id_salt("preset")
            .selected_text(self.preset.name())
            .width(240.0)
            .show_ui(ui, |ui| {
                for p in FabricPreset::ALL {
                    if ui.selectable_label(self.preset == p, p.name()).clicked() {
                        self.preset = p;
                        p.apply(&mut self.sim.params);
                    }
                }
            });
        let p = &mut self.sim.params;
        ui.add(egui::Slider::new(&mut p.density, 0.02..=1.0).logarithmic(true).text("densidade kg/m²"))
            .on_hover_text("Massa por área do tecido. Tecidos mais pesados caem mais rápido e dobram menos.");
        ui.add(egui::Slider::new(&mut p.stretch_warp, 50.0..=100_000.0).logarithmic(true).text("stretch urdume N/m"))
            .on_hover_text("Rigidez ao esticar na direção do urdume (fios longitudinais).");
        ui.add(egui::Slider::new(&mut p.stretch_weft, 50.0..=100_000.0).logarithmic(true).text("stretch trama N/m"))
            .on_hover_text("Rigidez ao esticar na direção da trama (fios transversais).");
        ui.add(egui::Slider::new(&mut p.shear, 5.0..=20_000.0).logarithmic(true).text("cisalhamento N/m"))
            .on_hover_text("Resistência do tecido a se deformar em losango (diagonais).");
        ui.add(egui::Slider::new(&mut p.bend, 1e-7..=1e-2).logarithmic(true).text("flexão N·m"))
            .on_hover_text("Rigidez à dobra. Valores baixos dão caimento solto (seda); altos, tecido encorpado (jeans).");
        ui.add(egui::Slider::new(&mut p.seam, 100.0..=100_000.0).logarithmic(true).text("costura N/m"))
            .on_hover_text("Rigidez das costuras. Valores altos fecham a costura com mais firmeza.");
        ui.add(egui::Slider::new(&mut p.damping, 0.0..=5.0).text("amortecimento 1/s"))
            .on_hover_text("Perda de velocidade ao longo do tempo. Mais amortecimento assenta o tecido mais rápido.");
        ui.add(egui::Slider::new(&mut p.friction, 0.0..=1.5).text("atrito"))
            .on_hover_text("Atrito do tecido com o corpo e com ele mesmo.");
        ui.add(egui::Slider::new(&mut p.thickness, 0.0..=0.03).text("espessura m"))
            .on_hover_text("Espessura de colisão: distância mínima mantida entre o tecido e o corpo.");
            });

        // ----- Solver
        ui.separator();
        egui::CollapsingHeader::new(egui::RichText::new("Solver e estabilidade").strong())
            .default_open(true)
            .show(ui, |ui| {
        let p = &mut self.sim.params;
        ui.add(egui::Slider::new(&mut p.substeps, 1..=80).text("substeps por quadro"))
            .on_hover_text("Mais substeps estabilizam movimentos rápidos, com custo proporcional.");
        ui.add(egui::Slider::new(&mut p.iterations, 1..=12).text("iterações por substep"))
            .on_hover_text("Mais iterações melhoram a convergência das restrições, com custo adicional.");
        ui.add(egui::Slider::new(&mut p.max_particle_speed, 2.0..=60.0)
            .text("limite de velocidade (m/s)"))
            .on_hover_text("Salvaguarda contra deslocamentos extremos de partículas.");
        ui.horizontal(|ui| {
            if ui.button("Equilibrado").on_hover_text("20 substeps · 2 iterações").clicked() {
                p.substeps = 20;
                p.iterations = 2;
                p.max_particle_speed = 20.0;
            }
            if ui.button("Mais estável").on_hover_text("40 substeps · 4 iterações").clicked() {
                p.substeps = 40;
                p.iterations = 4;
                p.max_particle_speed = 12.0;
            }
        });
        ui.checkbox(&mut p.self_collision, "Auto-colisão (mais custo)");
        ui.weak("A auto-colisão combina contatos vértice-vértice, vértice-triângulo e aresta-aresta; o processamento é serial e mostra quando limites descartam candidatos.");
        ui.checkbox(&mut p.parallel, "paralelo (rayon)");
            });

        ui.separator();
        ui.weak("Arraste: orbitar · botão direito/meio: mover · roda: zoom");

    }


    fn placement_controls(&mut self, ui: &mut egui::Ui) {
        ui.heading("Posicionamento 3D");
        ui.label("Organize os moldes ao redor do manequim antes de iniciar a simulação.");
        ui.label("Selecione um gizmo e arraste o eixo/anel/quadrado diretamente na viewport.");
        ui.horizontal(|ui| {
            ui.label("Ferramenta ativa:");
            ui.strong(match self.gizmo_mode {
                GizmoMode::Move => "Mover",
                GizmoMode::Rotate => "Girar",
                GizmoMode::Scale => "Escalar",
                GizmoMode::Transform => "Transformar",
            });
        });
        ui.weak("Selecione a ferramenta pelos ícones na barra vertical da viewport.");
        ui.horizontal(|ui| {
            if ui.button("Voltar ao 2D Viewport").clicked() {
                self.workspace = WorkspaceMode::Pattern;
                self.playing = false;
            }
            if ui.button("Atualizar prévia").clicked() {
                if let Err(error) = self.start_pattern_simulation() {
                    self.pattern_message = error;
                }
            }
        });
        if ui.button("Iniciar simulação de tecido").clicked() && self.using_pattern_sim {
            self.sim.reset();
            self.playing = true;
            self.workspace = WorkspaceMode::Simulation;
        }

        if self.pattern_project.pieces.is_empty() {
            ui.separator();
            ui.label("Crie pelo menos um molde no Pattern Studio antes de posicionar a roupa.");
            return;
        }

        let options: Vec<(u64, String)> = self.pattern_project.pieces.iter()
            .map(|piece| (piece.id, piece.name.clone())).collect();
        if self.selected_placement_piece.is_none()
            || !options.iter().any(|(id, _)| Some(*id) == self.selected_placement_piece)
        {
            self.selected_placement_piece = options.first().map(|(id, _)| *id);
        }
        egui::ComboBox::from_id_salt("placement_piece")
            .selected_text(self.selected_placement_piece
                .and_then(|id| options.iter().find(|(piece_id, _)| *piece_id == id))
                .map(|(_, name)| name.clone())
                .unwrap_or_else(|| "Selecionar molde".to_string()))
            .width(250.0)
            .show_ui(ui, |ui| {
                for (id, name) in &options {
                    ui.selectable_value(&mut self.selected_placement_piece, Some(*id), name);
                }
            });

        let selected_transform = self.selected_placement_piece
            .and_then(|id| self.pattern_project.pieces.iter().find(|piece| piece.id == id))
            .map(|piece| (
                piece.name.clone(),
                piece.assembly.translation_meters,
                piece.assembly.rotation_degrees,
                piece.assembly.scale,
            ));
        if let Some((name, translation, rotation, scale)) = selected_transform {
            ui.separator();
            ui.strong(format!("Transformação: {name}"));
            ui.label(format!(
                "Posição: X {:+.1} cm · Y {:+.1} cm · Z {:+.1} cm",
                translation[0] * 100.0, translation[1] * 100.0, translation[2] * 100.0
            ));
            ui.label(format!(
                "Rotação: X {:+.1}° · Y {:+.1}° · Z {:+.1}°",
                rotation[0], rotation[1], rotation[2]
            ));
            ui.label(format!(
                "Escala local: X {:.2}× · Y {:.2}× · Z {:.2}×",
                scale[0], scale[1], scale[2]
            ));
            if ui.button("Restaurar transformação deste molde").clicked() {
                self.reset_selected_placement_transform();
            }
        }

        if let Some(piece_id) = self.selected_placement_piece {
            ui.separator();
            ui.strong("Propriedades do molde");
            let mut layer_changed = false;
            if let Some(piece) = self.pattern_project.pieces.iter_mut().find(|piece| piece.id == piece_id) {
                ui.horizontal(|ui| {
                    ui.label("Camada");
                    let previous = piece.layer;
                    ui.add(egui::DragValue::new(&mut piece.layer).speed(0.1))
                        .on_hover_text("Camadas maiores ficam mais afastadas do corpo (por cima).");
                    piece.layer = piece.layer.clamp(-9, 9);
                    layer_changed = piece.layer != previous;
                });
                ui.checkbox(&mut piece.frozen, "Congelar molde (bloqueia mover, girar e escalar)");
            }
            if layer_changed {
                self.pattern_message = "Camada alterada: use \"Atualizar prévia\" para reposicionar.".into();
            }

            ui.separator();
            ui.strong("Posicionar no corpo");
            let avatar_config = AvatarConfig {
                height_m: self.pattern_project.avatar.height_meters as f32,
                shoulder_width_m: self.pattern_project.avatar.shoulder_width_meters as f32,
                waist_width_m: self.pattern_project.avatar.waist_width_meters as f32,
                hip_width_m: self.pattern_project.avatar.hip_width_meters as f32,
            };
            let points = arrangement_points(&avatar_config);
            let current_point = self.pattern_project.pieces.iter()
                .find(|piece| piece.id == piece_id)
                .and_then(|piece| piece.assembly.arrangement.clone());
            let current_name = current_point.as_ref()
                .and_then(|id| points.iter().find(|point| &point.id == id))
                .map(|point| point.name.clone())
                .unwrap_or_else(|| "layout automático".to_string());
            ui.label(format!("Ponto atual: {current_name}"));
            egui::ComboBox::from_id_salt("arrangement_point")
                .selected_text(self.selected_arrangement_point.as_ref()
                    .and_then(|id| points.iter().find(|point| &point.id == id))
                    .map(|point| point.name.clone())
                    .unwrap_or_else(|| "Escolher ponto do corpo".to_string()))
                .width(250.0)
                .show_ui(ui, |ui| {
                    for point in &points {
                        ui.selectable_value(
                            &mut self.selected_arrangement_point,
                            Some(point.id.clone()),
                            point.name.as_str(),
                        );
                    }
                });
            let mut apply_point = false;
            let mut clear_point = false;
            ui.horizontal(|ui| {
                if ui.add_enabled(
                    self.selected_arrangement_point.is_some(),
                    egui::Button::new("Posicionar neste ponto"),
                ).clicked() {
                    apply_point = true;
                }
                if ui.add_enabled(current_point.is_some(), egui::Button::new("Layout automático")).clicked() {
                    clear_point = true;
                }
            });
            ui.weak("Esquerdo = lado +X (o avatar olha para +Z). Depois de posicionar, refine com o gizmo.");
            if apply_point {
                let point = self.selected_arrangement_point.clone();
                self.set_piece_arrangement(piece_id, point);
            } else if clear_point {
                self.set_piece_arrangement(piece_id, None);
            }
        }

        ui.separator();
        ui.checkbox(&mut self.style.wireframe, "Mostrar triangulação 3D");
        ui.checkbox(&mut self.style.seams, "Mostrar costuras");
        ui.label(format!(
            "Moldes: {} · costuras: {} · partículas: {}",
            self.pattern_project.pieces.len(),
            self.pattern_project.seams.len(),
            self.sim.particle_count()
        ));
        if !self.pattern_message.is_empty() {
            ui.separator();
            ui.label(&self.pattern_message);
        }
    }

    /// Aplica a translação ao projeto e à pose atual/repouso do solver, para que Reiniciar
    /// preserve o posicionamento feito antes da simulação.
    fn apply_placement_translation(&mut self, piece_id: u64, delta: Vec3) {
        if self.piece_is_frozen(piece_id) { return; }
        let Some((first, end)) = self.placement_ranges.iter()
            .find(|range| range.piece_id == piece_id)
            .map(|range| (range.first, range.end))
        else { return; };
        let Some(current_translation) = self.pattern_project.pieces.iter()
            .find(|piece| piece.id == piece_id)
            .map(|piece| piece.assembly.translation_meters)
        else { return; };
        let Some((target_translation, applied_delta)) =
            bounded_placement_translation(current_translation, delta)
        else { return; };
        if applied_delta.length_squared() <= 1.0e-12 { return; }
        if !self.sim.translate_particles(first, end, applied_delta) {
            return;
        }
        if let Some(range) = self.placement_ranges.iter_mut().find(|range| range.piece_id == piece_id) {
            range.pivot += applied_delta;
        }
        if let Some(piece) = self.pattern_project.pieces.iter_mut().find(|piece| piece.id == piece_id) {
            piece.assembly.translation_meters = target_translation;
        }
    }

    fn translate_placement_by_screen_delta(&mut self, piece_id: u64, dx: f32, dy: f32, viewport_height: f32) {
        let view = self.camera.view();
        let right = view.row(0).truncate();
        let up = view.row(1).truncate();
        let world_per_pixel = 2.0 * self.camera.distance
            * (self.camera.fov_y * 0.5).tan() / viewport_height.max(1.0);
        // Inverte o deslocamento que a câmera usaria ao fazer pan: o painel acompanha o cursor.
        let delta = (right * dx - up * dy) * world_per_pixel;
        self.apply_placement_translation(piece_id, delta);
    }

    /// Centro e eixos do gizmo a partir da transformação atual do painel.
    fn placement_gizmo_axes(&self, piece_id: u64) -> Option<(Vec3, [Vec3; 3], [Vec3; 3])> {
        let range = self.placement_ranges.iter().find(|range| range.piece_id == piece_id)?;
        let positions = self.sim.pos.get(range.first..range.end)?;
        if positions.is_empty() { return None; }
        let center = range.pivot;
        if !center.is_finite() { return None; }
        let piece = self.pattern_project.pieces.iter().find(|piece| piece.id == piece_id)?;
        let [x, y, z] = piece.assembly.rotation_degrees.map(|degrees| (degrees as f32).to_radians());
        let qx = Quat::from_rotation_x(x);
        let qy = Quat::from_rotation_y(y);
        let rotation = range.base_rotation * Quat::from_euler(EulerRot::XYZ, x, y, z);
        let object_axes = [
            (rotation * Vec3::X).normalize_or_zero(),
            (rotation * Vec3::Y).normalize_or_zero(),
            (rotation * Vec3::Z).normalize_or_zero(),
        ];
        // Eixos incrementais XYZ: cada aro coincide com o componente Euler que será alterado.
        let rotation_axes = [
            (range.base_rotation * Vec3::X).normalize_or_zero(),
            (range.base_rotation * (qx * Vec3::Y)).normalize_or_zero(),
            (range.base_rotation * (qx * qy * Vec3::Z)).normalize_or_zero(),
        ];
        Some((center, object_axes, rotation_axes))
    }

    fn hit_test_placement_gizmo(
        &self,
        pointer: egui::Pos2,
        rect: egui::Rect,
        view_proj: glam::Mat4,
    ) -> Option<PlacementGizmoDrag> {
        let piece_id = self.selected_placement_piece?;
        let (center_world, object_axes, rotation_axes) = self.placement_gizmo_axes(piece_id)?;
        let screen_center = project_world_to_screen(center_world, view_proj, rect)?.0;
        let mode = self.gizmo_mode;

        if matches!(mode, GizmoMode::Rotate | GizmoMode::Transform) {
            let radius = if mode == GizmoMode::Transform {
                GIZMO_ROTATE_RADIUS * 0.84
            } else {
                GIZMO_ROTATE_RADIUS
            };
            let mut best: Option<(f32, usize)> = None;
            for (axis_index, axis) in rotation_axes.iter().enumerate() {
                let ring = projected_rotation_ring(center_world, *axis, radius, view_proj, rect);
                let distance = ring.windows(2)
                    .map(|pair| point_segment_distance(pointer, pair[0], pair[1]).0)
                    .fold(f32::INFINITY, f32::min);
                if distance <= 8.0 && best.map_or(true, |(d, _)| distance < d) {
                    best = Some((distance, axis_index));
                }
            }
            if let Some((_, axis)) = best {
                return Some(PlacementGizmoDrag {
                    piece_id,
                    axis,
                    action: GizmoAction::Rotate,
                    axis_world: rotation_axes[axis],
                    screen_axis: egui::Vec2::ZERO,
                    screen_center,
                    pixels_per_meter: 1.0,
                    last_pointer: pointer,
                });
            }
        }

        if matches!(mode, GizmoMode::Scale | GizmoMode::Transform) {
            for (axis_index, axis) in object_axes.iter().enumerate() {
                let length = if mode == GizmoMode::Transform { GIZMO_SCALE_LENGTH } else { GIZMO_SCALE_LENGTH };
                let Some((endpoint, _)) = project_world_to_screen(
                    center_world + *axis * length, view_proj, rect,
                ) else { continue; };
                let screen_delta = endpoint - screen_center;
                let screen_length = screen_delta.length();
                if screen_length <= 1.0 { continue; }
                let screen_axis = screen_delta / screen_length;
                let (distance, t) = point_segment_distance(pointer, screen_center, endpoint);
                let on_handle = pointer.distance(endpoint) <= 12.0;
                // No modo combinado, mantém os quadrados de escala fora da ponta das setas.
                let in_scale_region = mode != GizmoMode::Transform || t >= 0.72;
                if on_handle || (distance <= 7.5 && in_scale_region) {
                    return Some(PlacementGizmoDrag {
                        piece_id,
                        axis: axis_index,
                        action: GizmoAction::Scale,
                        axis_world: *axis,
                        screen_axis,
                        screen_center,
                        pixels_per_meter: screen_length / length,
                        last_pointer: pointer,
                    });
                }
            }
        }

        if matches!(mode, GizmoMode::Move | GizmoMode::Transform) {
            let length = if mode == GizmoMode::Transform { GIZMO_MOVE_LENGTH } else { 0.58 };
            let mut best: Option<(f32, PlacementGizmoDrag)> = None;
            for (axis_index, axis) in object_axes.iter().enumerate() {
                let Some((endpoint, _)) = project_world_to_screen(
                    center_world + *axis * length, view_proj, rect,
                ) else { continue; };
                let screen_delta = endpoint - screen_center;
                let screen_length = screen_delta.length();
                if screen_length <= 1.0 { continue; }
                let (distance, t) = point_segment_distance(pointer, screen_center, endpoint);
                if distance <= 8.5 && t >= 0.13 {
                    let drag = PlacementGizmoDrag {
                        piece_id,
                        axis: axis_index,
                        action: GizmoAction::Move,
                        axis_world: *axis,
                        screen_axis: screen_delta / screen_length,
                        screen_center,
                        pixels_per_meter: screen_length / length,
                        last_pointer: pointer,
                    };
                    if best.as_ref().map_or(true, |(d, _)| distance < *d) {
                        best = Some((distance, drag));
                    }
                }
            }
            if let Some((_, drag)) = best { return Some(drag); }
        }
        None
    }

    fn piece_is_frozen(&self, piece_id: u64) -> bool {
        self.pattern_project.pieces.iter()
            .find(|piece| piece.id == piece_id)
            .map(|piece| piece.frozen)
            .unwrap_or(false)
    }

    /// Define (ou limpa) o ponto de posicionamento do molde e refaz a prévia; restaura o
    /// projeto anterior se a montagem falhar.
    fn set_piece_arrangement(&mut self, piece_id: u64, point: Option<String>) {
        if self.piece_is_frozen(piece_id) {
            self.pattern_message = "O molde está congelado; descongele para reposicioná-lo.".into();
            return;
        }
        let before = self.editor_snapshot();
        let Some(piece) = self.pattern_project.pieces.iter_mut().find(|piece| piece.id == piece_id)
        else { return; };
        piece.assembly.arrangement = point;
        piece.assembly.translation_meters = [0.0; 3];
        piece.assembly.rotation_degrees = [0.0; 3];
        let selected = self.selected_placement_piece;
        if let Err(error) = self.start_pattern_simulation() {
            self.pattern_project = before.project.clone();
            self.pattern_message = format!("Posicionamento cancelado; estado anterior restaurado: {error}");
            return;
        }
        self.selected_placement_piece = selected;
        self.remember_pattern_change(before);
    }

    fn apply_placement_rotation_delta(
        &mut self,
        piece_id: u64,
        axis: usize,
        axis_world: Vec3,
        degrees: f32,
    ) {
        if axis >= 3 || !degrees.is_finite() || !axis_world.is_finite()
            || axis_world.length_squared() <= 1.0e-10
        {
            return;
        }
        let Some(range) = self.placement_ranges.iter().find(|range| range.piece_id == piece_id).copied()
        else { return; };
        if self.piece_is_frozen(piece_id) { return; }
        let center = range.pivot;
        let rotation = Quat::from_axis_angle(axis_world.normalize(), degrees.to_radians());
        if !self.sim.rotate_particles(range.first, range.end, center, rotation) {
            return;
        }
        if let Some(piece) = self.pattern_project.pieces.iter_mut().find(|piece| piece.id == piece_id) {
            piece.assembly.rotation_degrees[axis] =
                (piece.assembly.rotation_degrees[axis] + degrees as f64).rem_euclid(360.0);
        }
    }

    fn apply_placement_scale_factor(&mut self, piece_id: u64, axis: usize, factor: f32) {
        if axis >= 3 || !factor.is_finite() || factor <= 0.0 { return; }
        if self.piece_is_frozen(piece_id) { return; }
        let Some(range) = self.placement_ranges.iter().find(|range| range.piece_id == piece_id).copied()
        else { return; };
        let Some((_, object_axes, _)) = self.placement_gizmo_axes(piece_id) else { return; };
        let center = range.pivot;
        let scaled_axes: &[usize] = if axis == 2 { &[0, 1] } else { &[axis] };
        for &scale_axis in scaled_axes {
            let old_scale = self.pattern_project.pieces.iter()
                .find(|piece| piece.id == piece_id)
                .map(|piece| piece.assembly.scale[scale_axis] as f32)
                .unwrap_or(1.0)
                .clamp(0.1, 10.0);
            let new_scale = (old_scale * factor).clamp(0.1, 10.0);
            let applied_factor = new_scale / old_scale;
            if (applied_factor - 1.0).abs() <= 1.0e-6 { continue; }
            if self.sim.scale_particles_axis(
                range.first, range.end, center, object_axes[scale_axis], applied_factor,
            ) {
                if let Some(piece) = self.pattern_project.pieces.iter_mut()
                    .find(|piece| piece.id == piece_id)
                {
                    piece.assembly.scale[scale_axis] = new_scale as f64;
                }
            }
        }
    }

    fn draw_placement_gizmo(
        &self,
        painter: &egui::Painter,
        rect: egui::Rect,
        view_proj: glam::Mat4,
    ) {
        if self.workspace != WorkspaceMode::Placement || !self.using_pattern_sim { return; }
        let Some(piece_id) = self.selected_placement_piece else { return; };
        let Some((center_world, object_axes, rotation_axes)) = self.placement_gizmo_axes(piece_id)
        else { return; };
        let Some((center, _)) = project_world_to_screen(center_world, view_proj, rect) else { return; };
        let mode = self.gizmo_mode;
        let show_move = matches!(mode, GizmoMode::Move | GizmoMode::Transform);
        let show_rotate = matches!(mode, GizmoMode::Rotate | GizmoMode::Transform);
        let show_scale = matches!(mode, GizmoMode::Scale | GizmoMode::Transform);

        if show_rotate {
            let radius = if mode == GizmoMode::Transform {
                GIZMO_ROTATE_RADIUS * 0.84
            } else {
                GIZMO_ROTATE_RADIUS
            };
            for (axis_index, axis) in rotation_axes.iter().enumerate() {
                let ring = projected_rotation_ring(center_world, *axis, radius, view_proj, rect);
                for pair in ring.windows(2) {
                    painter.line_segment(
                        [pair[0], pair[1]],
                        egui::Stroke::new(2.4, gizmo_axis_color(axis_index)),
                    );
                }
            }
        }

        if show_move || show_scale {
            for (axis_index, axis) in object_axes.iter().enumerate() {
                let color = gizmo_axis_color(axis_index);
                let move_length = if mode == GizmoMode::Transform { GIZMO_MOVE_LENGTH } else { 0.58 };
                let scale_length = GIZMO_SCALE_LENGTH;
                let draw_length = if show_scale { scale_length } else { move_length };
                let Some((end, _)) = project_world_to_screen(
                    center_world + *axis * draw_length, view_proj, rect,
                ) else { continue; };
                if show_move && show_scale {
                    if let Some((move_end, _)) = project_world_to_screen(
                        center_world + *axis * move_length, view_proj, rect,
                    ) {
                        painter.line_segment([center, end], egui::Stroke::new(1.7, color.gamma_multiply(0.62)));
                        painter.line_segment([center, move_end], egui::Stroke::new(3.2, color));
                        draw_gizmo_arrow_head(painter, move_end, move_end - center, color);
                    }
                    let scale_rect = egui::Rect::from_center_size(end, egui::vec2(10.0, 10.0));
                    painter.rect_filled(scale_rect, 1.5, color);
                    painter.rect_stroke(scale_rect, 1.5, egui::Stroke::new(1.0, egui::Color32::WHITE), egui::StrokeKind::Inside);
                } else if show_scale {
                    painter.line_segment([center, end], egui::Stroke::new(2.5, color));
                    let scale_rect = egui::Rect::from_center_size(end, egui::vec2(11.0, 11.0));
                    painter.rect_filled(scale_rect, 1.5, color);
                    painter.rect_stroke(scale_rect, 1.5, egui::Stroke::new(1.0, egui::Color32::WHITE), egui::StrokeKind::Inside);
                } else {
                    painter.line_segment([center, end], egui::Stroke::new(3.2, color));
                    draw_gizmo_arrow_head(painter, end, end - center, color);
                    painter.text(
                        end + (end - center).normalized() * 8.0,
                        egui::Align2::CENTER_CENTER,
                        ["X", "Y", "Z"][axis_index],
                        egui::FontId::proportional(12.0),
                        color,
                    );
                }
            }
        }
        painter.circle_filled(center, 4.5, egui::Color32::WHITE);
        painter.circle_stroke(center, 6.5, egui::Stroke::new(1.5, egui::Color32::from_gray(30)));
    }

    fn draw_placement_toolbar(&mut self, ui: &mut egui::Ui, viewport_rect: egui::Rect) {
        let panel_rect = placement_toolbar_bounds(viewport_rect);
        let painter = ui.painter();
        painter.rect_filled(
            panel_rect,
            6.0,
            egui::Color32::from_rgba_unmultiplied(19, 21, 26, 242),
        );
        painter.rect_stroke(
            panel_rect,
            6.0,
            egui::Stroke::new(1.0, egui::Color32::from_rgb(62, 65, 74)),
            egui::StrokeKind::Inside,
        );

        let tools = [
            (GizmoMode::Rotate, "Girar", "Girar o molde usando os anéis dos eixos."),
            (GizmoMode::Move, "Mover", "Mover o molde pelos eixos X, Y e Z."),
            (GizmoMode::Scale, "Escalar", "Escalar o molde pelos controles nas extremidades."),
            (GizmoMode::Transform, "Transformar", "Combinar mover, girar e escalar."),
        ];
        let button_side = 36.0;
        let gap = 7.0;
        for (index, (mode, _label, tooltip)) in tools.iter().enumerate() {
            let button_rect = egui::Rect::from_min_size(
                egui::pos2(
                    panel_rect.center().x - button_side * 0.5,
                    panel_rect.top() + 7.0 + index as f32 * (button_side + gap),
                ),
                egui::vec2(button_side, button_side),
            );
            let response = ui.interact(
                button_rect,
                ui.make_persistent_id(("placement_gizmo_toolbar", index)),
                egui::Sense::click(),
            );
            let active = self.gizmo_mode == *mode;
            let fill = if active {
                egui::Color32::from_rgb(111, 48, 111)
            } else if response.hovered() {
                egui::Color32::from_rgb(48, 57, 74)
            } else {
                egui::Color32::from_rgb(29, 31, 37)
            };
            let icon_color = if active {
                egui::Color32::WHITE
            } else {
                egui::Color32::from_rgb(211, 214, 220)
            };
            painter.rect_filled(button_rect, 5.0, fill);
            if active {
                painter.rect_stroke(
                    button_rect,
                    5.0,
                    egui::Stroke::new(1.2, egui::Color32::from_rgb(218, 175, 91)),
                    egui::StrokeKind::Inside,
                );
            }
            paint_placement_tool_icon(&painter, button_rect, *mode, icon_color);
            let clicked = response.clicked();
            let _ = response.on_hover_text(*tooltip);
            if clicked {
                self.gizmo_mode = *mode;
                self.placement_gizmo_drag = None;
                self.placement_drag_piece = None;
            }
        }
    }

    fn rotate_selected_placement(&mut self, axis: usize, degrees: f64) {
        let Some(id) = self.selected_placement_piece else { return; };
        if self.piece_is_frozen(id) { return; }
        let before = self.editor_snapshot();
        let Some(piece) = self.pattern_project.pieces.iter_mut().find(|piece| piece.id == id) else { return; };
        if axis >= 3 { return; }
        piece.assembly.rotation_degrees[axis] =
            (piece.assembly.rotation_degrees[axis] + degrees).rem_euclid(360.0);
        let selected = self.selected_placement_piece;
        if let Err(error) = self.start_pattern_simulation() {
            self.pattern_project = before.project.clone();
            self.pattern_message = format!("Rotação cancelada; transformação anterior restaurada: {error}");
            return;
        }
        self.selected_placement_piece = selected;
        self.remember_pattern_change(before);
    }

    fn reset_selected_placement_transform(&mut self) {
        let Some(id) = self.selected_placement_piece else { return; };
        let before = self.editor_snapshot();
        let Some(piece) = self.pattern_project.pieces.iter_mut().find(|piece| piece.id == id) else {
            return;
        };
        piece.assembly.translation_meters = [0.0; 3];
        piece.assembly.rotation_degrees = [0.0; 3];
        piece.assembly.scale = [1.0; 3];
        let selected = self.selected_placement_piece;
        if let Err(error) = self.start_pattern_simulation() {
            self.pattern_project = before.project.clone();
            self.pattern_message = format!("Reset cancelado; transformação anterior restaurada: {error}");
            return;
        }
        self.selected_placement_piece = selected;
        self.remember_pattern_change(before);
    }

    fn hit_test_placement_piece(&self, pointer: egui::Pos2, rect: egui::Rect) -> Option<u64> {
        let aspect = rect.width() / rect.height().max(1.0);
        let view_proj = self.camera.view_proj(aspect);
        let mut best: Option<(f32, u64)> = None;
        for range in &self.placement_ranges {
            for triangle in &self.sim.triangles {
                if triangle.iter().any(|index| {
                    let index = *index as usize;
                    index < range.first || index >= range.end
                }) {
                    continue;
                }
                let projected = triangle.map(|index| project_world_to_screen(
                    self.sim.pos[index as usize], view_proj, rect,
                ));
                let [Some((a, za)), Some((b, zb)), Some((c, zc))] = projected else { continue; };
                if point_in_triangle(pointer, a, b, c) {
                    let depth = (za + zb + zc) / 3.0;
                    if best.map_or(true, |(current, _)| depth < current) {
                        best = Some((depth, range.piece_id));
                    }
                }
            }
        }
        best.map(|(_, id)| id)
    }

    fn editor_snapshot(&self) -> PatternEditorSnapshot {
        PatternEditorSnapshot {
            project: self.pattern_project.clone(),
            draft_points: self.draft_points.clone(),
            selected_piece: self.selected_piece,
            selected_target: self.selected_target,
        }
    }

    fn remember_pattern_change(&mut self, before: PatternEditorSnapshot) {
        if before == self.editor_snapshot() { return; }
        self.pattern_undo.push(before);
        if self.pattern_undo.len() > 100 {
            self.pattern_undo.remove(0);
        }
        self.pattern_redo.clear();
        self.pattern_dirty = true;
    }

    fn restore_editor_snapshot(&mut self, snapshot: PatternEditorSnapshot) {
        self.pattern_project = snapshot.project;
        self.draft_points = snapshot.draft_points;
        self.selected_piece = snapshot.selected_piece.min(
            self.pattern_project.pieces.len().saturating_sub(1)
        );
        self.selected_target = snapshot.selected_target.filter(|target| {
            target.piece() < self.pattern_project.pieces.len()
        });
        self.dragging_target = None;
        self.dragging_before = None;
        self.drag_last_world = None;
        self.pattern_dirty = true;
    }

    fn undo_pattern_edit(&mut self) {
        let Some(previous) = self.pattern_undo.pop() else {
            self.pattern_message = "Nada para desfazer.".into();
            return;
        };
        let current = self.editor_snapshot();
        self.pattern_redo.push(current);
        self.restore_editor_snapshot(previous);
        self.pattern_message = "Última edição desfeita.".into();
    }

    fn redo_pattern_edit(&mut self) {
        let Some(next) = self.pattern_redo.pop() else {
            self.pattern_message = "Nada para refazer.".into();
            return;
        };
        let current = self.editor_snapshot();
        self.pattern_undo.push(current);
        self.restore_editor_snapshot(next);
        self.pattern_message = "Edição refeita.".into();
    }

    fn confirm_discard_unsaved_changes(&self) -> bool {
        if !self.pattern_dirty { return true; }
        rfd::MessageDialog::new()
            .set_level(rfd::MessageLevel::Warning)
            .set_title("Alterações não salvas")
            .set_description("O projeto atual tem alterações não salvas. Descartar essas alterações?")
            .set_buttons(rfd::MessageButtons::YesNo)
            .show() == rfd::MessageDialogResult::Yes
    }

    fn new_pattern_project(&mut self) {
        if !self.confirm_discard_unsaved_changes() { return; }
        let avatar_path = self.avatar_mesh_path.clone();
        self.pattern_project = PatternProject::new("Novo projeto");
        self.workspace = WorkspaceMode::Pattern;
        self.playing = false;
        self.draft_points.clear();
        self.selected_piece = 0;
        self.selected_target = None;
        self.pattern_saved_path = None;
        self.pattern_undo.clear();
        self.pattern_redo.clear();
        self.dragging_target = None;
        self.dragging_before = None;
        self.drag_last_world = None;
        self.pattern_dirty = false;
        self.reset_to_empty_environment();
        if let Some(path) = avatar_path {
            self.schedule_avatar_import(&path, true, false);
        }
        self.pattern_message = "Novo projeto criado.".into();
    }

    /// Descarta a malha de tecido da sessão anterior e restaura o ambiente vazio,
    /// mantendo o avatar/collision settings atuais e os parâmetros globais do solver.
    fn reset_to_empty_environment(&mut self) {
        self.using_pattern_sim = false;
        self.placement_ranges.clear();
        self.selected_placement_piece = None;
        self.placement_drag_piece = None;
        self.placement_drag_before = None;
        self.placement_gizmo_drag = None;
        self.placement_drag_selected_body = false;

        let mut builder = ClothBuilder::new();
        builder.add_collider(Collider::ground());
        if let Ok(colliders) = self.avatar_body_colliders() {
            for collider in colliders {
                builder.add_collider(collider);
            }
        }
        self.sim = builder.build(self.sim.params.clone());
        self.seam_pairs = self.sim.seam_pairs();
        self.normals.clear();
        self.diverged = false;
        self.accumulator = 0.0;
        self.last_step_ms = 0.0;
        self.playing = false;
        self.upload_scene();
    }

    fn open_pattern_project_dialog(&mut self) {
        let mut dialog = rfd::FileDialog::new()
            .add_filter("Projeto ClothEvo / JSON", &["clothevo", "json"]);
        if let Some(parent) = self.pattern_saved_path.as_deref().and_then(std::path::Path::parent) {
            dialog = dialog.set_directory(parent);
        }
        let Some(path) = dialog.pick_file() else { return; };
        let loaded = std::fs::read_to_string(&path)
            .map_err(|e| format!("Falha ao ler {}: {e}", path.display()))
            .and_then(|json| {
                PatternProject::from_json(&json).map(|project| {
                    let app_state = serde_json::from_str::<serde_json::Value>(&json).ok()
                        .and_then(|document| document.get("clothevo_app").cloned());
                    (project, app_state)
                })
            });
        match loaded {
            Ok((project, app_state)) => {
                if !self.confirm_discard_unsaved_changes() { return; }
                self.pattern_project = project;
                self.workspace = WorkspaceMode::Pattern;
                self.playing = false;
                self.draft_points.clear();
                self.selected_piece = 0;
                self.selected_target = None;
                self.pattern_saved_path = Some(path.clone());
                self.pattern_undo.clear();
                self.pattern_redo.clear();
                self.dragging_target = None;
                self.dragging_before = None;
                self.drag_last_world = None;
                self.pattern_dirty = false;
                self.reset_to_empty_environment();
                self.pattern_message = format!("Projeto carregado: {}", path.display());
                if let Some(app_state) = app_state {
                    self.restore_project_app_state(&app_state);
                }
            }
            Err(error) => self.pattern_message = error,
        }
    }

    fn restore_project_app_state(&mut self, app_state: &serde_json::Value) {
        self.avatar_visible = app_state.get("avatar_visible")
            .and_then(serde_json::Value::as_bool).unwrap_or(true);
        self.avatar_wireframe = app_state.get("avatar_wireframe")
            .and_then(serde_json::Value::as_bool).unwrap_or(false);
        self.avatar_collision_enabled = app_state.get("avatar_collision_enabled")
            .and_then(serde_json::Value::as_bool).unwrap_or(true);
        self.avatar_collision_spacing_m = app_state.get("avatar_collision_spacing_m")
            .and_then(serde_json::Value::as_f64).unwrap_or(0.01)
            .clamp(0.004, 0.03) as f32;
        self.avatar_heading_degrees = app_state.get("avatar_heading_degrees")
            .and_then(serde_json::Value::as_f64).unwrap_or(0.0)
            .clamp(-180.0, 180.0) as f32;

        // Uma referência presente no documento substitui o avatar da sessão atual.
        self.avatar_import_receiver = None;
        self.avatar_import_pending_path = None;
        self.avatar_mesh = None;
        self.avatar_sdf = None;
        self.avatar_mesh_name = None;
        self.avatar_mesh_path = None;
        self.avatar_diagnostics = None;
        self.avatar_imported_height_m = None;
        self.avatar_imported_heading_degrees = None;
        self.avatar_imported_collision_spacing_m = None;

        if let Some(saved_path) = app_state.get("avatar_path").and_then(serde_json::Value::as_str) {
            let saved_path = std::path::PathBuf::from(saved_path);
            if saved_path.is_file() {
                self.schedule_avatar_import(&saved_path, true, false);
            } else {
                self.pattern_message = format!(
                    "Projeto carregado, mas o arquivo de avatar salvo não foi encontrado: {}. A visualização usa o manequim procedural.",
                    saved_path.display()
                );
            }
        }
        if let Err(error) = self.sync_pattern_avatar_colliders() {
            self.pattern_message = format!("Projeto carregado, mas não foi possível atualizar a colisão: {error}");
        }
        self.upload_scene();
    }

    fn save_pattern_project(&mut self) {
        if let Some(path) = self.pattern_saved_path.clone() {
            self.write_pattern_project(path);
        } else {
            self.save_pattern_project_as();
        }
    }

    fn save_pattern_project_as(&mut self) {
        let mut dialog = rfd::FileDialog::new()
            .add_filter("Projeto ClothEvo", &["clothevo", "json"])
            .set_file_name("projeto.clothevo");
        if let Some(parent) = self.pattern_saved_path.as_deref().and_then(std::path::Path::parent) {
            dialog = dialog.set_directory(parent);
        }
        if let Some(path) = dialog.save_file() {
            self.write_pattern_project(path);
        }
    }

    fn write_pattern_project(&mut self, mut path: std::path::PathBuf) {
        if path.extension().is_none() {
            path.set_extension("clothevo");
        }
        let result = self.pattern_project.to_json().and_then(|json| {
            let mut document = serde_json::from_str::<serde_json::Value>(&json)
                .map_err(|e| format!("Falha ao preparar o projeto para salvar: {e}"))?;
            document["clothevo_app"] = serde_json::json!({
                "version": 1,
                "avatar_path": self.avatar_mesh_path.as_ref()
                    .map(|saved_path| saved_path.to_string_lossy().into_owned()),
                "avatar_visible": self.avatar_visible,
                "avatar_wireframe": self.avatar_wireframe,
                "avatar_collision_enabled": self.avatar_collision_enabled,
                "avatar_collision_spacing_m": self.avatar_collision_spacing_m,
                "avatar_heading_degrees": self.avatar_heading_degrees,
            });
            let serialized = serde_json::to_string_pretty(&document)
                .map_err(|e| format!("Falha ao serializar o projeto: {e}"))?;
            std::fs::write(&path, serialized)
                .map_err(|e| format!("Falha ao salvar {}: {e}", path.display()))
        });
        match result {
            Ok(()) => {
                self.pattern_saved_path = Some(path.clone());
                self.pattern_dirty = false;
                self.pattern_message = format!("Projeto salvo: {}", path.display());
            }
            Err(error) => self.pattern_message = error,
        }
    }

    fn fit_pattern_view(&mut self, rect: egui::Rect) {
        let mut min_x = f64::INFINITY;
        let mut min_y = f64::INFINITY;
        let mut max_x = f64::NEG_INFINITY;
        let mut max_y = f64::NEG_INFINITY;
        for piece in &self.pattern_project.pieces {
            let points = sample_curve(&piece.outline, 16);
            for p in points {
                min_x = min_x.min(p.x);
                min_y = min_y.min(p.y);
                max_x = max_x.max(p.x);
                max_y = max_y.max(p.y);
            }
        }
        if !min_x.is_finite() {
            self.pattern_zoom = PATTERN_PIXELS_PER_METER;
            self.pattern_pan = egui::Vec2::ZERO;
            return;
        }
        let width = (max_x - min_x).max(0.1) as f32;
        let height = (max_y - min_y).max(0.1) as f32;
        let available_width = (rect.width() - 120.0).max(120.0);
        let available_height = (rect.height() - 120.0).max(120.0);
        self.pattern_zoom = (available_width / width).min(available_height / height).clamp(40.0, 1800.0);
        let mid_x = ((min_x + max_x) * 0.5) as f32;
        let mid_y = ((min_y + max_y) * 0.5) as f32;
        self.pattern_pan = egui::vec2(-mid_x * self.pattern_zoom, mid_y * self.pattern_zoom);
    }

    fn duplicate_selected_piece(&mut self) {
        if self.pattern_project.pieces.is_empty() { return; }
        let index = self.selected_piece.min(self.pattern_project.pieces.len() - 1);
        let before = self.editor_snapshot();
        let Some(mut duplicate) = self.pattern_project.pieces.get(index).cloned() else { return; };
        let new_id = self.pattern_project.pieces.iter().map(|p| p.id).max().unwrap_or(0) + 1;
        duplicate.id = new_id;
        duplicate.name = format!("{} (cópia)", duplicate.name);
        translate_piece(&mut duplicate, 0.20, 0.20);
        self.pattern_project.pieces.push(duplicate);
        self.selected_piece = self.pattern_project.pieces.len() - 1;
        self.selected_target = Some(PatternEditTarget::Piece { piece: self.selected_piece });
        self.pattern_message = "Peça duplicada e deslocada em 20 cm; as costuras não foram duplicadas.".into();
        self.remember_pattern_change(before);
    }

    fn pattern_controls(&mut self, ui: &mut egui::Ui) {
        ui.heading("2D VIEWPORT · GEOMETRIA VETORIAL");
        ui.label("Editor de moldes · medidas internas em metros");
        ui.horizontal(|ui| {
            ui.selectable_value(&mut self.pattern_tool, PatternTool::Draw, "Desenhar");
            ui.selectable_value(&mut self.pattern_tool, PatternTool::Select, "Selecionar / editar");
        });
        ui.horizontal(|ui| {
            for (mode, label) in [(PieceTransformMode::Move, "Mover (W)"), (PieceTransformMode::Rotate, "Girar (E)"), (PieceTransformMode::Scale, "Escalar (R)")] {
                if ui.selectable_value(&mut self.piece_transform_mode, mode, label).clicked() { self.pattern_tool = PatternTool::Select; }
            }
        });
        ui.weak("Atalhos W/E/R com o mouse sobre o canvas. Girar e escalar agem ao redor do centro do molde, somente no plano 2D (sem profundidade).");
        ui.horizontal(|ui| {
            if ui.add_enabled(!self.pattern_undo.is_empty(), egui::Button::new("Desfazer")).clicked() {
                self.undo_pattern_edit();
            }
            if ui.add_enabled(!self.pattern_redo.is_empty(), egui::Button::new("Refazer")).clicked() {
                self.redo_pattern_edit();
            }
        });
        if self.pattern_dirty {
            ui.colored_label(egui::Color32::from_rgb(240, 180, 75), "Modificações não salvas");
        } else if let Some(path) = &self.pattern_saved_path {
            ui.weak(format!("Arquivo: {}", path.file_name().unwrap_or_default().to_string_lossy()));
        }
        ui.horizontal(|ui| {
            if ui.button("Desfazer ponto").clicked() {
                let before = self.editor_snapshot();
                self.draft_points.pop();
                self.remember_pattern_change(before);
            }
            if ui.button("Limpar rascunho").clicked() {
                let before = self.editor_snapshot();
                self.draft_points.clear();
                self.remember_pattern_change(before);
            }
        });
        ui.checkbox(&mut self.pattern_snap, "Ajustar à grade (1 cm)");
        ui.checkbox(&mut self.show_avatar_guide, "Guia anatômico T-Pose");
        ui.add_enabled(
            self.show_avatar_guide,
            egui::Slider::new(&mut self.avatar_guide_backdrop, 0.0..=0.6).text("fundo do guia"),
        )
        .on_hover_text("Opacidade do fundo claro atrás da silhueta (0 = sem fundo).");
        ui.weak("A silhueta é uma referência proporcional de 1,72 m; os moldes continuam sendo editados em metros.");
        ui.horizontal(|ui| {
            if ui.button("Enquadrar todos os moldes").clicked() {
                self.pattern_fit_pending = true;
            }
            if ui.button("Vista padrão").clicked() {
                self.pattern_zoom = PATTERN_PIXELS_PER_METER;
                self.pattern_pan = egui::Vec2::ZERO;
            }
        });
        ui.separator();
        egui::CollapsingHeader::new(egui::RichText::new("Geometria vetorial").strong())
            .default_open(true)
            .show(ui, |ui| {
                ui.horizontal(|ui| {
                    if ui.button("Adicionar retângulo").clicked() {
                        self.create_rectangle_primitive();
                    }
                    if ui.button("Adicionar polígono regular").clicked() {
                        self.create_regular_polygon_primitive();
                    }
                });
                ui.add(egui::Slider::new(&mut self.pattern_polygon_sides, 3..=12).text("lados do polígono"));
            });
        ui.separator();
        egui::CollapsingHeader::new(egui::RichText::new("Simetria vinculada").strong())
            .default_open(true)
            .show(ui, |ui| {
                ui.add(egui::DragValue::new(&mut self.pattern_symmetry_axis_x)
                    .speed(0.005).prefix("Eixo X (m) "));
                ui.horizontal(|ui| {
                    if ui.button("Usar X mínimo do molde").clicked() {
                        self.set_symmetry_axis_to_min_x();
                    }
                    if ui.button("Criar par espelhado").clicked() {
                        self.create_mirrored_pair();
                    }
                    if ui.button("Desvincular par").clicked() {
                        self.unlink_selected_symmetry();
                    }
                });
                ui.weak("A edição de vértices e alças é refletida no molde vinculado.");
            });
        egui::CollapsingHeader::new(egui::RichText::new("Ferramentas de desenho").strong())
            .default_open(true)
            .show(ui, |ui| {
                ui.horizontal(|ui| {
                    if ui.button("Fechar peça a partir dos pontos").clicked() {
                        self.create_piece_from_draft();
                    }
                    if ui.button("Adicionar linha interna").clicked() {
                        self.create_internal_line_from_draft();
                    }
                    if ui.button("Cut & Sew").clicked() {
                        self.cut_and_sew_from_draft();
                    }
                });
                ui.weak("Cut & Sew: desenhe um caminho dentro do molde, começando e terminando no contorno.");
                ui.horizontal(|ui| {
                    if ui.button("Criar dobra (vale)").clicked() {
                        self.create_fold_line_from_draft();
                    }
                    if ui.button("Criar dobra (montanha)").clicked() {
                        self.create_mountain_fold_line_from_draft();
                    }
                    if ui.button("Criar prega").clicked() {
                        self.create_pleat_from_draft();
                    }
                    if ui.button("Definir linha de fio").clicked() {
                        self.create_grainline_from_draft();
                    }
                });
                ui.weak("Linha interna simples é visual. Dobras Vale/Montanha e pregas geram restrições físicas quando encontram arestas internas da malha.");
                ui.label(format!("Pontos no rascunho: {}", self.draft_points.len()));
            });
        if let Some(target) = self.selected_target {
            ui.label(format!("Selecionado: {}", match target {
                PatternEditTarget::Piece { .. } => "peça inteira".to_string(),
                PatternEditTarget::Anchor { index, .. } => format!("vértice {}", index + 1),
                PatternEditTarget::Control1 { segment, .. } => format!("alça 1 · segmento {}", segment + 1),
                PatternEditTarget::Control2 { segment, .. } => format!("alça 2 · segmento {}", segment + 1),
                PatternEditTarget::Edge { segment, .. } => format!("borda · segmento {}", segment + 1),
            }));
            ui.horizontal(|ui| {
                if matches!(target, PatternEditTarget::Edge { .. })
                    && ui.button("Inserir vértice").clicked() {
                    self.insert_selected_vertex();
                }
                if matches!(target, PatternEditTarget::Edge { .. })
                    && ui.button("Converter em Bézier").clicked() {
                    self.convert_selected_edge_to_bezier();
                }
                if matches!(target, PatternEditTarget::Anchor { .. })
                    && ui.button("Excluir vértice").clicked() {
                    self.delete_selected_vertex();
                }
            });
        }
        if !self.pattern_project.pieces.is_empty() {
            ui.horizontal(|ui| {
                if ui.button("Duplicar peça selecionada").clicked() {
                    self.duplicate_selected_piece();
                }
                if ui.button("Excluir peça").clicked() {
                    self.delete_selected_piece();
                }
            });
        }

        ui.separator();
        egui::CollapsingHeader::new(egui::RichText::new("Pré-visualização da malha 2D").strong())
            .default_open(false)
            .show(ui, |ui| {
                ui.add(egui::Slider::new(&mut self.pattern_spacing, 0.025..=0.20).text("espaçamento visual (m)"));
                ui.checkbox(&mut self.pattern_show_mesh, "Mostrar triangulação");
                ui.weak("A resolução da simulação 3D é definida separadamente em Física e montagem 3D por molde.");
                ui.label("Os projetos são armazenados em JSON no formato nativo .clothevo.");
            });

        ui.separator();
        ui.strong(format!("Peças ({})", self.pattern_project.pieces.len()));
        for piece in &self.pattern_project.pieces {
            ui.label(format!("{} · {} segmentos", piece.name, piece.outline.segment_count()));
        }
        if !self.pattern_project.pieces.is_empty() {
            let options: Vec<(u64, String)> = self.pattern_project.pieces.iter()
                .map(|piece| (piece.id, piece.name.clone())).collect();
            self.selected_piece = self.selected_piece.min(options.len() - 1);
            egui::ComboBox::from_id_salt("allowance_piece")
                .selected_text(format!("Margem: {}", options[self.selected_piece].1))
                .show_ui(ui, |ui| {
                    for (index, (_, name)) in options.iter().enumerate() {
                        ui.selectable_value(&mut self.selected_piece, index, name);
                    }
                });
            let allowance_before = self.editor_snapshot();
            let allowance_response = ui.add(egui::Slider::new(
                &mut self.pattern_project.pieces[self.selected_piece].seam_allowance_meters,
                0.0..=0.05,
            ).text("margem (m)"));
            if allowance_response.changed() {
                let piece_id = self.pattern_project.pieces[self.selected_piece].id;
                self.sync_symmetric_mate(piece_id);
                self.remember_pattern_change(allowance_before);
            }
        }

        if !self.pattern_project.pieces.is_empty() {
            ui.separator();
            ui.collapsing("Física e montagem 3D por molde", |ui| {
                let index = self.selected_piece.min(self.pattern_project.pieces.len() - 1);
                let piece_id = self.pattern_project.pieces[index].id;
                let before = self.editor_snapshot();
                let mut changed = false;
                {
                    let piece = &mut self.pattern_project.pieces[index];
                    ui.label(format!("Material: {}", piece.name));
                    changed |= ui.add(egui::Slider::new(
                        &mut piece.fabric.density_kg_m2, 0.02..=1.20,
                    ).text("Densidade (kg/m²)")).changed();
                    changed |= ui.add(egui::Slider::new(
                        &mut piece.fabric.stretch_warp_n_m, 100.0..=30000.0,
                    ).text("Tração no urdume (N/m)")).changed();
                    changed |= ui.add(egui::Slider::new(
                        &mut piece.fabric.stretch_weft_n_m, 100.0..=30000.0,
                    ).text("Tração na trama (N/m)")).changed();
                    changed |= ui.add(egui::Slider::new(
                        &mut piece.fabric.shear_n_m, 10.0..=5000.0,
                    ).text("Cisalhamento (N/m)")).changed();
                    changed |= ui.add(egui::Slider::new(
                        &mut piece.fabric.bend_nm, 0.000001..=0.001,
                    ).logarithmic(true).text("Flexão (N·m)")).changed();
                    changed |= ui.add(egui::Slider::new(
                        &mut piece.fabric.thickness_meters, 0.0001..=0.01,
                    ).text("Espessura (m)")).changed();
                    changed |= ui.add(egui::Slider::new(
                        &mut piece.fabric.friction_coefficient, 0.0..=1.5,
                    ).text("Atrito estático")).changed();
                    changed |= ui.add(egui::Slider::new(
                        &mut piece.fabric.friction_dynamic_coefficient, 0.0..=1.5,
                    ).text("Atrito dinâmico"))
                        .on_hover_text("O solver limita o atrito dinâmico para que ele não ultrapasse o estático.")
                        .changed();
                    changed |= ui.add(egui::Slider::new(
                        &mut piece.fabric.particle_distance_meters, 0.01..=0.20,
                    ).text("Distância entre partículas (m)")).changed();

                    ui.separator();
                    ui.label("Posição 3D (deslocamento relativo ao layout 2D)");
                    ui.horizontal(|ui| {
                        changed |= ui.add(egui::DragValue::new(
                            &mut piece.assembly.translation_meters[0],
                        ).speed(0.01).prefix("X ")).changed();
                        changed |= ui.add(egui::DragValue::new(
                            &mut piece.assembly.translation_meters[1],
                        ).speed(0.01).prefix("Y ")).changed();
                        changed |= ui.add(egui::DragValue::new(
                            &mut piece.assembly.translation_meters[2],
                        ).speed(0.01).prefix("Z ")).changed();
                    });
                    ui.label("Rotação local (graus)");
                    changed |= ui.add(egui::Slider::new(
                        &mut piece.assembly.rotation_degrees[0], -180.0..=180.0,
                    ).text("Rotação X")).changed();
                    changed |= ui.add(egui::Slider::new(
                        &mut piece.assembly.rotation_degrees[1], -180.0..=180.0,
                    ).text("Rotação Y")).changed();
                    changed |= ui.add(egui::Slider::new(
                        &mut piece.assembly.rotation_degrees[2], -180.0..=180.0,
                    ).text("Rotação Z")).changed();
                }
                if changed {
                    self.sync_symmetric_mate(piece_id);
                    self.remember_pattern_change(before);
                }

                let selected_anchor = match self.selected_target {
                    Some(PatternEditTarget::Anchor { piece, index: anchor }) if piece == index => Some(anchor),
                    _ => None,
                };
                if let Some(anchor) = selected_anchor {
                    let pinned = self.pattern_project.pieces[index].assembly.pinned_anchors.contains(&(anchor as u32));
                    if ui.button(if pinned {
                        format!("Desafixar vértice {}", anchor + 1)
                    } else {
                        format!("Fixar vértice {} na simulação", anchor + 1)
                    }).clicked() {
                        let before = self.editor_snapshot();
                        let assembly = &mut self.pattern_project.pieces[index].assembly;
                        if pinned {
                            assembly.pinned_anchors.retain(|candidate| *candidate != anchor as u32);
                        } else {
                            assembly.pinned_anchors.push(anchor as u32);
                            assembly.pinned_anchors.sort_unstable();
                            assembly.pinned_anchors.dedup();
                        }
                        self.sync_symmetric_mate(piece_id);
                        self.remember_pattern_change(before);
                    }
                } else {
                    ui.weak("Selecione um vértice do contorno para fixá-lo ou liberá-lo na simulação.");
                }
                if !self.pattern_project.pieces[index].assembly.pinned_anchors.is_empty() {
                    ui.label(format!(
                        "Vértices fixados: {}",
                        self.pattern_project.pieces[index].assembly.pinned_anchors.iter()
                            .map(|id| (id + 1).to_string()).collect::<Vec<_>>().join(", ")
                    ));
                }
            });
        }

        if self.pattern_project.pieces.len() >= 2 {
            ui.separator();
            ui.group(|ui| {
                ui.horizontal(|ui| {
                    ui.strong("Costura direta");
                    let quick_mode = self.seam_pick_mode && self.seam_pick_pair_mode;
                    if ui.selectable_label(
                        quick_mode,
                        if quick_mode { "Cancelar seleção" } else { "Costurar 2 bordas" },
                    ).clicked() {
                        if quick_mode {
                            self.seam_pick_mode = false;
                            self.seam_pick_pair_mode = false;
                            self.seam_canvas_pick_used = false;
                            self.seam_sides_a.clear();
                            self.seam_sides_b.clear();
                            self.pattern_message = "Seleção de costura cancelada.".into();
                        } else {
                            self.reset_seam_controls();
                            self.seam_pick_mode = true;
                            self.seam_pick_pair_mode = true;
                            self.seam_canvas_pick_used = true;
                            self.pattern_tool = PatternTool::Select;
                            self.seam_active_side_a = true;
                            self.pattern_message = "Selecione a borda A e depois a borda B; também pode usar Shift+LMB para clicar ou arrastar.".into();
                        }
                    }
                    ui.label(format!("{} costura(s)", self.pattern_project.seams.len()));
                });
                ui.weak("Atalho: Shift + botão esquerdo. Clique em A e depois B, ou arraste de A até B.");
                if let Some((piece_index, segment)) = self.direct_sewing_start {
                    if let Some(piece) = self.pattern_project.pieces.get(piece_index) {
                        ui.colored_label(
                            egui::Color32::from_rgb(82, 205, 245),
                            format!("A selecionada · {} · borda {}", piece.name, segment + 1),
                        );
                    }
                    ui.small("Selecione uma borda de outra peça. Esc cancela.");
                } else if self.seam_pick_mode && self.seam_pick_pair_mode {
                    ui.label(if self.seam_sides_a.is_empty() {
                        "Modo de costura ativo · selecione a borda A"
                    } else if self.seam_sides_b.is_empty() {
                        "Modo de costura ativo · selecione a borda B"
                    } else {
                        "Validando as bordas selecionadas…"
                    });
                } else {
                    ui.weak("As bordas A/B serão destacadas na viewport antes da união.");
                }
            });

            ui.collapsing("Ajustes avançados · cadeia M:N, folga e Auto Sewing", |ui| {
                ui.strong("Costura por múltiplos segmentos");
                ui.horizontal(|ui| {
                    let chain_label = if self.seam_pick_mode && !self.seam_pick_pair_mode {
                        "Concluir cadeia M:N"
                    } else {
                        "Costura livre / cadeia M:N"
                    };
                    if ui.button(chain_label).clicked() {
                        if self.seam_pick_mode && !self.seam_pick_pair_mode {
                            self.seam_pick_mode = false;
                            self.seam_pick_pair_mode = false;
                            self.pattern_message = "Seleção de cadeia pausada. Revise os lados e crie/salve a costura.".into();
                        } else {
                            if self.seam_pick_pair_mode {
                                self.reset_seam_controls();
                            }
                            self.seam_pick_mode = true;
                            self.seam_pick_pair_mode = false;
                            self.seam_canvas_pick_used = true;
                            self.pattern_tool = PatternTool::Select;
                            self.pattern_message = format!(
                                "Costura livre: selecione as bordas do lado {}. Alterne A/B para construir a outra cadeia.",
                                if self.seam_active_side_a { "A" } else { "B" }
                            );
                        }
                    }
                    ui.selectable_value(&mut self.seam_active_side_a, true, "Lado A");
                    ui.selectable_value(&mut self.seam_active_side_a, false, "Lado B");
                });
                ui.horizontal(|ui| {
                    if ui.button("Limpar A").clicked() {
                        self.seam_sides_a.clear();
                        self.seam_a_active_path = 0;
                        self.seam_canvas_pick_used = true;
                        self.seam_active_side_a = true;
                        self.pattern_message = "Lado A limpo. Selecione novamente as arestas no canvas.".into();
                    }
                    if ui.button("Limpar B").clicked() {
                        self.seam_sides_b.clear();
                        self.seam_b_active_path = 0;
                        self.seam_canvas_pick_used = true;
                        self.seam_active_side_a = false;
                        self.pattern_message = "Lado B limpo. Selecione novamente as arestas no canvas.".into();
                    }
                });
                if self.seam_pick_mode && !self.seam_pick_pair_mode {
                    ui.strong(format!(
                        "MODO COSTURA LIVRE · Lado {} ativo",
                        if self.seam_active_side_a { "A" } else { "B" }
                    ));
                    ui.weak("Clique nas bordas para formar cada cadeia; a correspondência é distribuída pelo comprimento de arco.");
                }
            let mut requested_seam = self.selected_seam_id;
            egui::ComboBox::from_id_salt("selected_seam")
                .selected_text(self.selected_seam_id
                    .and_then(|id| self.pattern_project.seams.iter().find(|seam| seam.id == id))
                    .map(|seam| format!("Editar: {}", seam.name))
                    .unwrap_or_else(|| "Nova costura".to_string()))
                .width(245.0)
                .show_ui(ui, |ui| {
                    ui.selectable_value(&mut requested_seam, None, "Nova costura");
                    for seam in &self.pattern_project.seams {
                        ui.selectable_value(
                            &mut requested_seam, Some(seam.id),
                            format!("{} · #{}", seam.name, seam.id),
                        );
                    }
                });
            if requested_seam != self.selected_seam_id {
                self.seam_pick_mode = false;
                self.seam_pick_pair_mode = false;
                self.seam_canvas_pick_used = false;
                self.selected_seam_id = requested_seam;
                if let Some(id) = requested_seam {
                    if let Some(seam) = self.pattern_project.seams.iter().find(|seam| seam.id == id).cloned() {
                        self.seam_sides_a = seam.sides_a.clone();
                        self.seam_sides_b = seam.sides_b.clone();
                        self.seam_a_active_path = 0;
                        self.seam_b_active_path = 0;
                        self.seam_active_side_a = true;
                        if let (Some(edge_a), Some(edge_b)) = (seam.sides_a.first(), seam.sides_b.first()) {
                            self.load_seam_edge_to_controls(true, edge_a);
                            self.load_seam_edge_to_controls(false, edge_b);
                            self.seam_free_mode = [edge_a, edge_b].iter().any(|edge|
                                edge.start.t > 1.0e-9 || edge.end.t < 1.0 - 1.0e-9);
                        }
                        self.seam_ease_m = seam.ease_meters;
                        self.seam_name = seam.name;
                    }
                } else {
                    self.reset_seam_controls();
                }
            }

            let options: Vec<(u64, String)> = self.pattern_project.pieces.iter()
                .map(|piece| (piece.id, piece.name.clone())).collect();
            self.seam_a_piece = self.seam_a_piece.min(options.len() - 1);
            self.seam_b_piece = self.seam_b_piece.min(options.len() - 1);
            egui::ComboBox::from_id_salt("seam_piece_a")
                .selected_text(format!("A: {}", options[self.seam_a_piece].1))
                .width(245.0)
                .show_ui(ui, |ui| {
                    for (index, (_, name)) in options.iter().enumerate() {
                        ui.selectable_value(&mut self.seam_a_piece, index, name);
                    }
                });
            egui::ComboBox::from_id_salt("seam_piece_b")
                .selected_text(format!("B: {}", options[self.seam_b_piece].1))
                .width(245.0)
                .show_ui(ui, |ui| {
                    for (index, (_, name)) in options.iter().enumerate() {
                        ui.selectable_value(&mut self.seam_b_piece, index, name);
                    }
                });

            let a_count = self.pattern_project.pieces[self.seam_a_piece].outline.segment_count().max(1);
            let b_count = self.pattern_project.pieces[self.seam_b_piece].outline.segment_count().max(1);
            self.seam_a_segment = self.seam_a_segment.min(a_count - 1);
            self.seam_b_segment = self.seam_b_segment.min(b_count - 1);
            self.seam_a_segment_end = self.seam_a_segment_end.clamp(self.seam_a_segment + 1, a_count);
            self.seam_b_segment_end = self.seam_b_segment_end.clamp(self.seam_b_segment + 1, b_count);

            ui.horizontal(|ui| {
                egui::ComboBox::from_id_salt("seam_a_start")
                    .selected_text(format!("Segmentos {} a {}", self.seam_a_segment + 1, self.seam_a_segment_end))
                    .show_ui(ui, |ui| {
                        for index in 0..a_count {
                            ui.selectable_value(&mut self.seam_a_segment, index, format!("Segmentos {} a {}", index + 1, self.seam_a_segment_end));
                        }
                    });
                egui::ComboBox::from_id_salt("seam_a_end")
                    .selected_text(format!("Segmentos {} a {}", self.seam_a_segment + 1, self.seam_a_segment_end))
                    .show_ui(ui, |ui| {
                        for end in (self.seam_a_segment + 1)..=a_count {
                            ui.selectable_value(&mut self.seam_a_segment_end, end, format!("Segmentos {} a {}", self.seam_a_segment + 1, end));
                        }
                    });
            });
            self.seam_a_segment_end = self.seam_a_segment_end.clamp(self.seam_a_segment + 1, a_count);
            ui.horizontal(|ui| {
                egui::ComboBox::from_id_salt("seam_b_start")
                    .selected_text(format!("Segmentos {} a {}", self.seam_b_segment + 1, self.seam_b_segment_end))
                    .show_ui(ui, |ui| {
                        for index in 0..b_count {
                            ui.selectable_value(&mut self.seam_b_segment, index, format!("Segmentos {} a {}", index + 1, self.seam_b_segment_end));
                        }
                    });
                egui::ComboBox::from_id_salt("seam_b_end")
                    .selected_text(format!("Segmentos {} a {}", self.seam_b_segment + 1, self.seam_b_segment_end))
                    .show_ui(ui, |ui| {
                        for end in (self.seam_b_segment + 1)..=b_count {
                            ui.selectable_value(&mut self.seam_b_segment_end, end, format!("Segmentos {} a {}", self.seam_b_segment + 1, end));
                        }
                    });
            });
            self.seam_b_segment_end = self.seam_b_segment_end.clamp(self.seam_b_segment + 1, b_count);

            ui.checkbox(&mut self.seam_free_mode, "Costura livre (extremidades parciais)");
            if self.seam_free_mode {
                ui.label("A — parâmetros das extremidades");
                ui.horizontal(|ui| {
                    ui.add(egui::Slider::new(&mut self.seam_a_start_t, 0.0..=1.0).text("Início t"));
                    ui.add(egui::Slider::new(&mut self.seam_a_end_t, 0.0..=1.0).text("Fim t"));
                });
                ui.label("B — parâmetros das extremidades");
                ui.horizontal(|ui| {
                    ui.add(egui::Slider::new(&mut self.seam_b_start_t, 0.0..=1.0).text("Início t"));
                    ui.add(egui::Slider::new(&mut self.seam_b_end_t, 0.0..=1.0).text("Fim t"));
                });
                ui.add(egui::Slider::new(&mut self.seam_snap_tolerance_px, 4.0..=32.0)
                    .text("Tolerância de encaixe (px)"));
                ui.weak("Arraste os pontos de início/fim no canvas; eles encaixam em vértices e piques.");
            }

            ui.horizontal(|ui| {
                if ui.button("Usar borda selecionada como A").clicked() {
                    if let Some(PatternEditTarget::Edge { piece, segment, .. }) = self.selected_target {
                        self.seam_a_piece = piece.min(options.len() - 1);
                        self.seam_a_segment = segment;
                        self.seam_a_segment_end = segment + 1;
                        self.seam_a_start_t = 0.0;
                        self.seam_a_end_t = 1.0;
                    } else {
                        self.pattern_message = "Selecione uma borda no canvas antes de defini-la como A.".into();
                    }
                }
                if ui.button("Usar borda selecionada como B").clicked() {
                    if let Some(PatternEditTarget::Edge { piece, segment, .. }) = self.selected_target {
                        self.seam_b_piece = piece.min(options.len() - 1);
                        self.seam_b_segment = segment;
                        self.seam_b_segment_end = segment + 1;
                        self.seam_b_start_t = 0.0;
                        self.seam_b_end_t = 1.0;
                    } else {
                        self.pattern_message = "Selecione uma borda no canvas antes de defini-la como B.".into();
                    }
                }
            });

            ui.add(egui::TextEdit::singleline(&mut self.seam_name).hint_text("Nome da costura"));
            let mut ease_cm = self.seam_ease_m * 100.0;
            if ui.add(egui::Slider::new(&mut ease_cm, -50.0..=50.0)
                .text("Diferença-alvo A − B (cm)")).changed()
            {
                self.seam_ease_m = ease_cm / 100.0;
            }
            ui.weak("Folga assinada: positivo = A mais longa; negativo = B mais longa. O valor é aplicado à simulação.");
            ui.checkbox(&mut self.seam_a_reversed, "Inverter direção da borda A");
            ui.checkbox(&mut self.seam_b_reversed, "Inverter direção da borda B");

            ui.weak("Cada lado é uma cadeia ordenada. Na seleção direta, clique nas arestas sem precisar manter Shift; fora dela, Shift+clique continua adicionando ao lado ativo.");
            for is_a in [true, false] {
                let paths = if is_a { self.seam_sides_a.clone() } else { self.seam_sides_b.clone() };
                let active = if is_a { self.seam_a_active_path } else { self.seam_b_active_path };
                let label = if is_a { "Lado A" } else { "Lado B" };
                ui.collapsing(format!("{label} · {} bordas", paths.len()), |ui| {
                    for (index, edge) in paths.iter().enumerate() {
                        let piece_name = self.pattern_project.pieces.iter()
                            .find(|piece| piece.id == edge.piece_id).map(|piece| piece.name.as_str()).unwrap_or("Peça ausente");
                        let len_cm = edge_length_meters(&self.pattern_project, edge) * 100.0;
                        let response = ui.selectable_label(active == index,
                            format!("{} · {} · segmentos {}–{} ({:.1} cm)",
                                index + 1, piece_name, edge.start.segment + 1, edge.end.segment + 1, len_cm));
                        let mut desired_active: Option<usize> = None;
                        let mut list_changed = false;
                        ui.horizontal(|ui| {
                            if response.clicked() { desired_active = Some(index); }
                            if ui.small_button("↑").clicked() && index > 0 {
                                let target = if is_a { &mut self.seam_sides_a } else { &mut self.seam_sides_b };
                                target.swap(index, index - 1);
                                desired_active = Some(index - 1);
                                list_changed = true;
                            }
                            if ui.small_button("↓").clicked() && index + 1 < paths.len() {
                                let target = if is_a { &mut self.seam_sides_a } else { &mut self.seam_sides_b };
                                target.swap(index, index + 1);
                                desired_active = Some(index + 1);
                                list_changed = true;
                            }
                            if paths.len() > 1 && ui.small_button("Remover").clicked() {
                                let target = if is_a { &mut self.seam_sides_a } else { &mut self.seam_sides_b };
                                target.remove(index);
                                desired_active = Some(index.min(target.len() - 1));
                                list_changed = true;
                            }
                        });
                        if let Some(active_index) = desired_active {
                            if is_a { self.seam_a_active_path = active_index; }
                            else { self.seam_b_active_path = active_index; }
                            self.seam_active_side_a = is_a;
                            self.load_active_chain_edge_to_controls(is_a);
                        } else if list_changed {
                            self.load_active_chain_edge_to_controls(is_a);
                        }
                    }
                });
            }

            let piece_a = &self.pattern_project.pieces[self.seam_a_piece];
            let piece_b = &self.pattern_project.pieces[self.seam_b_piece];
            let edge_a = EdgePath {
                piece_id: piece_a.id, curve: BoundaryCurve::Outline,
                start: CurveParam {
                    segment: self.seam_a_segment as u32,
                    t: if self.seam_free_mode { self.seam_a_start_t } else { 0.0 },
                },
                end: CurveParam {
                    segment: self.seam_a_segment_end.saturating_sub(1) as u32,
                    t: if self.seam_free_mode { self.seam_a_end_t } else { 1.0 },
                },
                reversed: self.seam_a_reversed,
            };
            let edge_b = EdgePath {
                piece_id: piece_b.id, curve: BoundaryCurve::Outline,
                start: CurveParam {
                    segment: self.seam_b_segment as u32,
                    t: if self.seam_free_mode { self.seam_b_start_t } else { 0.0 },
                },
                end: CurveParam {
                    segment: self.seam_b_segment_end.saturating_sub(1) as u32,
                    t: if self.seam_free_mode { self.seam_b_end_t } else { 1.0 },
                },
                reversed: self.seam_b_reversed,
            };
            let (display_a, display_b) = self.current_seam_sides_with_draft().unwrap_or((vec![edge_a.clone()], vec![edge_b.clone()]));
            let length_a = display_a.iter().map(|edge| edge_length_meters(&self.pattern_project, edge)).sum::<f64>();
            let length_b = display_b.iter().map(|edge| edge_length_meters(&self.pattern_project, edge)).sum::<f64>();
            let difference = length_a - length_b;
            ui.label(format!("A: {} bordas ({:.1} cm) · B: {} bordas ({:.1} cm)",
                display_a.len(), length_a * 100.0, display_b.len(), length_b * 100.0));
            ui.label(format!("Diferença geométrica A − B: {:+.1} cm · alvo: {:+.1} cm",
                difference * 100.0, self.seam_ease_m * 100.0));
            let shortest = length_a.min(length_b);
            let longest = length_a.max(length_b);
            if shortest > 1.0e-9 && longest / shortest > 3.0 {
                ui.colored_label(
                    egui::Color32::from_rgb(255, 175, 80),
                    format!("Aviso: a razão A/B é {:.2}:1 (acima de 3:1). Confira os segmentos; a costura continuará permitida.",
                        longest / shortest),
                );
            }
            if (difference - self.seam_ease_m).abs() > 0.02 {
                ui.colored_label(
                    egui::Color32::from_rgb(255, 190, 80),
                    "A diferença medida não coincide com o alvo; a simulação ajustará o repouso da borda excedente.",
                );
            }

            let notches_a = edge_chain_notch_positions(&self.pattern_project, &display_a);
            let notches_b = edge_chain_notch_positions(&self.pattern_project, &display_b);
            if notches_a.is_empty() && notches_b.is_empty() {
                ui.weak("Piques pareados: nenhum pique nas bordas selecionadas.");
            } else if notches_a.len() != notches_b.len() {
                ui.colored_label(
                    egui::Color32::from_rgb(255, 150, 90),
                    format!("Piques incompatíveis: A tem {}, B tem {}. Os que existirem serão pareados por ordem.",
                        notches_a.len(), notches_b.len()),
                );
            } else {
                let max_delta = notches_a.iter().zip(&notches_b)
                    .map(|(a, b)| (a - b).abs()).fold(0.0_f32, f32::max);
                if max_delta > 0.03 {
                    ui.colored_label(egui::Color32::from_rgb(255, 150, 90),
                        format!("Piques desalinhados: diferença de posição normalizada máxima {:.1}%.", max_delta * 100.0));
                } else {
                    ui.colored_label(egui::Color32::LIGHT_GREEN,
                        format!("Piques alinhados: {} correspondências.", notches_a.len()));
                }
            }

            ui.horizontal(|ui| {
                if self.selected_seam_id.is_some() {
                    if ui.button("Salvar alterações").clicked() {
                        self.update_selected_seam_from_controls();
                    }
                    if ui.button("Descosturar").clicked() {
                        self.remove_selected_seam();
                    }
                } else if ui.button("Criar costura").clicked() {
                    self.add_seam_from_controls();
                }
                if ui.button("Nova costura").clicked() {
                    self.reset_seam_controls();
                    self.seam_pick_mode = true;
                    self.seam_pick_pair_mode = true;
                    self.seam_canvas_pick_used = true;
                    self.pattern_tool = PatternTool::Select;
                    self.seam_active_side_a = true;
                    self.pattern_message = "Nova costura: clique na aresta A e depois na correspondente B. A costura será criada automaticamente.".into();
                }
            });
            ui.label(format!("Costuras no projeto: {}", self.pattern_project.seams.len()));

            ui.separator();
            egui::CollapsingHeader::new(egui::RichText::new("Verificar comprimentos das costuras").strong())
                .default_open(false)
                .show(ui, |ui| {
                    if self.pattern_project.seams.is_empty() {
                        ui.weak("Nenhuma costura no projeto.");
                    }
                    egui::ScrollArea::vertical().max_height(150.0).show(ui, |ui| {
                        for seam in &self.pattern_project.seams {
                            let length_a = edge_chain_length_meters(&self.pattern_project, &seam.sides_a);
                            let length_b = edge_chain_length_meters(&self.pattern_project, &seam.sides_b);
                            let difference = length_a - length_b;
                            let deviation = (difference - seam.ease_meters).abs();
                            let (color, status) = if deviation <= 0.005 {
                                (egui::Color32::LIGHT_GREEN, "ok")
                            } else if deviation <= SEAM_LENGTH_WARNING_METERS {
                                (egui::Color32::from_rgb(255, 190, 80), "atenção")
                            } else {
                                (egui::Color32::from_rgb(255, 120, 120), "revisar")
                            };
                            ui.colored_label(color, format!(
                                "{} · A {:.1} cm · B {:.1} cm · dif. {:+.1} cm ({})",
                                seam.name, length_a * 100.0, length_b * 100.0, difference * 100.0, status
                            ));
                        }
                    });
                    ui.weak("Verde: até 0,5 cm do alvo · âmbar: até 2,5 cm (1 polegada) · vermelho: acima disso.");
                });

            ui.separator();
            ui.strong("Auto Sewing — sugestões");
            ui.add(egui::Slider::new(&mut self.auto_seam_minimum_score, 0.0..=1.0)
                .text("Score mínimo").custom_formatter(|value, _| format!("{:.0}%", value * 100.0)));
            if ui.button("Sugerir costuras").clicked() {
                self.auto_seam_suggestions = suggest_seams(&self.pattern_project, &SeamSuggestionParams {
                    minimum_score: self.auto_seam_minimum_score,
                    ..SeamSuggestionParams::default()
                });
                self.auto_seam_selected = vec![false; self.auto_seam_suggestions.len()];
                self.hovered_auto_suggestion = None;
                self.pattern_message = format!(
                    "{} sugestão(ões) encontrada(s). Nenhuma costura foi criada.",
                    self.auto_seam_suggestions.len()
                );
            }
            self.hovered_auto_suggestion = None;
            for index in 0..self.auto_seam_suggestions.len() {
                let suggestion = self.auto_seam_suggestions[index].clone();
                ui.horizontal(|ui| {
                    ui.checkbox(&mut self.auto_seam_selected[index], "");
                    let response = ui.selectable_label(
                        false,
                        format!(
                            "#{:02} · {:.0}% · peça {} / seg. {} ↔ peça {} / seg. {}",
                            index + 1, suggestion.score * 100.0,
                            suggestion.a.piece_id, suggestion.a.start.segment + 1,
                            suggestion.b.piece_id, suggestion.b.start.segment + 1,
                        ),
                    );
                    if response.hovered() { self.hovered_auto_suggestion = Some(index); }
                });
                for reason in &suggestion.reasons {
                    ui.weak(reason);
                }
            }
            if !self.auto_seam_suggestions.is_empty()
                && ui.button("Aplicar selecionadas").clicked()
            {
                self.apply_selected_auto_seams();
            }

            });
        } else {
            ui.weak("Crie ao menos duas peças para configurar uma costura.");
        }

        if !self.pattern_project.pieces.is_empty() {
            ui.separator();
            ui.collapsing("Piques e correspondência", |ui| {
                let piece_index = self.selected_piece.min(self.pattern_project.pieces.len() - 1);
                self.selected_piece = piece_index;
                let piece_id = self.pattern_project.pieces[piece_index].id;
                let segment_count = self.pattern_project.pieces[piece_index].outline.segment_count().max(1);
                self.notch_segment = self.notch_segment.min(segment_count - 1);
                egui::ComboBox::from_id_salt("notch_segment")
                    .selected_text(format!("Borda: segmento {}", self.notch_segment + 1))
                    .show_ui(ui, |ui| {
                        for index in 0..segment_count {
                            ui.selectable_value(&mut self.notch_segment, index, format!("Segmento {}", index + 1));
                        }
                    });
                ui.add(egui::Slider::new(&mut self.notch_t, 0.01..=0.99).text("Posição no segmento"));
                ui.add(egui::Slider::new(&mut self.notch_depth_m, 0.002..=0.04).text("Profundidade (m)"));
                ui.add(egui::Slider::new(&mut self.notch_width_m, 0.003..=0.05).text("Largura (m)"));

                let notches = self.pattern_project.pieces[piece_index].notches.clone();
                if !notches.is_empty() {
                    self.selected_notch_index = self.selected_notch_index.min(notches.len() - 1);
                    let mut selected_notch = self.selected_notch_index;
                    egui::ComboBox::from_id_salt("selected_notch")
                        .selected_text(format!("Pique {}", notches[selected_notch].id))
                        .show_ui(ui, |ui| {
                            for (index, notch) in notches.iter().enumerate() {
                                ui.selectable_value(&mut selected_notch, index,
                                    format!("Pique {} · seg. {} · t={:.2}", notch.id, notch.segment_index + 1, notch.t));
                            }
                        });
                    if selected_notch != self.selected_notch_index {
                        self.selected_notch_index = selected_notch;
                        let notch = notches[selected_notch].clone();
                        self.notch_segment = notch.segment_index as usize;
                        self.notch_t = notch.t;
                        self.notch_depth_m = notch.depth_meters;
                        self.notch_width_m = notch.width_meters;
                    }
                    ui.horizontal(|ui| {
                        if ui.button("Aplicar edição do pique").clicked() {
                            self.update_notch_from_controls();
                        }
                        if ui.button("Remover pique").clicked() {
                            self.remove_selected_notch();
                        }
                    });
                } else {
                    ui.weak("Esta peça ainda não possui piques.");
                }
                if ui.button("Adicionar pique à borda").clicked() {
                    self.add_notch_from_controls();
                }
                let _ = piece_id;
                ui.weak("Piques posicionados no mesmo progresso de arco são pareados automaticamente entre as bordas da costura.");
            });
        }

        ui.separator();
        let avatar_before = self.pattern_project.avatar.clone();
        let mut avatar_changed = false;
        let import_pending = self.avatar_import_pending_path.is_some();
        let mut reprocess_avatar = false;
        let mut remove_avatar = false;
        ui.collapsing("Avatar e colisão corporal", |ui| {
            ui.label("Importe OBJ/FBX estáticos. A leitura e a preparação da colisão rodam em segundo plano.");
            avatar_changed |= ui.add(egui::Slider::new(
                &mut self.pattern_project.avatar.height_meters, 1.20..=2.20,
            ).text("Altura de referência (m)")).changed();
            avatar_changed |= ui.add(egui::Slider::new(
                &mut self.pattern_project.avatar.shoulder_width_meters, 0.25..=0.70,
            ).text("Largura dos ombros (m)")).changed();
            avatar_changed |= ui.add(egui::Slider::new(
                &mut self.pattern_project.avatar.waist_width_meters, 0.18..=0.60,
            ).text("Largura da cintura (m)")).changed();
            avatar_changed |= ui.add(egui::Slider::new(
                &mut self.pattern_project.avatar.hip_width_meters, 0.22..=0.70,
            ).text("Largura do quadril (m)")).changed();
            let heading_changed = ui.add(egui::Slider::new(
                &mut self.avatar_heading_degrees, -180.0..=180.0,
            ).text("Giro do avatar em Y (°)")).changed();
            ui.label("OBJ usa detecção automática de eixo; FBX é convertido para Y-up.");
            let spacing_changed = ui.add(egui::Slider::new(
                &mut self.avatar_collision_spacing_m, 0.004..=0.03,
            ).text("Precisão da colisão (m)"))
                .on_hover_text("Espaçamento menor preserva mais detalhes, com maior custo de memória e importação.")
                .changed();
            if heading_changed || spacing_changed {
                self.pattern_dirty = true;
            }

            ui.horizontal(|ui| {
                if self.avatar_mesh_path.is_some() {
                    reprocess_avatar |= ui.add_enabled(
                        !import_pending,
                        egui::Button::new("Aplicar escala e colisão"),
                    ).clicked();
                }
                if self.avatar_mesh.is_some() {
                    remove_avatar |= ui.add_enabled(
                        !import_pending,
                        egui::Button::new("Remover modelo"),
                    ).clicked();
                }
            });
            if import_pending {
                ui.horizontal(|ui| {
                    ui.spinner();
                    ui.label("Preparando malha visual e campo de colisão…");
                });
            }

            if let Some(name) = self.avatar_mesh_name.as_ref() {
                ui.colored_label(
                    egui::Color32::from_rgb(100, 220, 170),
                    format!("Arquivo: {name}"),
                );
            } else {
                ui.weak("Avatar visual atual: manequim procedural.");
            }
            if let Some(diag) = self.avatar_diagnostics.as_ref() {
                ui.label(format!(
                    "{} vértices · {} triângulos · dimensões {:.2} × {:.2} × {:.2} m",
                    diag.vertex_count,
                    diag.triangle_count,
                    diag.dimensions_meters[0],
                    diag.dimensions_meters[1],
                    diag.dimensions_meters[2],
                ));
                if let (Some(spacing), Some(dims)) =
                    (diag.sdf_spacing_meters, diag.sdf_dimensions)
                {
                    ui.label(format!(
                        "Colisão SDF: passo {:.1} mm · grade {} × {} × {} · importação {:.0} ms · colisão {:.0} ms",
                        spacing * 1000.0,
                        dims[0],
                        dims[1],
                        dims[2],
                        diag.import_milliseconds,
                        diag.collision_build_milliseconds,
                    ));
                }
                if let Some(topology) = diag.topology {
                    if topology.has_reliable_inside_outside() {
                        ui.colored_label(
                            egui::Color32::LIGHT_GREEN,
                            "Malha fechada e orientada: classificação interior/exterior habilitada.",
                        );
                    } else {
                        ui.colored_label(
                            egui::Color32::from_rgb(255, 180, 80),
                            format!(
                                "Sinal não confiável: {} arestas abertas, {} não-manifold, {} com orientação inconsistente.",
                                topology.boundary_edges,
                                topology.non_manifold_edges,
                                topology.inconsistent_winding_edges,
                            ),
                        );
                    }
                }
                if diag.discarded_degenerate_triangles > 0 {
                    ui.colored_label(
                        egui::Color32::from_rgb(255, 180, 80),
                        format!(
                            "{} triângulos degenerados foram removidos.",
                            diag.discarded_degenerate_triangles,
                        ),
                    );
                }
                if self.avatar_settings_are_stale() {
                    ui.colored_label(
                        egui::Color32::from_rgb(255, 180, 80),
                        "As opções de escala, orientação ou precisão mudaram. Aplique-as antes da simulação.",
                    );
                }
            }
            if self.avatar_mesh.is_some() {
                ui.weak(if self.avatar_collision_enabled {
                    "A colisão usa o SDF da malha; malhas abertas usam distância sem sinal e exibem aviso."
                } else {
                    "Colisão do avatar desativada; o tecido ainda pode colidir consigo mesmo."
                });
            } else {
                ui.weak("Sem malha importada, a colisão usa o manequim procedural quando ativada.");
            }
        });
        if remove_avatar {
            self.pattern_dirty = true;
            self.avatar_import_receiver = None;
            self.avatar_import_pending_path = None;
            self.avatar_mesh = None;
            self.avatar_sdf = None;
            self.avatar_mesh_name = None;
            self.avatar_mesh_path = None;
            self.avatar_diagnostics = None;
            self.avatar_imported_height_m = None;
            self.avatar_imported_heading_degrees = None;
            self.avatar_imported_collision_spacing_m = None;
            self.pattern_message = "Avatar importado removido; a colisão volta ao manequim procedural.".into();
            if let Err(error) = self.sync_pattern_avatar_colliders() {
                self.pattern_message.push_str(&format!(" {error}"));
            }
            self.upload_scene();
        }
        if reprocess_avatar {
            self.pattern_dirty = true;
            if let Some(path) = self.avatar_mesh_path.clone() {
                self.schedule_avatar_import(&path, true, true);
            }
        }
        if avatar_changed {
            let mut before = self.editor_snapshot();
            before.project.avatar = avatar_before;
            self.remember_pattern_change(before);
        }

        ui.separator();
        if ui.button("Posicionar moldes na cena 3D").clicked() {
            match self.start_pattern_simulation() {
                Ok(()) => {},
                Err(e) => self.pattern_message = e,
            }
        }
        ui.weak("A prévia permite selecionar e arrastar cada painel ao redor do manequim antes de iniciar a física.");
        if !self.pattern_message.is_empty() {
            ui.separator();
            ui.label(&self.pattern_message);
        }
        ui.separator();
        ui.weak("Canvas: clique para adicionar pontos; as peças fechadas aparecem com contorno azul.");
    }

    fn hit_test_sewing_edge(&self, pointer: egui::Pos2, center: egui::Pos2) -> Option<(usize, usize)> {
        let origin = center + self.pattern_pan;
        let scale = self.pattern_zoom.max(1.0);
        let to_screen = |point: PatternPoint2| {
            egui::pos2(origin.x + point.x as f32 * scale, origin.y - point.y as f32 * scale)
        };
        let mut best: Option<(f32, usize, usize)> = None;

        for (piece_index, piece) in self.pattern_project.pieces.iter().enumerate() {
            let sampled = sample_curve_with_ranges(&piece.outline, 16);
            let count = sampled.points.len();
            if count < 2 { continue; }
            for (segment_index, &(start, end)) in sampled.segment_vertex_ranges.iter().enumerate() {
                let sample_count = if end >= start { end - start } else { count - start };
                for step in 0..sample_count {
                    let ia = (start + step) % count;
                    let ib = (ia + 1) % count;
                    let (distance, _) = closest_segment_parameter(
                        pointer,
                        to_screen(sampled.points[ia]),
                        to_screen(sampled.points[ib]),
                    );
                    if distance <= 10.0
                        && best.map(|entry| distance < entry.0).unwrap_or(true)
                    {
                        best = Some((distance, piece_index, segment_index));
                    }
                }
            }
        }
        best.map(|(_, piece, segment)| (piece, segment))
    }

    fn hit_test_pattern(&self, pointer: egui::Pos2, center: egui::Pos2) -> Option<PatternEditTarget> {
        let origin = center + self.pattern_pan;
        let scale = self.pattern_zoom;
        let to_screen = |p: PatternPoint2| {
            egui::pos2(origin.x + p.x as f32 * scale, origin.y - p.y as f32 * scale)
        };

        let mut best_point: Option<(f32, PatternEditTarget)> = None;
        for (piece_index, piece) in self.pattern_project.pieces.iter().enumerate() {
            for (index, point) in outline_anchors(&piece.outline).iter().enumerate() {
                let d = to_screen(*point).distance(pointer);
                if d <= 12.0 && best_point.map(|best| d < best.0).unwrap_or(true) {
                    best_point = Some((d, PatternEditTarget::Anchor { piece: piece_index, index }));
                }
            }
            for (segment_index, segment) in piece.outline.segments.iter().enumerate() {
                if let CurveSegment::CubicBezier { control1, control2, .. } = segment {
                    for (point, target) in [
                        (*control1, PatternEditTarget::Control1 { piece: piece_index, segment: segment_index }),
                        (*control2, PatternEditTarget::Control2 { piece: piece_index, segment: segment_index }),
                    ] {
                        let d = to_screen(point).distance(pointer);
                        if d <= 12.0 && best_point.map(|best| d < best.0).unwrap_or(true) {
                            best_point = Some((d, target));
                        }
                    }
                }
            }
        }
        if let Some((_, target)) = best_point { return Some(target); }

        // Hit-test das bordas usando amostras de Bézier; o parâmetro local permite subdividir
        // o segmento selecionado sem alterar a forma original da curva.
        let mut best_edge: Option<(f32, PatternEditTarget)> = None;
        for (piece_index, piece) in self.pattern_project.pieces.iter().enumerate() {
            let sampled = sample_curve_with_ranges(&piece.outline, 16);
            let count = sampled.points.len();
            for (segment_index, &(start, end)) in sampled.segment_vertex_ranges.iter().enumerate() {
                if count < 2 { continue; }
                let sample_count = if end >= start { end - start } else { count - start };
                if sample_count == 0 { continue; }
                for step in 0..sample_count {
                    let ia = (start + step) % count;
                    let ib = (ia + 1) % count;
                    let a = to_screen(sampled.points[ia]);
                    let b = to_screen(sampled.points[ib]);
                    let (distance, segment_t) = closest_segment_parameter(pointer, a, b);
                    if distance <= 8.0 && best_edge.map(|best| distance < best.0).unwrap_or(true) {
                        let t = ((step as f64 + segment_t as f64) / sample_count as f64).clamp(0.02, 0.98);
                        best_edge = Some((distance, PatternEditTarget::Edge {
                            piece: piece_index,
                            segment: segment_index,
                            t,
                        }));
                    }
                }
            }
        }
        if let Some((_, target)) = best_edge { return Some(target); }

        let world = PatternPoint2::new(
            (pointer.x - origin.x) as f64 / scale as f64,
            (origin.y - pointer.y) as f64 / scale as f64,
        );
        for (piece_index, piece) in self.pattern_project.pieces.iter().enumerate().rev() {
            let polygon = sample_curve(&piece.outline, 16);
            if point_in_polygon(world, &polygon) {
                return Some(PatternEditTarget::Piece { piece: piece_index });
            }
        }
        None
    }

    fn move_pattern_target(&mut self, target: PatternEditTarget, point: PatternPoint2) {
        let piece_index = target.piece();
        let Some(piece_id) = self.pattern_project.pieces.get(piece_index).map(|piece| piece.id) else {
            return;
        };
        {
            let Some(piece) = self.pattern_project.pieces.get_mut(piece_index) else { return; };
            match target {
                PatternEditTarget::Anchor { index, .. } => {
                    set_curve_anchor(&mut piece.outline, index, point);
                }
                PatternEditTarget::Control1 { segment, .. } => {
                    if let Some(CurveSegment::CubicBezier { control1, .. }) = piece.outline.segments.get_mut(segment) {
                        *control1 = point;
                    }
                }
                PatternEditTarget::Control2 { segment, .. } => {
                    if let Some(CurveSegment::CubicBezier { control2, .. }) = piece.outline.segments.get_mut(segment) {
                        *control2 = point;
                    }
                }
                PatternEditTarget::Piece { .. } | PatternEditTarget::Edge { .. } => {}
            }
        }
        self.sync_symmetric_mate(piece_id);
    }

    fn insert_selected_vertex(&mut self) {
        let Some(PatternEditTarget::Edge { piece, segment, t }) = self.selected_target else {
            self.pattern_message = "Selecione uma borda no modo Editar antes de inserir um vértice.".into();
            return;
        };
        let Some(id) = self.pattern_project.pieces.get(piece).map(|piece| piece.id) else { return; };
        let before = self.editor_snapshot();
        let mut project = self.pattern_project.clone();
        let mate_id = symmetry_counterpart_id(&project, id);
        let result = split_project_segment(&mut project, id, segment, t);
        let new_anchor = match result {
            Ok(index) => index,
            Err(error) => { self.pattern_message = error; return; }
        };
        if let Some(mate_id) = mate_id {
            if let Err(error) = split_project_segment(&mut project, mate_id, segment, t) {
                self.pattern_message = format!("Não foi possível inserir o vértice nos dois moldes; nenhuma alteração foi aplicada: {error}");
                return;
            }
        }
        self.pattern_project = project;
        self.selected_target = Some(PatternEditTarget::Anchor { piece, index: new_anchor });
        self.pattern_message = if mate_id.is_some() {
            "Vértice inserido e refletido no molde vinculado."
        } else {
            "Vértice inserido; a curva mantém sua forma original."
        }.into();
        self.remember_pattern_change(before);
    }

    fn convert_selected_edge_to_bezier(&mut self) {
        let Some(PatternEditTarget::Edge { piece, segment, .. }) = self.selected_target else {
            self.pattern_message = "Selecione um segmento para convertê-lo em Bézier.".into();
            return;
        };
        let Some(id) = self.pattern_project.pieces.get(piece).map(|piece| piece.id) else { return; };
        let before = self.editor_snapshot();
        let mut project = self.pattern_project.clone();
        let mate_id = symmetry_counterpart_id(&project, id);
        if let Err(error) = convert_project_segment_to_bezier(&mut project, id, segment) {
            self.pattern_message = error;
            return;
        }
        if let Some(mate_id) = mate_id {
            if let Err(error) = convert_project_segment_to_bezier(&mut project, mate_id, segment) {
                self.pattern_message = format!("Conversão cancelada para preservar a simetria: {error}");
                return;
            }
        }
        self.pattern_project = project;
        self.selected_target = Some(PatternEditTarget::Control1 { piece, segment });
        self.pattern_message = if mate_id.is_some() {
            "Segmento convertido em Bézier nos dois moldes. Arraste as alças para editar."
        } else {
            "Segmento convertido. Arraste as alças roxas para editar a curva."
        }.into();
        self.remember_pattern_change(before);
    }

    fn delete_selected_vertex(&mut self) {
        let Some(PatternEditTarget::Anchor { piece, index }) = self.selected_target else { return; };
        let Some(id) = self.pattern_project.pieces.get(piece).map(|piece| piece.id) else { return; };
        let before = self.editor_snapshot();
        let mut project = self.pattern_project.clone();
        let mate_id = symmetry_counterpart_id(&project, id);
        if let Err(error) = delete_project_anchor(&mut project, id, index) {
            self.pattern_message = error;
            return;
        }
        if let Some(mate_id) = mate_id {
            if let Err(error) = delete_project_anchor(&mut project, mate_id, index) {
                self.pattern_message = format!("Exclusão cancelada para preservar a simetria: {error}");
                return;
            }
        }
        self.pattern_project = project;
        self.selected_target = None;
        self.dragging_target = None;
        self.pattern_message = if mate_id.is_some() {
            "Vértice removido nos dois moldes vinculados."
        } else {
            "Vértice removido."
        }.into();
        self.remember_pattern_change(before);
    }

    fn delete_selected_piece(&mut self) {
        if self.pattern_project.pieces.is_empty() { return; }
        let before = self.editor_snapshot();
        self.selected_piece = self.selected_piece.min(self.pattern_project.pieces.len() - 1);
        let id = self.pattern_project.pieces[self.selected_piece].id;
        self.pattern_project.pieces.remove(self.selected_piece);
        self.pattern_project.seams.retain(|seam| !seam.sides_a.iter().chain(seam.sides_b.iter()).any(|edge| edge.piece_id == id));
        self.pattern_project.symmetry_links.retain(|link| {
            link.piece_a_id != id && link.piece_b_id != id
        });
        self.selected_piece = self.selected_piece.min(self.pattern_project.pieces.len().saturating_sub(1));
        self.selected_target = None;
        self.dragging_target = None;
        self.pattern_message = "Peça excluída; costuras dependentes também foram removidas.".into();
        self.remember_pattern_change(before);
    }

    fn primitive_origin(&self) -> (f64, f64) {
        if self.pattern_project.pieces.is_empty() {
            return (0.0, 0.0);
        }
        let mut max_x = f64::NEG_INFINITY;
        let mut min_y = f64::INFINITY;
        for piece in &self.pattern_project.pieces {
            for point in sample_curve(&piece.outline, 24) {
                max_x = max_x.max(point.x);
                min_y = min_y.min(point.y);
            }
        }
        if !max_x.is_finite() || !min_y.is_finite() { (0.0, 0.0) }
        else { (max_x + 0.10, min_y) }
    }

    fn create_primitive_piece(&mut self, outline: Curve2D, label: &str) {
        let points = sample_curve(&outline, 1);
        if points.len() < 3 {
            self.pattern_message = "A primitiva gerada não formou um contorno fechado válido.".into();
            return;
        }
        let before = self.editor_snapshot();
        let next_id = self.pattern_project.pieces.iter()
            .map(|piece| piece.id).max().unwrap_or(0) + 1;
        let name = format!("{label} {next_id}");
        match PatternPiece::from_polygon(next_id, name, &points) {
            Ok(mut piece) => {
                // Mantém a curva original em vez de converter os segmentos para uma polilinha.
                piece.outline = outline;
                self.pattern_project.pieces.push(piece);
                self.selected_piece = self.pattern_project.pieces.len() - 1;
                self.selected_target = None;
                self.pattern_tool = PatternTool::Select;
                self.pattern_fit_pending = true;
                self.pattern_message = format!("{label} criado como molde vetorial editável.");
                self.remember_pattern_change(before);
            }
            Err(error) => self.pattern_message = error,
        }
    }

    fn set_symmetry_axis_to_min_x(&mut self) {
        let Some(piece) = self.pattern_project.pieces.get(self.selected_piece) else {
            self.pattern_message = "Crie ou selecione um molde primeiro.".into();
            return;
        };
        let points = sample_curve(&piece.outline, 24);
        if let Some(min_x) = points.iter().map(|p| p.x).reduce(f64::min) {
            self.pattern_symmetry_axis_x = min_x;
            self.pattern_message = format!("Eixo de simetria definido em X = {min_x:.3} m.");
        }
    }

    fn create_mirrored_pair(&mut self) {
        if !self.pattern_symmetry_axis_x.is_finite() {
            self.pattern_message = "O eixo de simetria precisa ser um valor finito.".into();
            return;
        }
        let Some(source) = self.pattern_project.pieces.get(self.selected_piece).cloned() else {
            self.pattern_message = "Crie ou selecione um molde primeiro.".into();
            return;
        };
        if self.pattern_project.symmetry_links.iter().any(|link| {
            link.piece_a_id == source.id || link.piece_b_id == source.id
        }) {
            self.pattern_message = "O molde selecionado já possui um par vinculado.".into();
            return;
        }

        let before = self.editor_snapshot();
        let new_id = self.pattern_project.pieces.iter().map(|piece| piece.id)
            .max().unwrap_or(0) + 1;
        let mirrored_name = format!("{} (espelho)", source.name);
        let mirrored = match mirror_pattern_piece(
            &source, new_id, mirrored_name, self.pattern_symmetry_axis_x,
        ) {
            Ok(piece) => piece,
            Err(error) => { self.pattern_message = error; return; }
        };

        let mut project = self.pattern_project.clone();
        project.pieces.push(mirrored);
        project.symmetry_links.push(SymmetryLink {
            piece_a_id: source.id,
            piece_b_id: new_id,
            axis_x: self.pattern_symmetry_axis_x,
        });

        // Replica a costura M:N apenas quando todos os moldes dos dois lados têm contrapartes.
        let original_seams = project.seams.clone();
        let mut next_seam_id = project.seams.iter().map(|seam| seam.id).max().unwrap_or(0) + 1;
        for seam in original_seams {
            let mut mirrored_seam = seam.clone();
            let mut changed = false;
            let mut complete = true;
            for edge in mirrored_seam.sides_a.iter_mut().chain(mirrored_seam.sides_b.iter_mut()) {
                let Some(mate) = symmetry_counterpart_id(&project, edge.piece_id) else {
                    complete = false; break;
                };
                changed |= mate != edge.piece_id;
                edge.piece_id = mate;
            }
            if !complete || !changed { continue; }
            mirrored_seam.id = next_seam_id;
            mirrored_seam.name = format!("{} (espelho)", seam.name);
            project.seams.push(mirrored_seam);
            next_seam_id += 1;
        }

        if let Err(error) = project.validate() {
            self.pattern_message = format!("Não foi possível criar o par espelhado: {error}");
            return;
        }
        self.pattern_project = project;
        self.selected_piece = self.pattern_project.pieces.len() - 1;
        self.selected_target = Some(PatternEditTarget::Piece { piece: self.selected_piece });
        self.pattern_fit_pending = true;
        self.pattern_message = format!(
            "Par espelhado criado para '{}'; edições de geometria serão sincronizadas nos dois sentidos.",
            source.name
        );
        self.remember_pattern_change(before);
    }

    fn unlink_selected_symmetry(&mut self) {
        let Some(piece) = self.pattern_project.pieces.get(self.selected_piece) else { return; };
        let id = piece.id;
        let before = self.editor_snapshot();
        let old_len = self.pattern_project.symmetry_links.len();
        self.pattern_project.symmetry_links.retain(|link| {
            link.piece_a_id != id && link.piece_b_id != id
        });
        if self.pattern_project.symmetry_links.len() == old_len {
            self.pattern_message = "O molde selecionado não possui vínculo de simetria.".into();
            return;
        }
        self.pattern_message = "Par desvinculado. Os dois moldes permanecem na cena como peças independentes.".into();
        self.remember_pattern_change(before);
    }

    fn sync_symmetric_mate(&mut self, piece_id: u64) {
        let Some(link) = self.pattern_project.symmetry_links.iter()
            .find(|link| link.piece_a_id == piece_id || link.piece_b_id == piece_id)
            .cloned() else { return; };
        let mate_id = if link.piece_a_id == piece_id { link.piece_b_id } else { link.piece_a_id };
        let Some(source) = self.pattern_project.pieces.iter().find(|piece| piece.id == piece_id).cloned() else {
            return;
        };
        let Some(target_name) = self.pattern_project.pieces.iter()
            .find(|piece| piece.id == mate_id).map(|piece| piece.name.clone()) else {
            return;
        };
        match mirror_pattern_piece(&source, mate_id, target_name, link.axis_x) {
            Ok(mirrored) => {
                if let Some(target) = self.pattern_project.pieces.iter_mut().find(|piece| piece.id == mate_id) {
                    *target = mirrored;
                }
            }
            Err(error) => self.pattern_message = format!("Falha ao sincronizar a simetria: {error}"),
        }
    }

    fn create_rectangle_primitive(&mut self) {
        let (x, y) = self.primitive_origin();
        match rectangle(
            PatternPoint2::new(x, y),
            PatternPoint2::new(x + 0.40, y + 0.60),
        ) {
            Ok(outline) => self.create_primitive_piece(outline, "Retângulo"),
            Err(error) => self.pattern_message = error.to_string(),
        }
    }

    fn create_regular_polygon_primitive(&mut self) {
        let (x, y) = self.primitive_origin();
        let sides = self.pattern_polygon_sides.clamp(3, 100);
        match regular_polygon(
            PatternPoint2::new(x + 0.20, y + 0.30),
            0.20,
            sides,
            -std::f64::consts::FRAC_PI_2,
        ) {
            Ok(outline) => self.create_primitive_piece(outline, "Polígono"),
            Err(error) => self.pattern_message = error.to_string(),
        }
    }

    fn drafted_internal_curve(&self) -> Result<(usize, Curve2D), String> {
        if self.draft_points.len() < 2 {
            return Err("Desenhe ao menos dois pontos para uma linha interna.".into());
        }
        if self.pattern_project.pieces.is_empty() {
            return Err("Crie ou selecione um molde antes de adicionar detalhes internos.".into());
        }
        let piece_index = self.selected_piece.min(self.pattern_project.pieces.len() - 1);
        let outline = sample_curve(&self.pattern_project.pieces[piece_index].outline, 24);
        let curve = Curve2D::polyline(&self.draft_points, false)?;
        // Evita gravar linhas que saiam do molde. Cada segmento é amostrado para detectar
        // trechos que atravessam o contorno, não apenas os seus pontos de controle.
        for pair in self.draft_points.windows(2) {
            for step in 0..=16 {
                let t = step as f64 / 16.0;
                let point = PatternPoint2::new(
                    pair[0].x + (pair[1].x - pair[0].x) * t,
                    pair[0].y + (pair[1].y - pair[0].y) * t,
                );
                if !point_in_polygon(point, &outline) {
                    return Err("A linha interna precisa permanecer dentro do contorno do molde.".into());
                }
            }
        }
        Ok((piece_index, curve))
    }

    fn create_internal_line_from_draft(&mut self) {
        let (piece_index, curve) = match self.drafted_internal_curve() {
            Ok(value) => value,
            Err(error) => { self.pattern_message = error; return; }
        };
        let before = self.editor_snapshot();
        let piece_id = self.pattern_project.pieces[piece_index].id;
        self.pattern_project.pieces[piece_index].internal_lines.push(curve);
        self.sync_symmetric_mate(piece_id);
        self.draft_points.clear();
        self.pattern_tool = PatternTool::Select;
        self.pattern_message = "Linha interna adicionada como detalhe visual; ela não altera a física. Use Criar dobra ou Criar prega para gerar vincos físicos.".into();
        self.remember_pattern_change(before);
    }

    fn create_fold_line_from_draft(&mut self) {
        self.create_fold_line_from_draft_with_kind(FoldKind::Valley);
    }

    fn create_mountain_fold_line_from_draft(&mut self) {
        self.create_fold_line_from_draft_with_kind(FoldKind::Mountain);
    }

    fn create_fold_line_from_draft_with_kind(&mut self, kind: FoldKind) {
        let (piece_index, curve) = match self.drafted_internal_curve() {
            Ok(value) => value,
            Err(error) => { self.pattern_message = error; return; }
        };
        let before = self.editor_snapshot();
        let piece_id = self.pattern_project.pieces[piece_index].id;
        self.pattern_project.pieces[piece_index].fold_lines.push(FoldLine {
            curve,
            kind,
            stiffness: 0.35,
            angle_radians: std::f64::consts::FRAC_PI_6,
        });
        self.sync_symmetric_mate(piece_id);
        self.draft_points.clear();
        self.pattern_tool = PatternTool::Select;
        let kind_name = if kind == FoldKind::Mountain { "montanha" } else { "vale" };
        self.pattern_message = format!(
            "Dobra física de {kind_name} criada: alvo de 30° e 35% da rigidez de flexão base."
        );
        self.remember_pattern_change(before);
    }

    fn create_pleat_from_draft(&mut self) {
        let (piece_index, axis) = match self.drafted_internal_curve() {
            Ok(value) => value,
            Err(error) => { self.pattern_message = error; return; }
        };
        let before = self.editor_snapshot();
        let piece_id = self.pattern_project.pieces[piece_index].id;
        self.pattern_project.pieces[piece_index].pleats.push(Pleat {
            axis,
            depth_meters: 0.02,
            angle_radians: std::f64::consts::FRAC_PI_4,
        });
        self.sync_symmetric_mate(piece_id);
        self.draft_points.clear();
        self.pattern_tool = PatternTool::Select;
        self.pattern_message = "Prega física criada: dois vincos opostos, separados por 2 cm, com alvo inicial de 45°.".into();
        self.remember_pattern_change(before);
    }

    fn create_grainline_from_draft(&mut self) {
        if self.draft_points.len() != 2 {
            self.pattern_message = "A linha de fio exige exatamente dois pontos.".into();
            return;
        }
        let (piece_index, curve) = match self.drafted_internal_curve() {
            Ok(value) => value,
            Err(error) => { self.pattern_message = error; return; }
        };
        let start = curve.start;
        let Some(end) = curve.endpoint() else { return; };
        let before = self.editor_snapshot();
        let piece_id = self.pattern_project.pieces[piece_index].id;
        self.pattern_project.pieces[piece_index].grain_line = Some(GrainLine { start, end });
        self.sync_symmetric_mate(piece_id);
        self.draft_points.clear();
        self.pattern_tool = PatternTool::Select;
        self.pattern_message = "Linha de fio atualizada.".into();
        self.remember_pattern_change(before);
    }

    fn cut_and_sew_from_draft(&mut self) {
        if self.draft_points.len() < 2 {
            self.pattern_message = "Para Cut & Sew, desenhe um caminho com pelo menos dois pontos.".into();
            return;
        }
        if self.pattern_project.pieces.is_empty() {
            self.pattern_message = "Crie um molde antes de executar Cut & Sew.".into();
            return;
        }

        let source_index = self.selected_piece.min(self.pattern_project.pieces.len() - 1);
        let source = self.pattern_project.pieces[source_index].clone();
        let cut_path = match Curve2D::polyline(&self.draft_points, false) {
            Ok(curve) => curve,
            Err(error) => { self.pattern_message = error; return; }
        };
        let cut_result = match cut_closed_outline(&source.outline, &cut_path, 0.035) {
            Ok(result) => result,
            Err(error) => {
                self.pattern_message = match error {
                    pattern_geom::GeometryError::SelfIntersection =>
                        "Corte inválido: o caminho se cruza. Use um caminho simples.".into(),
                    pattern_geom::GeometryError::InvalidParameter =>
                        "Corte inválido: mantenha o caminho dentro do molde e conecte as duas extremidades ao contorno (tolerância de 3,5 cm).".into(),
                    _ => format!("Não foi possível dividir o molde: {error}"),
                };
                return;
            }
        };

        let symmetry_link = self.pattern_project.symmetry_links.iter()
            .find(|link| link.piece_a_id == source.id || link.piece_b_id == source.id)
            .cloned();
        let max_piece_id = self.pattern_project.pieces.iter()
            .map(|piece| piece.id).max().unwrap_or(0);
        let second_id = max_piece_id + 1;

        // If the source has a linked mirror, cut the mate with the reflected path in the same
        // transaction. This keeps both sides topologically paired instead of silently diverging.
        let paired_cut: Option<(usize, PatternPiece, CutAndSewResult, u64, f64)> =
            if let Some(link) = &symmetry_link {
                let mate_id = if link.piece_a_id == source.id { link.piece_b_id } else { link.piece_a_id };
                let Some(mate_index) = self.pattern_project.pieces.iter()
                    .position(|piece| piece.id == mate_id) else {
                    self.pattern_message = "O par espelhado referencia um molde inexistente.".into();
                    return;
                };
                let mate_source = self.pattern_project.pieces[mate_index].clone();
                let mirrored_path = match mirror_curve_x(&cut_path, link.axis_x) {
                    Ok(path) => path,
                    Err(error) => { self.pattern_message = error.to_string(); return; }
                };
                let mate_result = match cut_closed_outline(&mate_source.outline, &mirrored_path, 0.035) {
                    Ok(result) => result,
                    Err(error) => {
                        self.pattern_message = format!(
                            "O corte do par espelhado não pôde ser validado ({error}); nenhum dos moldes foi alterado."
                        );
                        return;
                    }
                };
                Some((mate_index, mate_source, mate_result, max_piece_id + 2, link.axis_x))
            } else { None };

        let first_id = source.id;
        let mut first = source.clone();
        let mut second = source.clone();
        first.name = format!("{} (A)", source.name);
        second.id = second_id;
        second.name = format!("{} (B)", source.name);
        first.outline = cut_result.first.clone();
        second.outline = cut_result.second.clone();

        let polygon_a = sample_curve(&first.outline, 24);
        let polygon_b = sample_curve(&second.outline, 24);
        let sewing_side = match split_piece_metadata(
            &source, &mut first, &mut second, &cut_result, &polygon_a, &polygon_b,
        ) {
            Ok(side) => side,
            Err(error) => {
                self.pattern_message = format!("Cut & Sew cancelado sem alterar o molde: {error}");
                return;
            }
        };

        let mut mate_pieces: Option<(PatternPiece, PatternPiece, usize, u64, Option<usize>)> = None;
        if let Some((mate_index, mate_source, mate_result, mate_second_id, _)) = &paired_cut {
            let mut mate_first = mate_source.clone();
            let mut mate_second = mate_source.clone();
            mate_first.name = format!("{} (A)", mate_source.name);
            mate_second.id = *mate_second_id;
            mate_second.name = format!("{} (B)", mate_source.name);
            mate_first.outline = mate_result.first.clone();
            mate_second.outline = mate_result.second.clone();
            let mate_polygon_a = sample_curve(&mate_first.outline, 24);
            let mate_polygon_b = sample_curve(&mate_second.outline, 24);
            let mate_sewing_side = match split_piece_metadata(
                mate_source, &mut mate_first, &mut mate_second, mate_result,
                &mate_polygon_a, &mate_polygon_b,
            ) {
                Ok(side) => side,
                Err(error) => {
                    self.pattern_message = format!(
                        "Cut & Sew cancelado: não foi possível particionar os metadados do molde espelhado: {error}"
                    );
                    return;
                }
            };
            mate_pieces = Some((
                mate_first,
                mate_second,
                *mate_index,
                mate_source.id,
                mate_sewing_side,
            ));
        }

        let mut project = self.pattern_project.clone();
        // Remap existing references on both source pieces before replacing them. If a previous
        // seam crosses a segment split by the new cut, reject atomically rather than corrupt it.
        for seam in &mut project.seams {
            for edge in seam.sides_a.iter_mut().chain(seam.sides_b.iter_mut()) {
                let target = if edge.piece_id == first_id {
                    Some((&cut_result, first_id, second_id, sewing_side))
                } else if let Some((_, mate_source, mate_result, mate_second_id, _)) = &paired_cut {
                    if edge.piece_id == mate_source.id {
                        Some((mate_result, mate_source.id, *mate_second_id,
                            mate_pieces.as_ref().and_then(|entry| entry.4)))
                    } else { None }
                } else { None };
                let Some((result, base_id, new_id, source_sewing_side)) = target else { continue; };
                match edge.curve {
                    BoundaryCurve::Outline => {
                        if let Err(error) = remap_edge_after_cut(
                            edge, &result.source_segment_fragments, base_id, new_id,
                        ) {
                            self.pattern_message = format!(
                                "Cut & Sew cancelado: {error} A costura original foi preservada."
                            );
                            return;
                        }
                    }
                    BoundaryCurve::SewingLine => {
                        let Some(side) = source_sewing_side else {
                            self.pattern_message = "Cut & Sew cancelado: uma costura existente usa uma linha de costura que não pôde ser atribuída a um dos lados.".into();
                            return;
                        };
                        edge.piece_id = if side == 0 { base_id } else { new_id };
                    }
                }
            }
        }

        project.pieces[source_index] = first;
        project.pieces.push(second);
        if let Some((mate_first, mate_second, mate_index, mate_source_id, _)) = mate_pieces {
            project.pieces[mate_index] = mate_first;
            project.pieces.push(mate_second);

            if let Some((_, _, _, mate_second_id, axis_x)) = &paired_cut {
                project.symmetry_links.push(SymmetryLink {
                    piece_a_id: second_id,
                    piece_b_id: *mate_second_id,
                    axis_x: *axis_x,
                });
            }
            let _ = mate_source_id;
        }

        let mut next_seam_id = project.seams.iter()
            .map(|seam| seam.id).max().unwrap_or(0) + 1;
        let (a_start, a_end) = cut_result.first_cut_segment_range;
        let (b_start, b_end) = cut_result.second_cut_segment_range;
        project.seams.push(SeamDefinition {
            id: next_seam_id,
            name: format!("Cut & Sew {next_seam_id}"),
            sides_a: vec![EdgeRef {
                piece_id: first_id, curve: BoundaryCurve::Outline,
                segment_start: a_start as u32, segment_end: a_end as u32, reversed: false,
            }.into()],
            sides_b: vec![EdgeRef {
                piece_id: second_id, curve: BoundaryCurve::Outline,
                segment_start: b_start as u32, segment_end: b_end as u32, reversed: true,
            }.into()],
            ease_meters: 0.0,
        });
        next_seam_id += 1;

        if let Some((_, mate_source, mate_result, mate_second_id, _)) = &paired_cut {
            let (mate_a_start, mate_a_end) = mate_result.first_cut_segment_range;
            let (mate_b_start, mate_b_end) = mate_result.second_cut_segment_range;
            project.seams.push(SeamDefinition {
                id: next_seam_id,
                name: format!("Cut & Sew {next_seam_id}"),
                sides_a: vec![EdgeRef {
                    piece_id: mate_source.id, curve: BoundaryCurve::Outline,
                    segment_start: mate_a_start as u32, segment_end: mate_a_end as u32, reversed: false,
                }.into()],
                sides_b: vec![EdgeRef {
                    piece_id: *mate_second_id, curve: BoundaryCurve::Outline,
                    segment_start: mate_b_start as u32, segment_end: mate_b_end as u32, reversed: true,
                }.into()],
                ease_meters: 0.0,
            });
        }

        if let Err(error) = project.validate() {
            self.pattern_message = format!("Cut & Sew cancelado pela validação do projeto: {error}");
            return;
        }

        let before = self.editor_snapshot();
        self.pattern_project = project;
        self.selected_piece = source_index;
        self.selected_target = Some(PatternEditTarget::Piece { piece: source_index });
        self.draft_points.clear();
        self.pattern_tool = PatternTool::Select;
        self.pattern_fit_pending = true;
        self.pattern_message = if paired_cut.is_some() {
            format!(
                "Cut & Sew concluído nos dois moldes espelhados: '{}' foi dividido em quatro peças, com duas novas costuras e o vínculo de simetria preservado.",
                source.name
            )
        } else {
            format!(
                "Cut & Sew concluído: '{}' foi dividido em duas peças, com a nova costura {}.",
                source.name, next_seam_id - 1
            )
        };
        self.remember_pattern_change(before);
    }

        fn create_piece_from_draft(&mut self) {
        if self.draft_points.len() < 3 {
            self.pattern_message = "Uma peça precisa de pelo menos 3 pontos.".into();
            return;
        }
        let before = self.editor_snapshot();
        let next_id = self.pattern_project.pieces.iter().map(|piece| piece.id).max().unwrap_or(0) + 1;
        let piece_name = format!("Peça {next_id}");
        match PatternPiece::from_polygon(next_id, piece_name, &self.draft_points) {
            Ok(piece) => {
                self.pattern_project.pieces.push(piece);
                self.selected_piece = self.pattern_project.pieces.len() - 1;
                self.selected_target = None;
                self.draft_points.clear();
                self.pattern_tool = PatternTool::Select;
                self.pattern_message = "Peça criada. Use Selecionar/editar para mover a peça, vértices e alças.".into();
                self.remember_pattern_change(before);
            }
            Err(e) => self.pattern_message = e,
        }
    }

    fn create_direct_seam_from_edges(&mut self, a: (usize, usize), b: (usize, usize)) -> bool {
        if a.0 == b.0 {
            self.pattern_message = "A borda B precisa pertencer a outra peça.".into();
            return false;
        }
        let Some(piece_a_id) = self.pattern_project.pieces.get(a.0)
            .filter(|piece| a.1 < piece.outline.segment_count())
            .map(|piece| piece.id)
        else {
            self.pattern_message = "A borda A selecionada não existe mais.".into();
            return false;
        };
        let Some(piece_b_id) = self.pattern_project.pieces.get(b.0)
            .filter(|piece| b.1 < piece.outline.segment_count())
            .map(|piece| piece.id)
        else {
            self.pattern_message = "A borda B selecionada não existe mais.".into();
            return false;
        };

        let edge_a = EdgePath::from_segments(
            piece_a_id, BoundaryCurve::Outline, a.1 as u32, (a.1 + 1) as u32, false,
        );
        let mut edge_b = EdgePath::from_segments(
            piece_b_id, BoundaryCurve::Outline, b.1 as u32, (b.1 + 1) as u32, false,
        );
        // Escolhe automaticamente a orientação que melhor aproxima os extremos A↔B.
        // Isso evita que dois contornos com sentidos opostos sejam pareados torcidos.
        let points_a = edge_path_points(&self.pattern_project, &edge_a, 8);
        let points_b = edge_path_points(&self.pattern_project, &edge_b, 8);
        if points_a.len() >= 2 && points_b.len() >= 2 {
            let same_direction = points_a[0].distance(points_b[0])
                + points_a[points_a.len() - 1].distance(points_b[points_b.len() - 1]);
            let reversed_direction = points_a[0].distance(points_b[points_b.len() - 1])
                + points_a[points_a.len() - 1].distance(points_b[0]);
            edge_b.reversed = reversed_direction < same_direction;
        }

        self.reset_seam_controls();
        self.pattern_tool = PatternTool::Select;
        self.seam_pick_mode = true;
        self.seam_pick_pair_mode = true;
        self.seam_canvas_pick_used = true;
        self.seam_active_side_a = true;
        self.add_draft_edge_to_side(true, edge_a);
        self.add_draft_edge_to_side(false, edge_b);

        let old_count = self.pattern_project.seams.len();
        self.add_seam_from_controls();
        let created = self.pattern_project.seams.len() == old_count + 1;
        if created {
            self.pattern_message = format!(
                "Costura '{}' criada pelo atalho Shift+LMB.", 
                self.selected_seam_id.map(|id| format!("#{id}")).unwrap_or_default()
            );
            self.selected_target = Some(PatternEditTarget::Edge { piece: a.0, segment: a.1, t: 0.5 });
        }
        created
    }

    fn reset_seam_controls(&mut self) {
        self.selected_seam_id = None;
        self.seam_name.clear();
        self.seam_ease_m = 0.0;
        self.seam_a_segment = 0;
        self.seam_b_segment = 0;
        self.seam_a_segment_end = 1;
        self.seam_b_segment_end = 1;
        self.seam_sides_a.clear();
        self.seam_sides_b.clear();
        self.seam_a_active_path = 0;
        self.seam_b_active_path = 0;
        self.seam_active_side_a = true;
        self.seam_a_reversed = false;
        self.seam_b_reversed = false;
        self.seam_a_start_t = 0.0;
        self.seam_a_end_t = 1.0;
        self.seam_b_start_t = 0.0;
        self.seam_b_end_t = 1.0;
        self.seam_free_mode = false;
        self.seam_pick_mode = false;
        self.seam_pick_pair_mode = false;
        self.seam_canvas_pick_used = false;
        self.dragging_seam_handle = None;
        self.seam_drag_before = None;
        self.direct_sewing_start = None;
        self.direct_sewing_preselected = None;
        self.direct_sewing_press_edge = None;
        self.direct_sewing_preview = None;
        self.direct_sewing_dragging = false;
    }

    fn load_seam_edge_to_controls(&mut self, is_a: bool, edge: &EdgePath) {
        let Some(index) = self.pattern_project.pieces.iter().position(|piece| piece.id == edge.piece_id) else {
            return;
        };
        let segment_count = self.pattern_project.pieces[index].outline.segment_count().max(1);
        let start = (edge.start.segment as usize).min(segment_count - 1);
        let end = (edge.end.segment as usize
            + usize::from(edge.end.t >= 1.0 - 1.0e-9)).clamp(start + 1, segment_count);
        if is_a {
            self.seam_a_piece = index;
            self.seam_a_segment = start;
            self.seam_a_segment_end = end;
            self.seam_a_start_t = edge.start.t;
            self.seam_a_end_t = edge.end.t;
            self.seam_a_reversed = edge.reversed;
        } else {
            self.seam_b_piece = index;
            self.seam_b_segment = start;
            self.seam_b_segment_end = end;
            self.seam_b_start_t = edge.start.t;
            self.seam_b_end_t = edge.end.t;
            self.seam_b_reversed = edge.reversed;
        }
        self.seam_free_mode |= edge.start.t > 1.0e-9 || edge.end.t < 1.0 - 1.0e-9;
    }

    fn load_active_chain_edge_to_controls(&mut self, is_a: bool) {
        let index = if is_a { self.seam_a_active_path } else { self.seam_b_active_path };
        let edge = if is_a { self.seam_sides_a.get(index) } else { self.seam_sides_b.get(index) }.cloned();
        if let Some(edge) = edge { self.load_seam_edge_to_controls(is_a, &edge); }
    }

    fn current_seam_sides_with_draft(&self) -> Result<(Vec<EdgePath>, Vec<EdgePath>), String> {
        let (draft_a, draft_b) = self.seam_edges_from_controls()?;
        let direct_capture = self.seam_pick_mode || self.seam_canvas_pick_used;
        let merge = |stored: &[EdgePath], active: usize, draft: EdgePath| {
            let mut result = stored.to_vec();
            if result.is_empty() {
                if !direct_capture { result.push(draft); }
            } else if active < result.len() {
                result[active] = draft;
            } else if !direct_capture {
                result.push(draft);
            }
            result
        };
        Ok((
            merge(&self.seam_sides_a, self.seam_a_active_path, draft_a),
            merge(&self.seam_sides_b, self.seam_b_active_path, draft_b),
        ))
    }

    fn add_draft_edge_to_side(&mut self, is_a: bool, edge: EdgePath) {
        let target = if is_a { &mut self.seam_sides_a } else { &mut self.seam_sides_b };
        if target.iter().any(|existing| existing.piece_id == edge.piece_id
            && existing.curve == edge.curve
            && existing.start_position() < edge.end_position() - 1.0e-9
            && edge.start_position() < existing.end_position() - 1.0e-9)
        {
            self.pattern_message = "Essa borda já está presente ou sobreposta no lado da costura.".into();
            return;
        }
        target.push(edge);
        if is_a { self.seam_a_active_path = target.len() - 1; }
        else { self.seam_b_active_path = target.len() - 1; }
        self.seam_active_side_a = is_a;
        self.load_active_chain_edge_to_controls(is_a);
        self.pattern_message = format!("Borda adicionada ao lado {}.", if is_a { "A" } else { "B" });
    }

    fn seam_edges_from_controls(&self) -> Result<(EdgePath, EdgePath), String> {
        let count = self.pattern_project.pieces.len();
        if count < 2 { return Err("Crie pelo menos duas peças antes de costurá-las.".into()); }
        let a_piece = &self.pattern_project.pieces[self.seam_a_piece.min(count - 1)];
        let b_piece = &self.pattern_project.pieces[self.seam_b_piece.min(count - 1)];
        let make_edge = |piece: &PatternPiece, start: usize, end: usize,
                         start_t: f64, end_t: f64, reversed: bool| {
            let segment_count = piece.outline.segment_count();
            let start = start.min(segment_count.saturating_sub(1));
            let end = end.clamp(start + 1, segment_count);
            EdgePath {
                piece_id: piece.id,
                curve: BoundaryCurve::Outline,
                start: CurveParam {
                    segment: start as u32,
                    t: if self.seam_free_mode { start_t.clamp(0.0, 1.0) } else { 0.0 },
                },
                end: CurveParam {
                    segment: (end - 1) as u32,
                    t: if self.seam_free_mode { end_t.clamp(0.0, 1.0) } else { 1.0 },
                },
                reversed,
            }
        };
        Ok((
            make_edge(a_piece, self.seam_a_segment, self.seam_a_segment_end,
                self.seam_a_start_t, self.seam_a_end_t, self.seam_a_reversed),
            make_edge(b_piece, self.seam_b_segment, self.seam_b_segment_end,
                self.seam_b_start_t, self.seam_b_end_t, self.seam_b_reversed),
        ))
    }

    fn hit_test_seam_handle<F: Fn(PatternPoint2) -> egui::Pos2>(
        &self,
        pointer: egui::Pos2,
        to_screen: &F,
    ) -> Option<(bool, bool)> {
        if !self.seam_free_mode { return None; }
        let (edge_a, edge_b) = self.seam_edges_from_controls().ok()?;
        for (is_a, edge) in [(true, edge_a), (false, edge_b)] {
            let curve = edge_curve(&self.pattern_project, &edge)?;
            let start = curve_segment_point(curve, edge.start.segment as usize, edge.start.t)?;
            let end = curve_segment_point(curve, edge.end.segment as usize, edge.end.t)?;
            if pointer.distance(to_screen(start)) <= 13.0 { return Some((is_a, true)); }
            if pointer.distance(to_screen(end)) <= 13.0 { return Some((is_a, false)); }
        }
        None
    }

    fn update_seam_handle_from_pointer(&mut self, is_a: bool, is_start: bool, world: PatternPoint2) {
        let piece_index = if is_a { self.seam_a_piece } else { self.seam_b_piece };
        let Some(piece) = self.pattern_project.pieces.get(piece_index) else { return; };
        let curve = &piece.outline;
        let Some((mut param, _)) = closest_curve_param(curve, world) else { return; };
        param = snap_curve_param_to_features(
            curve, &piece.notches, param, world,
            self.seam_snap_tolerance_px as f64 / self.pattern_zoom.max(1.0) as f64,
        );
        let (start_t, end_t, start_segment, end_exclusive) = if is_a {
            (self.seam_a_start_t, self.seam_a_end_t,
             self.seam_a_segment, self.seam_a_segment_end)
        } else {
            (self.seam_b_start_t, self.seam_b_end_t,
             self.seam_b_segment, self.seam_b_segment_end)
        };
        let current_start = CurveParam { segment: start_segment as u32, t: start_t };
        let current_end = CurveParam {
            segment: end_exclusive.saturating_sub(1) as u32, t: end_t,
        };
        let segment_count = curve.segment_count();
        let new_param = if is_start {
            let maximum = current_end.segment as f64 + current_end.t - 1.0e-4;
            param = param_at_curve_position(param.segment as f64 + param.t, segment_count, maximum);
            if param.segment as f64 + param.t >= current_end.segment as f64 + current_end.t {
                return;
            }
            param
        } else {
            let minimum = current_start.segment as f64 + current_start.t + 1.0e-4;
            param = param_at_curve_position(param.segment as f64 + param.t, segment_count, f64::INFINITY);
            if param.segment as f64 + param.t <= minimum {
                param = param_at_curve_position(minimum, segment_count, f64::INFINITY);
            }
            param
        };

        if is_a {
            if is_start {
                self.seam_a_segment = new_param.segment as usize;
                self.seam_a_start_t = new_param.t;
            } else {
                self.seam_a_segment_end = new_param.segment as usize + 1;
                self.seam_a_end_t = new_param.t;
            }
            self.seam_a_segment_end = self.seam_a_segment_end.clamp(self.seam_a_segment + 1, segment_count);
        } else {
            if is_start {
                self.seam_b_segment = new_param.segment as usize;
                self.seam_b_start_t = new_param.t;
            } else {
                self.seam_b_segment_end = new_param.segment as usize + 1;
                self.seam_b_end_t = new_param.t;
            }
            self.seam_b_segment_end = self.seam_b_segment_end.clamp(self.seam_b_segment + 1, segment_count);
        }
    }

    fn commit_seam_handle_drag(&mut self) {
        let Some(before) = self.seam_drag_before.take() else { return; };
        let Some(id) = self.selected_seam_id else { return; };
        let (sides_a, sides_b) = match self.current_seam_sides_with_draft() {
            Ok(sides) => sides,
            Err(error) => { self.pattern_message = error; return; }
        };
        let mut staged = self.pattern_project.clone();
        let Some(seam) = staged.seams.iter_mut().find(|seam| seam.id == id) else { return; };
        seam.sides_a = sides_a;
        seam.sides_b = sides_b;
        if let Err(error) = staged.validate() {
            self.pattern_message = format!("A extremidade não foi aplicada: {error}");
            return;
        }
        self.pattern_project = staged;
        self.remember_pattern_change(before);
        self.pattern_message = format!("Extremidades da costura '{id}' atualizadas.");
    }

    fn add_seam_from_controls(&mut self) {
        if (self.seam_pick_mode || self.seam_canvas_pick_used)
            && (self.seam_sides_a.is_empty() || self.seam_sides_b.is_empty())
        {
            self.pattern_message = "A costura precisa de pelo menos uma aresta no lado A e uma no lado B. Selecione as duas cadeias no canvas.".into();
            return;
        }
        let (sides_a, sides_b) = match self.current_seam_sides_with_draft() {
            Ok(sides) => sides,
            Err(error) => { self.pattern_message = error; return; }
        };
        let next_id = self.pattern_project.seams.iter().map(|seam| seam.id).max().unwrap_or(0) + 1;
        let name = if self.seam_name.trim().is_empty() {
            format!("Costura {next_id}")
        } else { self.seam_name.trim().to_owned() };
        let seam = SeamDefinition {
            id: next_id,
            name: name.clone(),
            sides_a,
            sides_b,
            ease_meters: self.seam_ease_m,
        };
        let before = self.editor_snapshot();
        let mut staged = self.pattern_project.clone();
        staged.seams.push(seam);
        if let Err(error) = staged.validate() {
            self.pattern_message = error;
            return;
        }
        self.pattern_project = staged;
        self.selected_seam_id = Some(next_id);
        self.seam_name = name;
        self.seam_pick_mode = false;
        self.seam_pick_pair_mode = false;
        self.seam_canvas_pick_used = false;
        self.pattern_message = "Costura adicionada. A/B permanecem visíveis; as correspondências são distribuídas pelo arco acumulado e pelos piques.".into();
        self.remember_pattern_change(before);
    }

    fn update_selected_seam_from_controls(&mut self) {
        let Some(id) = self.selected_seam_id else {
            self.add_seam_from_controls();
            return;
        };
        let (sides_a, sides_b) = match self.current_seam_sides_with_draft() {
            Ok(sides) => sides,
            Err(error) => { self.pattern_message = error; return; }
        };
        let before = self.editor_snapshot();
        let mut staged = self.pattern_project.clone();
        let Some(seam) = staged.seams.iter_mut().find(|seam| seam.id == id) else {
            self.pattern_message = "A costura selecionada não existe mais.".into();
            self.selected_seam_id = None;
            return;
        };
        seam.name = if self.seam_name.trim().is_empty() {
            format!("Costura {id}")
        } else { self.seam_name.trim().to_owned() };
        seam.sides_a = sides_a;
        seam.sides_b = sides_b;
        seam.ease_meters = self.seam_ease_m;
        if let Err(error) = staged.validate() {
            self.pattern_message = error;
            return;
        }
        self.pattern_project = staged;
        self.seam_pick_mode = false;
        self.seam_pick_pair_mode = false;
        self.seam_canvas_pick_used = false;
        self.pattern_message = format!("Costura '{id}' atualizada.");
        self.remember_pattern_change(before);
    }

    fn apply_selected_auto_seams(&mut self) {
        let chosen = self.auto_seam_suggestions.iter().zip(self.auto_seam_selected.iter())
            .filter_map(|(suggestion, selected)| (*selected).then_some(suggestion.clone()))
            .collect::<Vec<_>>();
        if chosen.is_empty() {
            self.pattern_message = "Marque pelo menos uma sugestão antes de aplicar.".into();
            return;
        }

        let before = self.editor_snapshot();
        let mut staged = self.pattern_project.clone();
        let mut next_id = staged.seams.iter().map(|seam| seam.id).max().unwrap_or(0) + 1;
        for suggestion in &chosen {
            staged.seams.push(SeamDefinition {
                id: next_id,
                name: format!("Sugestão automática {next_id}"),
                sides_a: vec![suggestion.a.clone()],
                sides_b: vec![suggestion.b.clone()],
                ease_meters: 0.0,
            });
            next_id += 1;
        }
        if let Err(error) = staged.validate() {
            self.pattern_message = format!("Lote de sugestões não aplicado: {error}");
            return;
        }

        self.pattern_project = staged;
        self.selected_seam_id = Some(next_id - 1);
        if let Some(seam) = self.pattern_project.seams.iter().find(|seam| seam.id == next_id - 1).cloned() {
            self.seam_sides_a = seam.sides_a;
            self.seam_sides_b = seam.sides_b;
            self.seam_a_active_path = 0;
            self.seam_b_active_path = 0;
            if let Some(edge) = self.seam_sides_a.first().cloned() { self.load_seam_edge_to_controls(true, &edge); }
            if let Some(edge) = self.seam_sides_b.first().cloned() { self.load_seam_edge_to_controls(false, &edge); }
            self.seam_name = seam.name;
        }
        self.auto_seam_suggestions.clear();
        self.auto_seam_selected.clear();
        self.hovered_auto_suggestion = None;
        self.pattern_message = format!(
            "{} costura(s) sugerida(s) aplicada(s) como um único lote de undo.",
            chosen.len()
        );
        self.remember_pattern_change(before);
    }

    fn remove_selected_seam(&mut self) {
        let Some(id) = self.selected_seam_id else { return; };
        let before = self.editor_snapshot();
        let previous_len = self.pattern_project.seams.len();
        self.pattern_project.seams.retain(|seam| seam.id != id);
        if self.pattern_project.seams.len() == previous_len {
            self.pattern_message = "A costura selecionada não foi encontrada.".into();
            self.selected_seam_id = None;
            return;
        }
        self.selected_seam_id = None;
        self.seam_name.clear();
        self.pattern_message = format!("Costura '{id}' removida do projeto.");
        self.remember_pattern_change(before);
    }

    fn add_notch_from_controls(&mut self) {
        if self.pattern_project.pieces.is_empty() { return; }
        let index = self.selected_piece.min(self.pattern_project.pieces.len() - 1);
        let segment_count = self.pattern_project.pieces[index].outline.segment_count();
        if segment_count == 0 { return; }
        let before = self.editor_snapshot();
        let piece = &mut self.pattern_project.pieces[index];
        let id = piece.notches.iter().map(|notch| notch.id).max().unwrap_or(0) + 1;
        piece.notches.push(Notch {
            id,
            segment_index: self.notch_segment.min(segment_count - 1) as u32,
            t: self.notch_t.clamp(0.01, 0.99),
            depth_meters: self.notch_depth_m.clamp(0.001, 0.10),
            width_meters: self.notch_width_m.clamp(0.001, 0.10),
        });
        self.selected_notch_index = piece.notches.len() - 1;
        self.pattern_message = format!("Pique {id} adicionado ao molde '{}'.", piece.name);
        self.remember_pattern_change(before);
    }

    fn update_notch_from_controls(&mut self) {
        if self.pattern_project.pieces.is_empty() { return; }
        let index = self.selected_piece.min(self.pattern_project.pieces.len() - 1);
        let segment_count = self.pattern_project.pieces[index].outline.segment_count();
        let before = self.editor_snapshot();
        let piece = &mut self.pattern_project.pieces[index];
        let Some(notch) = piece.notches.get_mut(self.selected_notch_index) else { return; };
        notch.segment_index = self.notch_segment.min(segment_count.saturating_sub(1)) as u32;
        notch.t = self.notch_t.clamp(0.01, 0.99);
        notch.depth_meters = self.notch_depth_m.clamp(0.001, 0.10);
        notch.width_meters = self.notch_width_m.clamp(0.001, 0.10);
        self.pattern_message = format!("Pique {} atualizado.", notch.id);
        self.remember_pattern_change(before);
    }

    fn remove_selected_notch(&mut self) {
        if self.pattern_project.pieces.is_empty() { return; }
        let index = self.selected_piece.min(self.pattern_project.pieces.len() - 1);
        if self.selected_notch_index >= self.pattern_project.pieces[index].notches.len() { return; }
        let before = self.editor_snapshot();
        let notch = self.pattern_project.pieces[index].notches.remove(self.selected_notch_index);
        self.selected_notch_index = self.selected_notch_index.min(
            self.pattern_project.pieces[index].notches.len().saturating_sub(1)
        );
        self.pattern_message = format!("Pique {} removido.", notch.id);
        self.remember_pattern_change(before);
    }

    fn pattern_viewport(&mut self, ui: &mut egui::Ui) {
        let rect = ui.available_rect_before_wrap();
        if self.pattern_fit_pending {
            self.fit_pattern_view(rect);
            self.pattern_fit_pending = false;
        }
        let response = ui.allocate_rect(rect, egui::Sense::click_and_drag());
        let painter = ui.painter_at(rect);
        if response.hovered() {
            let requested = ui.input(|i| {
                if i.modifiers.command || i.modifiers.alt { return None; }
                if i.key_pressed(egui::Key::W) { Some(PieceTransformMode::Move) }
                else if i.key_pressed(egui::Key::E) { Some(PieceTransformMode::Rotate) }
                else if i.key_pressed(egui::Key::R) { Some(PieceTransformMode::Scale) }
                else { None }
            });
            if let Some(mode) = requested { self.piece_transform_mode = mode; self.pattern_tool = PatternTool::Select; }
        }
        let center = rect.center();

        // O zoom é medido em pixels por metro; a translação fica em pixels da viewport.
        let origin = center + self.pattern_pan;
        let scale = self.pattern_zoom;
        let to_screen = |p: PatternPoint2| {
            egui::pos2(origin.x + p.x as f32 * scale, origin.y - p.y as f32 * scale)
        };
        let to_world = |p: egui::Pos2| {
            PatternPoint2::new(
                (p.x - origin.x) as f64 / scale as f64,
                (origin.y - p.y) as f64 / scale as f64,
            )
        };

        painter.rect_filled(rect, 0.0, egui::Color32::from_gray(23));
        let grid_step = (scale * 0.1).max(4.0);
        let first_x = origin.x + ((rect.left() - origin.x) / grid_step).ceil() * grid_step;
        let first_y = origin.y + ((rect.top() - origin.y) / grid_step).ceil() * grid_step;
        let mut x = first_x;
        while x <= rect.right() {
            painter.line_segment(
                [egui::pos2(x, rect.top()), egui::pos2(x, rect.bottom())],
                egui::Stroke::new(1.0, egui::Color32::from_gray(39)),
            );
            x += grid_step;
        }
        let mut y = first_y;
        while y <= rect.bottom() {
            painter.line_segment(
                [egui::pos2(rect.left(), y), egui::pos2(rect.right(), y)],
                egui::Stroke::new(1.0, egui::Color32::from_gray(39)),
            );
            y += grid_step;
        }
        painter.line_segment(
            [egui::pos2(rect.left(), origin.y), egui::pos2(rect.right(), origin.y)],
            egui::Stroke::new(1.0, egui::Color32::from_gray(88)),
        );
        painter.line_segment(
            [egui::pos2(origin.x, rect.top()), egui::pos2(origin.x, rect.bottom())],
            egui::Stroke::new(1.0, egui::Color32::from_gray(88)),
        );

        // Usa o PNG original como textura raster, sem vetorizar, recortar ou modificar seus pixels.
        // A altura é 1,72 m; a largura deriva do aspect ratio original (585 × 632) para preservar a proporção.
        if self.show_avatar_guide {
            if self.avatar_guide_texture.is_none() {
                let decoded = image::load_from_memory(include_bytes!("../../../assets/t_pose_reference.png"))
                    .expect("assets/t_pose_reference.png deve ser um PNG válido")
                    .to_rgba8();
                let size = [decoded.width() as usize, decoded.height() as usize];
                let pixels = egui::ColorImage::from_rgba_unmultiplied(size, decoded.as_raw());
                self.avatar_guide_texture = Some(
                    ui.ctx().load_texture("t_pose_reference_original", pixels, egui::TextureOptions::LINEAR)
                );
            }

            if let Some(texture) = self.avatar_guide_texture.as_ref() {
                let pixel_size = texture.size();
                let reference_height_m = 1.72_f64;
                let reference_width_m = reference_height_m * pixel_size[0] as f64 / pixel_size[1].max(1) as f64;
                let top_left = to_screen(PatternPoint2::new(-reference_width_m * 0.5, 0.92));
                let bottom_right = to_screen(PatternPoint2::new(reference_width_m * 0.5, -0.80));
                let image_rect = egui::Rect::from_min_max(top_left, bottom_right);

                // O fundo translúcido é separado do PNG: ele só melhora o contraste sobre a grade.
                // A textura é desenhada com tint branca, mantendo as cores e o alpha originais.
                let backdrop_alpha = (self.avatar_guide_backdrop.clamp(0.0, 0.6) * 255.0).round() as u8;
                painter.rect_filled(image_rect, 6.0, egui::Color32::from_white_alpha(backdrop_alpha));
                painter.image(
                    texture.id(),
                    image_rect,
                    egui::Rect::from_min_max(egui::Pos2::ZERO, egui::pos2(1.0, 1.0)),
                    egui::Color32::WHITE,
                );
                painter.text(
                    to_screen(PatternPoint2::new(0.0, 0.99)),
                    egui::Align2::CENTER_CENTER,
                    "REFERÊNCIA T-POSE · 1,72 m",
                    egui::FontId::proportional(10.0),
                    egui::Color32::from_rgb(155, 199, 222),
                );
            }
        }

        let symmetry_screen_x = origin.x + self.pattern_symmetry_axis_x as f32 * scale;
        painter.line_segment(
            [egui::pos2(symmetry_screen_x, rect.top()), egui::pos2(symmetry_screen_x, rect.bottom())],
            egui::Stroke::new(1.0, egui::Color32::from_rgb(154, 111, 72)),
        );
        painter.text(
            egui::pos2(symmetry_screen_x + 4.0, rect.top() + 34.0),
            egui::Align2::LEFT_TOP, "EIXO SIMETRIA",
            egui::FontId::proportional(10.0),
            egui::Color32::from_rgb(198, 153, 108),
        );
        if self.pattern_tool == PatternTool::Select && self.piece_transform_mode != PieceTransformMode::Move {
            if let Some(piece) = self.pattern_project.pieces.get(self.selected_piece) {
                let pivot = self.piece_transform_drag.map(|drag| drag.pivot).unwrap_or_else(|| piece_pivot(piece));
                let screen = to_screen(pivot);
                let color = egui::Color32::from_rgb(255, 214, 102);
                painter.circle_stroke(screen, 7.0, egui::Stroke::new(1.5, color));
                painter.line_segment([screen - egui::vec2(12.0, 0.0), screen + egui::vec2(12.0, 0.0)], egui::Stroke::new(1.0, color));
                painter.line_segment([screen - egui::vec2(0.0, 12.0), screen + egui::vec2(0.0, 12.0)], egui::Stroke::new(1.0, color));
            }
        }
        let shift_down = ui.input(|input| input.modifiers.shift);
        let tool_label = if self.direct_sewing_dragging || self.direct_sewing_start.is_some() {
            "COSTURA DIRETA · arraste até outra borda · Esc cancela"
        } else if shift_down {
            "COSTURA DIRETA · Shift + LMB · clique A/B ou arraste de A até B"
        } else if self.seam_pick_mode && self.seam_pick_pair_mode {
            "COSTURA · selecione A e B no contorno"
        } else if self.pattern_tool == PatternTool::Draw {
            "DESENHAR · clique para adicionar pontos"
        } else if self.piece_transform_mode == PieceTransformMode::Rotate {
            "GIRAR (E) · arraste ao redor do centro do molde · somente no plano 2D · W mover · R escalar"
        } else if self.piece_transform_mode == PieceTransformMode::Scale {
            "ESCALAR (R) · arraste para aproximar/afastar do centro · somente no plano 2D · W mover · E girar"
        } else {
            "EDITAR (W) · arraste vértices/alças · botão direito/meio move a vista · E girar · R escalar"
        };
        painter.text(
            rect.left_top() + egui::vec2(12.0, 12.0),
            egui::Align2::LEFT_TOP,
            tool_label,
            egui::FontId::proportional(14.0),
            egui::Color32::from_gray(205),
        );

        for (piece_index, piece) in self.pattern_project.pieces.iter().enumerate() {
            let sampled = sample_curve_with_ranges(&piece.outline, 12);
            let points = &sampled.points;
            if points.len() < 3 { continue; }
            let selected_piece = self.selected_piece == piece_index
                || self.selected_target.map(|t| t.piece()) == Some(piece_index);
            let outline_color = if selected_piece {
                egui::Color32::from_rgb(108, 205, 255)
            } else {
                egui::Color32::from_rgb(91, 183, 245)
            };

            // Contorno de corte em azul; margem de costura independente em verde.
            if piece.seam_allowance_meters > 0.0 {
                if let Ok(margin) = offset_polygon(points, piece.seam_allowance_meters) {
                    for i in 0..margin.len() {
                        painter.line_segment(
                            [to_screen(margin[i]), to_screen(margin[(i + 1) % margin.len()])],
                            egui::Stroke::new(1.5, egui::Color32::from_rgb(104, 196, 131)),
                        );
                    }
                }
            }

            if self.pattern_show_mesh {
                if let Ok(mesh) = triangulate_outline(points, self.pattern_spacing) {
                    for triangle in &mesh.triangles {
                        let a = to_screen(mesh.vertices[triangle[0] as usize]);
                        let b = to_screen(mesh.vertices[triangle[1] as usize]);
                        let c = to_screen(mesh.vertices[triangle[2] as usize]);
                        let stroke = egui::Stroke::new(0.7, egui::Color32::from_rgb(63, 91, 105));
                        painter.line_segment([a, b], stroke);
                        painter.line_segment([b, c], stroke);
                        painter.line_segment([c, a], stroke);
                    }
                }
            }

            for i in 0..points.len() {
                painter.line_segment(
                    [to_screen(points[i]), to_screen(points[(i + 1) % points.len()])],
                    egui::Stroke::new(2.0, outline_color),
                );
            }

            // Piques do molde: traço transversal à borda e ponto dourado na posição paramétrica.
            for notch in &piece.notches {
                let segment_index = notch.segment_index as usize;
                if segment_index >= piece.outline.segments.len() { continue; }
                let t = notch.t.clamp(0.0, 1.0);
                let Some(point) = curve_segment_point(&piece.outline, segment_index, t) else { continue; };
                let before = curve_segment_point(&piece.outline, segment_index, (t - 0.005).max(0.0)).unwrap_or(point);
                let after = curve_segment_point(&piece.outline, segment_index, (t + 0.005).min(1.0)).unwrap_or(point);
                let dx = after.x - before.x;
                let dy = after.y - before.y;
                let length = (dx * dx + dy * dy).sqrt().max(1e-9);
                let normal = PatternPoint2::new(-dy / length, dx / length);
                let half_width = (notch.width_meters * 0.5).clamp(0.004, 0.035);
                let a = PatternPoint2::new(point.x - normal.x * half_width, point.y - normal.y * half_width);
                let b = PatternPoint2::new(point.x + normal.x * half_width, point.y + normal.y * half_width);
                let color = egui::Color32::from_rgb(255, 198, 72);
                painter.line_segment([to_screen(a), to_screen(b)], egui::Stroke::new(2.4, color));
                painter.circle_filled(to_screen(point), 3.0, color);
                if selected_piece {
                    painter.text(to_screen(point), egui::Align2::LEFT_BOTTOM,
                        format!("P{}", notch.id), egui::FontId::proportional(9.0), color);
                }
            }

            // Linhas internas, dobras, pregas e linha de fio têm estilos distintos do contorno.
            for line in &piece.internal_lines {
                let detail = sample_curve(line, 12);
                for pair in detail.windows(2) {
                    painter.line_segment(
                        [to_screen(pair[0]), to_screen(pair[1])],
                        egui::Stroke::new(1.7, egui::Color32::from_rgb(245, 178, 72)),
                    );
                }
            }
            for fold in &piece.fold_lines {
                let detail = sample_curve(&fold.curve, 12);
                let color = match fold.kind {
                    FoldKind::Valley => egui::Color32::from_rgb(185, 128, 255),
                    FoldKind::Mountain => egui::Color32::from_rgb(90, 220, 210),
                    FoldKind::Decorative => egui::Color32::from_rgb(230, 230, 230),
                };
                for pair in detail.windows(2) {
                    painter.line_segment(
                        [to_screen(pair[0]), to_screen(pair[1])],
                        egui::Stroke::new(1.8, color),
                    );
                }
            }
            for pleat in &piece.pleats {
                let detail = sample_curve(&pleat.axis, 12);
                for pair in detail.windows(2) {
                    painter.line_segment(
                        [to_screen(pair[0]), to_screen(pair[1])],
                        egui::Stroke::new(2.2, egui::Color32::from_rgb(255, 120, 170)),
                    );
                }
            }
            if let Some(grain) = &piece.grain_line {
                painter.line_segment(
                    [to_screen(grain.start), to_screen(grain.end)],
                    egui::Stroke::new(1.6, egui::Color32::from_rgb(120, 220, 135)),
                );
                let center = PatternPoint2::new(
                    (grain.start.x + grain.end.x) * 0.5,
                    (grain.start.y + grain.end.y) * 0.5,
                );
                painter.text(
                    to_screen(center), egui::Align2::CENTER_CENTER, "FIO",
                    egui::FontId::proportional(10.0),
                    egui::Color32::from_rgb(120, 220, 135),
                );
            }
            if let Some(PatternEditTarget::Edge { piece: selected_index, segment, .. }) = self.selected_target {
                if selected_index == piece_index {
                    if let Some(&(start, end)) = sampled.segment_vertex_ranges.get(segment) {
                        let sample_count = if end >= start { end - start } else { points.len() - start };
                        for step in 0..sample_count {
                            let ia = (start + step) % points.len();
                            let ib = (ia + 1) % points.len();
                            painter.line_segment(
                                [to_screen(points[ia]), to_screen(points[ib])],
                                egui::Stroke::new(4.5, egui::Color32::from_rgb(255, 196, 76)),
                            );
                        }
                    }
                }
            }

            let centroid = points.iter().fold(PatternPoint2::new(0.0, 0.0), |sum, point| {
                PatternPoint2::new(
                    sum.x + point.x / points.len() as f64,
                    sum.y + point.y / points.len() as f64,
                )
            });
            painter.text(
                to_screen(centroid),
                egui::Align2::CENTER_CENTER,
                &piece.name,
                egui::FontId::proportional(13.0),
                egui::Color32::WHITE,
            );

            if self.pattern_tool == PatternTool::Select {
                // Alças Bézier e linhas de controlo são exibidas apenas para a peça selecionada.
                if selected_piece {
                    let mut previous = piece.outline.start;
                    for (segment_index, segment) in piece.outline.segments.iter().enumerate() {
                        if let CurveSegment::CubicBezier { control1, control2, .. } = segment {
                            painter.line_segment(
                                [to_screen(previous), to_screen(*control1)],
                                egui::Stroke::new(1.0, egui::Color32::from_rgb(175, 119, 225)),
                            );
                            painter.line_segment(
                                [to_screen(segment.endpoint()), to_screen(*control2)],
                                egui::Stroke::new(1.0, egui::Color32::from_rgb(175, 119, 225)),
                            );
                            let c1 = to_screen(*control1);
                            let c2 = to_screen(*control2);
                            painter.circle_filled(c1, 4.0, egui::Color32::from_rgb(201, 151, 247));
                            painter.circle_filled(c2, 4.0, egui::Color32::from_rgb(201, 151, 247));
                            if self.selected_target == Some(PatternEditTarget::Control1 { piece: piece_index, segment: segment_index }) {
                                painter.circle_stroke(c1, 7.0, egui::Stroke::new(2.0, egui::Color32::WHITE));
                            }
                            if self.selected_target == Some(PatternEditTarget::Control2 { piece: piece_index, segment: segment_index }) {
                                painter.circle_stroke(c2, 7.0, egui::Stroke::new(2.0, egui::Color32::WHITE));
                            }
                        }
                        previous = segment.endpoint();
                    }
                }
                let anchors = outline_anchors(&piece.outline);
                for (anchor_index, anchor) in anchors.iter().enumerate() {
                    let screen = to_screen(*anchor);
                    let selected = self.selected_target == Some(PatternEditTarget::Anchor {
                        piece: piece_index,
                        index: anchor_index,
                    });
                    let pinned = piece.assembly.pinned_anchors.contains(&(anchor_index as u32));
                    let color = if pinned {
                        egui::Color32::from_rgb(255, 80, 95)
                    } else if selected {
                        egui::Color32::WHITE
                    } else {
                        egui::Color32::from_rgb(255, 190, 75)
                    };
                    painter.circle_filled(screen, if selected || pinned { 6.0 } else { 4.0 }, color);
                    if selected || pinned {
                        painter.circle_stroke(
                            screen, 9.0,
                            egui::Stroke::new(1.5, if pinned { color } else { egui::Color32::from_rgb(255, 190, 75) }),
                        );
                    }
                }
            }
        }

        // Desenha cada caminho sem criar segmentos artificiais entre bordas desconectadas.
        // Ao passar o cursor numa costura selecionada, realça os dois lados e liga piques pareados.
        let hovered_pointer = ui.input(|input| input.pointer.hover_pos());
        for seam in &self.pattern_project.seams {
            let selected = self.selected_seam_id == Some(seam.id);
            let (draw_a, draw_b) = if selected {
                self.current_seam_sides_with_draft()
                    .unwrap_or((seam.sides_a.clone(), seam.sides_b.clone()))
            } else {
                (seam.sides_a.clone(), seam.sides_b.clone())
            };
            let la = draw_a.iter().map(|edge| edge_length_meters(&self.pattern_project, edge)).sum::<f64>();
            let lb = draw_b.iter().map(|edge| edge_length_meters(&self.pattern_project, edge)).sum::<f64>();
            let mismatch = (la - lb - seam.ease_meters).abs();
            let screen_a = draw_a.iter().map(|edge| edge_path_points(&self.pattern_project, edge, 24)
                .into_iter().map(to_screen).collect::<Vec<_>>()).collect::<Vec<_>>();
            let screen_b = draw_b.iter().map(|edge| edge_path_points(&self.pattern_project, edge, 24)
                .into_iter().map(to_screen).collect::<Vec<_>>()).collect::<Vec<_>>();
            let hovering_selected_seam = selected && response.hovered() && hovered_pointer.map(|pointer| {
                screen_a.iter().chain(screen_b.iter())
                    .map(|path| distance_to_screen_path(pointer, path))
                    .fold(f32::INFINITY, f32::min) <= 9.0
            }).unwrap_or(false);
            let color = if mismatch > 0.02 {
                egui::Color32::from_rgb(255, 92, 92)
            } else if selected || hovering_selected_seam {
                egui::Color32::from_rgb(255, 220, 110)
            } else {
                egui::Color32::from_rgb(64, 221, 206)
            };
            for side in [&screen_a, &screen_b] {
                for path in side {
                    for pair in path.windows(2) {
                        painter.line_segment(
                            [pair[0], pair[1]],
                            egui::Stroke::new(if selected || hovering_selected_seam { 4.0 } else { 2.6 }, color),
                        );
                    }
                    if hovering_selected_seam { draw_path_direction_arrow(&painter, path, color); }
                }
            }
            for progress in [0.0_f32, 0.5, 1.0] {
                if let (Some(a), Some(b)) = (
                    point_on_edge_chain_at_progress(&self.pattern_project, &draw_a, progress),
                    point_on_edge_chain_at_progress(&self.pattern_project, &draw_b, progress),
                ) {
                    let alpha = if selected || hovering_selected_seam { 230 } else { 150 };
                    let dash_color = egui::Color32::from_rgba_unmultiplied(color.r(), color.g(), color.b(), alpha);
                    painter.extend(egui::Shape::dashed_line(&[to_screen(a), to_screen(b)], egui::Stroke::new(1.4, dash_color), 6.0, 4.0));
                }
            }
            if hovering_selected_seam {
                for (progress_a, progress_b) in paired_chain_notch_markers(&self.pattern_project, &draw_a, &draw_b) {
                    if let (Some(a), Some(b)) = (
                        point_on_edge_chain_at_progress(&self.pattern_project, &draw_a, progress_a),
                        point_on_edge_chain_at_progress(&self.pattern_project, &draw_b, progress_b),
                    ) {
                        painter.line_segment(
                            [to_screen(a), to_screen(b)],
                            egui::Stroke::new(1.1, egui::Color32::from_rgba_unmultiplied(255, 225, 130, 180)),
                        );
                    }
                }
            }
            if selected {
                let first_a = draw_a.first().and_then(|edge| edge_path_points(&self.pattern_project, edge, 12).get(6).copied());
                let first_b = draw_b.first().and_then(|edge| edge_path_points(&self.pattern_project, edge, 12).get(6).copied());
                if let (Some(a), Some(b)) = (first_a, first_b) {
                    painter.text(to_screen(a), egui::Align2::CENTER_BOTTOM,
                        format!("C{} · {:.1} cm", seam.id, mismatch * 100.0),
                        egui::FontId::proportional(10.0), color);
                    painter.text(to_screen(b), egui::Align2::CENTER_TOP,
                        format!("C{}", seam.id), egui::FontId::proportional(10.0), color);
                }
            }
        }

        // Prévia sem efeitos colaterais da sugestão sob o cursor na lista de Auto Sewing.
        if let Some(index) = self.hovered_auto_suggestion {
            if let Some(suggestion) = self.auto_seam_suggestions.get(index) {
                let color = egui::Color32::from_rgb(210, 115, 255);
                let path_a = edge_path_points(&self.pattern_project, &suggestion.a, 32);
                let path_b = edge_path_points(&self.pattern_project, &suggestion.b, 32);
                for path in [&path_a, &path_b] {
                    for pair in path.windows(2) {
                        painter.line_segment(
                            [to_screen(pair[0]), to_screen(pair[1])],
                            egui::Stroke::new(4.0, color),
                        );
                    }
                }
                if let (Some(a), Some(b)) = (
                    path_a.get(path_a.len() / 2),
                    path_b.get(path_b.len() / 2),
                ) {
                    painter.line_segment(
                        [to_screen(*a), to_screen(*b)],
                        egui::Stroke::new(1.8, color),
                    );
                    painter.text(
                        to_screen(*a), egui::Align2::CENTER_BOTTOM,
                        format!("Sugestão {:.0}%", suggestion.score * 100.0),
                        egui::FontId::proportional(11.0), color,
                    );
                }
            }
        }

        // Preview do lado atual: os extremos do caminho ativo podem ser arrastados livremente.
        if self.seam_free_mode || self.seam_pick_mode || self.seam_canvas_pick_used {
            if let Ok((draft_a, draft_b)) = self.seam_edges_from_controls() {
                let paths_a = self.seam_sides_a.clone();
                let paths_b = self.seam_sides_b.clone();
                let active_a = self.seam_a_active_path;
                let active_b = self.seam_b_active_path;
                let mut sides = [paths_a, paths_b];
                let drafts = [draft_a, draft_b];
                let active = [active_a, active_b];
                for side_index in 0..2 {
                    let is_a = side_index == 0;
                    let list = &mut sides[side_index];
                    let index = active[side_index];
                    if list.is_empty() {
                        if self.seam_canvas_pick_used {
                            continue;
                        }
                        list.push(drafts[side_index].clone());
                    } else if index < list.len() {
                        list[index] = drafts[side_index].clone();
                    } else if !self.seam_canvas_pick_used {
                        list.push(drafts[side_index].clone());
                    }
                    let color = if is_a { egui::Color32::from_rgb(255, 190, 74) } else { egui::Color32::from_rgb(94, 224, 207) };
                    for (path_index, edge) in list.iter().enumerate() {
                        let path = edge_path_points(&self.pattern_project, edge, 24);
                        let screen_path = path.iter().copied().map(to_screen).collect::<Vec<_>>();
                        for pair in screen_path.windows(2) {
                            painter.line_segment([pair[0], pair[1]], egui::Stroke::new(3.0, color));
                        }
                        // Direction arrows make a reversed/twisted edge visible before simulation.
                        draw_path_direction_arrow(&painter, &screen_path, color);
                        if path_index != index { continue; }
                        if let Some(curve) = edge_curve(&self.pattern_project, edge) {
                            for (is_start, param) in [(true, edge.start), (false, edge.end)] {
                                if let Some(point) = curve_segment_point(curve, param.segment as usize, param.t) {
                                    let screen = to_screen(point);
                                    painter.circle_filled(screen, 6.0, color);
                                    painter.circle_stroke(screen, 9.0, egui::Stroke::new(1.5, egui::Color32::WHITE));
                                    painter.text(screen + egui::vec2(8.0, if is_start { -8.0 } else { 8.0 }),
                                        egui::Align2::LEFT_CENTER,
                                        format!("{}{}", if is_a { "A" } else { "B" }, if is_start { " início" } else { " fim" }),
                                        egui::FontId::proportional(9.0), color);
                                }
                            }
                        }
                    }
                }

                // Correspondência visual aproximada por progresso do arco acumulado.
                if self.seam_canvas_pick_used && !sides[0].is_empty() && !sides[1].is_empty() {
                    for step in 0..=4 {
                        let progress = step as f32 / 4.0;
                        if let (Some(point_a), Some(point_b)) = (
                            point_on_edge_chain_at_progress(&self.pattern_project, &sides[0], progress),
                            point_on_edge_chain_at_progress(&self.pattern_project, &sides[1], progress),
                        ) {
                            let screen_a = to_screen(point_a);
                            let screen_b = to_screen(point_b);
                            painter.line_segment(
                                [screen_a, screen_b],
                                egui::Stroke::new(1.0, egui::Color32::from_rgba_unmultiplied(220, 230, 240, 125)),
                            );
                            painter.circle_filled(screen_a, 2.5, egui::Color32::from_rgb(255, 235, 185));
                            painter.circle_filled(screen_b, 2.5, egui::Color32::from_rgb(170, 250, 240));
                        }
                    }
                    // Piquer correspondences are stronger anchors than uniform arc progress.
                    for (progress_a, progress_b) in paired_chain_notch_markers(
                        &self.pattern_project, &sides[0], &sides[1],
                    ) {
                        if let (Some(point_a), Some(point_b)) = (
                            point_on_edge_chain_at_progress(&self.pattern_project, &sides[0], progress_a),
                            point_on_edge_chain_at_progress(&self.pattern_project, &sides[1], progress_b),
                        ) {
                            let screen_a = to_screen(point_a);
                            let screen_b = to_screen(point_b);
                            painter.line_segment(
                                [screen_a, screen_b],
                                egui::Stroke::new(2.0, egui::Color32::from_rgb(255, 145, 92)),
                            );
                            painter.circle_filled(screen_a, 4.0, egui::Color32::from_rgb(255, 145, 92));
                            painter.circle_filled(screen_b, 4.0, egui::Color32::from_rgb(255, 145, 92));
                        }
                    }
                }
            }
        }

        if let Some((piece_index, segment)) = self.direct_sewing_start {
            if let Some(piece) = self.pattern_project.pieces.get(piece_index) {
                let edge_a = EdgePath::from_segments(
                    piece.id, BoundaryCurve::Outline, segment as u32, (segment + 1) as u32, false,
                );
                let path_a = edge_path_points(&self.pattern_project, &edge_a, 16);
                for pair in path_a.windows(2) {
                    painter.line_segment(
                        [to_screen(pair[0]), to_screen(pair[1])],
                        egui::Stroke::new(5.0, egui::Color32::from_rgb(65, 210, 245)),
                    );
                }
                if let Some(mid_a) = path_a.get(path_a.len() / 2) {
                    painter.text(
                        to_screen(*mid_a), egui::Align2::CENTER_BOTTOM, "A",
                        egui::FontId::proportional(13.0),
                        egui::Color32::from_rgb(95, 225, 255),
                    );
                    if (shift_down || self.direct_sewing_dragging) && response.hovered() {
                        if let Some(pointer) = ui.input(|input| input.pointer.hover_pos()) {
                            painter.extend(egui::Shape::dashed_line(
                                &[to_screen(*mid_a), pointer],
                                egui::Stroke::new(2.0, egui::Color32::from_rgb(95, 225, 255)), 7.0, 5.0,
                            ));
                            if let Some((piece_b_index, segment_b)) =
                                self.hit_test_sewing_edge(pointer, center)
                            {
                                if piece_b_index != piece_index {
                                    if let Some(piece_b) = self.pattern_project.pieces.get(piece_b_index) {
                                        let edge_b = EdgePath::from_segments(
                                            piece_b.id, BoundaryCurve::Outline,
                                            segment_b as u32, (segment_b + 1) as u32, false,
                                        );
                                        let path_b = edge_path_points(&self.pattern_project, &edge_b, 16);
                                        for pair in path_b.windows(2) {
                                            painter.line_segment(
                                                [to_screen(pair[0]), to_screen(pair[1])],
                                                egui::Stroke::new(
                                                    5.0, egui::Color32::from_rgb(105, 235, 155),
                                                ),
                                            );
                                        }
                                        if let Some(mid_b) = path_b.get(path_b.len() / 2) {
                                            painter.text(
                                                to_screen(*mid_b), egui::Align2::CENTER_BOTTOM, "B",
                                                egui::FontId::proportional(13.0),
                                                egui::Color32::from_rgb(105, 235, 155),
                                            );
                                        }
                                    }
                                }
                            }
                        }
                    }
                }
            }
        }

        if self.draft_points.len() >= 2 {
            let polyline = self.draft_points.iter().copied().map(to_screen).collect::<Vec<_>>();
            painter.add(egui::Shape::line(
                polyline,
                egui::Stroke::new(2.0, egui::Color32::from_rgb(245, 180, 65)),
            ));
        }
        for point in &self.draft_points {
            painter.circle_filled(to_screen(*point), 4.5, egui::Color32::from_rgb(255, 194, 75));
        }

        // Botão direito/meio arrasta a viewport; a roda aproxima/afasta ao redor do cursor.
        if response.dragged_by(egui::PointerButton::Secondary)
            || response.dragged_by(egui::PointerButton::Middle)
        {
            self.pattern_pan += ui.input(|i| i.pointer.delta());
        }
        if response.hovered() {
            let (scroll, pointer) = ui.input(|i| (i.smooth_scroll_delta.y, i.pointer.hover_pos()));
            if scroll != 0.0 {
                let pointer = pointer.unwrap_or(rect.center());
                let before = to_world(pointer);
                self.pattern_zoom = (self.pattern_zoom * (scroll * 0.0015).exp()).clamp(80.0, 1800.0);
                self.pattern_pan.x = pointer.x - center.x - before.x as f32 * self.pattern_zoom;
                self.pattern_pan.y = pointer.y - center.y + before.y as f32 * self.pattern_zoom;
            }
        }

        let (primary_pressed, escape_pressed) = ui.input(|input| (
            input.pointer.button_pressed(egui::PointerButton::Primary),
            input.key_pressed(egui::Key::Escape),
        ));
        if escape_pressed && (
            self.direct_sewing_start.is_some()
                || self.direct_sewing_dragging
                || (self.seam_pick_mode && self.seam_pick_pair_mode)
        ) {
            self.direct_sewing_start = None;
            self.direct_sewing_preselected = None;
            self.direct_sewing_press_edge = None;
            self.direct_sewing_preview = None;
            self.direct_sewing_dragging = false;
            if self.seam_pick_mode && self.seam_pick_pair_mode {
                self.seam_pick_mode = false;
                self.seam_pick_pair_mode = false;
                self.seam_canvas_pick_used = false;
                self.seam_sides_a.clear();
                self.seam_sides_b.clear();
            }
            self.pattern_message = "Seleção de costura cancelada.".into();
        }

        // Em Costura livre M:N, Shift+clique continua adicionando segmentos à cadeia ativa.
        let direct_shortcut = shift_down && !(self.seam_pick_mode && !self.seam_pick_pair_mode);
        let direct_interaction = direct_shortcut || self.direct_sewing_dragging;
        if direct_interaction {
            if primary_pressed && direct_shortcut {
                self.direct_sewing_press_edge = response.interact_pointer_pos()
                    .and_then(|pointer| self.hit_test_sewing_edge(pointer, center));
                self.direct_sewing_preselected = if self.seam_pick_mode && self.seam_pick_pair_mode {
                    // Em uma costura nova, nunca reutilize uma borda selecionada antes de abrir a ferramenta.
                    None
                } else {
                    match self.selected_target {
                        Some(PatternEditTarget::Edge { piece, segment, .. }) => Some((piece, segment)),
                        _ => None,
                    }
                };
                self.direct_sewing_preview = None;
            }

            if response.dragged_by(egui::PointerButton::Primary)
                && (direct_shortcut || self.direct_sewing_press_edge.is_some())
            {
                if !self.direct_sewing_dragging {
                    self.direct_sewing_dragging = true;
                    if self.direct_sewing_start.is_none() {
                        self.direct_sewing_start = self.direct_sewing_press_edge;
                    }
                }
                self.direct_sewing_preview = response.interact_pointer_pos()
                    .and_then(|pointer| self.hit_test_sewing_edge(pointer, center));
            }

            if response.drag_stopped_by(egui::PointerButton::Primary) && self.direct_sewing_dragging {
                let start = self.direct_sewing_start;
                let end = response.interact_pointer_pos()
                    .and_then(|pointer| self.hit_test_sewing_edge(pointer, center));
                if let (Some(a), Some(b)) = (start, end) {
                    if a != b && self.create_direct_seam_from_edges(a, b) {
                        self.direct_sewing_start = None;
                    } else if a.0 == b.0 {
                        self.pattern_message = "A segunda borda precisa pertencer a outra peça.".into();
                    }
                } else if start.is_some() {
                    self.pattern_message = "Solte sobre uma borda válida de outra peça.".into();
                } else {
                    self.pattern_message = "Inicie o arraste sobre uma borda do contorno.".into();
                }
                self.direct_sewing_dragging = false;
                self.direct_sewing_preselected = None;
                self.direct_sewing_press_edge = None;
                self.direct_sewing_preview = None;
            }

            if response.clicked() && !self.direct_sewing_dragging {
                let edge = self.direct_sewing_press_edge.or_else(|| {
                    response.interact_pointer_pos()
                        .and_then(|pointer| self.hit_test_sewing_edge(pointer, center))
                });
                if let Some(edge) = edge {
                    if let Some(start) = self.direct_sewing_start {
                        if start != edge && self.create_direct_seam_from_edges(start, edge) {
                            self.direct_sewing_start = None;
                            self.direct_sewing_preview = None;
                        } else if start.0 == edge.0 {
                            self.pattern_message = "Escolha uma borda de outra peça.".into();
                        }
                    } else if let Some(start) = self.direct_sewing_preselected
                        .filter(|candidate| *candidate != edge)
                    {
                        if start.0 != edge.0 && self.create_direct_seam_from_edges(start, edge) {
                            self.direct_sewing_start = None;
                            self.direct_sewing_preview = None;
                        } else {
                            self.pattern_message = "A segunda borda precisa pertencer a outra peça.".into();
                        }
                    } else {
                        self.direct_sewing_start = Some(edge);
                        self.selected_target = Some(PatternEditTarget::Edge {
                            piece: edge.0, segment: edge.1, t: 0.5,
                        });
                        let piece_name = self.pattern_project.pieces.get(edge.0)
                            .map(|piece| piece.name.as_str()).unwrap_or("Peça");
                        self.pattern_message = format!(
                            "Borda A selecionada: {piece_name}, segmento {}. Selecione B.",
                            edge.1 + 1
                        );
                    }
                } else {
                    self.pattern_message = "Clique diretamente sobre uma aresta do contorno.".into();
                }
                self.direct_sewing_press_edge = None;
            }
        } else if self.pattern_tool == PatternTool::Select || self.seam_pick_mode {
            if response.drag_started_by(egui::PointerButton::Primary) {
                if let Some(pointer) = response.interact_pointer_pos() {
                    if let Some(handle) = self.hit_test_seam_handle(pointer, &to_screen) {
                        self.dragging_seam_handle = Some(handle);
                        self.seam_drag_before = Some(self.editor_snapshot());
                        self.dragging_target = None;
                        self.dragging_before = None;
                        self.drag_last_world = None;
                    } else {
                        let previous_target = self.selected_target;
                        self.selected_target = self.hit_test_pattern(pointer, center);
                        let transform_piece = if self.piece_transform_mode != PieceTransformMode::Move && !self.seam_pick_mode {
                            self.selected_target.or(previous_target).map(|target| target.piece())
                                .filter(|index| *index < self.pattern_project.pieces.len()
                                    && !self.pattern_project.pieces[*index].frozen)
                        } else { None };
                        if let Some(piece_index) = transform_piece {
                            let pivot = piece_pivot(&self.pattern_project.pieces[piece_index]);
                            self.selected_piece = piece_index;
                            self.piece_transform_drag = Some(PieceTransformDrag { piece_index, pivot, last_pointer: to_world(pointer) });
                            self.dragging_target = None;
                            self.dragging_before = None;
                            self.drag_last_world = None;
                        } else if let Some(target) = self.selected_target {
                            self.selected_piece = target.piece().min(self.pattern_project.pieces.len().saturating_sub(1));
                            let locked = self.pattern_project.pieces.get(target.piece())
                                .map(|piece| piece.frozen)
                                .unwrap_or(false);
                            self.dragging_target = if target.is_movable() && !locked { Some(target) } else { None };
                            self.dragging_before = self.dragging_target.map(|_| self.editor_snapshot());
                            self.drag_last_world = Some(to_world(pointer));
                        } else {
                            self.dragging_target = None;
                            self.dragging_before = None;
                            self.drag_last_world = None;
                        }
                    }
                }
            }
            if response.dragged_by(egui::PointerButton::Primary) {
                if let (Some((is_a, is_start)), Some(pointer)) = (self.dragging_seam_handle, response.interact_pointer_pos()) {
                    self.update_seam_handle_from_pointer(is_a, is_start, to_world(pointer));
                } else if let (Some(drag), Some(pointer)) = (self.piece_transform_drag, response.interact_pointer_pos()) {
                    let current = to_world(pointer);
                    let previous = (drag.last_pointer.x - drag.pivot.x, drag.last_pointer.y - drag.pivot.y);
                    let now = (current.x - drag.pivot.x, current.y - drag.pivot.y);
                    let previous_length = previous.0.hypot(previous.1);
                    let now_length = now.0.hypot(now.1);
                    let mode = self.piece_transform_mode;
                    if mode != PieceTransformMode::Move && previous_length > 1.0e-4 && now_length > 1.0e-4 {
                        let (amount, neutral) = if mode == PieceTransformMode::Rotate {
                            let mut delta = now.1.atan2(now.0) - previous.1.atan2(previous.0);
                            if delta > std::f64::consts::PI { delta -= std::f64::consts::TAU; }
                            if delta < -std::f64::consts::PI { delta += std::f64::consts::TAU; }
                            (delta, 0.0)
                        } else { ((now_length / previous_length).clamp(0.5, 2.0), 1.0) };
                        if (amount - neutral).abs() > 1.0e-9 {
                            if self.dragging_before.is_none() { self.dragging_before = Some(self.editor_snapshot()); }
                            let id = self.pattern_project.pieces.get(drag.piece_index).map(|piece| piece.id);
                            if let (Some(id), Some(piece)) = (id, self.pattern_project.pieces.get_mut(drag.piece_index)) {
                                if mode == PieceTransformMode::Rotate { rotate_piece_about(piece, drag.pivot, amount); }
                                else { scale_piece_about(piece, drag.pivot, amount); }
                                self.sync_symmetric_mate(id);
                            }
                        }
                    }
                    self.piece_transform_drag = Some(PieceTransformDrag { last_pointer: current, ..drag });
                } else if let (Some(target), Some(pointer)) = (self.dragging_target, response.interact_pointer_pos()) {
                    let mut world = to_world(pointer);
                    if self.pattern_snap {
                        world.x = (world.x * 100.0).round() / 100.0;
                        world.y = (world.y * 100.0).round() / 100.0;
                    }
                    if let PatternEditTarget::Piece { piece } = target {
                        if let Some(previous) = self.drag_last_world {
                            if let Some(id) = self.pattern_project.pieces.get(piece).map(|data| data.id) {
                                translate_project_piece(
                                    &mut self.pattern_project,
                                    id,
                                    world.x - previous.x,
                                    world.y - previous.y,
                                );
                                self.sync_symmetric_mate(id);
                            }
                        }
                    } else {
                        self.move_pattern_target(target, world);
                    }
                    self.drag_last_world = Some(world);
                }
            }
            if response.drag_stopped_by(egui::PointerButton::Primary) {
                if self.dragging_seam_handle.take().is_some() {
                    self.commit_seam_handle_drag();
                    self.seam_drag_before = None;
                } else if let Some(before) = self.dragging_before.take() {
                    self.remember_pattern_change(before);
                }
                self.piece_transform_drag = None;
                self.dragging_target = None;
                self.drag_last_world = None;
            }
            if response.clicked() && self.dragging_target.is_none() && self.dragging_seam_handle.is_none() {
                if let Some(pointer) = response.interact_pointer_pos() {
                    let target = self.hit_test_pattern(pointer, center);
                    let shift = ui.input(|input| input.modifiers.shift);
                    if self.seam_pick_mode || shift {
                        if let Some(PatternEditTarget::Edge { piece, segment, .. }) = target {
                            if let Some(data) = self.pattern_project.pieces.get(piece) {
                                let edge = EdgePath::from_segments(
                                    data.id, BoundaryCurve::Outline,
                                    segment as u32, segment as u32 + 1, false,
                                );
                                if self.seam_pick_mode && self.seam_pick_pair_mode {
                                    self.seam_canvas_pick_used = true;
                                    if self.seam_sides_a.is_empty() {
                                        self.add_draft_edge_to_side(true, edge);
                                        self.seam_active_side_a = false;
                                        self.pattern_message = "Lado A selecionado. Clique agora na aresta correspondente do lado B.".into();
                                    } else if self.seam_sides_b.is_empty() {
                                        self.add_draft_edge_to_side(false, edge);
                                        if !self.seam_sides_b.is_empty() {
                                            self.add_seam_from_controls();
                                        }
                                    } else {
                                        self.pattern_message = "O par já foi selecionado. Inicie outra costura ou use cadeia M:N para acrescentar segmentos.".into();
                                    }
                                } else {
                                    self.seam_canvas_pick_used |= self.seam_pick_mode;
                                    self.add_draft_edge_to_side(self.seam_active_side_a, edge);
                                }
                            }
                        } else if self.seam_pick_mode {
                            self.pattern_message = "Clique diretamente em uma aresta do contorno do molde; a área interna não é uma aresta.".into();
                        }
                    } else {
                        self.selected_target = target;
                        if let Some(target) = self.selected_target {
                            self.selected_piece = target.piece().min(self.pattern_project.pieces.len().saturating_sub(1));
                        }
                    }
                }
            }
            if response.hovered() && ui.input(|i| i.key_pressed(egui::Key::Delete)) {
                match self.selected_target {
                    Some(PatternEditTarget::Piece { .. }) => self.delete_selected_piece(),
                    Some(PatternEditTarget::Anchor { .. }) => self.delete_selected_vertex(),
                    _ => {}
                }
            }
        } else if response.clicked() {
            if let Some(pos) = response.interact_pointer_pos() {
                let before = self.editor_snapshot();
                let mut world = to_world(pos);
                if self.pattern_snap {
                    world.x = (world.x * 100.0).round() / 100.0;
                    world.y = (world.y * 100.0).round() / 100.0;
                }
                self.draft_points.push(world);
                self.remember_pattern_change(before);
            }
        }
    }

    fn start_pattern_simulation(&mut self) -> Result<(), String> {
        if self.avatar_import_pending_path.is_some() {
            return Err("A importação do avatar ainda está em andamento.".into());
        }
        if self.avatar_settings_are_stale() {
            return Err("As opções do avatar mudaram. Clique em 'Aplicar escala e colisão' antes de iniciar a simulação.".into());
        }
        let previous_selection = self.selected_placement_piece;
        self.pattern_project.validate()?;
        if self.pattern_project.pieces.is_empty() {
            return Err("O projeto não contém peças.".into());
        }

        // Força todos os piques e extremos de costura para dentro da amostragem de contorno
        // antes da triangulação. As extremidades não são aproximadas ao vértice vizinho.
        let mut forced_by_piece: std::collections::HashMap<u64, Vec<(u32, f64)>> =
            std::collections::HashMap::new();
        for piece in &self.pattern_project.pieces {
            forced_by_piece.entry(piece.id).or_default()
                .extend(piece.notches.iter().map(|notch| (notch.segment_index, notch.t)));
        }
        for seam in &self.pattern_project.seams {
            for edge in seam.sides_a.iter().chain(seam.sides_b.iter()) {
                if edge.curve != BoundaryCurve::Outline { continue; }
                let points = forced_by_piece.entry(edge.piece_id).or_default();
                points.push((edge.start.segment, edge.start.t));
                points.push((edge.end.segment, edge.end.t));
            }
        }
        for params in forced_by_piece.values_mut() {
            params.sort_by(|a, b| a.0.cmp(&b.0).then_with(|| {
                a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal)
            }));
            params.dedup_by(|a, b| a.0 == b.0 && (a.1 - b.1).abs() <= 1.0e-9);
        }

        // Cada molde usa a sua própria distância entre partículas. As coordenadas da malha
        // são recentradas por peça, para que a transformação 3D gire em torno do centro do painel.
        let mut prepared = Vec::with_capacity(self.pattern_project.pieces.len());
        let mut synchronized_samples = sample_pattern_outlines_with_seam_sync(
            &self.pattern_project.pieces,
            &self.pattern_project.seams,
        ).map_err(|error| format!("Falha ao preparar as bordas de costura: {error}"))?;
        for piece in &self.pattern_project.pieces {
            let outline_polygon = sample_curve(&piece.outline, 32);
            for curve in &piece.internal_lines {
                if !curve_samples_inside_polygon(curve, &outline_polygon, 32) {
                    return Err(format!("Uma linha interna de '{}' sai do contorno ou possui geometria inválida.", piece.name));
                }
            }
            for fold in &piece.fold_lines {
                if fold.kind != FoldKind::Decorative
                    && !curve_samples_inside_polygon(&fold.curve, &outline_polygon, 32)
                {
                    return Err(format!("A dobra física de '{}' sai do contorno ou possui geometria inválida.", piece.name));
                }
            }
            for pleat in &piece.pleats {
                if !curve_samples_inside_polygon(&pleat.axis, &outline_polygon, 32) {
                    return Err(format!("O eixo da prega de '{}' sai do contorno ou possui geometria inválida.", piece.name));
                }
            }
            let particle_distance = piece.fabric.particle_distance_meters;
            if !particle_distance.is_finite() || particle_distance <= 0.0 {
                return Err(format!("Distância entre partículas inválida em '{}'.", piece.name));
            }
            let sampled = synchronized_samples.remove(&piece.id)
                .ok_or_else(|| format!("A amostragem do molde '{}' não foi gerada.", piece.name))?;
            let params = forced_by_piece.get(&piece.id).cloned().unwrap_or_default();
            let forced_indices = forced_parameter_indices(piece, &sampled, &params)
                .map_err(|error| format!("Não foi possível mapear os piques/extremos de '{}': {error}", piece.name))?;
            validate_simple_polygon(&sampled.points)
                .map_err(|e| format!("Contorno inválido em '{}': {e}", piece.name))?;
            let mesh = triangulate_outline_with_forced_vertices(
                &sampled.points, particle_distance, &forced_indices,
            ).map_err(|e| format!("Não foi possível triangular '{}': {e}", piece.name))?;
            if mesh.boundary_vertex_count != sampled.points.len() {
                return Err(format!("Mapeamento de contorno inválido na peça '{}'.", piece.name));
            }
            let mapping = PieceSampleMapping {
                ranges: sampled.segment_vertex_ranges,
                forced_vertices: params.iter().copied().zip(forced_indices.iter().copied())
                    .map(|((segment, t), index)| (segment, t, index)).collect(),
            };
            let grain = piece.grain_line.as_ref().map(|line| {
                Vec2::new((line.end.x - line.start.x) as f32, (line.end.y - line.start.y) as f32)
            }).unwrap_or(Vec2::Y);
            prepared.push((
                piece.id,
                mapping,
                mesh,
                piece.fabric.clone(),
                piece.assembly.clone(),
                grain,
                piece.name.clone(),
            ));
        }

        let min_y = prepared.iter()
            .flat_map(|(_, _, mesh, _, _, _, _)| mesh.vertices.iter().map(|p| p.y))
            .fold(f64::INFINITY, f64::min);

        let avatar = AvatarConfig {
            height_m: self.pattern_project.avatar.height_meters as f32,
            shoulder_width_m: self.pattern_project.avatar.shoulder_width_meters as f32,
            waist_width_m: self.pattern_project.avatar.waist_width_meters as f32,
            hip_width_m: self.pattern_project.avatar.hip_width_meters as f32,
        };
        // A malha visual e o SDF usam a mesma altura, orientação e coordenadas.
        // O SDF do avatar importado tem prioridade; as formas analíticas são o fallback.
        let body_colliders = self.avatar_body_colliders()?;
        let mut builder = ClothBuilder::new();
        builder.add_collider(Collider::ground());
        for collider in &body_colliders {
            builder.add_collider(collider.clone());
        }

        struct PieceBuildInfo {
            id: u64,
            handle: cloth_sim::PanelHandle,
            sample: PieceSampleMapping,
            boundary_count: usize,
            base_rotation: Quat,
            pivot: Vec3,
        }
        let mut infos: Vec<PieceBuildInfo> = Vec::new();
        let mut physical_fold_edge_count = 0usize;
        let piece_count = prepared.len();
        // A origem 2D (y = 0) fica a 0,80 m do chão no guia T-Pose de 1,72 m e acompanha a altura do
        // avatar. Moldes desenhados abaixo dos pés sobem o conjunto para ficar acima do chão.
        let origin_y = (0.80 * f64::from(avatar.height_m) / 1.72).max(0.02 - min_y) as f32;

        // Empilhamento: camada menor fica mais perto do corpo; empate resolve pela ordem dos moldes.
        let mut layer_order: Vec<usize> = (0..piece_count).collect();
        layer_order.sort_by_key(|&index| {
            let piece_id = prepared[index].0;
            let layer = self.pattern_project.pieces.iter()
                .find(|piece| piece.id == piece_id)
                .map(|piece| piece.layer)
                .unwrap_or(0);
            (layer, index)
        });
        let mut layer_rank = vec![0usize; piece_count];
        for (rank, &index) in layer_order.iter().enumerate() {
            layer_rank[index] = rank;
        }
        let arrangement_list = arrangement_points(&avatar);

        for (piece_index, (id, sample, mesh, fabric, assembly, grain, name)) in
            prepared.into_iter().enumerate()
        {
            let mesh_min_x = mesh.vertices.iter().map(|p| p.x).fold(f64::INFINITY, f64::min);
            let mesh_max_x = mesh.vertices.iter().map(|p| p.x).fold(f64::NEG_INFINITY, f64::max);
            let mesh_min_y = mesh.vertices.iter().map(|p| p.y).fold(f64::INFINITY, f64::min);
            let mesh_max_y = mesh.vertices.iter().map(|p| p.y).fold(f64::NEG_INFINITY, f64::max);
            let center_x = (mesh_min_x + mesh_max_x) * 0.5;
            let center_y = (mesh_min_y + mesh_max_y) * 0.5;

            let panel_scale = assembly.scale.map(|value| value as f32);
            if panel_scale.iter().any(|value| !value.is_finite() || *value <= 0.0 || *value > 100.0) {
                return Err(format!("Escala inválida no molde '{}'.", name));
            }
            let uv = mesh.vertices.iter()
                .map(|p| Vec2::new(
                    ((p.x - center_x) as f32) * panel_scale[0],
                    ((p.y - center_y) as f32) * panel_scale[1],
                ))
                .collect();
            let piece_metadata = self.pattern_project.pieces.iter()
                .find(|piece| piece.id == id)
                .ok_or_else(|| format!("Metadados ausentes para o molde '{}'.", name))?;
            let fold_edges = physical_fold_edges(
                &mesh.vertices, &mesh.triangles, piece_metadata,
            );
            physical_fold_edge_count += fold_edges.len();
            let panel_mesh = PanelMesh { uv, triangles: mesh.triangles };
            let piece_thickness = fabric.thickness_meters as f32;
            let clearance = piece_thickness
                + PATTERN_SIM_PERTURBATION
                + layer_rank[piece_index] as f32 * PATTERN_SIM_LAYER_SPACING;
            // Com ponto de posicionamento, ele continua tendo prioridade e define a normal do painel.
            // Sem ponto, projetamos o layout 2D diretamente para a frente do corpo.
            let arrangement_point = assembly.arrangement.as_deref()
                .and_then(|point_id| arrangement_list.iter().find(|point| point.id == point_id));
            let (auto_origin, base_angle) = if let Some(point) = arrangement_point {
                let surface = arrangement_surface_position(&body_colliders, point, clearance);
                (Vec3::new(surface.x, point.anchor.y, surface.z), point.normal.x.atan2(point.normal.z))
            } else {
                // X/Y do layout 2D mapeiam diretamente para X/Y do mundo. Cada painel é independente;
                // painéis nas costas devem usar um ponto de posicionamento explícito.
                let front_z = front_projection_z(&body_colliders, &mesh.vertices, origin_y, clearance)
                    .unwrap_or_else(|| pattern_sim_shell_radius(layer_rank[piece_index], piece_thickness));
                (Vec3::new(center_x as f32, origin_y + center_y as f32, front_z), 0.0)
            };
            let translation = auto_origin + Vec3::new(
                assembly.translation_meters[0] as f32,
                assembly.translation_meters[1] as f32,
                assembly.translation_meters[2] as f32,
            );
            let local_rotation = Quat::from_euler(
                EulerRot::XYZ,
                assembly.rotation_degrees[0].rem_euclid(360.0).to_radians() as f32,
                assembly.rotation_degrees[1].rem_euclid(360.0).to_radians() as f32,
                assembly.rotation_degrees[2].rem_euclid(360.0).to_radians() as f32,
            );
            let base_rotation = Quat::from_rotation_y(base_angle);
            let rotation = base_rotation * local_rotation;
            let material = PanelMaterial {
                density_kg_m2: fabric.density_kg_m2 as f32,
                stretch_warp_n_m: fabric.stretch_warp_n_m as f32,
                stretch_weft_n_m: fabric.stretch_weft_n_m as f32,
                shear_n_m: fabric.shear_n_m as f32,
                bend_nm: fabric.bend_nm as f32,
                thickness_meters: fabric.thickness_meters as f32,
                friction_coefficient: fabric.friction_coefficient as f32,
                friction_dynamic_coefficient: fabric.friction_dynamic_coefficient as f32,
            };
            let handle = builder.add_panel_with_grain_and_material(
                &panel_mesh,
                Placement::new(translation, rotation),
                grain,
                material,
            ).map_err(|error| format!("Material inválido em '{}': {error}", name))?;
            for &(a, b, stiffness, rest_angle) in &fold_edges {
                builder.add_fold_edge(handle, a, b, stiffness, rest_angle);
            }

            // Âncoras de contorno: 0 é o início da curva; as demais correspondem ao fim do
            // segmento anterior. Os índices são os mesmos que o editor exibe ao selecionar nós.
            for anchor in &assembly.pinned_anchors {
                let anchor_index = *anchor as usize;
                if anchor_index >= sample.ranges.len() {
                    return Err(format!(
                        "O vértice fixado {} não existe no contorno de '{}'.",
                        anchor_index + 1, name
                    ));
                }
                let local_index = if anchor_index == 0 {
                    0
                } else {
                    sample.ranges[anchor_index - 1].1 % mesh.boundary_vertex_count
                };
                builder.pin(handle.global(local_index as u32));
            }

            infos.push(PieceBuildInfo {
                id,
                handle,
                sample,
                boundary_count: mesh.boundary_vertex_count,
                base_rotation,
                pivot: translation,
            });
        }

        for seam in &self.pattern_project.seams {
            let map_side = |paths: &[EdgePath]| -> Result<Vec<Vec<u32>>, String> {
                let mut result = Vec::with_capacity(paths.len());
                for edge in paths {
                    if edge.curve != BoundaryCurve::Outline {
                        return Err(format!(
                            "A costura '{}' usa uma linha interna; a simulação atual suporta bordas do contorno.",
                            seam.name
                        ));
                    }
                    let info = infos.iter().find(|info| info.id == edge.piece_id)
                        .ok_or_else(|| format!("Peça ausente na costura '{}'.", seam.name))?;
                    let locals = sampled_edge_indices(
                        edge, &info.sample.ranges, &info.sample.forced_vertices, info.boundary_count,
                    )?;
                    result.push(info.handle.globals(&locals));
                }
                Ok(result)
            };
            let globals_a = map_side(&seam.sides_a)?;
            let globals_b = map_side(&seam.sides_b)?;
            let notch_markers = paired_chain_notch_markers(
                &self.pattern_project, &seam.sides_a, &seam.sides_b,
            );
            builder.sew_with_path_chains(
                &globals_a, &globals_b, seam.ease_meters as f32, &notch_markers,
            );
        }

        // A simulação mantém o espaçamento real de cada molde; o ruído inicial é menor que
        // a folga aplicada contra o corpo para evitar penetração artificial no primeiro frame.
        builder.perturb(PATTERN_SIM_PERTURBATION);
        self.placement_ranges = infos.iter().map(|info| PanelPlacementRange {
            piece_id: info.id,
            first: info.handle.first as usize,
            end: (info.handle.first + info.handle.count) as usize,
            boundary_count: info.boundary_count,
            base_rotation: info.base_rotation,
            pivot: info.pivot,
        }).collect();
        self.sim = builder.build(self.sim.params.clone());
        self.seam_pairs = self.sim.seam_pairs();
        self.diverged = false;
        self.accumulator = 0.0;
        self.playing = false;
        self.using_pattern_sim = true;
        self.workspace = WorkspaceMode::Placement;
        self.placement_drag_piece = None;
        self.placement_drag_before = None;
        self.placement_gizmo_drag = None;
        self.placement_drag_selected_body = false;
        self.selected_placement_piece = previous_selection
            .filter(|id| self.placement_ranges.iter().any(|range| range.piece_id == *id))
            .or_else(|| self.placement_ranges.first().map(|range| range.piece_id));
        self.upload_scene();
        let pinned = self.pattern_project.pieces.iter()
            .map(|piece| piece.assembly.pinned_anchors.len())
            .sum::<usize>();
        let body_collider_count = self.sim.colliders.len().saturating_sub(1);
        let collision_label = if self.avatar_sdf.is_some() {
            "SDF da malha importada"
        } else {
            "volumes analíticos"
        };
        self.pattern_message = format!(
            "Prévia pronta: {} moldes, {} costuras, {} âncoras fixadas, {} arestas de dobra/prega mapeadas e {} colisor(es) corporal(is) ({collision_label}). Arraste os painéis em 3D; inicie a simulação quando estiver satisfeito. Avatar: {:.2} m; ombros {:.2} m, cintura {:.2} m e quadril {:.2} m.",
            piece_count, self.pattern_project.seams.len(), pinned, physical_fold_edge_count, body_collider_count,
            self.pattern_project.avatar.height_meters,
            self.pattern_project.avatar.shoulder_width_meters,
            self.pattern_project.avatar.waist_width_meters,
            self.pattern_project.avatar.hip_width_meters,
        );
        if physical_fold_edge_count == 0
            && self.pattern_project.pieces.iter().any(|piece| !piece.fold_lines.is_empty() || !piece.pleats.is_empty())
        {
            self.pattern_message.push_str(" Nenhuma dobra/prega encontrou arestas internas suficientes na malha; aumente a resolução do molde ou ajuste o traçado.");
        }
        Ok(())
    }

    /// Modo dividido: 3D e editor 2D lado a lado (ou empilhados), com divisor arrastável.
    /// O primeiro painel é redimensionável; o segundo ocupa o espaço restante.
    fn split_viewports(&mut self, ui: &mut egui::Ui) {
        const MIN_PANE: f32 = 160.0;
        let vertical = self.split_vertical;
        // `swapped` coloca o editor 2D no primeiro painel (esquerda ou topo).
        let pattern_first = self.split_swapped;
        let total = if vertical { ui.available_height() } else { ui.available_width() };
        let max_first = (total - MIN_PANE).max(MIN_PANE);
        // Ids distintos por orientação: o tamanho guardado de uma não vira o da outra.
        let first = if vertical {
            egui::Panel::top("split_first_vertical")
        } else {
            egui::Panel::left("split_first_horizontal")
        };
        first
            .resizable(true)
            .frame(egui::Frame::NONE)
            .default_size(total * 0.5)
            .size_range(egui::Rangef::new(MIN_PANE, max_first))
            .show(ui, |ui| self.split_pane(ui, pattern_first));
        egui::CentralPanel::default()
            .frame(egui::Frame::NONE)
            .show(ui, |ui| self.split_pane(ui, !pattern_first));
    }

    fn split_pane(&mut self, ui: &mut egui::Ui, pattern: bool) {
        if pattern {
            self.pattern_viewport(ui);
        } else {
            self.viewport(ui);
        }
    }

    fn camera_focus_bounds(&self) -> Option<(Vec3, f32)> {
        let mut min = Vec3::splat(f32::INFINITY);
        let mut max = Vec3::splat(f32::NEG_INFINITY);
        let mut include = |p: Vec3| {
            if p.is_finite() { min = min.min(p); max = max.max(p); }
        };
        if let Some(mesh) = self.avatar_mesh.as_ref() {
            for vertex in &mesh.vertices { include(Vec3::from_array(vertex.pos)); }
        } else if !self.validation_scene || self.using_pattern_sim {
            let height = self.pattern_project.avatar.height_meters as f32;
            include(Vec3::new(-0.30 * height, 0.0, -0.18 * height));
            include(Vec3::new(0.30 * height, height, 0.18 * height));
        }
        for &position in &self.sim.pos { include(position); }
        if min.x > max.x { return None; }
        Some(((min + max) * 0.5, ((max - min).length() * 0.5).max(0.25)))
    }
    fn focus_camera_preset(&mut self, preset: CameraPreset, aspect: f32) {
        let (center, radius) = self.camera_focus_bounds().unwrap_or((self.camera.target, 1.0));
        self.camera_goal = Some(self.camera.goal_for(preset, center, radius, aspect));
    }
    fn viewport(&mut self, ui: &mut egui::Ui) {
        let rect = ui.available_rect_before_wrap();
        let response = ui.allocate_rect(rect, egui::Sense::click_and_drag());
        let placement_mode = self.workspace == WorkspaceMode::Placement && self.using_pattern_sim;
        let toolbar_bounds = placement_mode.then(|| placement_toolbar_bounds(rect));
        let pointer_over_toolbar = toolbar_bounds
            .map(|bounds| ui.input(|i| i.pointer.hover_pos().is_some_and(|pointer| bounds.contains(pointer))))
            .unwrap_or(false);
        let aspect = rect.width() / rect.height().max(1.0);
        if response.hovered() {
            let preset = ui.input(|i| -> Option<CameraPreset> {
                if i.modifiers.command || i.modifiers.alt { return None; }
                if i.key_pressed(egui::Key::Num1) { Some(CameraPreset::Front) }
                else if i.key_pressed(egui::Key::Num2) { Some(CameraPreset::Side) }
                else if i.key_pressed(egui::Key::Num3) { Some(CameraPreset::Top) }
                else { None }
            });
            if let Some(preset) = preset { self.focus_camera_preset(preset, aspect); ui.ctx().request_repaint(); }
        }
        // Na prévia: primeiro tenta capturar um manipulador; só depois considera o painel.
        // Isso evita que um clique no eixo seja interpretado como arraste livre do tecido.
        let pointer_delta = ui.input(|i| i.pointer.delta());
        if placement_mode {
            if response.drag_started_by(egui::PointerButton::Primary) && !pointer_over_toolbar {
                if let Some(pointer) = response.interact_pointer_pos() {
                    let view_proj = self.camera.view_proj(aspect);
                    if let Some(gizmo) = self.hit_test_placement_gizmo(pointer, rect, view_proj) {
                        self.selected_placement_piece = Some(gizmo.piece_id);
                        self.placement_drag_piece = Some(gizmo.piece_id);
                        self.placement_gizmo_drag = Some(gizmo);
                        self.placement_drag_before = Some(self.editor_snapshot());
                    } else if let Some(id) = self.hit_test_placement_piece(pointer, rect) {
                        self.selected_placement_piece = Some(id);
                        if matches!(self.gizmo_mode, GizmoMode::Move | GizmoMode::Transform) {
                            self.placement_drag_piece = Some(id);
                            self.placement_drag_before = Some(self.editor_snapshot());
                        } else {
                            // Em Girar/Escalar, clicar no painel apenas o seleciona; o arraste
                            // do corpo não deve mover a câmera nem transformar o painel.
                            self.placement_drag_piece = None;
                            self.placement_drag_selected_body = true;
                        }
                    }
                }
            }
            if response.dragged_by(egui::PointerButton::Primary) && !pointer_over_toolbar {
                if let Some(mut gizmo) = self.placement_gizmo_drag {
                    let pointer = response.interact_pointer_pos()
                        .unwrap_or(gizmo.last_pointer + pointer_delta);
                    match gizmo.action {
                        GizmoAction::Move => {
                            let amount = pointer_delta.dot(gizmo.screen_axis)
                                / gizmo.pixels_per_meter.max(1.0e-3);
                            self.apply_placement_translation(
                                gizmo.piece_id, gizmo.axis_world * amount,
                            );
                        }
                        GizmoAction::Rotate => {
                            let from = gizmo.last_pointer - gizmo.screen_center;
                            let to = pointer - gizmo.screen_center;
                            if from.length() > 3.0 && to.length() > 3.0 {
                                let cross = from.x * to.y - from.y * to.x;
                                let degrees = cross.atan2(from.dot(to)).to_degrees();
                                self.apply_placement_rotation_delta(
                                    gizmo.piece_id, gizmo.axis, gizmo.axis_world, degrees,
                                );
                            }
                        }
                        GizmoAction::Scale => {
                            let exponent = pointer_delta.dot(gizmo.screen_axis) / 130.0;
                            let factor = exponent.clamp(-0.35, 0.35).exp();
                            self.apply_placement_scale_factor(
                                gizmo.piece_id, gizmo.axis, factor,
                            );
                        }
                    }
                    gizmo.last_pointer = pointer;
                    self.placement_gizmo_drag = Some(gizmo);
                } else if let Some(id) = self.placement_drag_piece {
                    self.translate_placement_by_screen_delta(
                        id, pointer_delta.x, pointer_delta.y, rect.height(),
                    );
                } else if !self.placement_drag_selected_body {
                    self.camera.orbit(pointer_delta.x, pointer_delta.y);
                }
            }
            if response.clicked() && !pointer_over_toolbar {
                if let Some(pointer) = response.interact_pointer_pos() {
                    let view_proj = self.camera.view_proj(aspect);
                    if self.hit_test_placement_gizmo(pointer, rect, view_proj).is_none() {
                        self.selected_placement_piece = self.hit_test_placement_piece(pointer, rect);
                    }
                }
            }
            if response.drag_stopped_by(egui::PointerButton::Primary) {
                let stopped_action = self.placement_gizmo_drag.take().map(|drag| drag.action);
                let before = self.placement_drag_before.take();
                self.placement_drag_selected_body = false;
                // Escalar altera comprimentos de repouso e áreas. Reconstrói uma única vez
                // ao soltar o mouse, e não a cada quadro do arraste.
                let mut commit_history = true;
                if stopped_action == Some(GizmoAction::Scale) {
                    if let Err(error) = self.start_pattern_simulation() {
                        commit_history = false;
                        if let Some(snapshot) = before.as_ref() {
                            self.pattern_project = snapshot.project.clone();
                            match self.start_pattern_simulation() {
                                Ok(()) => self.pattern_message = format!(
                                    "Escala cancelada e revertida após falha na reconstrução: {error}"
                                ),
                                Err(restore_error) => self.pattern_message = format!(
                                    "Falha ao aplicar a escala ({error}) e ao restaurar a prévia ({restore_error})."
                                ),
                            }
                        } else {
                            self.pattern_message = error;
                        }
                    }
                }
                if commit_history {
                    if let Some(before) = before {
                        self.remember_pattern_change(before);
                    }
                }
                self.placement_drag_piece = None;
            }
        } else if response.dragged_by(egui::PointerButton::Primary) {
            self.camera.orbit(pointer_delta.x, pointer_delta.y);
        }
        if !pointer_over_toolbar && (response.dragged_by(egui::PointerButton::Secondary)
            || response.dragged_by(egui::PointerButton::Middle))
        {
            self.camera.pan(pointer_delta.x, pointer_delta.y, rect.height());
        }
        if response.hovered() && !pointer_over_toolbar {
            let scroll = ui.input(|i| i.smooth_scroll_delta.y);
            if scroll != 0.0 {
                self.camera.zoom(scroll);
            }
        }

        if self.camera_goal.is_some() {
            let scrolling = response.hovered() && ui.input(|i| i.smooth_scroll_delta.y != 0.0);
            if response.dragged() || scrolling { self.camera_goal = None; }
        }
        // Dados do frame.
        self.sim.compute_vertex_normals(&mut self.normals);
        let cloth_vertices: Vec<Vertex> = self.sim.pos.iter()
            .zip(&self.normals)
            .map(|(p, n)| Vertex { pos: p.to_array(), normal: n.to_array() })
            .collect();

        // Fundo neutro atrás da cena (o chão desenhado pelo renderer cobre a parte de baixo).
        ui.painter().rect_filled(rect, 0.0, egui::Color32::from_rgb(52, 60, 74));
        let view_proj = self.camera.view_proj(aspect);
        ui.painter().add(egui_wgpu::Callback::new_paint_callback(
            rect,
            ViewportCallback {
                view_proj,
                eye: self.camera.eye(),
                cloth_vertices,
                warning_seam_pairs: self.sim.seam_warning_pairs(0.015),
                style: self.style.clone(),
            },
        ));

        if placement_mode {
            // Contorno do molde selecionado.
            if let Some(range) = self.selected_placement_piece
                .and_then(|id| self.placement_ranges.iter().find(|range| range.piece_id == id))
            {
                let boundary: Vec<usize> = (range.first..range.first + range.boundary_count).collect();
                for i in 0..boundary.len() {
                    let a = self.sim.pos[boundary[i]];
                    let b = self.sim.pos[boundary[(i + 1) % boundary.len()]];
                    if let (Some((a, _)), Some((b, _))) = (
                        project_world_to_screen(a, view_proj, rect),
                        project_world_to_screen(b, view_proj, rect),
                    ) {
                        ui.painter().line_segment(
                            [a, b],
                            egui::Stroke::new(2.5, egui::Color32::from_rgb(255, 205, 72)),
                        );
                    }
                }
            }
            self.draw_placement_gizmo(ui.painter(), rect, view_proj);
            self.draw_placement_toolbar(ui, rect);
        }
    }

    /// Modo de captura: simula N passos, pede um screenshot e salva em PPM.
    fn handle_capture(&mut self, ui: &egui::Ui) {
        let Some(cap) = &mut self.capture else {
            return;
        };
        if cap.frames > 0 {
            cap.frames -= 1;
            return;
        }
        if !cap.requested {
            cap.requested = true;
            ui.ctx().send_viewport_cmd(egui::ViewportCommand::Screenshot(egui::UserData::default()));
            return;
        }
        let image = ui.input(|i| {
            i.events.iter().find_map(|e| match e {
                egui::Event::Screenshot { image, .. } => Some(image.clone()),
                _ => None,
            })
        });
        if let Some(img) = image {
            let [w, h] = img.size;
            let mut data = format!("P6\n{w} {h}\n255\n").into_bytes();
            for px in &img.pixels {
                data.extend_from_slice(&[px.r(), px.g(), px.b()]);
            }
            match std::fs::write(&cap.path, data) {
                Ok(()) => eprintln!("screenshot salvo em {} ({w}x{h})", cap.path),
                Err(e) => eprintln!("falha ao salvar screenshot: {e}"),
            }
            ui.ctx().send_viewport_cmd(egui::ViewportCommand::Close);
        }
    }
}

/// Início geométrico de um segmento na curva.
fn curve_segment_start(curve: &Curve2D, segment_index: usize) -> Option<PatternPoint2> {
    if segment_index >= curve.segments.len() { return None; }
    if segment_index == 0 { Some(curve.start) }
    else { Some(curve.segments[segment_index - 1].endpoint()) }
}

/// Avalia um segmento linear ou Bézier cúbico no parâmetro local t ∈ [0, 1].
fn curve_segment_point(curve: &Curve2D, segment_index: usize, t: f64) -> Option<PatternPoint2> {
    let start = curve_segment_start(curve, segment_index)?;
    let segment = curve.segments.get(segment_index)?;
    let t = t.clamp(0.0, 1.0);
    let lerp = |a: PatternPoint2, b: PatternPoint2, t: f64| PatternPoint2::new(
        a.x + (b.x - a.x) * t,
        a.y + (b.y - a.y) * t,
    );
    Some(match segment {
        CurveSegment::Line { to } => lerp(start, *to, t),
        CurveSegment::CubicBezier { control1, control2, to } => {
            let a = lerp(start, *control1, t);
            let b = lerp(*control1, *control2, t);
            let c = lerp(*control2, *to, t);
            lerp(lerp(a, b, t), lerp(b, c, t), t)
        }
    })
}

/// Comprimento de arco entre dois parâmetros locais do mesmo segmento.
fn param_at_curve_position(position: f64, segment_count: usize, upper_bound: f64) -> CurveParam {
    let max_position = segment_count as f64;
    let upper = if upper_bound.is_finite() { upper_bound.min(max_position) } else { max_position };
    let position = position.clamp(0.0, upper.max(0.0));
    if segment_count == 0 { return CurveParam { segment: 0, t: 0.0 }; }
    if position >= max_position {
        return CurveParam { segment: (segment_count - 1) as u32, t: 1.0 };
    }
    let segment = position.floor() as usize;
    CurveParam { segment: segment as u32, t: (position - segment as f64).clamp(0.0, 1.0) }
}

/// Projeta um ponto sobre a curva amostrando cada segmento; retorna o parâmetro com menor erro.
fn closest_curve_param(curve: &Curve2D, point: PatternPoint2) -> Option<(CurveParam, f64)> {
    const STEPS: usize = 64;
    if curve.segments.is_empty() { return None; }
    let mut best: Option<(CurveParam, f64)> = None;
    for segment in 0..curve.segments.len() {
        for step in 0..STEPS {
            let t0 = step as f64 / STEPS as f64;
            let t1 = (step + 1) as f64 / STEPS as f64;
            let a = curve_segment_point(curve, segment, t0)?;
            let b = curve_segment_point(curve, segment, t1)?;
            let dx = b.x - a.x;
            let dy = b.y - a.y;
            let length_sq = dx * dx + dy * dy;
            let alpha = if length_sq > 1.0e-18 {
                (((point.x - a.x) * dx + (point.y - a.y) * dy) / length_sq).clamp(0.0, 1.0)
            } else { 0.0 };
            let projected = PatternPoint2::new(a.x + dx * alpha, a.y + dy * alpha);
            let distance = point.distance(projected);
            let param = CurveParam { segment: segment as u32, t: t0 + (t1 - t0) * alpha };
            if best.as_ref().map(|(_, d)| distance < *d).unwrap_or(true) {
                best = Some((param, distance));
            }
        }
    }
    best
}

fn snap_curve_param_to_features(
    curve: &Curve2D,
    notches: &[Notch],
    fallback: CurveParam,
    pointer: PatternPoint2,
    tolerance_world: f64,
) -> CurveParam {
    let mut best = (fallback, pointer.distance(
        curve_segment_point(curve, fallback.segment as usize, fallback.t).unwrap_or(pointer)
    ));
    let mut consider = |param: CurveParam| {
        if let Some(point) = curve_segment_point(curve, param.segment as usize, param.t) {
            let distance = pointer.distance(point);
            if distance < best.1 { best = (param, distance); }
        }
    };
    for segment in 0..curve.segments.len() {
        consider(CurveParam { segment: segment as u32, t: 0.0 });
        consider(CurveParam { segment: segment as u32, t: 1.0 });
    }
    for notch in notches {
        if (notch.segment_index as usize) < curve.segments.len() && notch.t.is_finite() {
            consider(CurveParam {
                segment: notch.segment_index,
                t: notch.t.clamp(0.0, 1.0),
            });
        }
    }
    if best.1 <= tolerance_world { best.0 } else { fallback }
}

fn edge_curve<'a>(project: &'a PatternProject, edge: &EdgePath) -> Option<&'a Curve2D> {
    let piece = project.pieces.iter().find(|piece| piece.id == edge.piece_id)?;
    match edge.curve {
        BoundaryCurve::Outline => Some(&piece.outline),
        BoundaryCurve::SewingLine => piece.sewing_line.as_ref(),
    }
}

/// Mede a extensão real do intervalo de segmentos, inclusive para intervalos Bézier.
fn edge_length_meters(project: &PatternProject, edge: &EdgePath) -> f64 {
    project.edge_length_meters(edge).unwrap_or(0.0)
}

/// Posição de cada pique como progresso de arco normalizado na borda de costura.
fn edge_notch_positions(project: &PatternProject, edge: &EdgePath) -> Vec<f32> {
    if edge.curve != BoundaryCurve::Outline { return Vec::new(); }
    let Some(piece) = project.pieces.iter().find(|piece| piece.id == edge.piece_id) else {
        return Vec::new();
    };
    let total = edge_length_meters(project, edge);
    if total <= 1.0e-9 { return Vec::new(); }

    let mut out = piece.notches.iter().filter_map(|notch| {
        let scalar = notch.segment_index as f64 + notch.t;
        if scalar < edge.start_position() - 1.0e-9
            || scalar > edge.end_position() + 1.0e-9
            || !notch.t.is_finite()
        { return None; }
        let prefix_path = EdgePath {
            piece_id: edge.piece_id,
            curve: edge.curve,
            start: edge.start,
            end: CurveParam { segment: notch.segment_index, t: notch.t },
            reversed: false,
        };
        let prefix = project.edge_length_meters(&prefix_path).unwrap_or(0.0);
        let mut progress = (prefix / total).clamp(0.0, 1.0);
        if edge.reversed { progress = 1.0 - progress; }
        Some(progress as f32)
    }).collect::<Vec<_>>();
    out.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
    out
}

/// Gera marcadores de notch para que cada pique pareado receba uma restrição explícita do solver.
fn paired_notch_markers(
    project: &PatternProject,
    edge_a: &EdgePath,
    edge_b: &EdgePath,
) -> Vec<(f32, f32)> {
    edge_notch_positions(project, edge_a).into_iter()
        .zip(edge_notch_positions(project, edge_b))
        .collect()
}

fn edge_chain_length_meters(project: &PatternProject, paths: &[EdgePath]) -> f64 {
    paths.iter().map(|edge| edge_length_meters(project, edge)).sum()
}

fn edge_chain_notch_positions(project: &PatternProject, paths: &[EdgePath]) -> Vec<f32> {
    let total = edge_chain_length_meters(project, paths);
    if total <= 1.0e-9 { return Vec::new(); }
    let mut prefix = 0.0;
    let mut out = Vec::new();
    for edge in paths {
        let length = edge_length_meters(project, edge);
        if length > 1.0e-9 {
            out.extend(edge_notch_positions(project, edge).into_iter()
                .map(|local| ((prefix + local as f64 * length) / total).clamp(0.0, 1.0) as f32));
            prefix += length;
        }
    }
    out.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
    out
}

fn paired_chain_notch_markers(
    project: &PatternProject,
    paths_a: &[EdgePath],
    paths_b: &[EdgePath],
) -> Vec<(f32, f32)> {
    edge_chain_notch_positions(project, paths_a).into_iter()
        .zip(edge_chain_notch_positions(project, paths_b))
        .collect()
}

fn point_on_edge_chain_at_progress(
    project: &PatternProject,
    paths: &[EdgePath],
    progress: f32,
) -> Option<PatternPoint2> {
    let total = edge_chain_length_meters(project, paths);
    if total <= 1.0e-9 { return None; }
    let mut target = total * progress.clamp(0.0, 1.0) as f64;
    for edge in paths {
        let length = edge_length_meters(project, edge);
        if length <= 1.0e-9 { continue; }
        if target <= length {
            let local = (target / length).clamp(0.0, 1.0) as f32;
            let points = edge_path_points(project, edge, 32);
            return point_on_path_at_progress(&points, local);
        }
        target -= length;
    }
    paths.last().and_then(|edge| edge_path_points(project, edge, 32).last().copied())
}

/// Amostra apenas o intervalo de segmentos referenciado pela costura, para o overlay do canvas.
fn edge_path_points(project: &PatternProject, edge: &EdgePath, steps: usize) -> Vec<PatternPoint2> {
    let Some(curve) = edge_curve(project, edge) else { return Vec::new(); };
    if edge.start_position() >= edge.end_position()
        || edge.end.segment as usize >= curve.segments.len()
    { return Vec::new(); }
    let steps = steps.max(1);
    let mut points = Vec::new();
    for index in edge.start.segment as usize..=edge.end.segment as usize {
        let t0 = if index == edge.start.segment as usize { edge.start.t } else { 0.0 };
        let t1 = if index == edge.end.segment as usize { edge.end.t } else { 1.0 };
        if t1 < t0 { return Vec::new(); }
        for step in 0..=steps {
            if index > edge.start.segment as usize && step == 0 { continue; }
            let t = t0 + (t1 - t0) * step as f64 / steps as f64;
            if let Some(point) = curve_segment_point(curve, index, t) { points.push(point); }
        }
    }
    if edge.reversed { points.reverse(); }
    points
}

fn point_on_path_at_progress(points: &[PatternPoint2], progress: f32) -> Option<PatternPoint2> {
    if points.len() < 2 { return points.first().copied(); }
    let lengths = points.windows(2).map(|pair| pair[0].distance(pair[1])).collect::<Vec<_>>();
    let total: f64 = lengths.iter().sum();
    if total <= 1.0e-9 { return points.first().copied(); }
    let mut target = total * progress.clamp(0.0, 1.0) as f64;
    for (index, length) in lengths.iter().copied().enumerate() {
        if target <= length || index + 1 == lengths.len() {
            let t = if length > 1.0e-9 { (target / length).clamp(0.0, 1.0) } else { 0.0 };
            return Some(PatternPoint2::new(
                points[index].x + (points[index + 1].x - points[index].x) * t,
                points[index].y + (points[index + 1].y - points[index].y) * t,
            ));
        }
        target -= length;
    }
    points.last().copied()
}

fn distance_to_screen_path(point: egui::Pos2, points: &[egui::Pos2]) -> f32 {
    let mut best = f32::INFINITY;
    for pair in points.windows(2) {
        let delta = pair[1] - pair[0];
        let length_sq = delta.length_sq();
        let t = if length_sq > 1.0e-6 {
            ((point - pair[0]).dot(delta) / length_sq).clamp(0.0, 1.0)
        } else { 0.0 };
        best = best.min(point.distance(pair[0] + delta * t));
    }
    best
}

fn draw_path_direction_arrow(
    painter: &egui::Painter,
    points: &[egui::Pos2],
    color: egui::Color32,
) {
    if points.len() < 3 { return; }
    let index = points.len() / 2;
    let tip = points[index];
    let direction = (points[index] - points[index - 1]).normalized();
    if direction.length_sq() <= 1.0e-6 { return; }
    let normal = egui::vec2(-direction.y, direction.x);
    let base = tip - direction * 9.0;
    painter.line_segment([tip, base + normal * 4.0], egui::Stroke::new(1.8, color));
    painter.line_segment([tip, base - normal * 4.0], egui::Stroke::new(1.8, color));
}

fn translate_curve(curve: &mut Curve2D, dx: f64, dy: f64) {
    let shift = |p: &mut PatternPoint2| {
        p.x += dx;
        p.y += dy;
    };
    shift(&mut curve.start);
    for segment in &mut curve.segments {
        match segment {
            CurveSegment::Line { to } => shift(to),
            CurveSegment::CubicBezier { control1, control2, to } => {
                shift(control1);
                shift(control2);
                shift(to);
            }
        }
    }
}

fn symmetry_counterpart_id(project: &PatternProject, piece_id: u64) -> Option<u64> {
    project.symmetry_links.iter().find_map(|link| {
        if link.piece_a_id == piece_id { Some(link.piece_b_id) }
        else if link.piece_b_id == piece_id { Some(link.piece_a_id) }
        else { None }
    })
}

fn mirror_pattern_piece(
    source: &PatternPiece,
    target_id: u64,
    target_name: String,
    axis_x: f64,
) -> Result<PatternPiece, String> {
    let mirror = |curve: &Curve2D| {
        mirror_curve_x(curve, axis_x).map_err(|error| error.to_string())
    };
    let mut piece = source.clone();
    piece.id = target_id;
    piece.name = target_name;
    piece.outline = mirror(&source.outline)?;
    piece.sewing_line = source.sewing_line.as_ref().map(mirror).transpose()?;
    piece.internal_lines = source.internal_lines.iter().map(mirror).collect::<Result<Vec<_>, _>>()?;
    for fold in &mut piece.fold_lines {
        fold.curve = mirror(&fold.curve)?;
    }
    for pleat in &mut piece.pleats {
        pleat.axis = mirror(&pleat.axis)?;
    }
    if let Some(grain) = &mut piece.grain_line {
        grain.start.x = 2.0 * axis_x - grain.start.x;
        grain.end.x = 2.0 * axis_x - grain.end.x;
    }
    for dart in &mut piece.darts {
        dart.apex.x = 2.0 * axis_x - dart.apex.x;
        dart.leg_a.x = 2.0 * axis_x - dart.leg_a.x;
        dart.leg_b.x = 2.0 * axis_x - dart.leg_b.x;
    }
    for hole in &mut piece.holes {
        hole.center.x = 2.0 * axis_x - hole.center.x;
    }
    for annotation in &mut piece.annotations {
        annotation.position.x = 2.0 * axis_x - annotation.position.x;
        annotation.rotation_degrees = 180.0 - annotation.rotation_degrees;
    }
    // Conjugate the 3D transform by reflection across X: translation X and Euler Y/Z change
    // sign while X rotation remains unchanged. Topology indices stay stable through reflection.
    piece.assembly.translation_meters[0] = -source.assembly.translation_meters[0];
    piece.assembly.rotation_degrees[0] = source.assembly.rotation_degrees[0];
    piece.assembly.rotation_degrees[1] = -source.assembly.rotation_degrees[1];
    piece.assembly.rotation_degrees[2] = -source.assembly.rotation_degrees[2];
    Ok(piece)
}

fn dense_curve_samples(curve: &Curve2D, steps: usize) -> Vec<PatternPoint2> {
    let steps = steps.max(1);
    let mut points = vec![curve.start];
    let mut start = curve.start;
    for segment in &curve.segments {
        for step in 1..=steps {
            let t = step as f64 / steps as f64;
            let point = match segment {
                CurveSegment::Line { to } => PatternPoint2::new(
                    start.x + (to.x - start.x) * t,
                    start.y + (to.y - start.y) * t,
                ),
                CurveSegment::CubicBezier { control1, control2, to } => {
                    let mt = 1.0 - t;
                    let a = mt * mt * mt;
                    let b = 3.0 * mt * mt * t;
                    let c = 3.0 * mt * t * t;
                    let d = t * t * t;
                    PatternPoint2::new(
                        a * start.x + b * control1.x + c * control2.x + d * to.x,
                        a * start.y + b * control1.y + c * control2.y + d * to.y,
                    )
                }
            };
            points.push(point);
        }
        start = segment.endpoint();
    }
    points
}

fn curve_samples_inside_polygon(
    curve: &Curve2D,
    polygon: &[PatternPoint2],
    steps: usize,
) -> bool {
    !curve.closed
        && !curve.segments.is_empty()
        && dense_curve_samples(curve, steps).iter()
            .all(|point| point_in_polygon(*point, polygon))
}

fn classify_detail_point(
    point: PatternPoint2,
    polygon_a: &[PatternPoint2],
    polygon_b: &[PatternPoint2],
) -> Result<usize, String> {
    match (
        point_in_polygon(point, polygon_a),
        point_in_polygon(point, polygon_b),
    ) {
        (true, false) => Ok(0),
        (false, true) => Ok(1),
        (true, true) => Err("um detalhe toca a linha de corte; separe-o antes de cortar".into()),
        (false, false) => Err("um detalhe está fora das duas novas peças".into()),
    }
}

fn classify_points_side(
    points: &[PatternPoint2],
    polygon_a: &[PatternPoint2],
    polygon_b: &[PatternPoint2],
) -> Result<usize, String> {
    let mut chosen = None;
    for point in points {
        let side = classify_detail_point(*point, polygon_a, polygon_b)?;
        if chosen.is_some_and(|previous| previous != side) {
            return Err("o corte atravessa um detalhe interno; divida ou reposicione esse detalhe antes de cortar".into());
        }
        chosen = Some(side);
    }
    chosen.ok_or_else(|| "não foi possível determinar a peça que contém o detalhe".to_string())
}

fn classify_curve_side(
    curve: &Curve2D,
    polygon_a: &[PatternPoint2],
    polygon_b: &[PatternPoint2],
) -> Result<usize, String> {
    classify_points_side(&dense_curve_samples(curve, 24), polygon_a, polygon_b)
}

fn side_for_cut_point(
    point: PatternPoint2,
    polygon_a: &[PatternPoint2],
    polygon_b: &[PatternPoint2],
) -> Option<usize> {
    match (
        point_in_polygon(point, polygon_a),
        point_in_polygon(point, polygon_b),
    ) {
        (true, false) => Some(0),
        (false, true) => Some(1),
        _ => None,
    }
}

fn cut_intersection_between(
    a: PatternPoint2,
    b: PatternPoint2,
    side_a: usize,
    side_b: usize,
    polygon_a: &[PatternPoint2],
    polygon_b: &[PatternPoint2],
) -> PatternPoint2 {
    let mut low = a;
    let mut high = b;
    for _ in 0..40 {
        let mid = PatternPoint2::new(
            (low.x + high.x) * 0.5,
            (low.y + high.y) * 0.5,
        );
        match side_for_cut_point(mid, polygon_a, polygon_b) {
            Some(side) if side == side_a => low = mid,
            Some(side) if side == side_b => high = mid,
            _ => return mid,
        }
    }
    PatternPoint2::new((low.x + high.x) * 0.5, (low.y + high.y) * 0.5)
}

fn append_split_segment(
    fragments: &mut [Vec<Vec<PatternPoint2>>; 2],
    side: usize,
    a: PatternPoint2,
    b: PatternPoint2,
) {
    let dx = b.x - a.x;
    let dy = b.y - a.y;
    if dx * dx + dy * dy <= 1.0e-20 {
        return;
    }
    let side_fragments = &mut fragments[side];
    if let Some(last) = side_fragments.last_mut() {
        if last.last().is_some_and(|previous| {
            let dx = previous.x - a.x;
            let dy = previous.y - a.y;
            dx * dx + dy * dy <= 1.0e-12
        }) {
            if last.last().is_none_or(|previous| {
                let dx = previous.x - b.x;
                let dy = previous.y - b.y;
                dx * dx + dy * dy > 1.0e-20
            }) {
                last.push(b);
            }
            return;
        }
    }
    side_fragments.push(vec![a, b]);
}

fn split_curve_fragments_by_cut(
    curve: &Curve2D,
    polygon_a: &[PatternPoint2],
    polygon_b: &[PatternPoint2],
) -> Result<[Vec<Curve2D>; 2], String> {
    // Preserve an untouched curve (including Bézier handles) whenever possible.
    if let Ok(side) = classify_curve_side(curve, polygon_a, polygon_b) {
        let mut result: [Vec<Curve2D>; 2] = [Vec::new(), Vec::new()];
        result[side].push(curve.clone());
        return Ok(result);
    }

    let points = dense_curve_samples(curve, 64);
    if points.len() < 2 {
        return Err("não foi possível amostrar o detalhe interno".into());
    }
    let mut point_fragments: [Vec<Vec<PatternPoint2>>; 2] = [Vec::new(), Vec::new()];
    for pair in points.windows(2) {
        let a = pair[0];
        let b = pair[1];
        match (
            side_for_cut_point(a, polygon_a, polygon_b),
            side_for_cut_point(b, polygon_a, polygon_b),
        ) {
            (Some(side_a), Some(side_b)) if side_a == side_b => {
                append_split_segment(&mut point_fragments, side_a, a, b);
            }
            (Some(side_a), Some(side_b)) => {
                let intersection = cut_intersection_between(
                    a, b, side_a, side_b, polygon_a, polygon_b,
                );
                append_split_segment(&mut point_fragments, side_a, a, intersection);
                append_split_segment(&mut point_fragments, side_b, intersection, b);
            }
            (Some(side), None) | (None, Some(side)) => {
                // An exact cut-boundary sample becomes an endpoint shared by both
                // resulting pieces. Each adjacent half-segment belongs to its interior side.
                append_split_segment(&mut point_fragments, side, a, b);
            }
            (None, None) => {
                let midpoint = PatternPoint2::new(
                    (a.x + b.x) * 0.5,
                    (a.y + b.y) * 0.5,
                );
                let Some(side) = side_for_cut_point(midpoint, polygon_a, polygon_b) else {
                    return Err(
                        "o detalhe coincide com a linha de corte ou sai das duas peças".into()
                    );
                };
                append_split_segment(&mut point_fragments, side, a, b);
            }
        }
    }

    let mut result: [Vec<Curve2D>; 2] = [Vec::new(), Vec::new()];
    for side in 0..2 {
        for fragment in point_fragments[side].drain(..) {
            if fragment.len() < 2 {
                continue;
            }
            let first = fragment[0];
            let last = *fragment.last().unwrap();
            let dx = last.x - first.x;
            let dy = last.y - first.y;
            if dx * dx + dy * dy <= 1.0e-16 {
                continue;
            }
            let clipped = Curve2D::polyline(&fragment, false)
                .map_err(|error| format!("fragmento de detalhe inválido: {error}"))?;
            result[side].push(clipped);
        }
    }
    if result[0].is_empty() && result[1].is_empty() {
        return Err("o detalhe não gerou fragmentos válidos após o corte".into());
    }
    Ok(result)
}

fn split_piece_metadata(
    source: &PatternPiece,
    first: &mut PatternPiece,
    second: &mut PatternPiece,
    cut: &CutAndSewResult,
    polygon_a: &[PatternPoint2],
    polygon_b: &[PatternPoint2],
) -> Result<Option<usize>, String> {
    first.internal_lines.clear();
    second.internal_lines.clear();
    first.fold_lines.clear();
    second.fold_lines.clear();
    first.pleats.clear();
    second.pleats.clear();
    first.darts.clear();
    second.darts.clear();
    first.notches.clear();
    second.notches.clear();
    first.holes.clear();
    second.holes.clear();
    first.annotations.clear();
    second.annotations.clear();
    first.grain_line = None;
    second.grain_line = None;
    first.sewing_line = None;
    second.sewing_line = None;

    for line in &source.internal_lines {
        let fragments = split_curve_fragments_by_cut(line, polygon_a, polygon_b)
            .map_err(|error| format!("linha interna: {error}"))?;
        first.internal_lines.extend(fragments[0].iter().cloned());
        second.internal_lines.extend(fragments[1].iter().cloned());
    }
    for fold in &source.fold_lines {
        let fragments = split_curve_fragments_by_cut(&fold.curve, polygon_a, polygon_b)
            .map_err(|error| format!("linha de dobra: {error}"))?;
        for fragment in &fragments[0] {
            let mut split_fold = fold.clone();
            split_fold.curve = fragment.clone();
            first.fold_lines.push(split_fold);
        }
        for fragment in &fragments[1] {
            let mut split_fold = fold.clone();
            split_fold.curve = fragment.clone();
            second.fold_lines.push(split_fold);
        }
    }
    for pleat in &source.pleats {
        let fragments = split_curve_fragments_by_cut(&pleat.axis, polygon_a, polygon_b)
            .map_err(|error| format!("prega: {error}"))?;
        for fragment in &fragments[0] {
            let mut split_pleat = pleat.clone();
            split_pleat.axis = fragment.clone();
            first.pleats.push(split_pleat);
        }
        for fragment in &fragments[1] {
            let mut split_pleat = pleat.clone();
            split_pleat.axis = fragment.clone();
            second.pleats.push(split_pleat);
        }
    }

    if let Some(grain) = &source.grain_line {
        let line = Curve2D::polyline(&[grain.start, grain.end], false)
            .map_err(|_| "linha de fio inválida".to_string())?;
        let side = classify_curve_side(&line, polygon_a, polygon_b)
            .map_err(|error| format!("linha de fio: {error}"))?;
        if side == 0 { first.grain_line = Some(grain.clone()); }
        else { second.grain_line = Some(grain.clone()); }
    }

    let sewing_side = if let Some(line) = &source.sewing_line {
        let side = classify_curve_side(line, polygon_a, polygon_b)
            .map_err(|error| format!("linha de costura: {error}"))?;
        if side == 0 { first.sewing_line = Some(line.clone()); }
        else { second.sewing_line = Some(line.clone()); }
        Some(side)
    } else { None };

    for dart in &source.darts {
        let leg_a = Curve2D::polyline(&[dart.apex, dart.leg_a], false)
            .map_err(|_| "perna A da pence inválida".to_string())?;
        let leg_b = Curve2D::polyline(&[dart.apex, dart.leg_b], false)
            .map_err(|_| "perna B da pence inválida".to_string())?;
        let side_a = classify_curve_side(&leg_a, polygon_a, polygon_b)
            .map_err(|error| format!("pence: {error}"))?;
        let side_b = classify_curve_side(&leg_b, polygon_a, polygon_b)
            .map_err(|error| format!("pence: {error}"))?;
        if side_a != side_b {
            return Err("o corte atravessa uma pence; feche ou recrie a pence antes de cortar".into());
        }
        if side_a == 0 { first.darts.push(dart.clone()); }
        else { second.darts.push(dart.clone()); }
    }

    for notch in &source.notches {
        let source_index = notch.segment_index as usize;
        let fragments = cut.source_segment_fragments.get(source_index)
            .ok_or_else(|| "um entalhe referencia um segmento inexistente".to_string())?;
        let matches = fragments.iter().filter(|fragment| {
            notch.t >= fragment.t_start - 1.0e-8 && notch.t <= fragment.t_end + 1.0e-8
        }).collect::<Vec<&SourceSegmentFragment>>();
        if matches.len() != 1 {
            return Err("um entalhe está exatamente sobre a linha de corte; mova-o antes de cortar".into());
        }
        let fragment = matches[0];
        let span = fragment.t_end - fragment.t_start;
        let mut moved = notch.clone();
        moved.segment_index = fragment.new_segment_index as u32;
        moved.t = if span.abs() <= 1.0e-12 { 0.0 }
            else { ((notch.t - fragment.t_start) / span).clamp(0.0, 1.0) };
        if fragment.piece_index == 0 { first.notches.push(moved); }
        else { second.notches.push(moved); }
    }

    for hole in &source.holes {
        if !hole.radius_meters.is_finite() || hole.radius_meters < 0.0 {
            return Err("um furo possui raio inválido".into());
        }
        let mut boundary = Vec::with_capacity(25);
        for index in 0..24 {
            let angle = std::f64::consts::TAU * index as f64 / 24.0;
            boundary.push(PatternPoint2::new(
                hole.center.x + hole.radius_meters * angle.cos(),
                hole.center.y + hole.radius_meters * angle.sin(),
            ));
        }
        let side = classify_detail_point(hole.center, polygon_a, polygon_b)
            .map_err(|error| format!("furo: {error}"))?;
        if boundary.iter().any(|point| {
            classify_detail_point(*point, polygon_a, polygon_b)
                .map(|boundary_side| boundary_side != side).unwrap_or(true)
        }) {
            return Err("o corte atravessa um furo; reposicione ou remova o furo antes de cortar".into());
        }
        if side == 0 { first.holes.push(hole.clone()); }
        else { second.holes.push(hole.clone()); }
    }
    for annotation in &source.annotations {
        let side = classify_detail_point(annotation.position, polygon_a, polygon_b)
            .map_err(|error| format!("anotação: {error}"))?;
        if side == 0 { first.annotations.push(annotation.clone()); }
        else { second.annotations.push(annotation.clone()); }
    }

    Ok(sewing_side)
}

fn remap_edge_after_cut(
    edge: &mut EdgePath,
    mapping: &[Vec<SourceSegmentFragment>],
    first_id: u64,
    second_id: u64,
) -> Result<(), String> {
    let start = edge.start.segment as usize;
    let end = edge.end.segment as usize;
    if start > end || end >= mapping.len() || edge.start_position() >= edge.end_position() {
        return Err("uma costura existente referencia um intervalo inválido".into());
    }
    let mut side = None;
    let mut first_new_index: Option<usize> = None;
    for source_index in start..=end {
        let fragments = &mapping[source_index];
        if fragments.len() != 1 {
            return Err(format!(
                "a costura existente usa o segmento {} que foi dividido pelo corte; remova ou refaça essa costura primeiro.",
                source_index + 1
            ));
        }
        let fragment = &fragments[0];
        if fragment.t_start.abs() > 1.0e-8 || (fragment.t_end - 1.0).abs() > 1.0e-8 {
            return Err(format!("a costura existente usa o segmento {} parcialmente cortado.", source_index + 1));
        }
        if side.is_some_and(|previous| previous != fragment.piece_index) {
            return Err("a costura existente atravessaria as duas novas peças; reconfigure essa costura antes do corte".into());
        }
        side = Some(fragment.piece_index);
        if let Some(first_index) = first_new_index {
            if fragment.new_segment_index != first_index + source_index - start {
                return Err("o intervalo da costura deixaria de ser contínuo depois do corte; refaça essa costura após dividir".into());
            }
        } else {
            first_new_index = Some(fragment.new_segment_index);
        }
    }
    let side = side.ok_or_else(|| "intervalo de costura vazio".to_string())?;
    edge.piece_id = if side == 0 { first_id } else { second_id };
    edge.start.segment = first_new_index.unwrap() as u32;
    edge.end.segment = edge.start.segment + (end - start) as u32;
    Ok(())
}

fn translate_piece(piece: &mut PatternPiece, dx: f64, dy: f64) {
    translate_curve(&mut piece.outline, dx, dy);
    if let Some(curve) = &mut piece.sewing_line { translate_curve(curve, dx, dy); }
    for curve in &mut piece.internal_lines { translate_curve(curve, dx, dy); }
    for fold in &mut piece.fold_lines { translate_curve(&mut fold.curve, dx, dy); }
    for pleat in &mut piece.pleats { translate_curve(&mut pleat.axis, dx, dy); }
    if let Some(grain) = &mut piece.grain_line {
        grain.start.x += dx; grain.start.y += dy;
        grain.end.x += dx; grain.end.y += dy;
    }
    for dart in &mut piece.darts {
        dart.apex.x += dx; dart.apex.y += dy;
        dart.leg_a.x += dx; dart.leg_a.y += dy;
        dart.leg_b.x += dx; dart.leg_b.y += dy;
    }
    for hole in &mut piece.holes {
        hole.center.x += dx; hole.center.y += dy;
    }
    for annotation in &mut piece.annotations {
        annotation.position.x += dx; annotation.position.y += dy;
    }
}

fn translate_project_piece(project: &mut PatternProject, piece_id: u64, dx: f64, dy: f64) {
    if let Some(piece) = project.pieces.iter_mut().find(|p| p.id == piece_id) {
        translate_piece(piece, dx, dy);
    }
}

fn map_curve_points(curve: &mut Curve2D, f: &impl Fn(PatternPoint2) -> PatternPoint2) {
    curve.start = f(curve.start);
    for segment in &mut curve.segments {
        match segment {
            CurveSegment::Line { to } => *to = f(*to),
            CurveSegment::CubicBezier { control1, control2, to } => {
                *control1 = f(*control1); *control2 = f(*control2); *to = f(*to);
            }
        }
    }
}
fn map_piece_points(piece: &mut PatternPiece, f: impl Fn(PatternPoint2) -> PatternPoint2) {
    map_curve_points(&mut piece.outline, &f);
    if let Some(curve) = &mut piece.sewing_line { map_curve_points(curve, &f); }
    for curve in &mut piece.internal_lines { map_curve_points(curve, &f); }
    for fold in &mut piece.fold_lines { map_curve_points(&mut fold.curve, &f); }
    for pleat in &mut piece.pleats { map_curve_points(&mut pleat.axis, &f); }
    if let Some(grain) = &mut piece.grain_line { grain.start = f(grain.start); grain.end = f(grain.end); }
    for dart in &mut piece.darts { dart.apex = f(dart.apex); dart.leg_a = f(dart.leg_a); dart.leg_b = f(dart.leg_b); }
    for hole in &mut piece.holes { hole.center = f(hole.center); }
    for annotation in &mut piece.annotations { annotation.position = f(annotation.position); }
}
fn piece_pivot(piece: &PatternPiece) -> PatternPoint2 {
    let anchors = outline_anchors(&piece.outline);
    if anchors.is_empty() { return PatternPoint2::new(0.0, 0.0); }
    let (mut min_x, mut max_x, mut min_y, mut max_y) = (f64::MAX, f64::MIN, f64::MAX, f64::MIN);
    for p in &anchors { min_x=min_x.min(p.x); max_x=max_x.max(p.x); min_y=min_y.min(p.y); max_y=max_y.max(p.y); }
    PatternPoint2::new((min_x+max_x)*0.5,(min_y+max_y)*0.5)
}
fn rotate_piece_about(piece: &mut PatternPiece, pivot: PatternPoint2, angle_radians: f64) {
    let (sin, cos) = angle_radians.sin_cos();
    map_piece_points(piece, |p| {
        let dx=p.x-pivot.x; let dy=p.y-pivot.y;
        PatternPoint2::new(pivot.x+dx*cos-dy*sin,pivot.y+dx*sin+dy*cos)
    });
    for annotation in &mut piece.annotations { annotation.rotation_degrees += angle_radians.to_degrees(); }
}
fn scale_piece_about(piece: &mut PatternPiece, pivot: PatternPoint2, factor: f64) {
    if !factor.is_finite() || factor <= 0.0 { return; }
    map_piece_points(piece, |p| PatternPoint2::new(pivot.x+(p.x-pivot.x)*factor,pivot.y+(p.y-pivot.y)*factor));
    for hole in &mut piece.holes { hole.radius_meters *= factor; }
    for dart in &mut piece.darts { dart.intake_meters *= factor; }
    for pleat in &mut piece.pleats { pleat.depth_meters *= factor; }
}
fn outline_anchors(curve: &Curve2D) -> Vec<PatternPoint2> {
    let mut anchors = vec![curve.start];
    let count = if curve.closed {
        curve.segments.len().saturating_sub(1)
    } else {
        curve.segments.len()
    };
    anchors.extend(curve.segments.iter().take(count).map(CurveSegment::endpoint));
    anchors
}

fn set_curve_anchor(curve: &mut Curve2D, index: usize, point: PatternPoint2) {
    if index == 0 {
        curve.start = point;
        if curve.closed {
            if let Some(last) = curve.segments.last_mut() {
                match last {
                    CurveSegment::Line { to } | CurveSegment::CubicBezier { to, .. } => *to = point,
                }
            }
        }
    } else if let Some(segment) = curve.segments.get_mut(index - 1) {
        match segment {
            CurveSegment::Line { to } | CurveSegment::CubicBezier { to, .. } => *to = point,
        }
    }
}

fn closest_segment_parameter(point: egui::Pos2, a: egui::Pos2, b: egui::Pos2) -> (f32, f32) {
    let ab = b - a;
    let length_squared = ab.length_sq();
    let t = if length_squared <= f32::EPSILON {
        0.0
    } else {
        ((point - a).dot(ab) / length_squared).clamp(0.0, 1.0)
    };
    (point.distance(a + ab * t), t)
}

/// Remapeia um pique quando o segmento que o contém é subdividido por De Casteljau.
fn remap_notches_after_segment_split(notches: &mut [Notch], segment_index: usize, split_t: f64) {
    let split_t = split_t.clamp(0.001, 0.999);
    for notch in notches {
        let old_segment = notch.segment_index as usize;
        if old_segment < segment_index {
            continue;
        }
        if old_segment > segment_index {
            notch.segment_index += 1;
            continue;
        }
        let old_t = notch.t.clamp(0.0, 1.0);
        if old_t <= split_t {
            notch.segment_index = segment_index as u32;
            notch.t = (old_t / split_t).clamp(0.0, 1.0);
        } else {
            notch.segment_index = (segment_index + 1) as u32;
            notch.t = ((old_t - split_t) / (1.0 - split_t)).clamp(0.0, 1.0);
        }
    }
}

/// Remapeia os piques para a nova topologia após excluir uma âncora num contorno linear.
/// O(s) segmento(s) adjacente(s) tornam-se uma única aresta; posições são preservadas como
/// fração do comprimento de arco antigo e reparametrizadas no segmento resultante.
fn remap_notches_after_anchor_delete(
    notches: &mut [Notch],
    anchor_index: usize,
    segment_count: usize,
    lengths: &[f64],
) {
    if segment_count < 2 || lengths.len() != segment_count { return; }
    let ratio = |first: f64, second: f64, t: f64| {
        let total = first + second;
        if total <= 1.0e-12 { 0.5 }
        else { ((first + second * t.clamp(0.0, 1.0)) / total).clamp(0.0, 1.0) }
    };
    for notch in notches {
        let old_segment = notch.segment_index as usize;
        let old_t = notch.t.clamp(0.0, 1.0);
        if anchor_index == 0 {
            let merged_index = (segment_count - 2) as u32;
            let denominator = (lengths[segment_count - 1] + lengths[0]).max(1.0e-12);
            if old_segment == segment_count - 1 {
                notch.segment_index = merged_index;
                notch.t = (old_t * lengths[segment_count - 1] / denominator).clamp(0.0, 1.0);
            } else if old_segment == 0 {
                notch.segment_index = merged_index;
                notch.t = ((lengths[segment_count - 1] + old_t * lengths[0]) / denominator)
                    .clamp(0.0, 1.0);
            } else {
                notch.segment_index = (old_segment - 1) as u32;
            }
        } else if old_segment == anchor_index - 1 {
            notch.segment_index = (anchor_index - 1) as u32;
            notch.t = (old_t * lengths[anchor_index - 1]
                / (lengths[anchor_index - 1] + lengths[anchor_index]).max(1.0e-12))
                .clamp(0.0, 1.0);
        } else if old_segment == anchor_index {
            notch.segment_index = (anchor_index - 1) as u32;
            notch.t = ratio(lengths[anchor_index - 1], lengths[anchor_index], old_t);
        } else if old_segment > anchor_index {
            notch.segment_index -= 1;
        }
    }
}

fn split_project_segment(
    project: &mut PatternProject,
    piece_id: u64,
    segment_index: usize,
    t: f64,
) -> Result<usize, String> {
    // Edição transacional: um pique ou intervalo de costura inválido não deixa o projeto
    // parcialmente subdividido.
    let mut staged = project.clone();
    let piece = staged.pieces.iter_mut().find(|p| p.id == piece_id)
        .ok_or_else(|| "Peça não encontrada.".to_string())?;
    if segment_index >= piece.outline.segments.len() {
        return Err("Segmento selecionado fora do contorno.".into());
    }
    let t = t.clamp(0.001, 0.999);
    remap_notches_after_segment_split(&mut piece.notches, segment_index, t);
    let curve = &mut piece.outline;
    let p0 = if segment_index == 0 {
        curve.start
    } else {
        curve.segments[segment_index - 1].endpoint()
    };
    let original = curve.segments[segment_index].clone();
    let (left, right) = match original {
        CurveSegment::Line { to } => {
            let split = PatternPoint2::new(
                p0.x + (to.x - p0.x) * t,
                p0.y + (to.y - p0.y) * t,
            );
            (CurveSegment::Line { to: split }, CurveSegment::Line { to })
        }
        CurveSegment::CubicBezier { control1, control2, to } => {
            let lerp = |a: PatternPoint2, b: PatternPoint2| PatternPoint2::new(
                a.x + (b.x - a.x) * t,
                a.y + (b.y - a.y) * t,
            );
            let p01 = lerp(p0, control1);
            let p12 = lerp(control1, control2);
            let p23 = lerp(control2, to);
            let p012 = lerp(p01, p12);
            let p123 = lerp(p12, p23);
            let split = lerp(p012, p123);
            (
                CurveSegment::CubicBezier { control1: p01, control2: p012, to: split },
                CurveSegment::CubicBezier { control1: p123, control2: p23, to },
            )
        }
    };
    curve.segments[segment_index] = left;
    curve.segments.insert(segment_index + 1, right);

    // Inserir um vértice desloca os índices das âncoras posteriores, inclusive as que estão
    // fixadas para a montagem 3D.
    let piece = staged.pieces.iter_mut().find(|p| p.id == piece_id).unwrap();
    let new_anchor_index = segment_index + 1;
    for anchor in &mut piece.assembly.pinned_anchors {
        if *anchor as usize >= new_anchor_index {
            *anchor += 1;
        }
    }

    // Reparametriza as extremidades sobre os segmentos Bézier/lineares recém-criados.
    for seam in &mut staged.seams {
        for edge in seam.sides_a.iter_mut().chain(seam.sides_b.iter_mut()) {
            if edge.piece_id != piece_id || edge.curve != BoundaryCurve::Outline { continue; }
            remap_param_after_segment_split(&mut edge.start, segment_index, t);
            remap_param_after_segment_split(&mut edge.end, segment_index, t);
        }
    }
    staged.validate()?;
    *project = staged;
    Ok(segment_index + 1)
}

fn remap_param_after_segment_split(param: &mut CurveParam, segment_index: usize, split_t: f64) {
    let index = segment_index as u32;
    if param.segment > index {
        param.segment += 1;
    } else if param.segment == index {
        if param.t < split_t - 1.0e-9 {
            param.t = (param.t / split_t).clamp(0.0, 1.0);
        } else if param.t > split_t + 1.0e-9 {
            param.segment += 1;
            param.t = ((param.t - split_t) / (1.0 - split_t)).clamp(0.0, 1.0);
        } else {
            param.t = 1.0;
        }
    }
}

fn convert_project_segment_to_bezier(
    project: &mut PatternProject,
    piece_id: u64,
    segment_index: usize,
) -> Result<(), String> {
    let piece = project.pieces.iter_mut().find(|p| p.id == piece_id)
        .ok_or_else(|| "Peça não encontrada.".to_string())?;
    let curve = &mut piece.outline;
    let segment = curve.segments.get(segment_index)
        .ok_or_else(|| "Segmento selecionado fora do contorno.".to_string())?;
    let end = segment.endpoint();
    let start = if segment_index == 0 {
        curve.start
    } else {
        curve.segments[segment_index - 1].endpoint()
    };
    let one_third = |a: PatternPoint2, b: PatternPoint2, t: f64| PatternPoint2::new(
        a.x + (b.x - a.x) * t,
        a.y + (b.y - a.y) * t,
    );
    curve.segments[segment_index] = CurveSegment::CubicBezier {
        control1: one_third(start, end, 1.0 / 3.0),
        control2: one_third(start, end, 2.0 / 3.0),
        to: end,
    };
    project.validate()?;
    Ok(())
}

fn delete_project_anchor(
    project: &mut PatternProject,
    piece_id: u64,
    anchor_index: usize,
) -> Result<(), String> {
    // Usa uma cópia transacional: se o remapeamento de costuras não puder ser representado
    // como intervalo contíguo, o projeto original permanece intacto.
    let mut staged = project.clone();
    let (n, segment_lengths) = {
        let piece = staged.pieces.iter().find(|p| p.id == piece_id)
            .ok_or_else(|| "Peça não encontrada.".to_string())?;
        let curve = &piece.outline;
        if !curve.closed || curve.segments.len() < 4 {
            return Err("O contorno deve ter pelo menos quatro vértices para excluir um ponto.".into());
        }
        if curve.segments.iter().any(|s| !matches!(s, CurveSegment::Line { .. })) {
            return Err("A exclusão de vértices está disponível para contornos lineares; em Bézier use Inserir vértice e edite as alças.".into());
        }
        let n = curve.segments.len();
        if anchor_index >= n { return Err("Vértice selecionado fora do contorno.".into()); }
        let mut lengths = Vec::with_capacity(n);
        for index in 0..n {
            let start = if index == 0 { curve.start } else { curve.segments[index - 1].endpoint() };
            lengths.push(start.distance(curve.segments[index].endpoint()));
        }
        (n, lengths)
    };

    {
        let piece = staged.pieces.iter_mut().find(|p| p.id == piece_id).unwrap();
        remap_notches_after_anchor_delete(
            &mut piece.notches, anchor_index, n, &segment_lengths,
        );
    }

    let map_segment = |old: usize| -> usize {
        if anchor_index == 0 {
            if old == 0 || old == n - 1 { n - 2 } else { old - 1 }
        } else if old < anchor_index - 1 {
            old
        } else if old <= anchor_index {
            anchor_index - 1
        } else {
            old - 1
        }
    };

    for seam in &mut staged.seams {
        for edge in seam.sides_a.iter_mut().chain(seam.sides_b.iter_mut()) {
            if edge.piece_id != piece_id || edge.curve != BoundaryCurve::Outline { continue; }
            let start_segment = edge.start.segment as usize;
            let end_segment = edge.end.segment as usize;
            if start_segment >= n || end_segment >= n || start_segment > end_segment {
                return Err("Uma costura referencia o segmento que seria removido.".into());
            }
            let affected = if anchor_index == 0 {
                [Some(0usize), Some(n - 1)]
            } else {
                [Some(anchor_index - 1), Some(anchor_index)]
            };
            for param in [&edge.start, &edge.end] {
                if affected.iter().flatten().any(|&segment| segment == param.segment as usize)
                    && param.t > 1.0e-9 && param.t < 1.0 - 1.0e-9
                {
                    return Err("Não é possível excluir o vértice porque uma extremidade de costura livre está em um dos segmentos que seriam unidos.".into());
                }
            }
            let original: Vec<usize> = (start_segment..=end_segment)
                .map(map_segment)
                .collect();
            if original.is_empty() {
                return Err("Uma costura referencia o segmento que seria removido.".into());
            }
            let mut unique = original.clone();
            unique.sort_unstable();
            unique.dedup();
            let min = *unique.first().unwrap();
            let max = *unique.last().unwrap();
            if max - min + 1 != unique.len() {
                return Err("Não é possível excluir esse vértice sem fragmentar um intervalo de costura. Ajuste a costura primeiro.".into());
            }
            edge.start.segment = map_segment(start_segment) as u32;
            edge.end.segment = map_segment(end_segment) as u32;
            if edge.start.segment > edge.end.segment {
                return Err("A exclusão fragmentaria a ordem dos parâmetros da costura.".into());
            }
        }
    }

    let curve = &mut staged.pieces.iter_mut().find(|p| p.id == piece_id).unwrap().outline;
    if anchor_index == 0 {
        let new_start = curve.segments[0].endpoint();
        curve.start = new_start;
        curve.segments.remove(0);
        if let Some(last) = curve.segments.last_mut() {
            match last {
                CurveSegment::Line { to } | CurveSegment::CubicBezier { to, .. } => *to = new_start,
            }
        }
    } else {
        let next = curve.segments[anchor_index].endpoint();
        match &mut curve.segments[anchor_index - 1] {
            CurveSegment::Line { to } => *to = next,
            _ => unreachable!("contornos Bézier foram rejeitados acima"),
        }
        curve.segments.remove(anchor_index);
    }
    // O ponto que deixa de existir perde a fixação; os vértices posteriores mudam de índice.
    {
        let piece = staged.pieces.iter_mut().find(|piece| piece.id == piece_id).unwrap();
        piece.assembly.pinned_anchors.retain(|anchor| *anchor as usize != anchor_index);
        for anchor in &mut piece.assembly.pinned_anchors {
            if *anchor as usize > anchor_index {
                *anchor -= 1;
            }
        }
    }
    staged.validate()?;
    *project = staged;
    Ok(())
}

fn forced_parameter_indices(
    piece: &PatternPiece,
    sampled: &pattern_geom::SampledCurve,
    params: &[(u32, f64)],
) -> Result<Vec<usize>, String> {
    if sampled.points.is_empty() { return Err("Contorno amostrado vazio.".into()); }
    let mut indices = Vec::with_capacity(params.len());
    for &(segment, t) in params {
        let segment_index = segment as usize;
        let (start, end) = *sampled.segment_vertex_ranges.get(segment_index)
            .ok_or_else(|| "Parâmetro referencia um segmento inexistente.".to_string())?;
        let target = curve_segment_point(&piece.outline, segment_index, t)
            .ok_or_else(|| "Não foi possível avaliar o parâmetro da curva.".to_string())?;
        let candidates: Box<dyn Iterator<Item = usize>> = if end >= start {
            Box::new(start..=end)
        } else {
            Box::new((start..sampled.points.len()).chain(0..=end))
        };
        let index = candidates
            .map(|index| index % sampled.points.len())
            .find(|&index| sampled.points[index].distance(target) <= 1.0e-7)
            .ok_or_else(|| format!(
                "O parâmetro (segmento {}, t={:.6}) não foi preservado exatamente pela amostragem.",
                segment + 1, t
            ))?;
        indices.push(index);
    }
    Ok(indices)
}

fn sampled_edge_indices(
    edge: &EdgePath,
    ranges: &[(usize, usize)],
    forced_vertices: &[(u32, f64, usize)],
    boundary_count: usize,
) -> Result<Vec<u32>, String> {
    if boundary_count < 2 || edge.start_position() >= edge.end_position() {
        return Err("Intervalo de costura sem vértices suficientes.".into());
    }
    if edge.end.segment as usize >= ranges.len() {
        return Err("Intervalo de costura fora da curva amostrada.".into());
    }

    let vertex_for_param = |param: CurveParam| -> Option<usize> {
        if param.t.abs() <= 1.0e-9 {
            return ranges.get(param.segment as usize).map(|range| range.0);
        }
        if (param.t - 1.0).abs() <= 1.0e-9 {
            return ranges.get(param.segment as usize).map(|range| range.1 % boundary_count);
        }
        forced_vertices.iter().find_map(|(segment, t, index)| {
            (*segment == param.segment && (*t - param.t).abs() <= 1.0e-9).then_some(*index)
        })
    };
    let start_index = vertex_for_param(edge.start)
        .ok_or_else(|| "O início da costura livre não está presente como vértice forçado na malha.".to_string())?;
    let end_index = vertex_for_param(edge.end)
        .ok_or_else(|| "O fim da costura livre não está presente como vértice forçado na malha.".to_string())?;

    let append = |indices: &mut Vec<u32>, index: usize| {
        let wrapped = (index % boundary_count) as u32;
        if indices.last().copied() != Some(wrapped) { indices.push(wrapped); }
    };
    let mut indices: Vec<u32> = Vec::new();
    for segment_index in edge.start.segment as usize..=edge.end.segment as usize {
        let (range_start, range_end) = ranges[segment_index];
        let begin = if segment_index == edge.start.segment as usize { start_index } else { range_start };
        let finish = if segment_index == edge.end.segment as usize { end_index } else { range_end % boundary_count };
        if begin <= finish {
            for index in begin..=finish { append(&mut indices, index); }
        } else {
            for index in begin..boundary_count { append(&mut indices, index); }
            for index in 0..=finish { append(&mut indices, index); }
        }
    }
    if edge.reversed { indices.reverse(); }
    indices.dedup();
    if indices.len() < 2 {
        return Err("A borda selecionada contém menos de dois vértices.".into());
    }
    Ok(indices)
}

fn metrics_panel(app: &ClothEvoApp, ui: &mut egui::Ui) {
        let sim = &app.sim;
        ui.horizontal_wrapped(|ui| {
            ui.label(egui::RichText::new("MÉTRICAS / SOLVER").strong().color(egui::Color32::from_rgb(255, 170, 53)));
            ui.separator();
            ui.label(format!("{} partículas", sim.particle_count()));
            ui.label(format!("{} restrições de distância · {} de flexão",
                sim.distance_constraint_count(), sim.bend_constraint_count()));
            ui.label(format!("passo {:.2} ms / 16,67 ms", app.last_step_ms));
            ui.label(format!("deformação {:.1} %", sim.max_strain() * 100.0));
            ui.label(format!("energia {:.4} J", sim.kinetic_energy()));
            ui.label(format!("velocidade máx. {:.2} m/s", sim.max_particle_speed()));
        });
        ui.separator();
        ui.columns(3, |columns| {
            columns[0].strong("CORPO");
            columns[0].label(format!("{} projeções · {} testes de trajetória · {} contatos CCD",
                sim.stats.body_contact_projections, sim.stats.body_sweep_tests, sim.stats.body_swept_contacts));
            columns[0].label(format!("penetração máx. {:.2} mm · {:.2} ms",
                sim.stats.body_max_penetration_m * 1000.0, sim.stats.body_collision_ms));

            columns[1].strong("AUTO-COLISÃO");
            columns[1].label(format!("VV {}/{} · VT {}/{} · EE {}/{} contatos/candidatos",
                sim.stats.self_collision_vv_contacts, sim.stats.self_collision_vv_candidates,
                sim.stats.self_collision_vt_contacts, sim.stats.self_collision_vt_candidates,
                sim.stats.self_collision_ee_contacts, sim.stats.self_collision_ee_candidates));
            columns[1].label(format!("CCD VT {}/{} · EE {}/{}",
                sim.stats.self_collision_vt_swept_contacts, sim.stats.self_collision_vt_sweep_tests,
                sim.stats.self_collision_ee_swept_contacts, sim.stats.self_collision_ee_sweep_tests));
            columns[1].label(format!("candidatos descartados {} · penetração {:.2} → {:.2} mm",
                sim.stats.self_collision_dropped_candidates,
                sim.stats.self_collision_penetration_before_m * 1000.0,
                sim.stats.self_collision_penetration_after_m * 1000.0));

            columns[2].strong("ATRITO E COSTURAS");
            columns[2].label(format!("{} contatos · {} estáticos · {} deslizando",
                sim.stats.contact_friction_contacts,
                sim.stats.contact_friction_static_contacts,
                sim.stats.contact_friction_sliding_contacts));
            columns[2].label(format!("velocidade tangencial {:.2} m/s",
                sim.stats.contact_max_tangential_speed_m_s));
            if !app.seam_pairs.is_empty() {
                columns[2].label(format!("abertura máx. {:.1} mm · tensão XPBD estimada {:.2} N",
                    app.seam_gap_max() * 1000.0, sim.seam_tension_estimate_max_n()));
            } else {
                columns[2].label("Nenhuma costura ativa.");
            }
            let unstable = sim.unstable_particle_count(sim.params.max_particle_speed.clamp(1.0, 100.0));
            if unstable > 0 {
                columns[2].colored_label(egui::Color32::from_rgb(255, 92, 92), format!("Partículas instáveis: {unstable}"));
            }
        });
        ui.weak("A carga de costura é uma estimativa do multiplicador XPBD, não uma força experimental calibrada.");
    }

impl eframe::App for ClothEvoApp {
    fn ui(&mut self, ui: &mut egui::Ui, _frame: &mut eframe::Frame) {
        if self.ui_theme_dirty {
            self.apply_ui_customization(ui.ctx());
            self.ui_theme_dirty = false;
        }
        self.poll_avatar_import();
        if let Some(goal) = self.camera_goal {
            let camera_dt = ui.input(|i| i.stable_dt);
            if self.camera.approach(&goal, camera_dt) { self.camera_goal = None; }
            else { ui.ctx().request_repaint(); }
        }
        if self.workspace == WorkspaceMode::Pattern {
            let (undo, redo) = ui.input(|i| (
                i.modifiers.ctrl && i.key_pressed(egui::Key::Z) && !i.modifiers.shift,
                (i.modifiers.ctrl && i.key_pressed(egui::Key::Y))
                    || (i.modifiers.ctrl && i.modifiers.shift && i.key_pressed(egui::Key::Z)),
            ));
            if undo { self.undo_pattern_edit(); }
            if redo { self.redo_pattern_edit(); }
        }
        // Avança depois de processar os controles deste frame. Assim, "Passo único" pausa
        // antes do solver rodar e nunca recebe um passo automático adicional no mesmo frame.
        let dt = if self.capture.is_some() { FIXED_DT } else { ui.input(|i| i.stable_dt) };
        let was_simulation_workspace = self.workspace.runs_simulation();
        let diverged_before = self.diverged;
        let steps_before = self.sim.stats.steps;

        egui::Panel::top("main_navigation")
            .frame(egui::Frame::new()
                .fill(egui::Color32::from_rgb(18, 54, 150))
                .stroke(egui::Stroke::new(1.0, egui::Color32::from_rgb(255, 170, 53)))
                .inner_margin(egui::Margin::symmetric(14, 8)))
            .show(ui, |ui| self.top_navigation(ui));

        egui::Panel::top("global_toolbar")
            .frame(egui::Frame::new()
                .fill(egui::Color32::from_rgb(20, 25, 39))
                .stroke(egui::Stroke::new(1.0, egui::Color32::from_rgb(54, 63, 91)))
                .inner_margin(egui::Margin::symmetric(14, 5)))
            .show(ui, |ui| self.global_toolbar(ui));

        if self.ui_customization.toolbar_buttons.iter().any(|button| button.visible) {
            egui::Panel::top("custom_command_toolbar")
                .frame(egui::Frame::new()
                    .fill(ui_rgb(self.ui_customization.theme.panel_bg))
                    .inner_margin(egui::Margin::symmetric(10, 3)))
                .show(ui, |ui| self.custom_toolbar(ui));
        }

        if self.show_metrics {
            egui::Panel::bottom("view_metrics")
                .resizable(true)
                .default_size(155.0)
                .frame(egui::Frame::new()
                    .fill(egui::Color32::from_rgb(12, 16, 29))
                    .stroke(egui::Stroke::new(1.0, egui::Color32::from_rgb(54, 63, 91)))
                    .inner_margin(egui::Margin::same(10)))
                .show(ui, |ui| {
                    ui.horizontal(|ui| {
                        ui.label(egui::RichText::new("Métricas").strong());
                        if ui.small_button("Fechar ×").clicked() {
                            self.show_metrics = false;
                        }
                    });
                    egui::ScrollArea::vertical().show(ui, |ui| metrics_panel(self, ui));
                });
        }

        egui::Panel::left("controls").default_size(310.0).show(ui, |ui| {
            egui::ScrollArea::vertical().show(ui, |ui| self.controls(ui));
        });
        if was_simulation_workspace {
            self.advance(dt);
        }
        if self.sim.stats.steps != steps_before {
            self.last_step_ms = self.sim.stats.step_ms;
        }

        egui::CentralPanel::default().frame(egui::Frame::NONE).show(ui, |ui| {
            match self.workspace {
                WorkspaceMode::Pattern => self.pattern_viewport(ui),
                WorkspaceMode::Simulation | WorkspaceMode::Placement => self.viewport(ui),
                WorkspaceMode::Split => self.split_viewports(ui),
            }
        });

        self.ui_customizer_window(ui.ctx());
        self.handle_capture(ui);
        if self.diverged && !diverged_before {
            // O diagnóstico foi definido após renderizar o painel de controles deste frame.
            ui.ctx().request_repaint();
        }
        if self.avatar_import_pending_path.is_some() {
            ui.ctx().request_repaint_after(std::time::Duration::from_millis(100));
        }
        if (self.playing && self.workspace.runs_simulation()) || self.capture.is_some() {
            ui.ctx().request_repaint();
        }
    }
}



fn ui_rgb(rgb: [u8; 3]) -> egui::Color32 {
    egui::Color32::from_rgb(rgb[0], rgb[1], rgb[2])
}

fn edit_ui_rgb(ui: &mut egui::Ui, label: &str, rgb: &mut [u8; 3]) -> bool {
    ui.horizontal(|ui| {
        ui.label(label);
        let mut color = ui_rgb(*rgb);
        let changed = ui.color_edit_button_srgba(&mut color).changed();
        if changed {
            *rgb = [color.r(), color.g(), color.b()];
        }
        changed
    }).inner
}

fn gizmo_axis_color(axis: usize) -> egui::Color32 {
    match axis {
        0 => egui::Color32::from_rgb(245, 86, 83),
        1 => egui::Color32::from_rgb(103, 220, 125),
        _ => egui::Color32::from_rgb(83, 164, 255),
    }
}

fn placement_toolbar_bounds(viewport_rect: egui::Rect) -> egui::Rect {
    const BUTTON_SIDE: f32 = 36.0;
    const BUTTON_GAP: f32 = 7.0;
    const PANEL_WIDTH: f32 = 50.0;
    const PANEL_PADDING: f32 = 7.0;
    let panel_height = PANEL_PADDING * 2.0 + BUTTON_SIDE * 4.0 + BUTTON_GAP * 3.0;
    egui::Rect::from_min_size(
        viewport_rect.left_top() + egui::vec2(12.0, 12.0),
        egui::vec2(PANEL_WIDTH, panel_height),
    )
}

fn paint_placement_tool_icon(
    painter: &egui::Painter,
    rect: egui::Rect,
    mode: GizmoMode,
    color: egui::Color32,
) {
    let center = rect.center();
    let stroke = egui::Stroke::new(2.0, color);
    let arrow = |tip: egui::Pos2, direction: egui::Vec2| {
        let direction = direction.normalized();
        if direction.length_sq() < 1.0e-6 { return; }
        let base = tip - direction * 4.5;
        let perpendicular = egui::vec2(-direction.y, direction.x) * 2.8;
        painter.line_segment([tip, base + perpendicular], stroke);
        painter.line_segment([tip, base - perpendicular], stroke);
    };

    match mode {
        GizmoMode::Rotate => {
            painter.circle_stroke(center, 8.0, stroke);
            let tip = center + egui::vec2(6.5, -5.0);
            arrow(tip, egui::vec2(1.0, -0.35));
        }
        GizmoMode::Move => {
            painter.line_segment([center - egui::vec2(10.0, 0.0), center + egui::vec2(10.0, 0.0)], stroke);
            painter.line_segment([center - egui::vec2(0.0, 10.0), center + egui::vec2(0.0, 10.0)], stroke);
            arrow(center + egui::vec2(12.0, 0.0), egui::vec2(1.0, 0.0));
            arrow(center - egui::vec2(12.0, 0.0), egui::vec2(-1.0, 0.0));
            arrow(center + egui::vec2(0.0, 12.0), egui::vec2(0.0, 1.0));
            arrow(center - egui::vec2(0.0, 12.0), egui::vec2(0.0, -1.0));
        }
        GizmoMode::Scale => {
            let start = center + egui::vec2(-9.0, 9.0);
            let end = center + egui::vec2(9.0, -9.0);
            painter.line_segment([start, end], stroke);
            arrow(start, egui::vec2(-1.0, 1.0));
            arrow(end, egui::vec2(1.0, -1.0));
        }
        GizmoMode::Transform => {
            let origin = center + egui::vec2(-6.5, 6.5);
            let x_end = origin + egui::vec2(13.5, 0.0);
            let y_end = origin + egui::vec2(0.0, -13.5);
            let z_end = origin + egui::vec2(9.0, -9.0);
            painter.line_segment([origin, x_end], stroke);
            painter.line_segment([origin, y_end], stroke);
            painter.line_segment([origin, z_end], stroke);
            arrow(x_end, egui::vec2(1.0, 0.0));
            arrow(y_end, egui::vec2(0.0, -1.0));
            arrow(z_end, egui::vec2(1.0, -1.0));
        }
    }
}

fn draw_gizmo_arrow_head(
    painter: &egui::Painter,
    tip: egui::Pos2,
    direction: egui::Vec2,
    color: egui::Color32,
) {
    let direction = direction.normalized();
    if direction.length_sq() < 1.0e-6 { return; }
    let perpendicular = egui::vec2(-direction.y, direction.x);
    let base = tip - direction * 10.0;
    painter.add(egui::Shape::convex_polygon(
        vec![
            tip,
            base + perpendicular * 4.0,
            base - perpendicular * 4.0,
        ],
        color,
        egui::Stroke::NONE,
    ));
}

/// Distância do ponteiro ao segmento e parâmetro projetado t em [0, 1].
fn point_segment_distance(
    point: egui::Pos2,
    a: egui::Pos2,
    b: egui::Pos2,
) -> (f32, f32) {
    let ab = b - a;
    let length_squared = ab.length_sq();
    if length_squared <= 1.0e-6 {
        return (point.distance(a), 0.0);
    }
    let t = ((point - a).dot(ab) / length_squared).clamp(0.0, 1.0);
    let closest = a + ab * t;
    (point.distance(closest), t)
}

fn projected_rotation_ring(
    center: Vec3,
    axis: Vec3,
    radius: f32,
    view_proj: glam::Mat4,
    rect: egui::Rect,
) -> Vec<egui::Pos2> {
    let axis = axis.normalize_or_zero();
    let helper = if axis.dot(Vec3::Y).abs() > 0.88 { Vec3::X } else { Vec3::Y };
    let u = axis.cross(helper).normalize_or_zero();
    let v = axis.cross(u).normalize_or_zero();
    if u.length_squared() < 1.0e-8 || v.length_squared() < 1.0e-8 { return Vec::new(); }
    let mut points = Vec::with_capacity(49);
    for step in 0..=48 {
        let angle = std::f32::consts::TAU * step as f32 / 48.0;
        let world = center + (u * angle.cos() + v * angle.sin()) * radius;
        if let Some((screen, _)) = project_world_to_screen(world, view_proj, rect) {
            points.push(screen);
        }
    }
    points
}

/// Projeta um ponto 3D para a viewport egui, mantendo a convenção de profundidade [0, 1] do wgpu.
fn project_world_to_screen(
    point: Vec3,
    view_proj: glam::Mat4,
    rect: egui::Rect,
) -> Option<(egui::Pos2, f32)> {
    let clip = view_proj * point.extend(1.0);
    if !clip.w.is_finite() || clip.w <= 1.0e-6 {
        return None;
    }
    let ndc = clip.truncate() / clip.w;
    if !ndc.is_finite() || !(0.0..=1.0).contains(&ndc.z) {
        return None;
    }
    Some((
        egui::pos2(
            rect.left() + (ndc.x + 1.0) * 0.5 * rect.width(),
            rect.top() + (1.0 - ndc.y) * 0.5 * rect.height(),
        ),
        ndc.z,
    ))
}

fn point_in_triangle(point: egui::Pos2, a: egui::Pos2, b: egui::Pos2, c: egui::Pos2) -> bool {
    let cross = |p1: egui::Pos2, p2: egui::Pos2, p3: egui::Pos2| {
        (p2.x - p1.x) * (p3.y - p1.y) - (p2.y - p1.y) * (p3.x - p1.x)
    };
    let d1 = cross(point, a, b);
    let d2 = cross(point, b, c);
    let d3 = cross(point, c, a);
    let has_negative = d1 < -0.5 || d2 < -0.5 || d3 < -0.5;
    let has_positive = d1 > 0.5 || d2 > 0.5 || d3 > 0.5;
    !(has_negative && has_positive)
}

#[cfg(test)]
mod pattern_editor_tests {
    use super::*;

    fn square_project() -> PatternProject {
        let mut project = PatternProject::new("Teste de edição");
        let square = [
            PatternPoint2::new(0.0, 0.0),
            PatternPoint2::new(1.0, 0.0),
            PatternPoint2::new(1.0, 1.0),
            PatternPoint2::new(0.0, 1.0),
        ];
        project.pieces.push(PatternPiece::from_polygon(1, "A", &square).unwrap());
        project.pieces.push(PatternPiece::from_polygon(
            2,
            "B",
            &[
                PatternPoint2::new(1.0, 0.0),
                PatternPoint2::new(2.0, 0.0),
                PatternPoint2::new(2.0, 1.0),
                PatternPoint2::new(1.0, 1.0),
            ],
        ).unwrap());
        project.seams.push(SeamDefinition {
            id: 1,
            name: "A-B".into(),
            sides_a: vec![EdgeRef::single_segment(1, 1).into()],
            sides_b: vec![EdgeRef::single_segment(2, 3).into()],
            ease_meters: 0.0,
        });
        project
    }

    #[test]
    fn simulation_runs_whenever_the_3d_viewport_is_visible() {
        assert!(WorkspaceMode::Simulation.runs_simulation());
        assert!(WorkspaceMode::Split.runs_simulation());
        assert!(!WorkspaceMode::Pattern.runs_simulation());
    }

    #[test]
    fn seam_measurements_and_notch_markers_follow_edge_arc_and_direction() {
        let mut project = square_project();
        project.pieces[0].notches.push(Notch {
            id: 1, segment_index: 1, t: 0.5, depth_meters: 0.008, width_meters: 0.012,
        });
        project.pieces[1].notches.push(Notch {
            id: 1, segment_index: 3, t: 0.5, depth_meters: 0.008, width_meters: 0.012,
        });
        let seam = project.seams[0].clone();
        assert!((edge_length_meters(&project, &seam.sides_a[0]) - 1.0).abs() < 1e-6);
        assert!((edge_length_meters(&project, &seam.sides_b[0]) - 1.0).abs() < 1e-6);
        assert_eq!(paired_notch_markers(&project, &seam.sides_a[0], &seam.sides_b[0]), vec![(0.5, 0.5)]);

        let mut reversed = seam.sides_b[0].clone();
        reversed.reversed = true;
        assert_eq!(edge_notch_positions(&project, &reversed), vec![0.5]);
    }

    #[test]
    fn pattern_sim_shell_layers_start_outside_the_body_collision_shell() {
        let thickness = 0.005;
        for index in 0..12 {
            let shell_radius = pattern_sim_shell_radius(index, thickness);
            // perturb() pode deslocar Z para ambos os lados em até sua amplitude.
            let lowest_radius = shell_radius - PATTERN_SIM_PERTURBATION;
            assert!(
                lowest_radius >= PATTERN_SIM_BODY_RADIUS + thickness - 1e-6,
                "a camada {index} começa dentro do colisor: raio mínimo {lowest_radius}"
            );
        }
    }

    #[test]
    fn cut_seam_remapping_keeps_untouched_edges_and_rejects_split_edges() {
        let outline = rectangle(
            PatternPoint2::new(0.0, 0.0),
            PatternPoint2::new(2.0, 1.0),
        ).unwrap();
        let path = Curve2D::polyline(&[
            PatternPoint2::new(1.0, 0.0),
            PatternPoint2::new(1.0, 1.0),
        ], false).unwrap();
        let cut = cut_closed_outline(&outline, &path, 0.02).unwrap();

        let mut untouched: EdgePath = EdgeRef::single_segment(1, 1).into();
        remap_edge_after_cut(&mut untouched, &cut.source_segment_fragments, 1, 10).unwrap();
        assert_eq!(untouched.piece_id, 1);
        assert_eq!(untouched.start.segment, 1);
        assert_eq!(untouched.end.segment, 1);

        let mut divided: EdgePath = EdgeRef::single_segment(1, 0).into();
        assert!(remap_edge_after_cut(&mut divided, &cut.source_segment_fragments, 1, 10).is_err());
    }

    #[test]
    fn physical_fold_lines_map_to_internal_bending_edges() {
        let mut piece = PatternPiece::from_polygon(21, "Painel com dobra", &[
            PatternPoint2::new(0.0, 0.0),
            PatternPoint2::new(1.0, 0.0),
            PatternPoint2::new(1.0, 1.0),
            PatternPoint2::new(0.0, 1.0),
        ]).unwrap();
        piece.fabric.particle_distance_meters = 0.20;
        piece.fold_lines.push(FoldLine {
            curve: Curve2D::polyline(&[
                PatternPoint2::new(0.5, 0.08),
                PatternPoint2::new(0.5, 0.92),
            ], false).unwrap(),
            kind: FoldKind::Valley,
            stiffness: 0.35,
            angle_radians: std::f64::consts::FRAC_PI_6,
        });
        let sampled = sample_pattern_outlines_with_seam_sync(&[piece.clone()], &[]).unwrap();
        let mesh = triangulate_outline(
            &sampled[&21].points,
            piece.fabric.particle_distance_meters,
        ).unwrap();
        let folds = physical_fold_edges(&mesh.vertices, &mesh.triangles, &piece);
        assert!(!folds.is_empty(), "a dobra deve mapear para dobradiças internas da malha");
        assert!(folds.iter().all(|(_, _, stiffness, angle)| {
            (*stiffness - 0.35).abs() < 1.0e-6
                && (angle.abs() - std::f32::consts::FRAC_PI_6).abs() < 1.0e-5
        }));
    }

    #[test]
    fn pleat_creates_two_opposing_physical_crease_targets() {
        let mut piece = PatternPiece::from_polygon(22, "Painel com prega", &[
            PatternPoint2::new(0.0, 0.0),
            PatternPoint2::new(1.0, 0.0),
            PatternPoint2::new(1.0, 1.0),
            PatternPoint2::new(0.0, 1.0),
        ]).unwrap();
        piece.fabric.particle_distance_meters = 0.20;
        piece.pleats.push(Pleat {
            axis: Curve2D::polyline(&[
                PatternPoint2::new(0.1, 0.5),
                PatternPoint2::new(0.9, 0.5),
            ], false).unwrap(),
            depth_meters: 0.20,
            angle_radians: 0.50,
        });
        let sampled = sample_pattern_outlines_with_seam_sync(&[piece.clone()], &[]).unwrap();
        let mesh = triangulate_outline(
            &sampled[&22].points,
            piece.fabric.particle_distance_meters,
        ).unwrap();
        let folds = physical_fold_edges(&mesh.vertices, &mesh.triangles, &piece);
        assert!(!folds.is_empty(), "a prega deve mapear seus dois vincos para a malha");
        assert!(folds.iter().any(|(_, _, _, angle)| *angle > 0.1));
        assert!(folds.iter().any(|(_, _, _, angle)| *angle < -0.1));
    }

    #[test]
    fn splitting_segment_remaps_notches_on_both_sides_and_following_segments() {
        let mut project = square_project();
        project.pieces[0].notches = vec![
            Notch { id: 1, segment_index: 1, t: 0.25, depth_meters: 0.008, width_meters: 0.012 },
            Notch { id: 2, segment_index: 1, t: 0.75, depth_meters: 0.008, width_meters: 0.012 },
            Notch { id: 3, segment_index: 3, t: 0.60, depth_meters: 0.008, width_meters: 0.012 },
        ];

        split_project_segment(&mut project, 1, 1, 0.5).unwrap();

        let notches = &project.pieces[0].notches;
        assert_eq!((notches[0].segment_index, notches[0].t), (1, 0.5));
        assert_eq!((notches[1].segment_index, notches[1].t), (2, 0.5));
        assert_eq!((notches[2].segment_index, notches[2].t), (4, 0.60));
        project.validate().unwrap();
    }

    #[test]
    fn deleting_anchor_remaps_notches_on_merged_and_shifted_segments() {
        let mut project = square_project();
        project.pieces[0].notches = vec![
            Notch { id: 1, segment_index: 0, t: 0.5, depth_meters: 0.008, width_meters: 0.012 },
            Notch { id: 2, segment_index: 1, t: 0.5, depth_meters: 0.008, width_meters: 0.012 },
            Notch { id: 3, segment_index: 3, t: 0.5, depth_meters: 0.008, width_meters: 0.012 },
        ];

        delete_project_anchor(&mut project, 1, 1).unwrap();

        let notches = &project.pieces[0].notches;
        assert_eq!(notches[0].segment_index, 0);
        assert!((notches[0].t - 0.25).abs() < 1.0e-9);
        assert_eq!(notches[1].segment_index, 0);
        assert!((notches[1].t - 0.75).abs() < 1.0e-9);
        assert_eq!((notches[2].segment_index, notches[2].t), (2, 0.5));
        project.validate().unwrap();
    }

    #[test]
    fn deleting_first_anchor_remaps_last_and_first_segment_notches_into_closure() {
        let mut project = square_project();
        project.pieces[0].notches = vec![
            Notch { id: 1, segment_index: 3, t: 0.5, depth_meters: 0.008, width_meters: 0.012 },
            Notch { id: 2, segment_index: 0, t: 0.5, depth_meters: 0.008, width_meters: 0.012 },
            Notch { id: 3, segment_index: 1, t: 0.5, depth_meters: 0.008, width_meters: 0.012 },
        ];

        delete_project_anchor(&mut project, 1, 0).unwrap();

        let notches = &project.pieces[0].notches;
        assert_eq!(notches[0].segment_index, 2);
        assert!((notches[0].t - 0.25).abs() < 1.0e-9);
        assert_eq!(notches[1].segment_index, 2);
        assert!((notches[1].t - 0.75).abs() < 1.0e-9);
        assert_eq!((notches[2].segment_index, notches[2].t), (0, 0.5));
        project.validate().unwrap();
    }

    #[test]
    fn mirrored_pair_keeps_topology_in_sync_after_anchor_insertion() {
        let mut project = square_project();
        let source = project.pieces[0].clone();
        let mirror = mirror_pattern_piece(&source, 3, "A espelho".into(), 1.0).unwrap();
        project.pieces.push(mirror);
        project.symmetry_links.push(SymmetryLink {
            piece_a_id: 1,
            piece_b_id: 3,
            axis_x: 1.0,
        });
        let mirrored_id = symmetry_counterpart_id(&project, 1).unwrap();
        split_project_segment(&mut project, 1, 1, 0.35).unwrap();
        split_project_segment(&mut project, mirrored_id, 1, 0.35).unwrap();
        assert_eq!(project.pieces[0].outline.segments.len(), project.pieces[2].outline.segments.len());
        project.validate().unwrap();
    }

    #[test]
    fn mirror_pattern_piece_reflects_outline_and_internal_metadata() {
        let mut source = PatternPiece::from_polygon(10, "Frente", &[
            PatternPoint2::new(0.0, 0.0),
            PatternPoint2::new(1.0, 0.0),
            PatternPoint2::new(1.0, 1.0),
            PatternPoint2::new(0.0, 1.0),
        ]).unwrap();
        source.internal_lines.push(Curve2D::polyline(&[
            PatternPoint2::new(0.2, 0.2),
            PatternPoint2::new(0.4, 0.5),
        ], false).unwrap());
        source.grain_line = Some(GrainLine {
            start: PatternPoint2::new(0.3, 0.1),
            end: PatternPoint2::new(0.3, 0.9),
        });
        source.darts.push(pattern_core::Dart {
            apex: PatternPoint2::new(0.5, 0.6),
            leg_a: PatternPoint2::new(0.4, 0.5),
            leg_b: PatternPoint2::new(0.6, 0.5),
            intake_meters: 0.02,
        });
        let mirrored = mirror_pattern_piece(&source, 11, "Costas".into(), 0.0).unwrap();
        assert_eq!(mirrored.id, 11);
        assert_eq!(mirrored.outline.start, PatternPoint2::new(0.0, 0.0));
        assert_eq!(mirrored.outline.segments[0].endpoint(), PatternPoint2::new(-1.0, 0.0));
        assert_eq!(mirrored.internal_lines[0].start, PatternPoint2::new(-0.2, 0.2));
        assert_eq!(mirrored.grain_line.as_ref().unwrap().start.x, -0.3);
        assert_eq!(mirrored.darts[0].apex.x, -0.5);
    }

    #[test]
    fn translating_piece_moves_every_curve_point_without_changing_seam_ids() {
        let mut project = square_project();
        let before = project.pieces[0].clone();
        translate_project_piece(&mut project, 1, 2.0, -0.5);
        let moved = &project.pieces[0];
        assert_eq!(moved.outline.start, PatternPoint2::new(before.outline.start.x + 2.0, before.outline.start.y - 0.5));
        assert_eq!(moved.id, before.id);
        assert_eq!(project.seams[0].sides_a[0].piece_id, 1);
        assert!((pattern_geom::curve_length(&moved.outline, 32) - pattern_geom::curve_length(&before.outline, 32)).abs() < 1.0e-6);
    }

    #[test]
    fn editor_snapshot_compares_project_and_draft_for_history() {
        let project = square_project();
        let a = PatternEditorSnapshot {
            project: project.clone(),
            draft_points: vec![PatternPoint2::new(0.0, 0.0)],
            selected_piece: 0,
            selected_target: None,
        };
        let mut b = a.clone();
        assert_eq!(a, b);
        b.draft_points.push(PatternPoint2::new(1.0, 0.0));
        assert_ne!(a, b);
    }

    #[test]
    fn moving_first_anchor_keeps_closed_curve_closed() {
        let mut curve = PatternPiece::from_polygon(
            1,
            "Quadrado",
            &[
                PatternPoint2::new(0.0, 0.0),
                PatternPoint2::new(1.0, 0.0),
                PatternPoint2::new(1.0, 1.0),
                PatternPoint2::new(0.0, 1.0),
            ],
        ).unwrap().outline;
        let moved = PatternPoint2::new(-0.25, 0.15);
        set_curve_anchor(&mut curve, 0, moved);
        assert_eq!(curve.start, moved);
        assert_eq!(curve.segments.last().unwrap().endpoint(), moved);
        assert_eq!(outline_anchors(&curve).len(), 4);
    }

    #[test]
    fn contour_edits_keep_fixed_anchors_attached_to_the_same_vertices() {
        let mut project = square_project();
        project.pieces[0].assembly.pinned_anchors = vec![1, 3];
        split_project_segment(&mut project, 1, 0, 0.5).unwrap();
        assert_eq!(project.pieces[0].assembly.pinned_anchors, vec![2, 4]);

        delete_project_anchor(&mut project, 1, 2).unwrap();
        assert_eq!(project.pieces[0].assembly.pinned_anchors, vec![3]);
        project.validate().unwrap();
    }

    #[test]
    fn splitting_seam_edge_updates_segment_ranges_and_keeps_geometry_length() {
        let mut project = square_project();
        let before = pattern_geom::curve_length(&project.pieces[0].outline, 32);
        split_project_segment(&mut project, 1, 1, 0.35).unwrap();
        let after = pattern_geom::curve_length(&project.pieces[0].outline, 32);
        assert!((before - after).abs() < 1.0e-6);
        assert_eq!(project.pieces[0].outline.segments.len(), 5);
        assert_eq!(project.seams[0].sides_a[0].start.segment, 1);
        assert_eq!(project.seams[0].sides_a[0].end.segment, 2);
        assert_eq!(project.seams[0].sides_b[0].start.segment, 3);
        assert_eq!(project.seams[0].sides_b[0].end.segment, 3);
        project.validate().unwrap();
    }

    #[test]
    fn deleting_polygon_anchor_keeps_project_valid_and_seam_reference_valid() {
        let mut project = square_project();
        delete_project_anchor(&mut project, 1, 1).unwrap();
        assert_eq!(project.pieces[0].outline.segments.len(), 3);
        assert_eq!(project.seams[0].sides_a[0].start.segment, 0);
        assert_eq!(project.seams[0].sides_a[0].end.segment, 0);
        project.validate().unwrap();
    }

    #[test]
    fn converting_line_to_bezier_preserves_straight_shape_and_seam_indices() {
        let mut project = square_project();
        convert_project_segment_to_bezier(&mut project, 1, 1).unwrap();
        let segment = &project.pieces[0].outline.segments[1];
        assert!(matches!(segment, CurveSegment::CubicBezier { .. }));
        assert_eq!(project.seams[0].sides_a[0].start.segment, 1);
        assert_eq!(project.seams[0].sides_a[0].end.segment, 1);
        assert!((pattern_geom::curve_length(&project.pieces[0].outline, 64) - 4.0).abs() < 1.0e-4);
        project.validate().unwrap();
    }

    #[test]
    fn deleting_anchor_on_bezier_outline_is_rejected_without_mutating_project() {
        let mut project = square_project();
        project.pieces[0].outline.segments[0] = CurveSegment::CubicBezier {
            control1: PatternPoint2::new(0.2, -0.1),
            control2: PatternPoint2::new(0.7, -0.1),
            to: PatternPoint2::new(1.0, 0.0),
        };
        let before = project.clone();
        assert!(delete_project_anchor(&mut project, 1, 1).is_err());
        assert_eq!(project, before);
    }
}

#[cfg(test)]
mod placement_ui_tests {
    use super::*;

    #[test]
    fn placement_translation_clamps_to_the_serialized_transform_bounds() {
        let (target, applied) = bounded_placement_translation(
            [99.0, -99.0, 0.0],
            Vec3::new(5.0, -5.0, 2.0),
        ).unwrap();
        assert_eq!(target, [100.0, -100.0, 2.0]);
        assert_eq!(applied, Vec3::new(1.0, -1.0, 2.0));
        assert!(bounded_placement_translation([0.0; 3], Vec3::splat(f32::NAN)).is_none());
    }

    #[test]
    fn rotate_and_scale_piece_stay_in_the_2d_plane_around_the_pivot() {
        let mut piece = PatternPiece::from_polygon(1, "teste", &[
            PatternPoint2::new(0.0,0.0), PatternPoint2::new(0.4,0.0),
            PatternPoint2::new(0.4,0.2), PatternPoint2::new(0.0,0.2),
        ]).unwrap();
        let pivot=piece_pivot(&piece);
        assert!((pivot.x-0.2).abs()<1.0e-9 && (pivot.y-0.1).abs()<1.0e-9);
        rotate_piece_about(&mut piece,pivot,std::f64::consts::FRAC_PI_2);
        let anchors=outline_anchors(&piece.outline);
        let width=anchors.iter().map(|p|p.x).fold(f64::MIN,f64::max)-anchors.iter().map(|p|p.x).fold(f64::MAX,f64::min);
        assert!((width-0.2).abs()<1.0e-9, "largura após girar 90°: {width}");
        let new_pivot=piece_pivot(&piece);
        assert!((new_pivot.x-pivot.x).abs()<1.0e-9 && (new_pivot.y-pivot.y).abs()<1.0e-9);
        scale_piece_about(&mut piece,pivot,2.0);
        let anchors=outline_anchors(&piece.outline);
        let width=anchors.iter().map(|p|p.x).fold(f64::MIN,f64::max)-anchors.iter().map(|p|p.x).fold(f64::MAX,f64::min);
        assert!((width-0.4).abs()<1.0e-9, "largura após escalar 2×: {width}");
    }
    #[test]
    fn screen_projection_maps_camera_target_to_viewport_center() {
        let camera = OrbitCamera::default();
        let rect = egui::Rect::from_min_size(
            egui::pos2(30.0, 50.0),
            egui::vec2(600.0, 400.0),
        );
        let (point, depth) = project_world_to_screen(
            camera.target,
            camera.view_proj(1.5),
            rect,
        ).expect("camera target should be in view");
        assert!((point.x - rect.center().x).abs() < 1.0e-3);
        assert!((point.y - rect.center().y).abs() < 1.0e-3);
        assert!((0.0..=1.0).contains(&depth));
    }

    #[test]
    fn t_pose_reference_png_is_loaded_as_the_original_raster_dimensions() {
        let reference = image::load_from_memory(include_bytes!("../../../assets/t_pose_reference.png"))
            .expect("T-pose reference PNG must be a decodable raster image");
        assert_eq!(reference.width(), 585);
        assert_eq!(reference.height(), 632);
    }

    #[test]
    fn triangle_hit_test_accepts_inside_and_rejects_outside_points() {
        let a = egui::pos2(0.0, 0.0);
        let b = egui::pos2(10.0, 0.0);
        let c = egui::pos2(0.0, 10.0);
        assert!(point_in_triangle(egui::pos2(2.0, 2.0), a, b, c));
        assert!(!point_in_triangle(egui::pos2(9.0, 9.0), a, b, c));
    }
}



#[cfg(test)]
mod cut_feature_tests {
    use super::*;

    fn point(x: f64, y: f64) -> PatternPoint2 {
        PatternPoint2::new(x, y)
    }

    fn split_rectangles() -> (Vec<PatternPoint2>, Vec<PatternPoint2>) {
        (
            vec![point(-1.0, -1.0), point(0.0, -1.0), point(0.0, 1.0), point(-1.0, 1.0)],
            vec![point(0.0, -1.0), point(1.0, -1.0), point(1.0, 1.0), point(0.0, 1.0)],
        )
    }

    #[test]
    fn cut_crossing_internal_curve_creates_fragments_on_both_pieces() {
        let curve = Curve2D::polyline(&[point(-0.75, 0.2), point(0.75, 0.2)], false).unwrap();
        let (polygon_a, polygon_b) = split_rectangles();

        let fragments = split_curve_fragments_by_cut(&curve, &polygon_a, &polygon_b).unwrap();

        assert_eq!(fragments[0].len(), 1);
        assert_eq!(fragments[1].len(), 1);
        let left_end = fragments[0][0].endpoint().unwrap();
        let right_start = fragments[1][0].start;
        assert!(left_end.x.abs() < 1.0e-4, "left fragment ended at {}", left_end.x);
        assert!(right_start.x.abs() < 1.0e-4, "right fragment started at {}", right_start.x);
        assert!((left_end.y - right_start.y).abs() < 1.0e-5);
    }

    #[test]
    fn cut_preserves_curves_that_are_wholly_on_one_side() {
        let curve = Curve2D::polyline(&[
            point(-0.8, -0.5),
            point(-0.7, 0.0),
            point(-0.8, 0.5),
        ], false).unwrap();
        let (polygon_a, polygon_b) = split_rectangles();

        let fragments = split_curve_fragments_by_cut(&curve, &polygon_a, &polygon_b).unwrap();

        assert_eq!(fragments[0], vec![curve]);
        assert!(fragments[1].is_empty());
    }
}
