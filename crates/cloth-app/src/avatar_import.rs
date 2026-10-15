//! Importação e validação de avatares estáticos OBJ/FBX para renderização e colisão.
//! A geometria visual preserva a malha triangulada; a representação física é um SDF
//! construído em segundo plano a partir dos mesmos vértices já normalizados.

use cloth_render::{MeshKind, StaticMeshData, Vertex};
use cloth_sim::sdf::MeshTopologyDiagnostics;
use cloth_sim::SignedDistanceField;
use glam::Vec3;
use std::collections::HashMap;
use std::fs;
use std::path::Path;
use std::sync::Arc;
use std::time::Instant;

const MAX_VERTICES: usize = 2_000_000;
const MAX_TRIANGLES: usize = 2_000_000;

#[derive(Clone, Debug)]
pub struct AvatarDiagnostics {
    pub vertex_count: usize,
    pub triangle_count: usize,
    pub dimensions_meters: [f32; 3],
    pub discarded_degenerate_triangles: usize,
    pub sdf_spacing_meters: Option<f32>,
    pub sdf_dimensions: Option<[usize; 3]>,
    pub topology: Option<MeshTopologyDiagnostics>,
    pub import_milliseconds: f32,
    pub collision_build_milliseconds: f32,
}

pub struct ImportedAvatar {
    pub display_name: String,
    pub mesh: StaticMeshData,
    /// SDF em metros, gerado com a mesma transformação/escala da malha visual.
    pub collision_sdf: Option<Arc<SignedDistanceField>>,
    pub collision_warning: Option<String>,
    pub diagnostics: AvatarDiagnostics,
}

pub fn load_model(
    path: &Path,
    target_height_meters: f32,
    requested_sdf_spacing_meters: f32,
    heading_degrees: f32,
) -> Result<ImportedAvatar, String> {
    let import_started = Instant::now();
    let extension = path.extension().and_then(|s| s.to_str()).unwrap_or("").to_ascii_lowercase();
    let name = path.file_name().and_then(|s| s.to_str()).unwrap_or("Avatar importado").to_owned();
    let target_height = if target_height_meters.is_finite() {
        target_height_meters.clamp(0.5, 2.5)
    } else {
        1.72
    };

    let (positions, indices) = match extension.as_str() {
        "obj" => {
            let text = fs::read_to_string(path)
                .map_err(|e| format!("Não foi possível ler o OBJ '{}': {e}", path.display()))?;
            parse_obj(&text)?
        }
        "fbx" => parse_fbx(path)?,
        _ => return Err("Formato não suportado. Escolha um arquivo .obj ou .fbx.".into()),
    };

    let (positions, indices, discarded_degenerate_triangles) =
        normalize_mesh_with_heading(positions, indices, target_height, heading_degrees)?;
    let min = bounds_min(&positions);
    let max = bounds_max(&positions);
    let dimensions_meters = [
        max[0] - min[0],
        max[1] - min[1],
        max[2] - min[2],
    ];
    let vertices = make_vertices(&positions, &indices)?;
    let wire_indices = make_wire_indices(&indices);
    let import_milliseconds = import_started.elapsed().as_secs_f32() * 1000.0;

    // A construção do campo usa BVH sobre os triângulos e roda fora da thread da interface.
    let collision_started = Instant::now();
    let collision_result = SignedDistanceField::from_triangle_mesh(
        &positions,
        &indices,
        requested_sdf_spacing_meters,
    );
    let collision_build_milliseconds = collision_started.elapsed().as_secs_f32() * 1000.0;
    let (collision_sdf, collision_warning, sdf_spacing_meters, sdf_dimensions, topology) =
        match collision_result {
            Ok(field) => {
                let topology = field.topology_diagnostics();
                let warning = if field.has_reliable_inside_outside() {
                    None
                } else {
                    Some(format!(
                        "A malha tem {} arestas abertas, {} não-manifold, {} com orientação inconsistente e {} triângulos degenerados. O campo usa distância sem sinal; revise a topologia para uma colisão interior/exterior confiável.",
                        topology.boundary_edges,
                        topology.non_manifold_edges,
                        topology.inconsistent_winding_edges,
                        topology.degenerate_triangles,
                    ))
                };
                let spacing = field.voxel_spacing();
                let dimensions = field.dimensions();
                (
                    Some(Arc::new(field)),
                    warning,
                    Some(spacing),
                    Some(dimensions),
                    Some(topology),
                )
            }
            Err(error) => (None, Some(error), None, None, None),
        };

    Ok(ImportedAvatar {
        display_name: name,
        collision_sdf,
        collision_warning,
        diagnostics: AvatarDiagnostics {
            vertex_count: positions.len(),
            triangle_count: indices.len() / 3,
            dimensions_meters,
            discarded_degenerate_triangles,
            sdf_spacing_meters,
            sdf_dimensions,
            topology,
            import_milliseconds,
            collision_build_milliseconds,
        },
        mesh: StaticMeshData {
            vertices,
            indices,
            wire_indices,
            color: [0.76, 0.75, 0.73],
            kind: MeshKind::Shaded,
        },
    })
}

fn parse_obj(text: &str) -> Result<(Vec<[f32; 3]>, Vec<u32>), String> {
    let mut positions: Vec<[f32; 3]> = Vec::new();
    let mut indices: Vec<u32> = Vec::new();

    for (line_index, line) in text.lines().enumerate() {
        let mut words = line.split_whitespace();
        let Some(kind) = words.next() else { continue };
        match kind {
            "v" => {
                let values = words.take(3).map(str::parse::<f32>).collect::<Result<Vec<_>, _>>()
                    .map_err(|_| format!("Coordenada inválida na linha {} do OBJ.", line_index + 1))?;
                if values.len() != 3 || values.iter().any(|v| !v.is_finite()) {
                    return Err(format!("Vértice inválido na linha {} do OBJ.", line_index + 1));
                }
                positions.push([values[0], values[1], values[2]]);
                if positions.len() > MAX_VERTICES {
                    return Err("O OBJ excede o limite seguro de 2 milhões de vértices.".into());
                }
            }
            "f" => {
                let mut polygon: Vec<u32> = Vec::new();
                for token in words {
                    let raw = token.split('/').next().unwrap_or("");
                    let parsed = raw.parse::<i64>()
                        .map_err(|_| format!("Índice de face inválido na linha {} do OBJ.", line_index + 1))?;
                    if parsed == 0 {
                        return Err(format!("Índice OBJ zero na linha {}.", line_index + 1));
                    }
                    let vertex_index = if parsed > 0 {
                        parsed - 1
                    } else {
                        positions.len() as i64 + parsed
                    };
                    if vertex_index < 0 || vertex_index as usize >= positions.len() {
                        return Err(format!("Face fora dos limites na linha {} do OBJ.", line_index + 1));
                    }
                    polygon.push(vertex_index as u32);
                }
                if polygon.len() < 3 { continue; }
                // Triangulação em leque para polígonos convexos, comum em OBJ exportado por DCCs.
                for corner in 1..polygon.len() - 1 {
                    indices.extend_from_slice(&[polygon[0], polygon[corner], polygon[corner + 1]]);
                    if indices.len() / 3 > MAX_TRIANGLES {
                        return Err("O OBJ excede o limite seguro de 2 milhões de triângulos.".into());
                    }
                }
            }
            _ => {}
        }
    }

    if positions.is_empty() || indices.is_empty() {
        return Err("O OBJ não contém uma malha de faces triangulável.".into());
    }

    // O OBJ não codifica um eixo vertical padrão. Use o maior eixo como altura;
    // modelos exportados do Blender (Z-up) são girados para o Y-up do ClothEvo.
    let min = bounds_min(&positions);
    let max = bounds_max(&positions);
    let extent = [max[0] - min[0], max[1] - min[1], max[2] - min[2]];
    let vertical_axis = if extent[2] > extent[1] && extent[2] >= extent[0] {
        2
    } else if extent[0] > extent[1] {
        0
    } else {
        1
    };
    if vertical_axis != 1 {
        for position in &mut positions {
            let p = *position;
            *position = match vertical_axis {
                2 => [p[0], p[2], -p[1]],
                0 => [p[1], p[0], p[2]],
                _ => p,
            };
        }
    }
    Ok((positions, indices))
}

fn parse_fbx(path: &Path) -> Result<(Vec<[f32; 3]>, Vec<u32>), String> {
    let options = ufbx::LoadOpts {
        target_axes: ufbx::CoordinateAxes::right_handed_y_up(),
        target_unit_meters: 1.0,
        target_camera_axes: ufbx::CoordinateAxes::right_handed_y_up(),
        target_light_axes: ufbx::CoordinateAxes::right_handed_y_up(),
        space_conversion: ufbx::SpaceConversion::ModifyGeometry,
        geometry_transform_handling: ufbx::GeometryTransformHandling::ModifyGeometry,
        generate_missing_normals: true,
        ..Default::default()
    };
    let filename = path.to_str().ok_or_else(|| {
        format!("O caminho do arquivo não pode ser representado em UTF-8: {}", path.display())
    })?;
    let scene = ufbx::load_file(filename, options)
        .map_err(|e| format!("Falha ao importar FBX '{}': {}", path.display(), e.description))?;

    let mut positions: Vec<[f32; 3]> = Vec::new();
    let mut indices: Vec<u32> = Vec::new();
    let mut position_lookup: HashMap<(i64, i64, i64), u32> = HashMap::new();

    for node in &scene.nodes {
        if !node.visible { continue; }
        let Some(mesh) = node.mesh.as_ref() else { continue };
        let mut triangle_indices = vec![0u32; (mesh.max_face_triangles as usize).saturating_mul(3).max(3)];

        for face in &mesh.faces {
            if face.num_indices < 3 { continue; }
            let triangle_count = mesh.triangulate_face(&mut triangle_indices, *face) as usize;
            for &corner_index in triangle_indices.iter().take(triangle_count.saturating_mul(3)) {
                let local = mesh.vertex_position[corner_index as usize];
                let world = ufbx::transform_position(&node.geometry_to_world, local);
                let position = [world.x as f32, world.y as f32, world.z as f32];
                if position.iter().any(|v| !v.is_finite()) {
                    return Err("O FBX contém vértices com coordenadas não finitas.".into());
                }
                // Solda vértices de canto coincidentes para reconstruir normais suaves.
                let key = (
                    (position[0] as f64 * 1_000_000.0).round() as i64,
                    (position[1] as f64 * 1_000_000.0).round() as i64,
                    (position[2] as f64 * 1_000_000.0).round() as i64,
                );
                let index = if let Some(&existing) = position_lookup.get(&key) {
                    existing
                } else {
                    if positions.len() >= MAX_VERTICES {
                        return Err("O FBX excede o limite seguro de 2 milhões de vértices.".into());
                    }
                    let next = positions.len() as u32;
                    positions.push(position);
                    position_lookup.insert(key, next);
                    next
                };
                indices.push(index);
                if indices.len() / 3 > MAX_TRIANGLES {
                    return Err("O FBX excede o limite seguro de 2 milhões de triângulos.".into());
                }
            }
        }
    }

    if positions.is_empty() || indices.is_empty() {
        return Err("Nenhuma malha poligonal visível foi encontrada no FBX.".into());
    }
    Ok((positions, indices))
}

fn bounds_min(positions: &[[f32; 3]]) -> [f32; 3] {
    let mut min = [f32::INFINITY; 3];
    for p in positions {
        for axis in 0..3 { min[axis] = min[axis].min(p[axis]); }
    }
    min
}

fn bounds_max(positions: &[[f32; 3]]) -> [f32; 3] {
    let mut max = [f32::NEG_INFINITY; 3];
    for p in positions {
        for axis in 0..3 { max[axis] = max[axis].max(p[axis]); }
    }
    max
}

fn normalize_mesh(
    positions: Vec<[f32; 3]>,
    indices: Vec<u32>,
    target_height: f32,
) -> Result<(Vec<[f32; 3]>, Vec<u32>), String> {
    let (positions, indices, _) =
        normalize_mesh_with_heading(positions, indices, target_height, 0.0)?;
    Ok((positions, indices))
}

fn normalize_mesh_with_heading(
    mut positions: Vec<[f32; 3]>,
    indices: Vec<u32>,
    target_height: f32,
    heading_degrees: f32,
) -> Result<(Vec<[f32; 3]>, Vec<u32>, usize), String> {
    if positions.len() < 3 || indices.len() < 3 || indices.len() % 3 != 0 {
        return Err("A malha importada não contém triângulos válidos.".into());
    }
    if positions.iter().flatten().any(|v| !v.is_finite()) {
        return Err("A malha importada contém coordenadas inválidas.".into());
    }
    if indices.iter().any(|i| *i as usize >= positions.len()) {
        return Err("A malha importada contém índices fora dos limites.".into());
    }
    if !target_height.is_finite() || target_height <= 0.0 || !heading_degrees.is_finite() {
        return Err("Escala e orientação do avatar precisam ser valores finitos.".into());
    }

    // OBJ não armazena eixo frontal; FBX já chega em Y-up via ufbx. A heurística
    // mantém +Z como frente antes de aplicar a rotação explícita escolhida pelo usuário.
    orient_front_to_positive_z(&mut positions);

    let min = bounds_min(&positions);
    let max = bounds_max(&positions);
    let height = max[1] - min[1];
    if !height.is_finite() || height <= 1.0e-6 {
        return Err("Não foi possível determinar a altura do modelo. Verifique o eixo vertical do arquivo.".into());
    }
    let scale = target_height / height;
    let center_x = (min[0] + max[0]) * 0.5;
    let center_z = (min[2] + max[2]) * 0.5;
    let heading = heading_degrees.to_radians();
    let (sin_heading, cos_heading) = heading.sin_cos();
    for p in &mut positions {
        let x = (p[0] - center_x) * scale;
        let y = (p[1] - min[1]) * scale;
        let z = (p[2] - center_z) * scale;
        p[0] = cos_heading * x + sin_heading * z;
        p[1] = y;
        p[2] = -sin_heading * x + cos_heading * z;
    }

    // Une apenas vértices coincidentes em 1 µm, necessário para OBJ com vértices
    // duplicados em juntas/materiales e para avaliar corretamente as arestas abertas.
    let mut welded_positions = Vec::with_capacity(positions.len());
    let mut position_lookup: HashMap<(i64, i64, i64), u32> = HashMap::with_capacity(positions.len());
    let mut remap = vec![0_u32; positions.len()];
    for (index, p) in positions.iter().enumerate() {
        let key = (
            (p[0] as f64 * 1_000_000.0).round() as i64,
            (p[1] as f64 * 1_000_000.0).round() as i64,
            (p[2] as f64 * 1_000_000.0).round() as i64,
        );
        let new_index = if let Some(&existing) = position_lookup.get(&key) {
            existing
        } else {
            let new_index = welded_positions.len() as u32;
            welded_positions.push(*p);
            position_lookup.insert(key, new_index);
            new_index
        };
        remap[index] = new_index;
    }

    let mut welded_indices = Vec::with_capacity(indices.len());
    for index in indices {
        welded_indices.push(remap[index as usize]);
    }
    let mut valid_indices = Vec::with_capacity(welded_indices.len());
    let mut discarded_degenerate_triangles = 0;
    for tri in welded_indices.chunks_exact(3) {
        let (a, b, c) = (
            Vec3::from_array(welded_positions[tri[0] as usize]),
            Vec3::from_array(welded_positions[tri[1] as usize]),
            Vec3::from_array(welded_positions[tri[2] as usize]),
        );
        if tri[0] == tri[1]
            || tri[1] == tri[2]
            || tri[2] == tri[0]
            || (b - a).cross(c - a).length_squared() <= 1.0e-16
        {
            discarded_degenerate_triangles += 1;
        } else {
            valid_indices.extend_from_slice(tri);
        }
    }
    if welded_positions.len() < 3 || valid_indices.is_empty() {
        return Err("A malha não contém triângulos não degenerados depois da normalização.".into());
    }
    Ok((welded_positions, valid_indices, discarded_degenerate_triangles))
}

fn make_wire_indices(indices: &[u32]) -> Vec<u32> {
    // Limita o overlay a 300 mil arestas, mantendo o custo previsível em avatares densos.
    const MAX_WIRE_EDGES: usize = 300_000;
    let triangles = indices.len() / 3;
    if triangles == 0 {
        return Vec::new();
    }
    let stride = triangles.div_ceil(MAX_WIRE_EDGES).max(1);
    let mut output = Vec::with_capacity(triangles.min(MAX_WIRE_EDGES) * 6);
    for tri in indices.chunks_exact(3).step_by(stride) {
        output.extend_from_slice(&[tri[0], tri[1], tri[1], tri[2], tri[2], tri[0]]);
    }
    output
}

/// Orienta malhas humanoides em pose T/A para que a frente fique em +Z e os braços em X.
fn orient_front_to_positive_z(positions: &mut [[f32; 3]]) {
    let min_y = positions.iter().map(|p| p[1]).fold(f32::INFINITY, f32::min);
    let max_y = positions.iter().map(|p| p[1]).fold(f32::NEG_INFINITY, f32::max);
    let height = max_y - min_y;
    if !height.is_finite() || height <= 1.0e-6 { return; }
    let (mut x_lo, mut x_hi, mut z_lo, mut z_hi) =
        (f32::INFINITY, f32::NEG_INFINITY, f32::INFINITY, f32::NEG_INFINITY);
    for p in positions.iter().filter(|p| p[1] >= min_y + 0.70 * height && p[1] <= min_y + 0.88 * height) {
        x_lo = x_lo.min(p[0]); x_hi = x_hi.max(p[0]);
        z_lo = z_lo.min(p[2]); z_hi = z_hi.max(p[2]);
    }
    if x_hi < x_lo { return; }
    if (z_hi - z_lo) > (x_hi - x_lo) {
        // Corrige o sentido: +X passa a apontar para +Z.
        for p in positions.iter_mut() {
            let (x, z) = (p[0], p[2]);
            p[0] = -z;
            p[2] = x;
        }
    }
    let (mut shin_sum, mut shin_count) = (0.0_f32, 0_u32);
    let (mut foot_lo, mut foot_hi) = (f32::INFINITY, f32::NEG_INFINITY);
    for p in positions.iter() {
        if p[1] >= min_y + 0.15 * height && p[1] <= min_y + 0.30 * height {
            shin_sum += p[2]; shin_count += 1;
        }
        if p[1] <= min_y + 0.04 * height {
            foot_lo = foot_lo.min(p[2]); foot_hi = foot_hi.max(p[2]);
        }
    }
    if shin_count == 0 || foot_hi < foot_lo { return; }
    let ankle = shin_sum / shin_count as f32;
    if ankle - foot_lo > (foot_hi - ankle) * 1.15 {
        for p in positions.iter_mut() { p[0] = -p[0]; p[2] = -p[2]; }
    }
}
fn make_vertices(positions: &[[f32; 3]], indices: &[u32]) -> Result<Vec<Vertex>, String> {
    let mut normals = vec![Vec3::ZERO; positions.len()];
    for triangle in indices.chunks_exact(3) {
        let a = positions[triangle[0] as usize];
        let b = positions[triangle[1] as usize];
        let c = positions[triangle[2] as usize];
        let normal = (Vec3::from_array(b) - Vec3::from_array(a))
            .cross(Vec3::from_array(c) - Vec3::from_array(a));
        if normal.is_finite() && normal.length_squared() > 1.0e-16 {
            for index in triangle {
                normals[*index as usize] += normal;
            }
        }
    }
    Ok(positions.iter().enumerate().map(|(index, p)| {
        let normal = normals[index].try_normalize().unwrap_or(Vec3::Y);
        Vertex { pos: *p, normal: normal.to_array() }
    }).collect())
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn orientation_puts_arms_along_x_and_toes_toward_plus_z() {
        let mut positions = vec![
            [0.0, 1.7, 0.0],
            [-0.05, 0.0, 0.1], [0.20, 0.0, 0.1], [-0.05, 0.0, -0.1], [0.20, 0.0, -0.1],
            [0.0, 0.4, 0.1], [0.0, 0.4, -0.1],
            [0.0, 1.4, -0.8], [0.0, 1.4, 0.8],
        ];
        orient_front_to_positive_z(&mut positions);
        let shoulders: Vec<_> = positions.iter().filter(|p| (p[1] - 1.4).abs() < 1.0e-6).collect();
        let x_extent = shoulders.iter().map(|p| p[0]).fold(f32::NEG_INFINITY, f32::max)
            - shoulders.iter().map(|p| p[0]).fold(f32::INFINITY, f32::min);
        assert!(x_extent > 1.5, "braços deveriam ficar ao longo de X: {x_extent}");
        let toe_z = positions.iter().filter(|p| p[1] < 1.0e-6).map(|p| p[2]).fold(f32::NEG_INFINITY, f32::max);
        assert!((toe_z - 0.20).abs() < 1.0e-5, "pontas dos pés deveriam apontar para +Z: {toe_z}");
    }
    #[test]
    fn orientation_keeps_an_already_frontal_figure() {
        let mut positions = vec![
            [0.0, 1.7, 0.0],
            [0.1, 0.0, -0.05], [0.1, 0.0, 0.20], [-0.1, 0.0, -0.05], [-0.1, 0.0, 0.20],
            [0.1, 0.4, 0.0], [-0.1, 0.4, 0.0],
            [-0.8, 1.4, 0.0], [0.8, 1.4, 0.0],
        ];
        let before = positions.clone();
        orient_front_to_positive_z(&mut positions);
        assert_eq!(positions, before);
    }

    #[test]
    fn normalization_welds_duplicate_vertices_and_discards_degenerate_triangles() {
        let (positions, indices, discarded) = normalize_mesh_with_heading(
            vec![
                [0.0, 0.0, 0.0], [0.0, 0.0, 0.0],
                [1.0, 0.0, 0.0], [0.0, 1.0, 0.0],
            ],
            vec![0, 1, 2, 0, 2, 3],
            1.0,
            0.0,
        ).unwrap();
        assert_eq!(discarded, 1);
        assert_eq!(indices.len(), 3);
        assert_eq!(positions.len(), 3);
    }

    #[test]
    fn obj_parser_triangulates_polygons_and_supports_negative_indices() {
        let obj = "\
v 0 0 0
v 1 0 0
v 1 0 1
v 0 0 1
f -4 -3 -2 -1
";
        let (positions, indices) = parse_obj(obj).unwrap();
        assert_eq!(positions.len(), 4);
        assert_eq!(indices.len(), 6);
    }

    #[test]
    fn normalizing_places_model_on_floor_and_sets_requested_height() {
        let (positions, indices) = normalize_mesh(
            vec![[-1.0, 3.0, -1.0], [1.0, 3.0, -1.0], [0.0, 5.0, 1.0]],
            vec![0, 1, 2],
            1.8,
        ).unwrap();
        assert!((positions.iter().map(|p| p[1]).fold(f32::INFINITY, f32::min)).abs() < 1.0e-6);
        let height = bounds_max(&positions)[1] - bounds_min(&positions)[1];
        assert!((height - 1.8).abs() < 1.0e-5);
        assert_eq!(indices.len(), 3);
    }
}
