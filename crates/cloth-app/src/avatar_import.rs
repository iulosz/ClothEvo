//! Importação simples de malhas estáticas OBJ/FBX para a representação visual do avatar.
//! A simulação continua usando os colisores SDF analíticos existentes; o mesh importado
//! substitui apenas a geometria visual do boneco.

use cloth_render::{MeshKind, StaticMeshData, Vertex};
use glam::Vec3;
use std::collections::HashMap;
use std::fs;
use std::path::Path;

const MAX_VERTICES: usize = 2_000_000;
const MAX_TRIANGLES: usize = 2_000_000;

pub struct ImportedAvatar {
    pub display_name: String,
    pub mesh: StaticMeshData,
}

pub fn load_model(path: &Path, target_height_meters: f32) -> Result<ImportedAvatar, String> {
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

    let (positions, indices) = normalize_mesh(positions, indices, target_height)?;
    let vertices = make_vertices(&positions, &indices)?;
    Ok(ImportedAvatar {
        display_name: name,
        mesh: StaticMeshData {
            vertices,
            indices,
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
    let scene = ufbx::load_file(path, options)
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
    mut positions: Vec<[f32; 3]>,
    indices: Vec<u32>,
    target_height: f32,
) -> Result<(Vec<[f32; 3]>, Vec<u32>), String> {
    if positions.len() < 3 || indices.len() < 3 || indices.len() % 3 != 0 {
        return Err("A malha importada não contém triângulos válidos.".into());
    }
    if positions.iter().flatten().any(|v| !v.is_finite()) {
        return Err("A malha importada contém coordenadas inválidas.".into());
    }
    if indices.iter().any(|i| *i as usize >= positions.len()) {
        return Err("A malha importada contém índices fora dos limites.".into());
    }

    let min = bounds_min(&positions);
    let max = bounds_max(&positions);
    let height = max[1] - min[1];
    if !height.is_finite() || height <= 1.0e-6 {
        return Err("Não foi possível determinar a altura do modelo. Verifique o eixo vertical do arquivo.".into());
    }
    let scale = target_height / height;
    let center_x = (min[0] + max[0]) * 0.5;
    let center_z = (min[2] + max[2]) * 0.5;
    for p in &mut positions {
        p[0] = (p[0] - center_x) * scale;
        p[1] = (p[1] - min[1]) * scale;
        p[2] = (p[2] - center_z) * scale;
    }
    Ok((positions, indices))
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
