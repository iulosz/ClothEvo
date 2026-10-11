//! Valida `shader.wgsl` offline (parser + validador do naga), sem GPU.

use naga::valid::{Capabilities, ValidationFlags, Validator};

#[test]
fn wgsl_shader_is_valid_and_has_entry_points() {
    let src = include_str!("../src/shader.wgsl");
    let module = naga::front::wgsl::parse_str(src).expect("WGSL não compila");
    Validator::new(ValidationFlags::all(), Capabilities::empty())
        .validate(&module)
        .expect("WGSL inválido");

    let names: Vec<_> = module.entry_points.iter().map(|e| e.name.as_str()).collect();
    assert!(names.contains(&"vs_main") && names.contains(&"fs_main"), "entry points: {names:?}");
}
