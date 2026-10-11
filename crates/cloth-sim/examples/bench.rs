//! Benchmark do solver em CPU: `cargo run --release -p cloth-sim --example bench`
//!
//! Mede o custo de um passo de 1/60 s (todos os substeps) em várias resoluções,
//! em série e em paralelo. Serve para validar o risco de desempenho da Fase 0.

use std::time::Instant;

use cloth_sim::{build_scene, SceneKind, SimParams};

fn main() {
    println!("threads do rayon: {}", rayon::current_num_threads());
    println!(
        "{:>5} {:>9} {:>9} {:>8} {:>9} {:>9} {:>9}",
        "res", "partíc.", "dist.", "flexão", "paralelo", "ms/passo", "FPS máx"
    );

    for &res in &[40u32, 70, 100, 141] {
        for &parallel in &[false, true] {
            let params = SimParams { parallel, ..SimParams::default() };
            let mut sim = build_scene(SceneKind::SphereDrop, res, params);
            for _ in 0..5 {
                sim.step(1.0 / 60.0);
            }
            let steps = 40;
            let t = Instant::now();
            for _ in 0..steps {
                sim.step(1.0 / 60.0);
            }
            let ms = t.elapsed().as_secs_f64() * 1000.0 / steps as f64;
            println!(
                "{:>5} {:>9} {:>9} {:>8} {:>9} {:>9.2} {:>9.0}",
                res,
                sim.particle_count(),
                sim.distance_constraint_count(),
                sim.bend_constraint_count(),
                if parallel { "sim" } else { "não" },
                ms,
                1000.0 / ms
            );
            assert!(sim.is_finite(), "divergiu em res={res}");
        }
    }
}
