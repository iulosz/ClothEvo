//! Diagnóstico de estabilidade: imprime deformação máxima e energia cinética ao longo do tempo.
//!
//! `cargo run --release -p cloth-sim --example diag -- [cena 1-3] [res] [substeps] [iterações] [segundos]`

use cloth_sim::{build_scene, SceneKind, SimParams};

fn main() {
    let args: Vec<String> = std::env::args().skip(1).collect();
    let get = |i: usize, default: f32| -> f32 {
        args.get(i).and_then(|s| s.parse().ok()).unwrap_or(default)
    };
    let kind = SceneKind::ALL[(get(0, 1.0) as usize).clamp(1, 3) - 1];
    let res = get(1, 24.0) as u32;
    let seconds = get(4, 10.0);

    let mut params = SimParams::default();
    params.substeps = get(2, params.substeps as f32) as u32;
    params.iterations = get(3, params.iterations as f32) as u32;

    println!(
        "{} | res={} substeps={} iterações={}",
        kind.name(),
        res,
        params.substeps,
        params.iterations
    );
    let mut sim = build_scene(kind, res, params);
    println!(
        "{:>6} {:>10} {:>10} {:>10} {:>12} {:>8} {:>8}",
        "t(s)", "strain máx", "p99", "p50", "energia cin.", "y mín", "pior"
    );

    let dt = 1.0 / 60.0;
    let frames = (seconds / dt) as usize;
    for f in 0..=frames {
        if f % 30 == 0 {
            let lowest = sim.pos.iter().map(|p| p.y).fold(f32::MAX, f32::min);
            let worst = sim
                .worst_strain_edge()
                .map(|(a, b, k, _)| format!("{:?}({a},{b})", k))
                .unwrap_or_default();
            println!(
                "{:>6.2} {:>10.4} {:>10.4} {:>10.4} {:>12.6} {:>8.3} {}",
                f as f32 * dt,
                sim.max_strain(),
                sim.strain_percentile(0.99),
                sim.strain_percentile(0.5),
                sim.kinetic_energy(),
                lowest,
                worst
            );
        }
        sim.step(dt);
        if !sim.is_finite() {
            println!("DIVERGIU em t = {:.2} s", f as f32 * dt);
            break;
        }
    }
}
