//! Diagnóstico de estabilidade: imprime deformação máxima e energia cinética ao longo do tempo.
//!
//! `cargo run --release -p cloth-sim --example diag -- [cena 1-6] [res] [substeps] [iterações] [segundos]`

use cloth_sim::{build_scene, SceneKind, SimParams};

fn main() {
    let args: Vec<String> = std::env::args().skip(1).collect();
    let get = |i: usize, default: f32| -> f32 {
        args.get(i).and_then(|s| s.parse().ok()).unwrap_or(default)
    };
    let kind = SceneKind::DIAGNOSTIC[(get(0, 1.0) as usize).clamp(1, SceneKind::DIAGNOSTIC.len()) - 1];
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
        "{:>6} {:>10} {:>10} {:>10} {:>12} {:>8} {:>10} {:>12} {:>10} {:>10} {:>10} {:>8} {:>8} {:>20}",
        "t(s)", "strain máx", "p99", "p50", "energia cin.", "y mín",
        "contatos", "penet. máx", "passo ms", "restr. ms", "colisor ms", "sweeps", "CCD", "pior"
    );

    let dt = 1.0 / 60.0;
    let frames = (seconds / dt) as usize;
    for f in 0..=frames {
        let report_every = if kind == SceneKind::FastSdfCrossing { 1 } else { 30 };
        if f % report_every == 0 {
            let lowest = sim.pos.iter().map(|p| p.y).fold(f32::MAX, f32::min);
            let worst = sim
                .worst_strain_edge()
                .map(|(a, b, k, _)| format!("{:?}({a},{b})", k))
                .unwrap_or_default();
            println!(
                "{:>6.2} {:>10.4} {:>10.4} {:>10.4} {:>12.6} {:>8.3} {:>10} {:>12.5} {:>10.3} {:>10.3} {:>10.3} {:>8} {:>8} {:>20}",
                f as f32 * dt,
                sim.max_strain(),
                sim.strain_percentile(0.99),
                sim.strain_percentile(0.5),
                sim.kinetic_energy(),
                lowest,
                sim.stats.body_contact_projections,
                sim.stats.body_max_penetration_m,
                sim.stats.step_ms,
                sim.stats.constraint_ms,
                sim.stats.body_collision_ms,
                sim.stats.body_sweep_tests,
                sim.stats.body_swept_contacts,
                worst
            );
            if sim.params.self_collision {
                println!(
                    "  auto: VV {} candidatos/{} contatos · VT {} candidatos/{} verificados/{} contatos · EE {} candidatos/{} verificados/{} contatos · descartados {} · penetração {:.3} → {:.3} mm · broadphase {:.3} ms · resolução {:.3} ms",
                    sim.stats.self_collision_vv_candidates,
                    sim.stats.self_collision_vv_contacts,
                    sim.stats.self_collision_vt_candidates,
                    sim.stats.self_collision_vt_verified,
                    sim.stats.self_collision_vt_contacts,
                    sim.stats.self_collision_ee_candidates,
                    sim.stats.self_collision_ee_verified,
                    sim.stats.self_collision_ee_contacts,
                    sim.stats.self_collision_dropped_candidates,
                    sim.stats.self_collision_penetration_before_m * 1000.0,
                    sim.stats.self_collision_penetration_after_m * 1000.0,
                    sim.stats.self_collision_broadphase_ms,
                    sim.stats.self_collision_solve_ms,
                );
            }
        }
        sim.step(dt);
        if !sim.is_finite() {
            println!("DIVERGIU em t = {:.2} s", f as f32 * dt);
            break;
        }
    }
}
