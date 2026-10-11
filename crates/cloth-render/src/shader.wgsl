// Shader único do viewport: malha de tecido (dois lados), colisores, chão com grade e wireframe.

struct Uniforms {
    view_proj: mat4x4<f32>,
    color_front: vec4<f32>,
    color_back: vec4<f32>,
    light_dir: vec4<f32>, // aponta PARA a luz
    eye: vec4<f32>,
    // x: tipo (0 = sombreado, 1 = chão com grade, 2 = linha)
    // y: 1.0 se o shader deve aplicar gamma (alvo não-sRGB)
    params: vec4<f32>,
};

@group(0) @binding(0) var<uniform> u: Uniforms;

struct VsIn {
    @location(0) pos: vec3<f32>,
    @location(1) normal: vec3<f32>,
};

struct VsOut {
    @builtin(position) clip: vec4<f32>,
    @location(0) world: vec3<f32>,
    @location(1) normal: vec3<f32>,
};

@vertex
fn vs_main(in: VsIn) -> VsOut {
    var out: VsOut;
    var clip = u.view_proj * vec4<f32>(in.pos, 1.0);
    if (u.params.x > 1.5) {
        // Linhas de wireframe: puxa levemente para a câmera para não brigar com a superfície.
        clip.z = clip.z - 2.0e-4 * clip.w;
    }
    out.clip = clip;
    out.world = in.pos;
    out.normal = in.normal;
    return out;
}

fn encode(c: vec3<f32>) -> vec3<f32> {
    if (u.params.y > 0.5) {
        return pow(max(c, vec3<f32>(0.0)), vec3<f32>(1.0 / 2.2));
    }
    return c;
}

@fragment
fn fs_main(in: VsOut, @builtin(front_facing) front: bool) -> @location(0) vec4<f32> {
    if (u.params.x > 1.5) {
        return vec4<f32>(encode(u.color_front.rgb), 1.0);
    }

    var n = normalize(in.normal);
    if (!front) {
        n = -n; // tecido de dois lados: inverte a normal nas costas
    }
    let base = select(u.color_back.rgb, u.color_front.rgb, front);

    let l = normalize(u.light_dir.xyz);
    let v = normalize(u.eye.xyz - in.world);
    let h = normalize(l + v);
    let ndl = max(dot(n, l), 0.0);
    let hemi = mix(0.22, 0.55, n.y * 0.5 + 0.5);
    // Brilho de borda leve (aproxima o "sheen" de tecidos), útil para ler as dobras.
    let rim = pow(1.0 - max(dot(n, v), 0.0), 3.0) * 0.12;
    let spec = pow(max(dot(n, h), 0.0), 40.0) * 0.12;

    var color = base * (hemi + 0.8 * ndl) + vec3<f32>(spec + rim);

    if (u.params.x > 0.5) {
        // Chão: grade de 0,5 m com linhas mais fortes a cada 1 m, esmaecendo com a distância.
        let p = in.world.xz;
        let w = max(fwidth(p), vec2<f32>(1.0e-4));
        let g = abs(fract(p * 2.0 - 0.5) - 0.5) / (w * 2.0);
        let line = 1.0 - clamp(min(g.x, g.y), 0.0, 1.0);
        let fade = 1.0 - clamp(length(u.eye.xz - p) / 14.0, 0.0, 1.0);
        color = mix(color, color * 0.55, line * 0.7 * fade);
    }

    return vec4<f32>(encode(color), 1.0);
}
