// dataset-dreamer WebGPU Latent Drift Shader
// 90s/00s aesthetic: wireframe, scanlines, phosphor glow, sigil geometry

const LATENT_VERTEX_SHADER = `
struct VertexInput {
    @location(0) position: vec3<f32>,
    @location(1) latent: vec3<f32>,
    @location(2) signal: vec4<f32>,
    @location(3) idx: u32,
};

struct VertexOutput {
    @builtin(position) position: vec4<f32>,
    @location(0) latent: vec3<f32>,
    @location(1) signal: vec4<f32>,
    @location(2) idx: u32,
    @location(3) world_pos: vec3<f32>,
};

@group(0) @binding(0) var<uniform> camera: mat4x4<f32>;
@group(0) @binding(1) var<uniform> projection: mat4x4<f32>;
@group(0) @binding(2) var<uniform> time: f32;
@group(0) @binding(3) var<uniform> params: Params;

struct Params {
    point_size: f32,
    trail_length: f32,
    signal_influence: f32,
    scanline_intensity: f32,
    phosphor_decay: f32,
    wireframe: u32,
}

@vertex
fn vs_main(input: VertexInput) -> VertexOutput {
    var output: VertexOutput;

    // animate position based on latent coords and signals
    var pos = input.position;

    // signal-driven displacement
    let signal_strength = length(input.signal.xyz) * params.signal_influence;
    let wave = sin(pos.x * 2.0 + time * 0.5) * cos(pos.y * 3.0 + time * 0.3) * signal_strength * 0.1;
    pos.z += wave;

    // latent space drift
    let drift = vec3<f32>(
        sin(time * 0.15 + input.latent.x * 6.28) * 0.05,
        cos(time * 0.12 + input.latent.y * 6.28) * 0.05,
        sin(time * 0.18 + input.latent.z * 6.28) * 0.03
    );
    pos += drift * params.signal_influence;

    output.world_pos = pos;
    output.position = projection * camera * vec4<f32>(pos, 1.0);
    output.latent = input.latent;
    output.signal = input.signal;
    output.idx = input.idx;

    return output;
}
`;

const LATENT_FRAGMENT_SHADER = `
@group(0) @binding(4) var<uniform> frame_params: FrameParams;

struct FrameParams {
    resolution: vec2<f32>,
    time: f32,
    scanline_intensity: f32,
    phosphor_decay: f32,
    wireframe: u32,
    point_size: f32,
}

@fragment
fn fs_main(@builtin(position) frag_coord: vec2<f32>) -> @location(0) vec4<f32> {
    // this is a full-screen quad shader for post-processing
    // actual point rendering uses point_sprites below
    return vec4<f32>(0.0);
}
`;

const POINT_SPRITE_VERTEX = `
struct VertexInput {
    @location(0) position: vec3<f32>,
    @location(1) latent: vec3<f32>,
    @location(2) signal: vec4<f32>,
    @location(3) idx: u32,
};

struct VertexOutput {
    @builtin(position) position: vec4<f32>,
    @builtin(point_size) point_size: f32,
    @location(0) latent: vec3<f32>,
    @location(1) signal: vec4<f32>,
    @location(2) idx: u32,
    @location(3) world_pos: vec3<f32>,
};

@group(0) @binding(0) var<uniform> camera: mat4x4<f32>;
@group(0) @binding(1) var<uniform> projection: mat4x4<f32>;
@group(0) @binding(2) var<uniform> time: f32;
@group(0) @binding(3) var<uniform> params: Params;

struct Params {
    point_size: f32,
    trail_length: f32,
    signal_influence: f32,
    scanline_intensity: f32,
    phosphor_decay: f32,
    wireframe: u32,
    camera_pos: vec3<f32>,
}

@vertex
fn vs_main(input: VertexInput) -> VertexOutput {
    var output: VertexOutput;

    var pos = input.position;

    // signal-driven organic motion
    let signal_strength = length(input.signal.xyz) * params.signal_influence;

    // multi-octave noise for organic drift
    let t = time;
    let seed = f32(input.idx) * 0.01;
    let noise1 = sin(pos.x * 1.7 + t * 0.3 + seed) * cos(pos.y * 1.3 + t * 0.2 + seed);
    let noise2 = sin(pos.x * 3.1 + t * 0.5 + seed * 2.0) * cos(pos.y * 2.7 + t * 0.4 + seed * 2.0);
    let noise3 = sin(pos.x * 5.3 + t * 0.8 + seed * 3.0) * cos(pos.y * 4.1 + t * 0.6 + seed * 3.0);

    let drift = vec3<f32>(
        (noise1 * 0.5 + noise2 * 0.25 + noise3 * 0.125) * 0.15 * signal_strength,
        (noise1 * 0.5 + noise2 * 0.25 + noise3 * 0.125) * 0.15 * signal_strength * 0.7,
        (noise1 * 0.3 + noise2 * 0.15) * 0.1 * signal_strength
    );

    pos += drift;

    // latent coordinate influence
    let latent_pull = input.latent * 0.5;
    pos += latent_pull * 0.3;

    output.world_pos = pos;
    output.position = projection * camera * vec4<f32>(pos, 1.0);

    // perspective-correct point size
    let dist = distance(params.camera_pos, pos);
    output.point_size = params.point_size * (50.0 / max(dist, 1.0));

    output.latent = input.latent;
    output.signal = input.signal;
    output.idx = input.idx;

    return output;
}
`;

const POINT_SPRITE_FRAGMENT = `
struct VertexOutput {
    @builtin(position) position: vec4<f32>,
    @builtin(point_size) point_size: f32,
    @location(0) latent: vec3<f32>,
    @location(1) signal: vec4<f32>,
    @location(2) idx: u32,
    @location(3) world_pos: vec3<f32>,
};

@group(0) @binding(4) var<uniform> frame_params: FrameParams;

struct FrameParams {
    resolution: vec2<f32>,
    time: f32,
    scanline_intensity: f32,
    phosphor_decay: f32,
    wireframe: u32,
    point_size: f32,
}

@fragment
fn fs_main(input: VertexOutput) -> @location(0) vec4<f32> {
    // circular point sprite
    let center = vec2<f32>(0.5, 0.5);
    let coord = (vec2<f32>(gl_PointCoord.x, gl_PointCoord.y) - center) * 2.0;
    let dist = length(coord);

    if (dist > 1.0) {
        discard;
    }

    // core color from latent coordinates
    let latent_hue = input.latent.x * 0.3 + input.latent.y * 0.5 + input.latent.z * 0.2;
    let signal_intensity = length(input.signal.xyz);

    // Y2K violet palette with signal modulation
    let base_hue = 270.0 + latent_hue * 60.0 + signal_intensity * 40.0;
    let saturation = 0.6 + signal_intensity * 0.4;
    let brightness = 0.4 + signal_intensity * 0.6;

    // phosphor glow falloff
    let falloff = 1.0 - smoothstep(0.0, 1.0, dist);
    falloff = pow(falloff, 2.0);

    // pulsing
    let pulse = 0.7 + 0.3 * sin(frame_params.time * 2.0 + f32(input.idx) * 0.1);

    // hsv to rgb
    let h = base_hue / 360.0;
    let s = saturation;
    let v = brightness * pulse;

    let i = floor(h * 6.0);
    let f = h * 6.0 - i;
    let p = v * (1.0 - s);
    let q = v * (1.0 - f * s);
    let t = v * (1.0 - (1.0 - f) * s);

    var rgb: vec3<f32>;
    if (i == 0.0 || i == 6.0) { rgb = vec3<f32>(v, t, p); }
    else if (i == 1.0) { rgb = vec3<f32>(q, v, p); }
    else if (i == 2.0) { rgb = vec3<f32>(p, v, t); }
    else if (i == 3.0) { rgb = vec3<f32>(p, q, v); }
    else if (i == 4.0) { rgb = vec3<f32>(t, p, v); }
    else { rgb = vec3<f32>(v, p, q); }

    // phosphor glow
    let glow = falloff * (1.0 + signal_intensity);

    // scanline effect
    let scanline = sin(input.world_pos.y * 50.0 + frame_params.time * 10.0) * frame_params.scanline_intensity;
    rgb *= 1.0 + scanline * 0.1;

    // vignette
    let vignette = 1.0 - length(coord) * 0.3;
    rgb *= vignette;

    let alpha = falloff * glow * 0.8;

    return vec4<f32>(rgb, alpha);
}
`;

const WIREFRAME_VERTEX = `
struct VertexInput {
    @location(0) position: vec3<f32>,
    @location(1) latent: vec3<f32>,
    @location(2) signal: vec4<f32>,
    @location(3) idx: u32,
};

struct VertexOutput {
    @builtin(position) position: vec4<f32>,
    @location(0) latent: vec3<f32>,
    @location(1) signal: vec4<f32>,
    @location(2) idx: u32,
    @location(3) world_pos: vec3<f32>,
    @location(4) edge_factor: f32,
};

@group(0) @binding(0) var<uniform> camera: mat4x4<f32>;
@group(0) @binding(1) var<uniform> projection: mat4x4<f32>;
@group(0) @binding(2) var<uniform> time: f32;
@group(0) @binding(3) var<uniform> params: Params;

struct Params {
    point_size: f32,
    trail_length: f32,
    signal_influence: f32,
    scanline_intensity: f32,
    phosphor_decay: f32,
    wireframe: u32,
    camera_pos: vec3<f32>,
}

@vertex
fn vs_main(input: VertexInput) -> VertexOutput {
    var output: VertexOutput;

    var pos = input.position;

    // same drift as points
    let signal_strength = length(input.signal.xyz) * params.signal_influence;
    let t = time;
    let seed = f32(input.idx) * 0.01;
    let noise1 = sin(pos.x * 1.7 + t * 0.3 + seed) * cos(pos.y * 1.3 + t * 0.2 + seed);
    let noise2 = sin(pos.x * 3.1 + t * 0.5 + seed * 2.0) * cos(pos.y * 2.7 + t * 0.4 + seed * 2.0);

    let drift = vec3<f32>(
        (noise1 * 0.5 + noise2 * 0.25) * 0.15 * signal_strength,
        (noise1 * 0.5 + noise2 * 0.25) * 0.15 * signal_strength * 0.7,
        (noise1 * 0.3) * 0.1 * signal_strength
    );

    pos += drift;
    pos += input.latent * 0.5 * 0.3;

    output.world_pos = pos;
    output.position = projection * camera * vec4<f32>(pos, 1.0);
    output.latent = input.latent;
    output.signal = input.signal;
    output.idx = input.idx;
    output.edge_factor = 1.0;

    return output;
}
`;

const WIREFRAME_FRAGMENT = `
struct VertexOutput {
    @builtin(position) position: vec4<f32>,
    @location(0) latent: vec3<f32>,
    @location(1) signal: vec4<f32>,
    @location(2) idx: u32,
    @location(3) world_pos: vec3<f32>,
    @location(4) edge_factor: f32,
};

@group(0) @binding(4) var<uniform> frame_params: FrameParams;

struct FrameParams {
    resolution: vec2<f32>,
    time: f32,
    scanline_intensity: f32,
    phosphor_decay: f32,
    wireframe: u32,
    point_size: f32,
}

@fragment
fn fs_main(input: VertexOutput) -> @location(0) vec4<f32> {
    // wireframe lines - subtle grid
    let signal_intensity = length(input.signal.xyz);

    // color from latent
    let latent_hue = input.latent.x * 0.3 + input.latent.y * 0.5;
    let base_hue = 260.0 + latent_hue * 80.0;
    let h = base_hue / 360.0;
    let s = 0.4 + signal_intensity * 0.3;
    let v = 0.2 + signal_intensity * 0.4;

    let i = floor(h * 6.0);
    let f = h * 6.0 - i;
    let p = v * (1.0 - s);
    let q = v * (1.0 - f * s);
    let t = v * (1.0 - (1.0 - f) * s);

    var rgb: vec3<f32>;
    if (i == 0.0 || i == 6.0) { rgb = vec3<f32>(v, t, p); }
    else if (i == 1.0) { rgb = vec3<f32>(q, v, p); }
    else if (i == 2.0) { rgb = vec3<f32>(p, v, t); }
    else if (i == 3.0) { rgb = vec3<f32>(p, q, v); }
    else if (i == 4.0) { rgb = vec3<f32>(t, p, v); }
    else { rgb = vec3<f32>(v, p, q); }

    // scanline
    let scanline = sin(input.world_pos.y * 50.0 + frame_params.time * 10.0) * frame_params.scanline_intensity;
    rgb *= 1.0 + scanline * 0.2;

    return vec4<f32>(rgb, 0.3 + signal_intensity * 0.3);
}
`;

// Post-process shader for CRT/phosphor effect
const POST_PROCESS_SHADER = `
@group(0) @binding(0) var texture: texture_2d<f32>;
@group(0) @binding(1) var sampler: sampler;
@group(0) @binding(2) var<uniform> params: PostParams;

struct PostParams {
    resolution: vec2<f32>,
    time: f32,
    scanline_intensity: f32,
    phosphor_decay: f32,
    vignette: f32,
    chromatic_aberration: f32,
    noise_intensity: f32,
}

@vertex
fn vs_main(@builtin(vertex_index) idx: u32) -> @builtin(position) vec4<f32> {
    var pos = array<vec2<f32>, 4>(
        vec2<f32>(-1.0, -1.0),
        vec2<f32>(3.0, -1.0),
        vec2<f32>(-1.0, 3.0),
        vec2<f32>(3.0, 3.0)
    );
    return vec4<f32>(pos[idx], 0.0, 1.0);
}

@fragment
fn fs_main(@builtin(position) frag_coord: vec2<f32>) -> @location(0) vec4<f32> {
    let uv = frag_coord / params.resolution;

    // chromatic aberration
    let ca = params.chromatic_aberration * 0.002;
    let r = textureSample(texture, sampler, uv + vec2<f32>(ca, 0.0)).r;
    let g = textureSample(texture, sampler, uv).g;
    let b = textureSample(texture, sampler, uv - vec2<f32>(ca, 0.0)).b;

    var color = vec3<f32>(r, g, b);

    // scanlines
    let scanline = sin(uv.y * params.resolution.y * 0.5 + params.time * 50.0) * params.scanline_intensity;
    color *= 1.0 + scanline * 0.15;

    // vignette
    let vignette = 1.0 - length(uv - 0.5) * params.vignette * 1.5;
    color *= vignette;

    // film grain / noise
    let noise = fract(sin(dot(frag_coord + params.time, vec2<f32>(12.9898, 78.233))) * 43758.5453);
    color += (noise - 0.5) * params.noise_intensity * 0.02;

    // phosphor ghosting (simulated via temporal feedback would need ping-pong)
    // approximate with subtle glow
    let glow = length(color) * 0.1;
    color += color * glow * params.phosphor_decay;

    // gamma
    color = pow(max(color, vec3<f32>(0.0)), vec3<f32>(1.0 / 2.2));

    return vec4<f32>(color, 1.0);
}
`;

// Sigil geometry generator - creates Organization XIII-style glyphs
const SIGIL_GEOMETRY = `
// generates sigil vertices for latent space anchors
// each sigil = 13 vertices (Organization XIII seats)

fn generate_sigil(seat: u32, radius: f32, time: f32) -> array<vec3<f32>, 13> {
    var vertices: array<vec3<f32>, 13>;
    let base_angle = f32(seat) * 6.28318 / 13.0;
    let rotation = time * 0.05 + base_angle;

    for (var i = 0u; i < 13u; i++) {
        let angle = f32(i) * 6.28318 / 13.0 + rotation;
        let r = radius * (0.8 + 0.4 * sin(time * 0.3 + f32(i) * 0.5));
        vertices[i] = vec3<f32>(
            r * cos(angle),
            r * sin(angle),
            sin(time * 0.2 + f32(i)) * 0.2
        );
    }
    return vertices;
}
`;

// Export all shaders as a module
export const shaders = {
    latent_vertex: LATENT_VERTEX_SHADER,
    latent_fragment: LATENT_FRAGMENT_SHADER,
    point_sprite_vertex: POINT_SPRITE_VERTEX,
    point_sprite_fragment: POINT_SPRITE_FRAGMENT,
    wireframe_vertex: WIREFRAME_VERTEX,
    wireframe_fragment: WIREFRAME_FRAGMENT,
    post_process: POST_PROCESS_SHADER,
    sigil_geometry: SIGIL_GEOMETRY,
};

export type ShaderSource = keyof typeof shaders;
export function getShader(name: ShaderSource): string {
    return shaders[name];
}