#version 460

layout(location = 0) in vec3 vNormal;
layout(location = 1) in vec2 vUV;
layout(location = 2) in vec3 vWorld;

layout(set = 0, binding = 0) uniform sampler2D tex;
// Half-res key/tip shadow visibilities from wash_shadow.comp: R = key, G = tip light.
layout(set = 1, binding = 1) uniform sampler2D washShadow;

#include "shadow_common.glsl"   // Camera (set 1, binding 0), key/tip light constants

layout(push_constant) uniform PC {
    mat4 model;
    vec4 color;
    float fade;
    int mode;
    float barLen;          // M_BAR: the fill rod's current world length
    float tessLevel;
    float liveBar;         // >0 = live-stream fill: uniform glow, no white-hot tip
    float washDim;         // M_WASH: per-cover darkening factor for the blurred background
} pc;

layout(location = 0) out vec4 outColor;

const int M_LIT  = 0;
const int M_WASH = 1;
const int M_BAR  = 2;
const int M_FLAT = 3;

// How strongly the key light's cast shadow darkens the card: 0 leaves it fully lit, 1 applies the full
// filtered-occlusion depth.
const float KEY_SHADOW_STRENGTH = 1.0;

// How strongly the tip light's cast shadow removes the playhead glow: 0 leaves the pool fully lit, 1
// applies the full filtered-occlusion depth. At TIP_SHADOW_STRENGTH the deepest streak keeps 1 - TIP_SHADOW_STRENGTH of the added glow.
const float TIP_SHADOW_STRENGTH = 0.90;

// How far the tip's own letter-streak dims the bare wash, on top of removing the glow. Kept modest:
// the projected tip silhouette slides with the playhead, and a strong wash dim looks like letter
// shadows crawling. Scaled by the light's reach so it fades with the pool.
const float TIP_SHADOW_DARKEN = 0.45;

float ditherHash(vec2 p) { return fract(sin(dot(p, vec2(12.9898, 78.233))) * 43758.5453); }

// Card background wash. A square crop of the album cover is centred over the wide card, read from a
// near-top mip (textureQueryLevels - 1.5) so a single tap yields a heavy blur, then desaturated,
// darkened, and vignetted. No time term: the wash holds still behind the foreground. `dim` is the
// per-cover darkening the CPU scaled to the cover's brightness, so every wash reads against the same
// backdrop level regardless of how light or dark its cover is.
vec3 blurWash(vec2 uv, float dim) {
    float aspect = cam.params.x;
    vec2 p = clamp(vec2((uv.x - 0.5) / aspect, uv.y - 0.5) * 0.9 + 0.5, 0.0, 1.0);
    vec3 acc = textureLod(tex, p, max(0.0, float(textureQueryLevels(tex)) - 1.5)).rgb;
    float luma = dot(acc, vec3(0.299, 0.587, 0.114));
    acc = mix(vec3(luma), acc, 0.82);
    acc *= dim;
    acc *= smoothstep(1.05, 0.2, distance(uv, vec2(0.5)));   // vignette toward the card edges
    return acc;
}

// Progress-bar tip light: a coloured point light with a smooth distance falloff to zero at its range.
// When `shadowed`, `shVis` is the cast-shadow visibility toward the light (the half-res buffer sampled
// by the caller), turned into the glow attenuation `sh`; the rod is left out of the tip occluder set
// since the light sits on it. `washOcc` dims the bare wash where a letter blocks the streak. M_LIT
// passes shadowed = false: the letter faces are frontmost, so nothing can fall between them and a light
// in front of them. A dark light (intensity 0 or out of range) returns before using `shVis`.
vec3 tipLight(vec3 P, vec3 N, vec3 V, vec3 albedo, bool shadowed, float shVis, vec4 light, vec4 color, out float washOcc) {
    washOcc = 1.0;
    float range = light.w, intensity = color.a;
    vec3 toL = light.xyz - P;
    float dist = length(toL);
    if (intensity <= 0.0 || dist >= range) return vec3(0.0);
    vec3 L = toL / max(dist, 1e-4);
    float at = 1.0 - dist / range;
    at *= at;
    float sh = 1.0;
    if (shadowed) {
        sh = mix(1.0, shVis, TIP_SHADOW_STRENGTH);
        washOcc = mix(1.0, sh, TIP_SHADOW_DARKEN * at);
    }
    float diff = max(dot(N, L), 0.0);
    float spec = pow(max(dot(N, normalize(L + V)), 0.0), 28.0);
    return color.rgb * (intensity * at * sh) * (albedo * diff + vec3(spec * 0.12));
}

void main() {
    if (pc.mode == M_FLAT) {
        vec4 tx = texture(tex, vUV);
        outColor = vec4(tx.rgb * pc.color.rgb, tx.a * pc.fade);
        return;
    }

    vec3 N = normalize(vNormal);

    if (pc.mode == M_WASH) {
        vec3 w = blurWash(vUV, pc.washDim);
        // The wash quad covers the full window at z=0, so vUV is the screen UV: it indexes the half-res
        // shadow buffer at the same world point wash_shadow.comp filtered. R = key, G = the tip light.
        vec3 vis = texture(washShadow, vUV).rgb;
        vec3 Vw = normalize(cam.camPos.xyz - vWorld);
        float occ;
        vec3 glow = tipLight(vWorld, N, Vw, vec3(0.8), true, vis.g, cam.tipLight, cam.tipColor, occ);
        // Key letter shadows stay put. The playhead still lights the wash additively (`glow`); mixing
        // the key term toward 1 with tip fill made those shadows crawl and punch out as the bar moved.
        w *= mix(1.0, vis.r, KEY_SHADOW_STRENGTH);
        w *= occ;
        w += glow;
        // Triangular-PDF dither (two hashes summed, recentred) breaks 8-bit banding in the dark wash
        // gradient; the lit solids carry no smooth gradient to band.
        float dz = (ditherHash(gl_FragCoord.xy) + ditherHash(gl_FragCoord.xy + 13.37) - 1.0) * (0.5 / 255.0);
        outColor = vec4(w + dz, pc.fade);
        return;
    }

    if (pc.mode == M_BAR) {
        // Emissive neon progress rod. localx runs 0 at the bar's start to 1 at the playhead tip; the
        // cylinder's uv.y is 1 at the start cap and 0 at the tip cap, hence the 1.0 - vUV.y.
        float localx = clamp(1.0 - vUV.y, 0.0, 1.0);
        vec3 accent = pc.color.rgb;
        vec3 V = normalize(cam.camPos.xyz - vWorld);
        float fres = pow(1.0 - max(dot(N, V), 0.0), 3.0);
        bool live = pc.liveBar > 0.0;
        // A live stream has no playhead, so the rod glows at a uniform mid brightness with neither the
        // length ramp nor the white-hot tip below; nothing along it reads as an end point.
        float ramp = live ? 0.85 : mix(0.45, 1.15, localx * localx * localx);
        // White-hot leading edge of fixed world length TIP_HOT_LEN at the playhead. localx is normalised
        // over the fill's current world length (pc.barLen), so dividing keeps the hot tip a constant
        // physical size at any played fraction. The clamp caps tipSpan at the whole rod when barLen is
        // shorter than TIP_HOT_LEN, so a stub runs fully hot.
        const float TIP_HOT_LEN = 0.07;
        float tipSpan = clamp(TIP_HOT_LEN / max(pc.barLen, 1e-4), 0.0, 1.0);
        float tip = live ? 0.0 : smoothstep(1.0 - tipSpan, 1.0, localx);
        vec3 body = accent * 0.20;
        vec3 emit = accent * ramp + accent * fres * 0.8;
        // TIP_WHITE < 1 keeps an accent tint at the peak, not full white
        const float TIP_WHITE = 0.7;
        emit += mix(accent, vec3(1.0), tip * TIP_WHITE) * tip;
        outColor = vec4(body + emit, pc.fade);
        return;
    }

    vec4 tx = texture(tex, vUV);
    vec3 albedo = tx.rgb * pc.color.rgb;
    vec3 V = normalize(cam.camPos.xyz - vWorld);
    vec3 L = normalize(KEY_LIGHT_DIR);
    vec3 H = normalize(L + V);

    float diff = max(dot(N, L), 0.0);
    float spec = pow(max(dot(N, H), 0.0), 28.0);
    float ambient = 0.36;

    // ambient + 0.60 * diff stays at or below 1 so a lit face never brightens past its albedo and clips
    // to white; the specular is a small additive term on top. The tip light passes shadowed = false: the
    // letter faces are frontmost, so nothing can fall between them and a light in front of them.
    vec3 lit = albedo * (ambient + 0.60 * diff) + vec3(spec * 0.08);
    float washOcc;   // written by tipLight but unused here (shadowed = false)
    lit += tipLight(vWorld, N, V, albedo, false, 1.0, cam.tipLight, cam.tipColor, washOcc);
    outColor = vec4(lit, tx.a * pc.fade);
}
