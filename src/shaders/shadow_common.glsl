// Shared by mesh.frag, the shadow projection shaders, and wash_shadow.comp: the scene camera, the
// disintegration death field, and the wash-plane projection used to rasterize letter/rod shadows.
// Keeping the death field here is load-bearing: the shadow discard and the fragment's visible
// discard must read the same field, or the cast shadow would not match the burned-away geometry.
//
// CAM_SET selects the descriptor set that holds the camera UBO (1 for the mesh/compute paths, 0
// for the shadow-projection pass which has no texture set).

#ifndef CAM_SET
#define CAM_SET 1
#endif

layout(set = CAM_SET, binding = 0) uniform Camera {
    mat4 viewProj;
    vec4 camPos;
    vec4 params;     // x = card aspect (W/H), y = card world half-height, zw unused
    vec4 tipLight;   // xyz = world position, w = range (0 = off)
    vec4 tipColor;   // rgb = colour, a = intensity
    vec4 tipLight2;  // second tip light, same layout as tipLight
    vec4 tipColor2;  // rgb = colour, a = intensity (0 = off)
    vec4 tipShadow;  // x = tipLight cast-shadow gate, y = tipLight2 gate (0 = casts no shadow, 1 = full)
} cam;

// Fixed directional key light, direction toward the source (+x right, +y up, +z toward viewer): the
// dominant +z keeps front faces lit; the upper-left tilt throws each letter's shadow to its lower-right.
const vec3 KEY_LIGHT_DIR = vec3(-0.4, 0.4, 1.0);
// Direction the key shadow is projected along, separate from the shading direction. A directional
// shadow's displacement scales with L.xy / L.z, so halving xy shortens the cast-shadow throw by 50%.
const vec3 KEY_SHADOW_DIR = vec3(KEY_LIGHT_DIR.xy * 0.5, KEY_LIGHT_DIR.z);

// World z of the virtual source the tip light's cast shadow is projected from, distinct from the
// real shading light: held near the card plane for a long grazing streak, just in front of the glyph faces.
const float TIP_SHADOW_Z = 0.0978;
// Disk radius for the soft-shadow PCF (wider = softer, more scattered penumbra). Key and tip share
// one radius; wash_shadow.comp samples all three channels at the same offsets.
const float SHADOW_PCF_RADIUS = 0.02;

float hash13(vec3 p) {
    p = fract(p * 0.1031);
    p += dot(p, p.zyx + 31.32);
    return fract((p.x + p.y) * p.z);
}
// Value noise on the card plane; reuses the 3D hash at z = 0 so four lattice taps suffice.
float vnoise(vec2 x) {
    vec3 i = vec3(floor(x), 0.0);
    vec2 f = fract(x);
    vec2 u = f * f * (3.0 - 2.0 * f);
    float n0 = mix(hash13(i + vec3(0,0,0)), hash13(i + vec3(1,0,0)), u.x);
    float n1 = mix(hash13(i + vec3(0,1,0)), hash13(i + vec3(1,1,0)), u.x);
    return mix(n0, n1, u.y);
}
// Card-plane death field for the disintegration front; no time term, so it is stable frame to frame
// and the front advances by raising the dissolve progress, not by animating the field.
float dissolveField(vec2 p) {
    return 0.6 * vnoise(p * 28.0) + 0.4 * vnoise(p * 75.6);
}
// Per-fragment death threshold: the noise field plus a bottom-to-top rise bias. mesh.frag and
// shadow.frag must use this same value so a burning letter's shadow erodes with its visible bits.
float dissolveDeath(vec3 world) {
    float halfH = max(cam.params.y, 1e-4);
    float rise = clamp((world.y + halfH) / (2.0 * halfH), 0.0, 1.0);
    return mix(dissolveField(world.xy), rise, 0.3);
}

// Map a card-plane world XY to the wash-shadow image's UV (u across x in [-1,1], v down y from
// +halfH to -halfH). Inverse of the reconstruction in wash_shadow.comp.
vec2 worldToWashUv(vec2 w) {
    float halfH = max(cam.params.y, 1e-4);
    return vec2(w.x * 0.5 + 0.5, 0.5 - w.y / (2.0 * halfH));
}

// Project a world-space occluder point onto the card plane (z = 0) and return Vulkan clip for the
// wash-shadow target: x = world x in [-1,1], y flipped so +world-y is the top of the image.
// lightMode 0 = key (directional), 1 = tipLight, 2 = tipLight2.
vec4 projectToWashClip(vec3 world, int lightMode) {
    vec3 projP;
    if (lightMode <= 0) {
        vec3 L = KEY_SHADOW_DIR;
        projP = world + L * (-world.z / max(L.z, 1e-6));
    } else {
        vec4 light = lightMode == 1 ? cam.tipLight : cam.tipLight2;
        vec3 S = vec3(light.xy, TIP_SHADOW_Z);
        vec3 d = world - S;
        float dz = abs(d.z) < 1e-6 ? 1e-6 : d.z;
        projP = S + d * (-S.z / dz);
    }
    float halfH = max(cam.params.y, 1e-4);
    return vec4(projP.x, -projP.y / halfH, 0.0, 1.0);
}
