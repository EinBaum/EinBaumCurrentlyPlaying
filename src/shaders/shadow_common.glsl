// Shared by mesh.frag, the shadow projection shaders, and wash_shadow.comp: the scene camera and
// the wash-plane projection used to rasterize letter/rod shadows.
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
// one radius; wash_shadow.comp samples all channels at the same offsets.
const float SHADOW_PCF_RADIUS = 0.02;

// Map a card-plane world XY to the wash-shadow image's UV (u across x in [-1,1], v down y from
// +halfH to -halfH). Inverse of the reconstruction in wash_shadow.comp.
vec2 worldToWashUv(vec2 w) {
    float halfH = max(cam.params.y, 1e-4);
    return vec2(w.x * 0.5 + 0.5, 0.5 - w.y / (2.0 * halfH));
}

// Project a world-space occluder point onto the card plane (z = 0) and return Vulkan clip for the
// wash-shadow target: x = world x in [-1,1], y flipped so +world-y is the top of the image.
// lightMode 0 = key (directional), 1 = tipLight.
vec4 projectToWashClip(vec3 world, int lightMode) {
    vec3 projP;
    if (lightMode <= 0) {
        vec3 L = KEY_SHADOW_DIR;
        projP = world + L * (-world.z / max(L.z, 1e-6));
    } else {
        vec3 S = vec3(cam.tipLight.xy, TIP_SHADOW_Z);
        vec3 d = world - S;
        float dz = abs(d.z) < 1e-6 ? 1e-6 : d.z;
        projP = S + d * (-S.z / dz);
    }
    float halfH = max(cam.params.y, 1e-4);
    return vec4(projP.x, -projP.y / halfH, 0.0, 1.0);
}
