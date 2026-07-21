// Shared by mesh.frag and wash_shadow.comp: the scene camera, the disintegration death field, and the
// soft ray-traced shadow trace. Included after #version and `#extension GL_EXT_ray_query : require`.
// Keeping the death field here is load-bearing: the shadow erosion and the fragment's visible discard
// must read the same field, or the cast shadow would not match the burned-away geometry.

layout(set = 1, binding = 0) uniform Camera {
    mat4 viewProj;
    vec4 camPos;
    vec4 params;     // x = card aspect (W/H), y = card world half-height, zw = marquee column edges (world x; w<=z off)
    vec4 tipLight;   // xyz = world position, w = range (0 = off)
    vec4 tipColor;   // rgb = colour, a = intensity
    vec4 tipLight2;  // second tip light, same layout as tipLight
    vec4 tipColor2;  // rgb = colour, a = intensity (0 = off)
    vec4 tipShadow;  // x = tipLight cast-shadow gate, y = tipLight2 gate (0 = casts no shadow, 1 = full)
} cam;
layout(set = 2, binding = 0) uniform accelerationStructureEXT sceneTlas;

// Fixed directional key light, direction toward the source (+x right, +y up, +z toward viewer): the
// dominant +z keeps front faces lit; the upper-left tilt throws each letter's shadow to its lower-right.
const vec3 KEY_LIGHT_DIR = vec3(-0.4, 0.4, 1.0);
// Direction the key shadow is traced toward, separate from the shading direction. A directional
// shadow's displacement scales with L.xy / L.z, so halving xy shortens the cast-shadow throw by 50%.
const vec3 KEY_SHADOW_DIR = vec3(KEY_LIGHT_DIR.xy * 0.5, KEY_LIGHT_DIR.z);

// World z of the virtual source the tip light's cast shadow is traced toward, distinct from the real
// shading light: held near the card plane for a long grazing streak, just in front of the glyph faces.
const float TIP_SHADOW_Z = 0.0978;
// Disk radius for the tip light's soft shadow (wider = softer penumbra).
const float TIP_SHADOW_RADIUS = 0.03;

// Fixed-point scale a dissolving glyph's progress is packed into its TLAS custom index with (matches
// DISS_ENCODE in renderer.cpp). DISS_ENC keeps the 0..~1.05 range well inside the 24-bit field.
const float DISS_ENC = 1048576.0;

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

// Soft ray-traced shadow toward L: eight rays across a disk of `radius`, tested against `mask` out to
// `tMax`, origin lifted off the surface to avoid acne. Returns 1 fully lit; at full occlusion 1 - hardness.
// A dissolving letter is non-opaque and carries its dissolve progress in the custom index, so each
// candidate hit is confirmed only where the death field shows the letter has not yet burned away.
float traceShadow(vec3 P, vec3 N, vec3 L, float tMax, uint mask, float hardness, float radius) {
    vec3 up = abs(L.y) < 0.99 ? vec3(0.0, 1.0, 0.0) : vec3(1.0, 0.0, 0.0);
    vec3 t = normalize(cross(up, L));
    vec3 b = cross(L, t);
    const vec2 J[8] = vec2[](
        vec2( 0.40,  0.0),  vec2(-0.40,  0.0),  vec2( 0.0,  0.40),  vec2( 0.0, -0.40),
        vec2( 0.64,  0.64), vec2(-0.64,  0.64), vec2( 0.64, -0.64), vec2(-0.64, -0.64));
    vec3 origin = P + N * 0.003 + L * 0.003;
    float halfH = max(cam.params.y, 1e-4);
    float occ = 0.0;
    for (int i = 0; i < 8; i++) {
        vec3 d = normalize(L + (t * J[i].x + b * J[i].y) * radius);
        rayQueryEXT rq;
        rayQueryInitializeEXT(rq, sceneTlas, gl_RayFlagsTerminateOnFirstHitEXT, mask, origin, 0.0, d, tMax);
        while (rayQueryProceedEXT(rq)) {
            float diss = float(rayQueryGetIntersectionInstanceCustomIndexEXT(rq, false)) * (1.0 / DISS_ENC);
            vec3 hp = origin + d * rayQueryGetIntersectionTEXT(rq, false);
            float rise = clamp((hp.y + halfH) / (2.0 * halfH), 0.0, 1.0);
            float death = mix(dissolveField(hp.xy), rise, 0.3);
            if (death - diss > 0.0) rayQueryConfirmIntersectionEXT(rq);
        }
        if (rayQueryGetIntersectionTypeEXT(rq, true) != gl_RayQueryCommittedIntersectionNoneEXT)
            occ += 1.0;
    }
    return 1.0 - (occ * 0.125) * hardness;
}
