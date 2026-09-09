#version 460

// Phong-smoothed rod, then projected onto the card plane. Same smoothing as mesh.tese so the
// groove's cast shadow matches the rounded silhouette rather than the coarse control cage.

layout(triangles, equal_spacing, ccw) in;

#define CAM_SET 0
#include "shadow_common.glsl"

layout(location = 0) in vec3 tcNormal[];
layout(location = 1) in vec2 tcUV[];
layout(location = 2) in vec3 tcWorld[];

layout(push_constant) uniform PC {
    mat4 model;
    vec4 color;
    float fade;
    int mode;
    float barLen;
    float tessLevel;
} pc;

layout(location = 0) out vec3 vNormal;
layout(location = 1) out vec2 vUV;
layout(location = 2) out vec3 vWorld;

vec3 project(vec3 p, vec3 c, vec3 n) {
    return p - dot(p - c, n) * n;
}

void main() {
    vec3 bc = gl_TessCoord;

    vec3 n0 = normalize(tcNormal[0]);
    vec3 n1 = normalize(tcNormal[1]);
    vec3 n2 = normalize(tcNormal[2]);

    vec3 flat_p = bc.x * tcWorld[0] + bc.y * tcWorld[1] + bc.z * tcWorld[2];
    vec3 n      = normalize(bc.x * n0 + bc.y * n1 + bc.z * n2);
    vec2 uv     = bc.x * tcUV[0] + bc.y * tcUV[1] + bc.z * tcUV[2];

    vec3 phong = bc.x * project(flat_p, tcWorld[0], n0)
               + bc.y * project(flat_p, tcWorld[1], n1)
               + bc.z * project(flat_p, tcWorld[2], n2);
    vec3 pos = mix(flat_p, phong, 0.75);

    vNormal = n;
    vUV = uv;
    vWorld = pos;
    gl_Position = projectToWashClip(pos, pc.mode);
}
