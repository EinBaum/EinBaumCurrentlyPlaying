#version 460

// Shadow-pass vertex stage. Transforms into world, then projects onto the card plane along the
// light selected by pc.mode. The tessellated rod path overwrites gl_Position in shadow.tese after
// Phong smoothing.

#define CAM_SET 0
#include "shadow_common.glsl"

layout(location = 0) in vec3 inPos;
layout(location = 1) in vec3 inNormal;
layout(location = 2) in vec2 inUV;

layout(push_constant) uniform PC {
    mat4 model;
    vec4 color;
    float fade;
    int mode;
} pc;

layout(location = 0) out vec3 vNormal;
layout(location = 1) out vec2 vUV;
layout(location = 2) out vec3 vWorld;

void main() {
    vec4 world = pc.model * vec4(inPos, 1.0);
    vWorld = world.xyz;
    vNormal = mat3(pc.model) * inNormal;
    vUV = inUV;
    gl_Position = projectToWashClip(vWorld, pc.mode);
}
