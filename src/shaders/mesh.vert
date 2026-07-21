#version 450

// model is a per-element translate/scale with no rotation. Normals are axis-aligned (quads),
// uniformly scaled (glyphs), or radial with equal cross-section scale (rod), so mat3(model)
// transforms them correctly and no inverse-transpose is needed.

layout(location = 0) in vec3 inPos;
layout(location = 1) in vec3 inNormal;
layout(location = 2) in vec2 inUV;

layout(set = 1, binding = 0) uniform Camera {
    mat4 viewProj;
    vec4 camPos;
    vec4 params;
} cam;

layout(push_constant) uniform PC {
    mat4 model;
    vec4 color;
    float fade;
    int mode;
    float barLen;
} pc;

layout(location = 0) out vec3 vNormal;
layout(location = 1) out vec2 vUV;
layout(location = 2) out vec3 vWorld;

void main() {
    vec4 world = pc.model * vec4(inPos, 1.0);
    vWorld = world.xyz;
    vNormal = mat3(pc.model) * inNormal;
    vUV = inUV;
    gl_Position = cam.viewProj * world;
}
