#version 460

// Hard projected occlusion for the wash-shadow target. Writes 0 into the channel selected by
// pc.mode (R = key, G = tip 1, B = tip 2) and 1 elsewhere; MIN blending against a clear of 1
// accumulates overlapping letters without touching the other lights' channels. Dissolve uses the
// same death field as mesh.frag so a burning letter's shadow erodes with its visible fragments.

#define CAM_SET 0
#include "shadow_common.glsl"

layout(location = 2) in vec3 vWorld;

layout(push_constant) uniform PC {
    mat4 model;
    vec4 color;
    float fade;
    int mode;
    float barLen;
    float tessLevel;
    float barDissolve;
    float dissolve;
} pc;

layout(location = 0) out vec4 outColor;

void main() {
    if (pc.dissolve > 0.0 && dissolveDeath(vWorld) - pc.dissolve <= 0.0) discard;
    vec4 o = vec4(1.0);
    if (pc.mode <= 0) o.r = 0.0;
    else if (pc.mode == 1) o.g = 0.0;
    else o.b = 0.0;
    outColor = o;
}
