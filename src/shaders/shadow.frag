#version 460

// Hard projected occlusion for the wash-shadow target. Writes (1 - fade) into the channel selected
// by pc.mode (R = key, G = tip) and 1 elsewhere; MIN blending against a clear of 1 accumulates
// overlapping letters without touching the other light's channel. fade = 1 is a solid occluder
// (writes 0); fade = 0 casts no shadow (writes 1). Text and its shadow share this fade.

layout(push_constant) uniform PC {
    mat4 model;
    vec4 color;
    float fade;
    int mode;
} pc;

layout(location = 0) out vec4 outColor;

void main() {
    float occ = 1.0 - pc.fade;
    vec4 o = vec4(1.0);
    if (pc.mode <= 0) o.r = occ;
    else o.g = occ;
    outColor = o;
}
