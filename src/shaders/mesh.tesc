#version 460

layout(vertices = 3) out;

layout(location = 0) in vec3 vNormal[];
layout(location = 1) in vec2 vUV[];
layout(location = 2) in vec3 vWorld[];

layout(push_constant) uniform PC {
    mat4 model;
    vec4 color;
    float fade;
    int mode;
    float barLen;
    float tessLevel;
} pc;

layout(location = 0) out vec3 tcNormal[];
layout(location = 1) out vec2 tcUV[];
layout(location = 2) out vec3 tcWorld[];

void main() {
    tcNormal[gl_InvocationID] = vNormal[gl_InvocationID];
    tcUV[gl_InvocationID]     = vUV[gl_InvocationID];
    tcWorld[gl_InvocationID]  = vWorld[gl_InvocationID];
    gl_out[gl_InvocationID].gl_Position = gl_in[gl_InvocationID].gl_Position;

    if (gl_InvocationID == 0) {
        float level = max(1.0, pc.tessLevel);
        gl_TessLevelInner[0] = level;
        gl_TessLevelOuter[0] = level;
        gl_TessLevelOuter[1] = level;
        gl_TessLevelOuter[2] = level;
    }
}
