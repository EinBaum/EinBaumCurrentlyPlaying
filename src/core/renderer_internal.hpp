// Shared between the renderer translation units (renderer.cpp, renderer_vk.cpp): the Vulkan
// result check and the GPU-facing data blocks. CameraUBO / MeshPush must match the GLSL
// Camera block in shadow_common.glsl and the PC blocks in the mesh and shadow shaders.
#pragma once
#include <vulkan/vulkan.h>
#include <cstdint>
#include <source_location>
#include "core/math3d.hpp"

void vkCheck(VkResult r, std::source_location loc = std::source_location::current());

// Fragment path selector carried in MeshPush::mode; names and values match the M_* constants in
// mesh.frag (Flat = unlit, used for the album cover).
enum class MeshMode : int32_t { Lit = 0, Wash = 1, Bar = 2, Flat = 3 };

// set 1 of the 3D mesh pipeline: bound once per frame. std140 layout (mat4 64B, vec4s 16B).
struct CameraUBO {
    mat4 viewProj;
    vec4 camPos;
    vec4 params;     // x = card aspect (W/H); y = card world half-height; zw unused (std140 padding)
    vec4 tipLight;   // progress-bar tip point light: xyz = world position, w = range (0 = off)
    vec4 tipColor;   // tip light: rgb = colour, a = intensity
};

// Push constant for the mesh and shadow pipelines. Matches the PC blocks in
// mesh.vert/mesh.tesc/mesh.tese/mesh.frag and the shadow shaders; earlier stages declare a
// shorter prefix of this block, which the shared layout permits. In the shadow pass, `mode`
// selects the light (0 = key, 1 = tip) rather than MeshMode.
struct MeshPush {
    mat4 model;
    vec4 color;
    float fade;
    int32_t mode;      // MeshMode
    float barLen;      // Bar: the fill rod's world length, sizing the white-hot tip in fixed world units
    float tessLevel;   // tessellation subdivision factor (ignored by the non-tessellated pipeline)
    float liveBar;     // Bar: >0 = live-stream fill — uniform glow, no playhead tip
    float washDim;     // Wash: per-cover darkening factor
};
