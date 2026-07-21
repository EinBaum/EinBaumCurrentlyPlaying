// Shared between the renderer translation units (renderer.cpp, renderer_vk.cpp,
// renderer_raytracing.cpp): the Vulkan result check and the GPU-facing data blocks.
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
    vec4 params;     // x = card aspect (W/H); y = card world half-height; zw = marquee shadow-clip column
    vec4 tipLight;   // progress-bar tip point light: xyz = world position, w = range (0 = off)
    vec4 tipColor;   // tip light: rgb = colour, a = intensity
    vec4 tipLight2;  // second tip point light: the outgoing playhead during a song change
    vec4 tipColor2;  // second tip light: rgb = colour, a = intensity (0 = off)
    vec4 tipShadow;  // x = tipLight cast-shadow gate, y = tipLight2 gate (0 = casts no shadow)
};

// Push constant for the 3D mesh pipeline (128 B). Matches mesh.vert/mesh.tesc/mesh.tese/mesh.frag;
// the earlier stages declare a shorter prefix of this block, which the shared layout permits.
struct MeshPush {
    mat4 model;
    vec4 color;
    float fade;
    int32_t mode;      // MeshMode
    float barLen;      // Bar: the fill rod's world length, sizing the white-hot tip in fixed world units
    float tessLevel;   // tessellation subdivision factor (ignored by the non-tessellated pipeline)
    float barDissolve; // >0 = progress (0..1) of the bar fill's left-to-right dissolve-out
    float dissolve;    // >0 = text-disintegration progress (0..1)
    float liveBar;     // Bar: >0 = live-stream fill — uniform glow, no playhead tip
    float washDim;     // Wash: per-cover darkening factor (also pads dissolveColor to offset 112)
    vec4 dissolveColor;
};
