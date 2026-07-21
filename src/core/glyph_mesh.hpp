#pragma once
#include "core/math3d.hpp"
#include <cstdint>
#include <vector>

// Half the glyph extrusion depth, in em-normalized units: a letter spans z in [-this, +this].
inline constexpr float GLYPH_EXTRUDE_HALF = 0.07f;

// A glyph outline as flattened polygon contours, em-normalized (1.0 == em), y up. The platform font
// backend fills this; tessellateGlyph turns it into the extruded 3D mesh. Whitespace carries only
// `advance` and no contours.
struct GlyphOutline {
    std::vector<std::vector<vec2>> contours;  // each contour is a closed ring
    float advance = 0.0f;                     // horizontal pen advance, em-normalized
};

struct GlyphMesh3 {
    std::vector<Vertex3> verts;
    std::vector<uint32_t> indices;
    float advance = 0.0f;   // horizontal pen advance, em-normalized
    [[nodiscard]] bool empty() const noexcept { return indices.empty(); }
};

// Triangulate and extrude an outline into a 3D mesh (front/back caps + side walls). An outline with
// no contours yields an empty mesh that still carries its advance.
[[nodiscard]] GlyphMesh3 tessellateGlyph(const GlyphOutline& outline);
