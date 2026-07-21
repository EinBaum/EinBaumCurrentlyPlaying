// Linux album-art decode via stb_image (vendored): decode -> center-crop to square -> bilinear
// scale to target -> RGBA. Accent extraction is shared (core/image.cpp). Only PNG and JPEG are
// compiled in (the thumbnail formats in use).
#include "core/image.hpp"
#include <algorithm>
#include <cmath>

#define STB_IMAGE_IMPLEMENTATION
#define STBI_ONLY_PNG
#define STBI_ONLY_JPEG
#define STBI_NO_STDIO            // we only decode from memory; no file paths
#define STBI_ASSERT(x) ((void)0) // a malformed thumbnail must fail soft, never abort the overlay
#include "stb_image.h"

DecodedImage decodeAlbumArt(const std::vector<std::uint8_t>& bytes, int target) {
    DecodedImage out;
    if (bytes.empty() || target <= 0) return out;

    int w = 0, h = 0, comp = 0;
    stbi_uc* px = stbi_load_from_memory(bytes.data(), static_cast<int>(bytes.size()),
                                        &w, &h, &comp, 4);  // force 4 channels (RGBA)
    if (!px || w <= 0 || h <= 0) { if (px) stbi_image_free(px); return out; }

    // Center-crop to the largest centered square, matching the WIC clipper.
    const int side = std::min(w, h);
    const int cx = (w - side) / 2;
    const int cy = (h - side) / 2;

    out.w = target;
    out.h = target;
    out.rgba.resize(static_cast<size_t>(target) * target * 4);

    // Bilinear resample. Source sample centers map to output pixel centers so the scaling is
    // symmetric and introduces no half-texel shift.
    const float scale = static_cast<float>(side) / static_cast<float>(target);
    auto srcPixel = [&](int sx, int sy, int c) -> float {
        sx = std::clamp(sx, 0, side - 1);
        sy = std::clamp(sy, 0, side - 1);
        const size_t idx = (static_cast<size_t>(cy + sy) * w + (cx + sx)) * 4 + c;
        return static_cast<float>(px[idx]);
    };
    for (int oy = 0; oy < target; ++oy) {
        const float fy = (oy + 0.5f) * scale - 0.5f;
        const int y0 = static_cast<int>(std::floor(fy));
        const float ty = fy - y0;
        for (int ox = 0; ox < target; ++ox) {
            const float fx = (ox + 0.5f) * scale - 0.5f;
            const int x0 = static_cast<int>(std::floor(fx));
            const float tx = fx - x0;
            uint8_t* dst = &out.rgba[(static_cast<size_t>(oy) * target + ox) * 4];
            for (int c = 0; c < 4; ++c) {
                float top = srcPixel(x0, y0, c) * (1 - tx) + srcPixel(x0 + 1, y0, c) * tx;
                float bot = srcPixel(x0, y0 + 1, c) * (1 - tx) + srcPixel(x0 + 1, y0 + 1, c) * tx;
                float v = top * (1 - ty) + bot * ty;
                dst[c] = static_cast<uint8_t>(std::clamp(v + 0.5f, 0.0f, 255.0f));
            }
        }
    }
    stbi_image_free(px);
    computeAccent(out);
    return out;
}
