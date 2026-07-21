#pragma once
#include <cstdint>
#include <vector>

struct DecodedImage {
    std::vector<std::uint8_t> rgba;  // target*target*4 bytes
    int w = 0, h = 0;
    bool hasAccent = false;
    float ar = 0, ag = 0, ab = 0;  // accent colour, channel range 0..1 (not 0..255 like rgba)
    float washDim = 1.0f;          // darkening factor applied to the blurred wash (see computeAccent)
};

// Decode raw thumbnail bytes to a square target x target RGBA image with accent extracted.
// Implemented per platform: WIC (win_image.cpp) or stb_image (lin_image.cpp).
[[nodiscard]] DecodedImage decodeAlbumArt(const std::vector<std::uint8_t>& bytes, int target);

// Derive the dominant accent colour and washDim from decoded RGBA pixels (image.cpp, shared by
// both platform decoders).
void computeAccent(DecodedImage& img);
