#include "core/image.hpp"
#include <algorithm>

// Weighting the per-pixel average by saturation^2 * value suppresses near-black and near-grey
// pixels, which otherwise dominate the mean; the result is brightness-normalised and
// saturation-boosted. The same pass accumulates mean luma to derive washDim.
void computeAccent(DecodedImage& img) {
    if (img.rgba.empty()) return;
    constexpr float DARK_CUTOFF = 0.06f;
    constexpr double MIN_MEAN_WEIGHT = 0.004;
    constexpr float NORM_BRIGHTNESS = 0.92f;
    constexpr float SAT_BOOST = 1.35f;
    double sr = 0, sg = 0, sb = 0, sw = 0, sl = 0;
    size_t n = img.rgba.size() / 4;
    for (size_t i = 0; i < n; ++i) {
        float r = img.rgba[i * 4 + 0] / 255.0f;
        float g = img.rgba[i * 4 + 1] / 255.0f;
        float b = img.rgba[i * 4 + 2] / 255.0f;
        sl += 0.299 * r + 0.587 * g + 0.114 * b;
        float mx = std::max(r, std::max(g, b));
        float mn = std::min(r, std::min(g, b));
        if (mx <= DARK_CUTOFF) continue;
        float sat = (mx - mn) / mx;
        float w = sat * sat * mx;
        sr += r * w; sg += g * w; sb += b * w; sw += w;
    }
    // Wash brightness = luma * washDim rolls off toward WASH_MAX for a bright cover and falls to black
    // for a dark one. WASH_KNEE > WASH_MAX keeps washDim < 1, so a cover is only ever dimmed.
    constexpr double WASH_MAX  = 0.13;   // brightness a near-white cover's wash approaches
    constexpr double WASH_KNEE = 0.35;   // mean luma at which the wash reaches half of WASH_MAX
    if (n > 0) {
        double luma = sl / static_cast<double>(n);
        img.washDim = static_cast<float>(WASH_MAX / (luma + WASH_KNEE));
    }
    if (sw < static_cast<double>(n) * MIN_MEAN_WEIGHT) return;  // near-grayscale: no usable accent
    float r = static_cast<float>(sr / sw), g = static_cast<float>(sg / sw), b = static_cast<float>(sb / sw);
    float mx = std::max(r, std::max(g, b));
    if (mx > 0.0f) { float k = NORM_BRIGHTNESS / mx; r *= k; g *= k; b *= k; }
    float l = 0.299f * r + 0.587f * g + 0.114f * b;
    auto boostSat = [l](float x) { return std::clamp(l + (x - l) * SAT_BOOST, 0.0f, 1.0f); };
    r = boostSat(r); g = boostSat(g); b = boostSat(b);
    img.ar = r; img.ag = g; img.ab = b; img.hasAccent = true;
}
