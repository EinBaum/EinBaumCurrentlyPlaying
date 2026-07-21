// Linux font backend: FreeType face loading and glyph-outline fetch (FT_Outline_Decompose),
// flattening the conic/cubic Bezier segments to em-normalized contours for the portable tessellator.
#include "core/font.hpp"
#include "core/fatal.hpp"
#include <ft2build.h>
#include FT_FREETYPE_H
#include FT_OUTLINE_H
#include <array>
#include <cmath>
#include <format>
#include <string>
#include <vector>
#include <fontconfig/fontconfig.h>

namespace {

constexpr int CURVE_STEPS = 8;   // subdivisions per Bezier segment; matches the Windows GDI path

// One process-wide FreeType library and one resolved font path per weight, initialized on first
// use. FontFace construction is single-threaded (all faces are built on the main thread), so no
// locking.
FT_Library g_ft = nullptr;
std::array<std::string, 2> g_fontPath;  // [0] regular, [1] bold

void ensureFreeType() {
    if (g_ft) return;
    if (FT_Init_FreeType(&g_ft))
        fatal("FreeType: FT_Init_FreeType failed.");
    if (!FcInit()) fatal("fontconfig: FcInit failed.");
}

const std::string& fontPath(bool bold) {
    std::string& cached = g_fontPath[bold ? 1 : 0];
    if (!cached.empty()) return cached;
    // "Segoe UI" (the Windows face) is requested first so a system that has it matches; fontconfig
    // falls back to the default sans otherwise.
    FcPattern* pat = FcNameParse(reinterpret_cast<const FcChar8*>("Segoe UI,sans-serif"));
    FcPatternAddInteger(pat, FC_WEIGHT, bold ? FC_WEIGHT_BOLD : FC_WEIGHT_REGULAR);
    FcConfigSubstitute(nullptr, pat, FcMatchPattern);
    FcDefaultSubstitute(pat);
    FcResult res;
    FcPattern* matched = FcFontMatch(nullptr, pat, &res);
    if (matched) {
        FcChar8* file = nullptr;
        if (FcPatternGetString(matched, FC_FILE, 0, &file) == FcResultMatch && file)
            cached = reinterpret_cast<const char*>(file);
        FcPatternDestroy(matched);
    }
    FcPatternDestroy(pat);
    if (cached.empty())
        fatal(std::format("fontconfig: no {} sans-serif font found.", bold ? "bold" : "regular"));
    return cached;
}

// Accumulates FT_Outline_Decompose callbacks into em-normalized contours (y up). FreeType point
// coordinates are 26.6 fixed point at the size set with FT_Set_Pixel_Sizes, so dividing by em*64
// normalizes to 1.0 == em.
struct Decomposer {
    std::vector<std::vector<vec2>> contours;
    std::vector<vec2> cur;
    float em = 1.0f;

    [[nodiscard]] vec2 norm(const FT_Vector* p) const {
        return {static_cast<float>(p->x) / em, static_cast<float>(p->y) / em};
    }
    void finishContour() {
        if (cur.size() >= 3) {
            // Weld a closing point coincident with the start (FreeType's contours are implicitly closed).
            if (cur.size() > 1) {
                vec2 a = cur.front(), b = cur.back();
                if (std::fabs(a.x - b.x) < 1e-6f && std::fabs(a.y - b.y) < 1e-6f) cur.pop_back();
            }
            if (cur.size() >= 3) contours.push_back(cur);
        }
        cur.clear();
    }
};

int moveTo(const FT_Vector* to, void* user) {
    auto* d = static_cast<Decomposer*>(user);
    d->finishContour();
    d->cur.push_back(d->norm(to));
    return 0;
}
int lineTo(const FT_Vector* to, void* user) {
    auto* d = static_cast<Decomposer*>(user);
    d->cur.push_back(d->norm(to));
    return 0;
}
int conicTo(const FT_Vector* control, const FT_Vector* to, void* user) {
    auto* d = static_cast<Decomposer*>(user);
    if (d->cur.empty()) return 0;
    vec2 p0 = d->cur.back(), c = d->norm(control), p1 = d->norm(to);
    for (int i = 1; i <= CURVE_STEPS; ++i) {
        float t = static_cast<float>(i) / CURVE_STEPS, u = 1.0f - t;
        d->cur.push_back({u * u * p0.x + 2 * u * t * c.x + t * t * p1.x,
                          u * u * p0.y + 2 * u * t * c.y + t * t * p1.y});
    }
    return 0;
}
int cubicTo(const FT_Vector* c1, const FT_Vector* c2, const FT_Vector* to, void* user) {
    auto* d = static_cast<Decomposer*>(user);
    if (d->cur.empty()) return 0;
    vec2 p0 = d->cur.back(), a = d->norm(c1), b = d->norm(c2), p1 = d->norm(to);
    for (int i = 1; i <= CURVE_STEPS; ++i) {
        float t = static_cast<float>(i) / CURVE_STEPS, u = 1.0f - t;
        float b0 = u * u * u, b1 = 3 * u * u * t, b2 = 3 * u * t * t, b3 = t * t * t;
        d->cur.push_back({b0 * p0.x + b1 * a.x + b2 * b.x + b3 * p1.x,
                          b0 * p0.y + b1 * a.y + b2 * b.y + b3 * p1.y});
    }
    return 0;
}

}  // namespace

struct FontFace::Impl {
    FT_Face face = nullptr;
};

FontFace::FontFace(int emPx, bool bold) : impl_(std::make_unique<Impl>()), emPx_(emPx) {
    ensureFreeType();
    const std::string& path = fontPath(bold);
    if (FT_New_Face(g_ft, path.c_str(), 0, &impl_->face))
        fatal(std::format("FreeType: cannot open font '{}'.", path));
    FT_Set_Pixel_Sizes(impl_->face, 0, static_cast<FT_UInt>(emPx));

    // FreeType's size metrics are 26.6 fixed point; round to whole pixels to match the renderer's
    // integer layout (the same values GDI's tmHeight/tmAscent provide on Windows).
    const FT_Size_Metrics& m = impl_->face->size->metrics;
    ascent_ = static_cast<int>(m.ascender >> 6);
    lineHeight_ = static_cast<int>(m.height >> 6);
}

FontFace::~FontFace() {
    if (impl_ && impl_->face) FT_Done_Face(impl_->face);
}

GlyphOutline FontFace::loadOutline(uint32_t cp) const {
    GlyphOutline out;
    FT_Face face = impl_->face;
    const float em = static_cast<float>(emPx_);

    FT_UInt gi = FT_Get_Char_Index(face, cp);
    if (FT_Load_Glyph(face, gi, FT_LOAD_NO_BITMAP | FT_LOAD_NO_HINTING)) return out;

    out.advance = static_cast<float>(face->glyph->advance.x) / 64.0f / em;  // 26.6 px -> em
    if (face->glyph->format != FT_GLYPH_FORMAT_OUTLINE) return out;          // whitespace / bitmap

    Decomposer d;
    d.em = em * 64.0f;
    FT_Outline_Funcs funcs{};
    funcs.move_to = moveTo;
    funcs.line_to = lineTo;
    funcs.conic_to = conicTo;
    funcs.cubic_to = cubicTo;
    funcs.shift = 0;
    funcs.delta = 0;
    if (FT_Outline_Decompose(&face->glyph->outline, &funcs, &d) != 0) return out;
    d.finishContour();
    out.contours = std::move(d.contours);
    return out;
}
