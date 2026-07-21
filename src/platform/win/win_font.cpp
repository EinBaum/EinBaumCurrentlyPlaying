// Windows font backend: GDI font creation and glyph-outline fetch (GetGlyphOutlineW / GGO_NATIVE),
// flattening the TrueType/OpenType curves to em-normalized contours for the portable tessellator.
#include "core/font.hpp"
#include <windows.h>
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cwchar>
#include <cwctype>
#include <vector>

struct FontFace::Impl {
    HFONT hfont = nullptr;
};

namespace {

constexpr int   CURVE_STEPS = 8;
constexpr float WELD_EPS    = 1e-4f;   // tolerance in font units; coordinates here are raw fx2f values

using Contour = std::vector<vec2>;

[[nodiscard]] float fx2f(const FIXED& f) {
    return static_cast<float>(f.value) + static_cast<float>(f.fract) / 65536.0f;
}
[[nodiscard]] vec2 toVec(const POINTFX& p) { return {fx2f(p.x), fx2f(p.y)}; }  // GGO_NATIVE: font units, y up

[[nodiscard]] bool coincident(vec2 a, vec2 b) {
    return std::fabs(a.x - b.x) < WELD_EPS && std::fabs(a.y - b.y) < WELD_EPS;
}

void addQuadratic(Contour& out, vec2 p0, vec2 c, vec2 p1) {
    for (int i = 1; i <= CURVE_STEPS; ++i) {
        float t = static_cast<float>(i) / CURVE_STEPS, u = 1.0f - t;
        out.push_back({u * u * p0.x + 2 * u * t * c.x + t * t * p1.x,
                       u * u * p0.y + 2 * u * t * c.y + t * t * p1.y});
    }
}
void addCubic(Contour& out, vec2 p0, vec2 c1, vec2 c2, vec2 p1) {
    for (int i = 1; i <= CURVE_STEPS; ++i) {
        float t = static_cast<float>(i) / CURVE_STEPS, u = 1.0f - t;
        float b0 = u * u * u, b1 = 3 * u * u * t, b2 = 3 * u * t * t, b3 = t * t * t;
        out.push_back({b0 * p0.x + b1 * c1.x + b2 * c2.x + b3 * p1.x,
                       b0 * p0.y + b1 * c1.y + b2 * c2.y + b3 * p1.y});
    }
}

[[nodiscard]] std::vector<Contour> parseOutline(const std::vector<uint8_t>& buf) {
    std::vector<Contour> contours;
    size_t off = 0, size = buf.size();
    while (off + sizeof(TTPOLYGONHEADER) <= size) {
        const auto* hdr = reinterpret_cast<const TTPOLYGONHEADER*>(buf.data() + off);
        if (hdr->dwType != TT_POLYGON_TYPE || hdr->cb < sizeof(TTPOLYGONHEADER)) break;
        size_t end = std::min<size_t>(off + hdr->cb, size);
        Contour c;
        vec2 cur = toVec(hdr->pfxStart);
        c.push_back(cur);
        size_t p = off + sizeof(TTPOLYGONHEADER);
        while (p + 4 <= end) {                       // 4 = wType + cpfx
            const auto* cv = reinterpret_cast<const TTPOLYCURVE*>(buf.data() + p);
            int n = cv->cpfx;
            size_t bytes = 4 + static_cast<size_t>(n) * sizeof(POINTFX);
            if (n <= 0 || p + bytes > end) break;
            const POINTFX* pts = cv->apfx;
            if (cv->wType == TT_PRIM_LINE) {
                for (int i = 0; i < n; ++i) { cur = toVec(pts[i]); c.push_back(cur); }
            } else if (cv->wType == TT_PRIM_QSPLINE) {
                for (int i = 0; i < n - 1; ++i) {
                    vec2 ctrl = toVec(pts[i]);
                    vec2 nxt = toVec(pts[i + 1]);
                    vec2 endp = (i == n - 2) ? nxt : vec2{(ctrl.x + nxt.x) * 0.5f, (ctrl.y + nxt.y) * 0.5f};
                    addQuadratic(c, cur, ctrl, endp);
                    cur = endp;
                }
            } else if (cv->wType == TT_PRIM_CSPLINE) {
                for (int i = 0; i + 2 < n; i += 3) {
                    addCubic(c, cur, toVec(pts[i]), toVec(pts[i + 1]), toVec(pts[i + 2]));
                    cur = toVec(pts[i + 2]);
                }
            }
            p += bytes;
        }
        if (c.size() > 1 && coincident(c.front(), c.back())) c.pop_back();
        Contour dedup;
        for (const vec2& q : c)
            if (dedup.empty() || !coincident(dedup.back(), q)) dedup.push_back(q);
        if (dedup.size() >= 3) contours.push_back(std::move(dedup));
        off = end;
    }
    return contours;
}

}  // namespace

FontFace::FontFace(int emPx, bool bold) : impl_(std::make_unique<Impl>()), emPx_(emPx) {
    // Negative height = character (em) height in pixels. ANTIALIASED_QUALITY forces grayscale AA,
    // so coverage lands equally in R/G/B with no ClearType colour fringe.
    LOGFONTW lf{};
    lf.lfHeight = -emPx;
    lf.lfWeight = bold ? FW_BOLD : FW_NORMAL;
    lf.lfQuality = ANTIALIASED_QUALITY;
    lf.lfCharSet = DEFAULT_CHARSET;
    std::wcsncpy(lf.lfFaceName, L"Segoe UI", LF_FACESIZE);
    impl_->hfont = CreateFontIndirectW(&lf);

    HDC dc = CreateCompatibleDC(nullptr);
    HGDIOBJ old = SelectObject(dc, impl_->hfont);
    TEXTMETRICW tm{};
    GetTextMetricsW(dc, &tm);
    lineHeight_ = tm.tmHeight;
    ascent_ = tm.tmAscent;
    SelectObject(dc, old);
    DeleteDC(dc);
}

FontFace::~FontFace() {
    if (impl_ && impl_->hfont) DeleteObject(impl_->hfont);
}

GlyphOutline FontFace::loadOutline(uint32_t cp) const {
    GlyphOutline out;
    const float em = static_cast<float>(emPx_);

    HDC dc = CreateCompatibleDC(nullptr);
    HGDIOBJ oldFont = SelectObject(dc, impl_->hfont);
    MAT2 mat{};
    mat.eM11.value = 1;
    mat.eM22.value = 1;

    GLYPHMETRICS gm{};
    const UINT fmt = GGO_NATIVE | GGO_UNHINTED;
    DWORD size = GetGlyphOutlineW(dc, cp, fmt, &gm, 0, nullptr, &mat);

    if (size == GDI_ERROR) {
        int w = 0;
        if (GetCharWidth32W(dc, cp, cp, &w)) out.advance = static_cast<float>(w) / em;
        SelectObject(dc, oldFont); DeleteDC(dc);
        return out;
    }
    out.advance = static_cast<float>(gm.gmCellIncX) / em;
    if (size == 0) { SelectObject(dc, oldFont); DeleteDC(dc); return out; }  // whitespace

    std::vector<uint8_t> buf(size);
    GetGlyphOutlineW(dc, cp, fmt, &gm, size, buf.data(), &mat);
    SelectObject(dc, oldFont);
    DeleteDC(dc);

    // Flatten + weld in font units, then normalize to em.
    std::vector<Contour> contours = parseOutline(buf);
    out.contours.reserve(contours.size());
    for (const Contour& c : contours) {
        std::vector<vec2> em_c;
        em_c.reserve(c.size());
        for (vec2 p : c) em_c.push_back({p.x / em, p.y / em});
        out.contours.push_back(std::move(em_c));
    }
    return out;
}
