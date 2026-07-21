#pragma once
#include "core/glyph_mesh.hpp"
#include <cstdint>
#include <memory>

// A font at a fixed em size. Platform-neutral interface; implemented with GDI (win_font.cpp) on
// Windows and FreeType (lin_font.cpp) on Linux.
class FontFace {
public:
    FontFace(int emPx, bool bold);
    ~FontFace();
    FontFace(const FontFace&) = delete;
    FontFace& operator=(const FontFace&) = delete;

    [[nodiscard]] int lineHeight() const { return lineHeight_; }
    [[nodiscard]] int ascent() const { return ascent_; }
    [[nodiscard]] int emPx() const { return emPx_; }

    // Stable per-face identity used purely as a glyph-cache key; never dereferenced.
    [[nodiscard]] const void* handle() const { return this; }

    // Flatten one glyph's outline to em-normalized polygon contours (y up). Whitespace and
    // outline-less glyphs return contours.empty() with advance still set.
    [[nodiscard]] GlyphOutline loadOutline(uint32_t cp) const;

private:
    struct Impl;
    std::unique_ptr<Impl> impl_;  // platform handle(s): HFONT, or FT_Face
    int lineHeight_ = 0;
    int ascent_ = 0;
    int emPx_ = 0;
};
