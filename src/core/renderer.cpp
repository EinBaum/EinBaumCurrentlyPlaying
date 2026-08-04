// Scene composition and per-track state: text layout, the song-change state machine, and draw().
// Vulkan plumbing lives in renderer_vk.cpp, acceleration structures in renderer_raytracing.cpp.
#include "core/renderer.hpp"
#include "core/renderer_internal.hpp"
#include "core/glyph_mesh.hpp"
#include "core/image.hpp"
#include "core/track_text.hpp"
#include "platform/platform.hpp"
#include <algorithm>
#include <array>
#include <cmath>
#include <cstring>
#include <format>
#include <limits>
#include <numbers>
#include <utility>

namespace {

constexpr double TRANSITION_SEC = 3.0;

// A frame-to-frame change in the true playhead larger than SEEK_JUMP_SEC, beyond what elapsed
// playback explains, is treated as a seek and glided over SEEK_GLIDE_SEC.
constexpr double SEEK_GLIDE_SEC = 1.0;
constexpr double SEEK_JUMP_SEC = 1.5;

// Ease time between the accent/PAUSED_FG colours and full/PAUSED_TIP_SCALE intensities on a
// play/pause change.
constexpr double PLAY_FADE_SEC = 0.5;

// The text swap runs in two sequential halves of TEXT_DISSOLVE_SEC each, together filling
// TRANSITION_SEC. DISSOLVE_END is slightly past 1 so the last glyph fragments and their ember clear.
constexpr double TEXT_DISSOLVE_SEC = 1.5;
constexpr float DISSOLVE_END = 1.05f;

// Fixed-point scale packing a dissolving glyph's progress into its 24-bit TLAS instance custom
// index (matches DISS_ENC in shadow_common.glsl); DISSOLVE_END * 2^20 stays well under 2^24.
constexpr float DISS_ENCODE = static_cast<float>(1u << 20);

// A line too long for its column holds left-aligned for MARQUEE_HOLD_SEC (> TRANSITION_SEC, so a
// new line is solid before it moves), then drifts left at MARQUEE_EM_PER_SEC of its own em-widths
// per second, wrapping with a MARQUEE_GAP_EM blank gap.
constexpr double MARQUEE_HOLD_SEC = 4.5;
constexpr float MARQUEE_EM_PER_SEC = 0.512f;
constexpr float MARQUEE_GAP_EM = 2.0f;

// Ray-tracing occluder masks: the key light traces text and rod, the tip light only text (it sits
// on the rod), and the wash shadow pass clips marquee-text shadows to the column.
constexpr uint32_t MASK_TEXT = 0x01;
constexpr uint32_t MASK_ROD = 0x02;
constexpr uint32_t MASK_MARQUEE_TEXT = 0x04;

// `occluder` items are instanced into the scene TLAS so they cast ray-traced shadows; the flat
// receivers (card wash, album) are not.
struct DrawItem {
    const Mesh* mesh;
    mat4 model;
    vec4 color;
    float fade;
    MeshMode mode;
    bool occluder;
    VkDescriptorSet tex;
    bool tess = false;
    float tessLevel = 0.0f;
    float barDissolve = 0.0f; // bar fill only: >0 runs the left-to-right dissolve-out (0..1 progress)
    float dissolve = 0.0f;    // >0 runs the per-glyph disintegration
    vec4 dissolveColor{};
    float barLen = 0.0f;      // bar fill only: the rod's world length, sizing the tip in fixed world units
    float liveBar = 0.0f;     // bar fill only: >0 = live stream, glows uniformly with no playhead tip
    float washDim = 1.0f;     // wash only: per-cover darkening factor for the blurred background
    uint32_t mask = MASK_TEXT;
    bool clipColumn = false;  // scissor the raster to the text column (a scrolling marquee line)
};

// Column-major mat4 -> the row-major 3x4 affine an acceleration-structure instance expects.
[[nodiscard]] VkTransformMatrixKHR toVkTransform(const mat4& m) {
    VkTransformMatrixKHR t;
    for (int r = 0; r < 3; ++r)
        for (int c = 0; c < 4; ++c) t.matrix[r][c] = m.at(c, r);
    return t;
}

// The card maps to a rectangle 2 units wide on the z=0 plane at the window's aspect. worldX/worldY
// map a pixel (origin top-left, y down) into it (origin centre, y up); both axes share one scale
// so the extruded letters keep the font's proportions.
constexpr float WORLD_PER_PX = 2.0f / WIN_W;
constexpr float CARD_HALF_W = 1.0f;
constexpr float CARD_HALF_H = static_cast<float>(WIN_H) / WIN_W;

[[nodiscard]] float worldX(float px) { return px * WORLD_PER_PX - CARD_HALF_W; }
[[nodiscard]] float worldY(float py) { return CARD_HALF_H - py * WORLD_PER_PX; }

// Element depths (world z; larger z is nearer). Z_ALBUM is coplanar with Z_CARD: the album is
// drawn after the card background, so it wins the shared depth under LESS_OR_EQUAL.
constexpr float Z_CARD = 0.0f;
constexpr float Z_ALBUM = 0.0f;
constexpr float Z_BAR = 0.008f;
constexpr float Z_BAR_FILL = 0.010f;
constexpr float Z_TEXT = 0.020f;

// Two progress fills coincide during a song change; the incoming fill is lifted this far (nearer)
// to win the shared depth, staying in the bar's z-band.
constexpr float Z_FILL_OVER = 0.001f;

// Oblique-projection shear per world unit of depth, bringing the right and bottom side walls of
// every letter into view. Anchored at the card plane (z = 0) so the full-window background fills
// the viewport with no border.
constexpr float OBLIQUE_X = -0.40f;   // front shifts left per world z
constexpr float OBLIQUE_Y =  0.40f;   // front shifts up per world z

// Progress-bar tip light: the playhead end of the neon fill doubles as a point light just in
// front of the card.
constexpr float TIP_LIGHT_Z         = 0.10f;
constexpr float TIP_LIGHT_RANGE     = 0.70f;
constexpr float TIP_LIGHT_INTENSITY = 0.96f;
constexpr float PAUSED_TIP_SCALE    = 0.4f;

// World-space progress bar: a rod from leftXw to rightXw at height cyW, radius half the bar
// thickness. The left edge clears the album cover (WIN_H + PAD) when one is shown, else sits at PAD.
struct BarGeometry {
    float cyW;
    float radius;
    float leftXw;
    float rightXw;
};

[[nodiscard]] BarGeometry barGeometry(bool hasAlbum) {
    float txPx = hasAlbum ? static_cast<float>(WIN_H) + PAD : static_cast<float>(PAD);
    float by1 = WIN_H - PAD_BOTTOM, by0 = by1 - BAR_H;
    return {
        .cyW = worldY((by0 + by1) * 0.5f),
        .radius = (BAR_H * 0.5f) * WORLD_PER_PX,
        .leftXw = worldX(txPx),
        .rightXw = worldX(WIN_W - PAD),
    };
}

[[nodiscard]] mat4 quadModel(float left, float top, float right, float bottom, float z, float zBias) {
    float x0 = worldX(left), x1 = worldX(right);
    float yTop = worldY(top), yBot = worldY(bottom);   // yTop > yBot
    return translate({x0, yBot, z + zBias}) * scale({x1 - x0, yTop - yBot, 1.0f});
}

// A unit cylinder along +x: x in [0, 1], radius 1 in the y-z plane. Smooth radial side normals and
// flat ±x end caps. scale({length, r, r}) keeps the cross-section circular, so mat3(model) leaves
// the side normals' direction unchanged — no inverse-transpose needed.
[[nodiscard]] std::pair<std::vector<Vertex3>, std::vector<uint32_t>> makeCylinder(int seg) {
    constexpr float TAU = 2.0f * std::numbers::pi_v<float>;
    std::vector<Vertex3> v;
    std::vector<uint32_t> idx;
    for (int i = 0; i <= seg; ++i) {
        float a = TAU * static_cast<float>(i) / seg, cy = std::cos(a), cz = std::sin(a);
        vec3 n{0.0f, cy, cz};
        v.push_back({{0.0f, cy, cz}, n, {static_cast<float>(i) / seg, 1.0f}});
        v.push_back({{1.0f, cy, cz}, n, {static_cast<float>(i) / seg, 0.0f}});
    }
    for (int i = 0; i < seg; ++i) {
        uint32_t a0 = static_cast<uint32_t>(i) * 2;
        idx.insert(idx.end(), {a0, a0 + 2, a0 + 1, a0 + 1, a0 + 2, a0 + 3});
    }
    for (int end = 0; end < 2; ++end) {
        float x = static_cast<float>(end);
        vec3 n{end ? 1.0f : -1.0f, 0.0f, 0.0f};
        uint32_t centre = static_cast<uint32_t>(v.size());
        v.push_back({{x, 0.0f, 0.0f}, n, {0.5f, 0.5f}});
        uint32_t ring = static_cast<uint32_t>(v.size());
        for (int i = 0; i <= seg; ++i) {
            float a = TAU * static_cast<float>(i) / seg;
            v.push_back({{x, std::cos(a), std::sin(a)}, n, {0.5f, 0.5f}});
        }
        for (int i = 0; i < seg; ++i)
            idx.insert(idx.end(), {centre, ring + static_cast<uint32_t>(i), ring + static_cast<uint32_t>(i) + 1});
    }
    return {std::move(v), std::move(idx)};
}

// Combine a UTF-16 surrogate pair into one codepoint; an unpaired surrogate passes through so a
// malformed string can't desync measurement from layout.
uint32_t nextCodepoint(const std::wstring& s, size_t& i) {
    const uint32_t hi = static_cast<uint32_t>(s[i++]);
    if (hi >= 0xD800 && hi <= 0xDBFF && i < s.size()) {
        const uint32_t lo = static_cast<uint32_t>(s[i]);
        if (lo >= 0xDC00 && lo <= 0xDFFF) { ++i; return 0x10000u + ((hi - 0xD800u) << 10) + (lo - 0xDC00u); }
    }
    return hi;
}

}  // namespace

void Renderer::init(PlatformWindow& window) {
    platformInitThread();  // setTaskbarArt decodes album art via WIC on this thread; no-op on Linux
    fontTitle_ = std::make_unique<FontFace>(TITLE_PX, true);
    fontTitleSmall_ = std::make_unique<FontFace>(TITLE_PX / 2, true);
    fontArtist_ = std::make_unique<FontFace>(ARTIST_PX, false);
    fontFps_ = std::make_unique<FontFace>(40, false);

    initVulkan(window);

    // 1x1 white texture: solid (untextured) elements bind this and carry colour in the push constant.
    std::array<uint8_t, 4> whitePx{255, 255, 255, 255};
    white_ = createTextureRGBA(whitePx.data(), 1, 1);

    // uv (0,1) at the bottom-left so the album cover samples upright.
    std::vector<Vertex3> qv = {
        {{0, 0, 0}, {0, 0, 1}, {0, 1}},
        {{1, 0, 0}, {0, 0, 1}, {1, 1}},
        {{1, 1, 0}, {0, 0, 1}, {1, 0}},
        {{0, 1, 0}, {0, 0, 1}, {0, 0}},
    };
    std::vector<uint32_t> qi = {0, 1, 2, 0, 2, 3};
    unitQuad_ = createMesh(qv, qi);

    auto [cv, ci] = makeCylinder(28);
    barRod_ = createMesh(cv, ci);

    createSwapchain();

    loadRayTracingFns();
    // The rod is present every frame, so build its BLAS eagerly; glyph BLASes are queued as
    // letters first appear and built together per track change.
    ensureMeshBlas(barRod_);
    initSceneTlas();
    makeMeshPipeline();
    // After the camera/TLAS sets exist: build the shadow compute pipeline and point its
    // descriptors at the image createSwapchain already made.
    makeWashShadowPipeline();
    writeWashShadowDescriptors();
}

GpuGlyph Renderer::glyphGpuMesh(const FontFace& f, uint32_t cp) {
    std::pair<const void*, uint32_t> key{f.handle(), cp};
    if (auto it = glyphCache_.find(key); it != glyphCache_.end())
        return {&it->second, advanceCache_[key]};
    // Advance cache hit with no GPU mesh: a whitespace/outline-less glyph.
    if (auto it = advanceCache_.find(key); it != advanceCache_.end())
        return {nullptr, it->second};

    GlyphMesh3 g = tessellateGlyph(f.loadOutline(cp));
    advanceCache_[key] = g.advance;
    if (g.empty()) return {nullptr, g.advance};
    auto ins = glyphCache_.emplace(key, createMesh(g.verts, g.indices)).first;
    prepareMeshBlas(ins->second);
    return {&ins->second, g.advance};
}

void Renderer::layoutLine(std::vector<GlyphInstance>& out, const std::wstring& s, FontFace& f,
                          float emWorld, float penX, float baselineY, float z, vec4 color, float maxX) {
    // A scrolling line passes maxX = +inf so it lays out in full and draw() applies the offset.
    for (size_t i = 0; i < s.size();) {
        auto [m, adv] = glyphGpuMesh(f, nextCodepoint(s, i));
        float advW = adv * emWorld;
        if (penX + advW > maxX) return;
        if (m) out.push_back({m, {penX, baselineY, z}, emWorld, color});
        penX += advW;
    }
}

float Renderer::measureLine(const FontFace& f, const std::wstring& s, float emWorld) {
    float w = 0.0f;
    for (size_t i = 0; i < s.size();)
        w += glyphGpuMesh(f, nextCodepoint(s, i)).advanceEm * emWorld;
    return w;
}

// Arm a line's scroll if it overflows; return layoutLine's clamp (maxX when it fits, +inf when it
// scrolls). gap/speed scale to the line's own em so both lines scroll at one visual pace.
float Renderer::armLine(Marquee& m, float laidW, float colW, float emWorld, float maxX) {
    if (laidW <= colW) return maxX;
    m.active = true;
    m.width = laidW;
    m.gap = MARQUEE_GAP_EM * emWorld;
    m.speed = MARQUEE_EM_PER_SEC * emWorld;
    m.armed = true;
    return std::numeric_limits<float>::max();
}

void Renderer::buildCurrent(const Track& t) {
    // Keep burn: beginTextBurn latched it from the outgoing line before this runs.
    for (Marquee& m : mq_) { auto burn = m.burn; m = Marquee{}; m.burn = burn; }
    // The cover slot is reserved from the bytes alone so the layout stands before the decode lands;
    // an already-valid album_ (setTrack kept an identical cover) needs no decode.
    artWait_.reserved = !t.artPng.empty();
    if (artWait_.reserved && !album_.valid) {
        artWait_.seq = artDecoder_.submit(t.artPng, WIN_H);
        artWait_.pending = true;
    }
    layoutText(t);
}

void Renderer::layoutText(const Track& t) {
    const float txPx = artWait_.reserved ? static_cast<float>(WIN_H) + PAD : static_cast<float>(PAD);
    const float penX0 = worldX(txPx);
    const float maxX = worldX(WIN_W - PAD);
    colLeft_ = penX0;
    colRight_ = maxX;

    const int gap = 4;
    int titleLH = fontTitle_->lineHeight(), artistLH = fontArtist_->lineHeight();
    bool hasTitle = !t.title.empty(), hasArtist = !t.artist.empty();
    int blockH = (hasTitle ? titleLH : 0) + (hasArtist ? artistLH : 0) + (hasTitle && hasArtist ? gap : 0);
    int top = PAD_TOP;
    int rowTop = (t.duration > 0.0 || t.live) ? (WIN_H - PAD_BOTTOM - BAR_H - 8) : (WIN_H - PAD_BOTTOM);
    int lift = static_cast<int>(std::lround(WIN_H * 0.05));
    int cyTop = std::max(4, top + std::max(0, (rowTop - top - blockH) / 2) - lift);

    if (hasTitle) {
        float emWorld = fontTitle_->emPx() * WORLD_PER_PX;
        float emSmall = fontTitleSmall_->emPx() * WORLD_PER_PX;
        float baselineY = worldY(static_cast<float>(cyTop + fontTitle_->ascent()));
        const vec4 titleCol{TITLE_FG.r, TITLE_FG.g, TITLE_FG.b, 1.0f};
        const float colW = maxX - penX0;

        // The name is full size; a trailing bracket group ("(Official Video)") is half size.
        auto bs = bracketSuffixStart(t.title);
        const bool hasSuffix = bs && *bs > 0;
        const std::wstring mainStr = hasSuffix ? t.title.substr(0, *bs) : t.title;
        const std::wstring suffix = hasSuffix ? t.title.substr(*bs) : std::wstring{};
        float mainW = measureLine(*fontTitle_, mainStr, emWorld);
        float suffixW = hasSuffix ? measureLine(*fontTitleSmall_, suffix, emSmall) : 0.0f;

        const float lim = armLine(mq(Line::Title), mainW + suffixW, colW, emWorld, maxX);
        layoutLine(titleGlyphs_, mainStr, *fontTitle_, emWorld, penX0, baselineY, Z_TEXT, titleCol, lim);
        if (hasSuffix)
            layoutLine(titleGlyphs_, suffix, *fontTitleSmall_, emSmall, penX0 + mainW, baselineY, Z_TEXT,
                       titleCol, lim);
    }
    if (hasArtist) {
        float emWorld = fontArtist_->emPx() * WORLD_PER_PX;
        const float colW = maxX - penX0;
        int artistTop = cyTop + (hasTitle ? titleLH + gap : 0);
        float baselineY = worldY(static_cast<float>(artistTop + fontArtist_->ascent()));
        const vec4 artistCol{ARTIST_FG.r, ARTIST_FG.g, ARTIST_FG.b, 1.0f};
        float artistW = measureLine(*fontArtist_, t.artist, emWorld);
        const float lim = armLine(mq(Line::Artist), artistW, colW, emWorld, maxX);
        layoutLine(artistGlyphs_, t.artist, *fontArtist_, emWorld, penX0, baselineY, Z_TEXT, artistCol, lim);
    }
    // Build every BLAS the layout queued in one submit, rather than one stall per glyph.
    flushPendingBlas();
}

void Renderer::beginTextBurn(bool burnTitle, bool burnArtist) {
    outgoingTitleGlyphs_.clear();
    outgoingArtistGlyphs_.clear();
    // Freeze the scroll where the last drawn frame left it (fpsLastSteady_ is that frame's clock).
    auto latch = [&](Marquee& m, bool burning) {
        m.burn = Marquee::Burn{};
        if (!burning || !m.active) return;
        m.burn.on = true;
        m.burn.period = m.width + m.gap;
        // An armed line has no latched start (never drawn), so it froze at offset zero.
        double scroll = m.armed ? 0.0 : std::max(0.0, (fpsLastSteady_ - m.start) - MARQUEE_HOLD_SEC) * m.speed;
        m.burn.shift = m.burn.period > 0.0f ? static_cast<float>(std::fmod(scroll, m.burn.period)) : 0.0f;
    };
    latch(mq(Line::Title), burnTitle);
    latch(mq(Line::Artist), burnArtist);
    if (burnTitle)  outgoingTitleGlyphs_  = std::move(titleGlyphs_);
    if (burnArtist) outgoingArtistGlyphs_ = std::move(artistGlyphs_);
    titleGlyphs_.clear();  artistGlyphs_.clear();
    outgoingEmberAccent_ = hasAlbumAccent_ ? albumAccent_ : ACCENT;
}

void Renderer::setTrack(const Track& t) {
    // A change requested mid-dissolve does not interrupt it: buffer the latest request (overwritten
    // by each further change, so a fast-skip burst collapses to the final song) and promote it when
    // the running dissolve completes (see draw). A media-cleared request is a hard cut, never buffered.
    if (t.valid && state_ == PlaybackState::Transitioning) {
        pendingTrack_.track = t;
        pendingTrack_.queued = true;
        return;
    }

    vkDeviceWaitIdle(device_);

    if (!t.valid) {
        // Media cleared. The card-to-key transition is a hard cut: drop everything with no fade.
        state_ = PlaybackState::NoMedia;  transitionArmed_ = false;
        pendingTrack_.queued = false;
        for (Marquee& m : mq_) m = Marquee{};
        outgoingTitleGlyphs_.clear();  outgoingArtistGlyphs_.clear();
        destroyTexture(outgoingAlbum_);
        destroyTexture(album_);
        titleGlyphs_.clear();       artistGlyphs_.clear();
        hasAlbumAccent_ = false;
        artWait_ = ArtWait{};
        currentTrack_ = Track{};
        playFade_ = playFadeTarget_ = 1.0f;
        playFading_ = false;
        return;
    }

    // A changed title/artist line moves to the outgoing set via beginTextBurn, which must run
    // before buildCurrent rebuilds the new text; an unchanged line stays solid through the change.
    titleChanged_ = (t.title != currentTrack_.title);
    artistChanged_ = (t.artist != currentTrack_.artist);
    beginTextBurn(titleChanged_, artistChanged_);
    outgoingTrack_ = currentTrack_;
    outgoingAccent_ = albumAccent_;  outgoingHasAccent_ = hasAlbumAccent_;
    outgoingWashDim_ = washDim_;
    outgoingHadAlbum_ = artWait_.reserved;
    sameCover_ = !t.artPng.empty() && t.artPng == outgoingTrack_.artPng;

    // Hand the outgoing wash over so the new song's wash fades in over it. An identical cover skips
    // the handoff and stays bound: draw never samples outgoingAlbum_ when sameCover_, and
    // re-decoding would blank the cover until the async result lands.
    destroyTexture(outgoingAlbum_);
    if (!sameCover_) {
        outgoingAlbum_ = album_;
        album_ = Texture{};
        hasAlbumAccent_ = false;
    }

    // The cross-dissolve runs only on a song-to-song change; a first song reached from a blank
    // card appears instantly. transitionArmed_ latches transitionStart_ on the next draw frame.
    if (outgoingTrack_.valid) {
        state_ = PlaybackState::Transitioning;
        transitionArmed_ = true;
    } else {
        state_ = PlaybackState::Active;
    }

    // Re-anchor the jump detector so the song change itself is not mistaken for a seek.
    havePlayhead_ = false;
    seekGliding_ = false;
    playOffset_ = 0.0;

    // The new song's bar starts at its own play state; a fade from the prior song is dropped.
    playFade_ = playFadeTarget_ = t.playing ? 1.0f : 0.0f;
    playFading_ = false;

    buildCurrent(t);
    currentTrack_ = t;
}

void Renderer::refreshArt(const Track& t) {
    // A late cover for a song buffered behind a running dissolve updates the pending song so the
    // promotion rebuilds with it. A cover for the song on screen falls through and rebuilds in
    // place: mid-dissolve the running incoming-cover fade carries it in.
    if (state_ == PlaybackState::Transitioning && t.identity() != currentTrack_.identity()) {
        pendingTrack_.track = t;
        pendingTrack_.queued = true;
        return;
    }
    // album_ may still be bound by in-flight command buffers, so stall the GPU before destroying it.
    vkDeviceWaitIdle(device_);
    destroyTexture(album_);
    titleGlyphs_.clear();  artistGlyphs_.clear();
    hasAlbumAccent_ = false;
    buildCurrent(t);
    currentTrack_ = t;
}

// A mid-fade toggle re-anchors playFadeFrom_ to the current playFade_, so a reversal stays continuous.
float Renderer::advancePlayPauseFade(bool playing, double nowSteady) {
    float target = playing ? 1.0f : 0.0f;
    if (target != playFadeTarget_) {
        playFadeFrom_ = playFade_;
        playFadeTarget_ = target;
        playFadeStart_ = nowSteady;
        playFading_ = true;
    }
    if (playFading_) {
        double g = (nowSteady - playFadeStart_) / PLAY_FADE_SEC;
        if (g >= 1.0) { playFade_ = playFadeTarget_; playFading_ = false; }
        else {
            float s = static_cast<float>(g * g * g * (g * (g * 6.0 - 15.0) + 10.0));  // smootherstep
            playFade_ = playFadeFrom_ + (playFadeTarget_ - playFadeFrom_) * s;
        }
    }
    return playFade_;
}

void Renderer::draw(const Track& t, double nowSteady) {
    // Land a finished cover decode; a result whose seq is stale (superseded submit) is dropped.
    if (artWait_.pending) {
        if (auto r = artDecoder_.take(); r && r->seq == artWait_.seq) {
            artWait_.pending = false;
            const DecodedImage& img = r->img;
            if (img.w > 0 && !album_.valid) {
                album_ = createTextureRGBA(img.rgba.data(), img.w, img.h, true);  // mips: the wash mip-taps for its blur
                if (img.hasAccent) { albumAccent_ = {img.ar, img.ag, img.ab, 1.0f}; hasAlbumAccent_ = true; }
                washDim_ = img.washDim;
            } else if (img.w <= 0) {
                // Undecodable cover: drop the reserved slot and re-lay the text at the coverless offset.
                artWait_.reserved = false;
                titleGlyphs_.clear();
                artistGlyphs_.clear();
                for (Marquee& m : mq_) { auto burn = m.burn; m = Marquee{}; m.burn = burn; }
                layoutText(currentTrack_);
            }
        }
    }

    vkWaitForFences(device_, 1, &fence_, VK_TRUE, UINT64_MAX);
    uint32_t idx = 0;
    VkResult acq = vkAcquireNextImageKHR(device_, swapchain_, UINT64_MAX, semAcquire_, VK_NULL_HANDLE, &idx);
    if (acq == VK_ERROR_OUT_OF_DATE_KHR) { destroySwapchain(); createSwapchain(); return; }
    if (acq != VK_SUCCESS && acq != VK_SUBOPTIMAL_KHR) vkCheck(acq);
    vkResetFences(device_, 1, &fence_);

    // Frame-rate EMA with the alpha derived from dt, so it converges in fixed wall-clock time
    // regardless of frame cadence.
    if (fpsLastSteady_ > 0.0) {
        double dt = nowSteady - fpsLastSteady_;
        if (dt > 1e-6) {
            double inst = 1.0 / dt;
            double alpha = 1.0 - std::exp(-dt / 0.25);
            fpsSmoothed_ = fpsSmoothed_ > 0.0 ? fpsSmoothed_ + (inst - fpsSmoothed_) * alpha : inst;
        }
    }
    fpsLastSteady_ = nowSteady;

    if (transitionArmed_) { transitionStart_ = nowSteady; transitionArmed_ = false; }
    for (Marquee& m : mq_)
        if (m.armed) { m.start = nowSteady; m.armed = false; }

    // A finished cross-dissolve releases the outgoing song, then promotes the pending song (a
    // change requested mid-dissolve) into its own fresh dissolve. setTrack runs the real rotation
    // here because state_ is Active; the re-armed transition's start is latched on this same frame
    // so the parameter block below renders its dt=0 start.
    if (state_ == PlaybackState::Transitioning && nowSteady - transitionStart_ >= TRANSITION_SEC) {
        state_ = PlaybackState::Active;
        // Safe to free the outgoing wash: the fence wait above ordered the last frame to sample it,
        // and this frame will not reference it (crossBlur is false once state_ leaves Transitioning).
        destroyTexture(outgoingAlbum_);
        outgoingTitleGlyphs_.clear();
        outgoingArtistGlyphs_.clear();
        if (pendingTrack_.queued) {
            Track next = std::move(pendingTrack_.track);
            pendingTrack_.queued = false;
            if (next.identity() != currentTrack_.identity())  setTrack(next);
            else if (next.artPng != currentTrack_.artPng)     refreshArt(next);
        }
        if (transitionArmed_) { transitionStart_ = nowSteady; transitionArmed_ = false; }
        for (Marquee& m : mq_)
            if (m.armed) { m.start = nowSteady; m.armed = false; }
    }

    // Song-change cross-dissolve parameters. The text runs in two sequential halves: the outgoing
    // text burns over the first TEXT_DISSOLVE_SEC, then the incoming text materializes over the
    // second, so the new text appears only once the old has fully burned. The defaults draw the
    // incoming card solid with no transition.
    float fillIn = 1.0f, dissolve = 0.0f, oldTextDissolve = 0.0f, newTextDissolve = 0.0f;
    if (state_ == PlaybackState::Transitioning) {
        double dt = nowSteady - transitionStart_;
        float p = static_cast<float>(dt / TRANSITION_SEC);
        float s = p * p * (3.0f - 2.0f * p);
        fillIn = s;
        dissolve = s;
        oldTextDissolve = static_cast<float>(std::clamp(dt / TEXT_DISSOLVE_SEC, 0.0, 1.0)) * DISSOLVE_END;
        newTextDissolve = (1.0f - static_cast<float>(std::clamp((dt - TEXT_DISSOLVE_SEC) / TEXT_DISSOLVE_SEC, 0.0, 1.0))) * DISSOLVE_END;
    }

    const bool transitioning = state_ == PlaybackState::Transitioning;
    const float blurIn = fillIn;
    const bool crossBlur = transitioning && outgoingAlbum_.valid;

    // The bar, playhead, tip light, and play/pause mix read the on-screen song. During a buffered
    // transition the draw argument t is the pending song (not yet shown); reading it would drive
    // the visible card's bar from a song whose title and cover are not on screen.
    const Track& onScreen = (t.valid && t.identity() == currentTrack_.identity()) ? t : currentTrack_;

    // Smoothed playhead, shared by the progress fill and the tip light. A jump beyond the expected
    // per-frame advance by more than SEEK_JUMP_SEC is folded into playOffset_ (= shown - true) so
    // the shown value stays continuous, then playOffset_ eases to zero over SEEK_GLIDE_SEC.
    double curLive = 0.0;
    if (onScreen.valid && onScreen.duration > 0.0) {
        double trueLive = std::clamp(onScreen.playing ? onScreen.position + (nowSteady - onScreen.posBase) : onScreen.position,
                                     0.0, onScreen.duration);
        if (!havePlayhead_) {
            havePlayhead_ = true;
            playOffset_ = 0.0;
            seekGliding_ = false;
        } else {
            double advance = onScreen.playing ? (nowSteady - lastNowSteady_) : 0.0;
            double jump = trueLive - (lastTrueLive_ + advance);
            if (std::fabs(jump) > SEEK_JUMP_SEC) {
                playOffset_ -= jump;
                glideFromOffset_ = playOffset_;
                glideStart_ = nowSteady;
                seekGliding_ = true;
            }
        }
        if (seekGliding_) {
            double g = (nowSteady - glideStart_) / SEEK_GLIDE_SEC;
            if (g >= 1.0) { playOffset_ = 0.0; seekGliding_ = false; }
            else {
                // smootherstep: zero velocity at both ends, so the catch-up eases from rest and settles
                double s = g * g * g * (g * (g * 6.0 - 15.0) + 10.0);
                playOffset_ = glideFromOffset_ * (1.0 - s);
            }
        }
        curLive = std::clamp(trueLive + playOffset_, 0.0, onScreen.duration);
        lastTrueLive_ = trueLive;
        lastNowSteady_ = nowSteady;
    }

    const float playMix = onScreen.valid ? advancePlayPauseFade(onScreen.playing, nowSteady) : playFade_;

    // Orthographic projection with the oblique depth shear layered in. Only the projection is
    // oblique: world-space lighting and ray-traced shadows are unaffected.
    mat4 shear;                                  // x' = x + OBLIQUE_X*z; y' = y + OBLIQUE_Y*z
    shear.at(2, 0) = OBLIQUE_X;
    shear.at(2, 1) = OBLIQUE_Y;
    CameraUBO cam;
    cam.viewProj = ortho(CARD_HALF_W, CARD_HALF_H, -1.0f, 1.0f) * shear;
    // Ortho view looks down -z; a far point on that axis gives the fragment shader a uniform view
    // vector for the specular term.
    cam.camPos = {0.0f, 0.0f, 100.0f, 0.0f};
    // params.y carries the card's world half-height so the fragment shader can normalize a glyph
    // fragment's height into the text-disintegration front's 0..1 gradient.
    cam.params = {static_cast<float>(WIN_W) / static_cast<float>(WIN_H), CARD_HALF_H, 0.0f, 0.0f};

    const bool drawCard = transitioning ||
                          (t.valid && (!titleGlyphs_.empty() || !artistGlyphs_.empty() || artWait_.reserved));
    const RGBA accent = hasAlbumAccent_ ? albumAccent_ : ACCENT;

    // Point light at the playhead x on the bar's row, coloured like the fill. The incoming light
    // scales by fillIn; during a song change the outgoing playhead drives a second light scaling by
    // (1 - fillIn), so the two pools cross-fade.
    cam.tipLight  = {0.0f, 0.0f, 0.0f, 0.0f};
    cam.tipColor  = {0.0f, 0.0f, 0.0f, 0.0f};
    cam.tipLight2 = {0.0f, 0.0f, 0.0f, 0.0f};
    cam.tipColor2 = {0.0f, 0.0f, 0.0f, 0.0f};
    auto placeTip = [&](vec4& lightSlot, vec4& colorSlot, const Track& tk, bool hasAlbum,
                        RGBA c, double live, float intensity) {
        if (intensity <= 0.0f || tk.duration <= 0.0) return;
        BarGeometry bar = barGeometry(hasAlbum);
        float frac = static_cast<float>(std::clamp(live / tk.duration, 0.0, 1.0));
        float tipXw = bar.leftXw + (bar.rightXw - bar.leftXw) * frac;
        lightSlot = {tipXw, bar.cyW, TIP_LIGHT_Z, TIP_LIGHT_RANGE};
        colorSlot = {c.r, c.g, c.b, intensity};
    };
    if (drawCard && onScreen.valid) {
        float intensity = TIP_LIGHT_INTENSITY * std::lerp(PAUSED_TIP_SCALE, 1.0f, playMix) * fillIn;
        placeTip(cam.tipLight, cam.tipColor, onScreen, artWait_.reserved, mix(PAUSED_FG, accent, playMix),
                 curLive, intensity);
    }
    if (transitioning && outgoingTrack_.valid) {
        double outLive = outgoingTrack_.playing ? outgoingTrack_.position + (nowSteady - outgoingTrack_.posBase)
                                            : outgoingTrack_.position;
        RGBA c = outgoingTrack_.playing ? (outgoingHasAccent_ ? outgoingAccent_ : ACCENT) : PAUSED_FG;
        // A live incoming song has no playhead light to crossfade in, so fade the outgoing pool
        // with the old-text burn, not over the full transition, else it lights the new title.
        float outFade = (onScreen.valid && onScreen.live) ? (1.0f - oldTextDissolve / DISSOLVE_END) : (1.0f - fillIn);
        float intensity = TIP_LIGHT_INTENSITY * (outgoingTrack_.playing ? 1.0f : PAUSED_TIP_SCALE) * outFade;
        placeTip(cam.tipLight2, cam.tipColor2, outgoingTrack_, outgoingHadAlbum_, c, outLive, intensity);
    }
    // params.zw: the shared text column for the wash shadow pass's marquee clip, shifted by the
    // oblique shear at Z_TEXT so the cut aligns with the scissored letters, not the world edge.
    // Zeroed (w <= z) when nothing scrolls.
    const bool clipShadow = mq(Line::Title).active || mq(Line::Artist).active ||
        (transitioning && (mq(Line::Title).burn.on || mq(Line::Artist).burn.on));
    cam.params.z = clipShadow ? colLeft_ + OBLIQUE_X * Z_TEXT : 0.0f;
    cam.params.w = clipShadow ? colRight_ + OBLIQUE_X * Z_TEXT : 0.0f;
    // The incoming tip's cast shadow is gated to the new title's own materialization so the streak
    // appears only once the geometry it shadows is the incoming title, not the outgoing one still
    // burning over the first half.
    cam.tipShadow = {1.0f - newTextDissolve / DISSOLVE_END, 1.0f, 0.0f, 0.0f};
    std::memcpy(camUboMapped_, &cam, sizeof(cam));

    // Collect the scene on the CPU first so the occluders can be gathered into the acceleration
    // structure before the render pass records its draws.
    std::vector<DrawItem> items;
    auto addCard = [&](const Track& tk, Texture& album) {
        const mat4 washQuad = quadModel(0, 0, WIN_W, WIN_H, Z_CARD, 0.0f);
        const mat4 coverQuad = quadModel(0, 0, WIN_H, WIN_H, Z_ALBUM, 0.0f);
        // An unchanged cover (same album) is held static; only a different cover crossfades.
        const bool coverHandoff = crossBlur && !sameCover_;

        // Drawn first so the incoming wash composites over it under LESS_OR_EQUAL at the shared Z_CARD.
        if (coverHandoff)
            items.push_back({.mesh = &unitQuad_, .model = washQuad, .color = {1, 1, 1, 1}, .fade = 1.0f, .mode = MeshMode::Wash, .occluder = false, .tex = outgoingAlbum_.dset, .washDim = outgoingWashDim_});
        if (album.valid)
            items.push_back({.mesh = &unitQuad_, .model = washQuad, .color = {1, 1, 1, 1}, .fade = coverHandoff ? blurIn : 1.0f, .mode = MeshMode::Wash, .occluder = false, .tex = album.dset, .washDim = washDim_});
        else
            items.push_back({.mesh = &unitQuad_, .model = washQuad, .color = {CARD_BG.r, CARD_BG.g, CARD_BG.b, 1.0f}, .fade = blurIn, .mode = MeshMode::Lit, .occluder = false, .tex = white_.dset});
        // The outgoing cover clears over the first half and the incoming over the second, so the
        // two coplanar quads never blend into a double image.
        if (coverHandoff) {
            float outFade = album.valid ? (1.0f - oldTextDissolve / DISSOLVE_END) : (1.0f - blurIn);
            items.push_back({.mesh = &unitQuad_, .model = coverQuad, .color = {1, 1, 1, 1},
                             .fade = outFade, .mode = MeshMode::Flat, .occluder = false, .tex = outgoingAlbum_.dset});
        }
        if (album.valid)
            items.push_back({.mesh = &unitQuad_, .model = coverQuad, .color = {1, 1, 1, 1},
                             .fade = coverHandoff ? (1.0f - newTextDissolve / DISSOLVE_END) : 1.0f, .mode = MeshMode::Flat, .occluder = false, .tex = album.dset});
        if (tk.duration > 0.0 || tk.live) {
            // Unfilled groove: the rod's centre is lifted one radius off Z_BAR so it sits in front
            // of the card rather than sinking into it. Sized by the reserved cover slot so the bar
            // doesn't jump when a decoding cover lands.
            BarGeometry bar = barGeometry(artWait_.reserved);
            items.push_back({.mesh = &barRod_,
                             .model = translate({bar.leftXw, bar.cyW, Z_BAR + bar.radius}) *
                                      scale({bar.rightXw - bar.leftXw, bar.radius, bar.radius}),
                             .color = {TRACK_FG.r, TRACK_FG.g, TRACK_FG.b, 1.0f}, .fade = 1.0f, .mode = MeshMode::Lit,
                             .occluder = true, .tex = white_.dset, .tess = true, .tessLevel = 3.0f, .mask = MASK_ROD});
        }
    };

    // Letters cast ray-traced shadows. A dissolving letter stays in the occluder set: its TLAS
    // instance carries the dissolve progress so the shadow ray erodes the cast shadow against the
    // same death field. A marquee glyph goes on MASK_MARQUEE_TEXT: the wash shadow pass traces it
    // separately and discards the cast shadow outside the column, matching the scissored raster.
    auto addGlyphs = [&](const std::vector<GlyphInstance>& glyphs, float dissolve, vec4 ember,
                         float penShift = 0.0f, bool clip = false) {
        for (const GlyphInstance& gi : glyphs) {
            float sx = gi.pos.x + penShift;
            items.push_back({.mesh = gi.mesh,
                             .model = translate({sx, gi.pos.y, gi.pos.z}) * scale(gi.scale),
                             .color = gi.color, .fade = 1.0f, .mode = MeshMode::Lit, .occluder = true, .tex = white_.dset,
                             .dissolve = dissolve, .dissolveColor = ember,
                             .mask = clip ? MASK_MARQUEE_TEXT : MASK_TEXT, .clipColumn = clip});
        }
    };

    // Neon progress fill: the played span from the bar's left edge to the playhead, one Z step
    // nearer than the groove. Built apart from the card so a song change can cross-dissolve two
    // fills over the single groove.
    auto addFill = [&](const Track& tk, bool hasAlbum, RGBA fillColor, float fade, float barDissolve, double playhead,
                       float zBias) {
        if (fade <= 0.0f || (tk.duration <= 0.0 && !tk.live)) return;
        double frac = tk.live ? 1.0 : std::clamp(std::min(playhead, tk.duration) / tk.duration, 0.0, 1.0);
        if (frac <= 0.0) return;
        BarGeometry bar = barGeometry(hasAlbum);
        float tipXw = bar.leftXw + static_cast<float>((bar.rightXw - bar.leftXw) * frac);
        float fillLen = std::max(tipXw - bar.leftXw, bar.radius);
        items.push_back({.mesh = &barRod_,
                         .model = translate({bar.leftXw, bar.cyW, Z_BAR_FILL + bar.radius + zBias}) *
                                  scale({fillLen, bar.radius, bar.radius}),
                         .color = {fillColor.r, fillColor.g, fillColor.b, 1.0f}, .fade = fade, .mode = MeshMode::Bar, .occluder = false,
                         .tex = white_.dset, .tess = true, .tessLevel = 24.0f, .barDissolve = barDissolve,
                         .barLen = fillLen, .liveBar = tk.live ? 1.0f : 0.0f});
    };

    if (drawCard) {
        vec4 inEmber{accent.r, accent.g, accent.b, 1.0f};
        addCard(onScreen, album_);
        // A scrolling line draws two copies a period apart so one enters from the right as the
        // other leaves left, both column-clipped; a line that fits draws one unclipped copy.
        auto drawLine = [&](const std::vector<GlyphInstance>& glyphs, const Marquee& m, float diss) {
            if (!m.active) { addGlyphs(glyphs, diss, inEmber); return; }
            double scroll = std::max(0.0, (nowSteady - m.start) - MARQUEE_HOLD_SEC) * m.speed;
            double period = m.width + m.gap;
            double sMod = period > 0.0 ? std::fmod(scroll, period) : 0.0;
            addGlyphs(glyphs, diss, inEmber, -static_cast<float>(sMod), true);
            addGlyphs(glyphs, diss, inEmber, static_cast<float>(period - sMod), true);
        };
        drawLine(titleGlyphs_, mq(Line::Title), titleChanged_ ? newTextDissolve : 0.0f);
        drawLine(artistGlyphs_, mq(Line::Artist), artistChanged_ ? newTextDissolve : 0.0f);
        if (transitioning) {
            vec4 outEmber{outgoingEmberAccent_.r, outgoingEmberAccent_.g, outgoingEmberAccent_.b, 1.0f};
            // A burning line that was scrolling holds its frozen offset; both wrapped copies stay
            // clipped so whatever spanned the column keeps spanning it while it burns.
            auto drawOutgoing = [&](const std::vector<GlyphInstance>& glyphs, const Marquee& m) {
                if (!m.burn.on) { addGlyphs(glyphs, oldTextDissolve, outEmber); return; }
                addGlyphs(glyphs, oldTextDissolve, outEmber, -m.burn.shift, true);
                addGlyphs(glyphs, oldTextDissolve, outEmber, m.burn.period - m.burn.shift, true);
            };
            drawOutgoing(outgoingTitleGlyphs_, mq(Line::Title));
            drawOutgoing(outgoingArtistGlyphs_, mq(Line::Artist));
        }
        // outgoing fill on the bar plane; incoming lifted by Z_FILL_OVER to own the shared span
        if (transitioning && outgoingTrack_.valid)
            addFill(outgoingTrack_, outgoingHadAlbum_,
                    outgoingTrack_.playing ? (outgoingHasAccent_ ? outgoingAccent_ : ACCENT) : PAUSED_FG, 1.0f, dissolve,
                    outgoingTrack_.playing ? outgoingTrack_.position + (nowSteady - outgoingTrack_.posBase) : outgoingTrack_.position,
                    0.0f);
        addFill(onScreen, artWait_.reserved, mix(PAUSED_FG, accent, playMix), fillIn, 0.0f, curLive, transitioning ? Z_FILL_OVER : 0.0f);
    }

    // Refresh the displayed song's snapshot each frame so the next change dissolves the bar from
    // its real last-shown position (a seek does not update the setTrack snapshot). Guarded to the
    // song on screen: during a buffered transition the poll carries the pending song, and adopting
    // it would defeat the promotion's identity check. artPng is preserved, not advanced: only
    // setTrack/refreshArt rebuild album_, so currentTrack_.artPng must keep mirroring the bytes
    // album_ was built from, or the promotion's same-song late-cover check compares already-equal
    // values and skips the rebuild.
    if (t.valid && t.identity() == currentTrack_.identity()) {
        std::vector<uint8_t> builtArt = std::move(currentTrack_.artPng);
        currentTrack_ = t;
        currentTrack_.artPng = std::move(builtArt);
    }

    if (showFps_) {
        float emWorld = fontFps_->emPx() * WORLD_PER_PX;
        // Right-aligned debug line, row-th from the top (row 0 = fps).
        auto drawDebugLine = [&](const std::wstring& s, int row) {
            float w = 0.0f;
            for (size_t i = 0; i < s.size();) w += glyphGpuMesh(*fontFps_, nextCodepoint(s, i)).advanceEm * emWorld;
            float baselineY = worldY(static_cast<float>(PAD_TOP + fontFps_->ascent() + row * fontFps_->lineHeight()));
            float penX = worldX(WIN_W - PAD) - w;
            std::vector<GlyphInstance> glyphs;
            layoutLine(glyphs, s, *fontFps_, emWorld, penX, baselineY, Z_TEXT,
                       {TITLE_FG.r, TITLE_FG.g, TITLE_FG.b, 1.0f}, worldX(WIN_W));
            for (const GlyphInstance& gi : glyphs)
                items.push_back({.mesh = gi.mesh, .model = translate({gi.pos.x, gi.pos.y, gi.pos.z}) * scale(gi.scale),
                                 .color = gi.color, .fade = 1.0f, .mode = MeshMode::Lit, .occluder = false, .tex = white_.dset});
        };
        int fps = std::max(0, static_cast<int>(std::lround(fpsSmoothed_)));
        drawDebugLine(std::format(L"{} fps", fps), 0);
        // The wash darkening only applies when a cover backs the wash; a coverless card is Lit.
        if (album_.valid) drawDebugLine(std::format(L"dim {:.3f}", washDim_), 1);
        // A first-seen fps digit queues its BLAS in glyphGpuMesh; no buildCurrent follows here to flush it.
        flushPendingBlas();
    }

    std::vector<VkAccelerationStructureInstanceKHR> insts;
    insts.reserve(items.size());
    for (const DrawItem& it : items) {
        if (!it.occluder || !it.mesh->blas) continue;
        VkAccelerationStructureInstanceKHR inst{};
        inst.transform = toVkTransform(it.model);
        inst.mask = it.mask;
        inst.flags = VK_GEOMETRY_INSTANCE_TRIANGLE_FACING_CULL_DISABLE_BIT_KHR;
        // A dissolving letter is forced non-opaque and carries its dissolve progress in the custom
        // index, so the shadow ray's any-hit test erodes the cast shadow against the death field;
        // solid occluders stay opaque and skip that test.
        if (it.dissolve > 0.0f) {
            inst.flags |= VK_GEOMETRY_INSTANCE_FORCE_NO_OPAQUE_BIT_KHR;
            inst.instanceCustomIndex = static_cast<uint32_t>(it.dissolve * DISS_ENCODE) & 0xFFFFFFu;
        }
        inst.accelerationStructureReference = it.mesh->blasAddr;
        insts.push_back(inst);
    }

    VkCommandBuffer cb = cmds_[idx];
    vkResetCommandBuffer(cb, 0);
    VkCommandBufferBeginInfo bi{VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO};
    vkBeginCommandBuffer(cb, &bi);

    // The TLAS is a pure function of the occluder instances, so an unchanged set leaves the prior
    // build valid; the fence wait at draw's top already ordered the last build's reads.
    const bool occludersChanged = !tlasBuilt_ || !std::ranges::equal(insts, lastBuiltInsts_,
        [](const VkAccelerationStructureInstanceKHR& a, const VkAccelerationStructureInstanceKHR& b) {
            return std::memcmp(&a, &b, sizeof a) == 0;
        });
    if (occludersChanged) {
        buildSceneTlas(cb, insts);
        lastBuiltInsts_ = insts;
        tlasBuilt_ = true;
    }

    // Only a card frame samples the wash shadow, so dispatch only when drawCard.
    if (drawCard) {
        vkCmdBindPipeline(cb, VK_PIPELINE_BIND_POINT_COMPUTE, washShadowPipeline_);
        std::array<VkDescriptorSet, 3> washSets{washStoreSet_, camSet_, sceneAsSet_};
        vkCmdBindDescriptorSets(cb, VK_PIPELINE_BIND_POINT_COMPUTE, washShadowPipeLayout_, 0,
                                static_cast<uint32_t>(washSets.size()), washSets.data(), 0, nullptr);
        vkCmdDispatch(cb, (washShadowExtent_.width + 7) / 8, (washShadowExtent_.height + 7) / 8, 1);
        VkImageMemoryBarrier ib{VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER};
        ib.srcQueueFamilyIndex = ib.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
        ib.image = washShadowImage_;
        ib.subresourceRange = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1};
        ib.oldLayout = ib.newLayout = VK_IMAGE_LAYOUT_GENERAL;
        ib.srcAccessMask = VK_ACCESS_SHADER_WRITE_BIT;
        ib.dstAccessMask = VK_ACCESS_SHADER_READ_BIT;
        vkCmdPipelineBarrier(cb, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT,
                             0, 0, nullptr, 0, nullptr, 1, &ib);
    }

    // With a card up the frame clears to black; with no card it clears to the chroma key so OBS
    // keys it out. The card-to-key swap is a hard cut: the text disintegration burns over the card
    // and never reveals the key.
    std::array<VkClearValue, 3> clears{};
    clears[0].color = drawCard ? VkClearColorValue{{0.0f, 0.0f, 0.0f, 1.0f}}
                               : VkClearColorValue{{KEY_COLOR.r, KEY_COLOR.g, KEY_COLOR.b, 1.0f}};
    clears[1].depthStencil = {1.0f, 0};
    VkRenderPassBeginInfo rbi{VK_STRUCTURE_TYPE_RENDER_PASS_BEGIN_INFO};
    rbi.renderPass = renderPass_;
    rbi.framebuffer = framebuffers_[idx];
    rbi.renderArea.extent = extent_;
    rbi.clearValueCount = static_cast<uint32_t>(clears.size());
    rbi.pClearValues = clears.data();
    vkCmdBeginRenderPass(cb, &rbi, VK_SUBPASS_CONTENTS_INLINE);

    float cw = static_cast<float>(extent_.width), ch = static_cast<float>(extent_.height);
    VkViewport vpt{0, 0, cw, ch, 0, 1};
    VkRect2D sc{{0, 0}, extent_};
    vkCmdSetViewport(cb, 0, 1, &vpt);
    vkCmdSetScissor(cb, 0, 1, &sc);

    // Scissor a scrolling line to its text column. The edges are projected through the same oblique
    // x-shear the vertex stage applies at Z_TEXT, so the cut lands on the glyph pen.
    auto colEdgePx = [&](float worldXv) {
        float ndc = (worldXv + OBLIQUE_X * Z_TEXT) / CARD_HALF_W;
        return (ndc * 0.5f + 0.5f) * cw;
    };
    int colL = std::clamp(static_cast<int>(std::floor(colEdgePx(colLeft_))), 0, static_cast<int>(extent_.width));
    int colR = std::clamp(static_cast<int>(std::ceil(colEdgePx(colRight_))), colL, static_cast<int>(extent_.width));
    VkRect2D colSc{{colL, 0}, {static_cast<uint32_t>(colR - colL), extent_.height}};
    bool colScissor = false;
    // set 1 (camera) and set 2 (scene TLAS) are bound once: both pipelines share meshPipeLayout_,
    // so switching between the flat and tessellated pipelines leaves them in place.
    vkCmdBindDescriptorSets(cb, VK_PIPELINE_BIND_POINT_GRAPHICS, meshPipeLayout_, 1, 1, &camSet_, 0, nullptr);
    vkCmdBindDescriptorSets(cb, VK_PIPELINE_BIND_POINT_GRAPHICS, meshPipeLayout_, 2, 1, &sceneAsSet_, 0, nullptr);

    constexpr VkShaderStageFlags pushStages = VK_SHADER_STAGE_VERTEX_BIT |
        VK_SHADER_STAGE_TESSELLATION_CONTROL_BIT | VK_SHADER_STAGE_TESSELLATION_EVALUATION_BIT |
        VK_SHADER_STAGE_FRAGMENT_BIT;
    VkDescriptorSet boundTex = VK_NULL_HANDLE;
    VkPipeline boundPipe = VK_NULL_HANDLE;
    for (const DrawItem& it : items) {
        const Mesh& m = *it.mesh;
        if (m.indexCount == 0) continue;
        if (it.clipColumn != colScissor) {
            vkCmdSetScissor(cb, 0, 1, it.clipColumn ? &colSc : &sc);
            colScissor = it.clipColumn;
        }
        VkPipeline want = it.tess ? tessPipeline_ : meshPipeline_;
        if (want != boundPipe) {
            vkCmdBindPipeline(cb, VK_PIPELINE_BIND_POINT_GRAPHICS, want);
            boundPipe = want;
        }
        if (it.tex != boundTex) {
            vkCmdBindDescriptorSets(cb, VK_PIPELINE_BIND_POINT_GRAPHICS, meshPipeLayout_, 0, 1, &it.tex, 0, nullptr);
            boundTex = it.tex;
        }
        MeshPush pc;
        pc.model = it.model;
        pc.color = it.color;
        pc.fade = it.fade;
        pc.mode = std::to_underlying(it.mode);
        pc.barLen = it.barLen;
        pc.tessLevel = it.tessLevel;
        pc.barDissolve = it.barDissolve;
        pc.dissolve = it.dissolve;
        pc.liveBar = it.liveBar;
        pc.washDim = it.washDim;
        pc.dissolveColor = it.dissolveColor;
        vkCmdPushConstants(cb, meshPipeLayout_, pushStages, 0, sizeof(pc), &pc);
        VkDeviceSize off = 0;
        vkCmdBindVertexBuffers(cb, 0, 1, &m.vbo, &off);
        vkCmdBindIndexBuffer(cb, m.ibo, 0, VK_INDEX_TYPE_UINT32);
        vkCmdDrawIndexed(cb, m.indexCount, 1, 0, 0, 0);
    }

    vkCmdEndRenderPass(cb);
    vkEndCommandBuffer(cb);

    VkPipelineStageFlags wait = VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT;
    VkSubmitInfo si{VK_STRUCTURE_TYPE_SUBMIT_INFO};
    si.waitSemaphoreCount = 1;
    si.pWaitSemaphores = &semAcquire_;
    si.pWaitDstStageMask = &wait;
    si.commandBufferCount = 1;
    si.pCommandBuffers = &cb;
    si.signalSemaphoreCount = 1;
    si.pSignalSemaphores = &semRender_;
    vkCheck(vkQueueSubmit(queue_, 1, &si, fence_));

    VkPresentInfoKHR pi{VK_STRUCTURE_TYPE_PRESENT_INFO_KHR};
    pi.waitSemaphoreCount = 1;
    pi.pWaitSemaphores = &semRender_;
    pi.swapchainCount = 1;
    pi.pSwapchains = &swapchain_;
    pi.pImageIndices = &idx;
    VkResult pres = vkQueuePresentKHR(queue_, &pi);
    if (pres == VK_ERROR_OUT_OF_DATE_KHR || pres == VK_SUBOPTIMAL_KHR) { destroySwapchain(); createSwapchain(); }
    else vkCheck(pres);
}

void Renderer::shutdown() {
    if (!device_) return;
    artDecoder_.stop();
    vkDeviceWaitIdle(device_);
    destroyTexture(white_);
    destroyTexture(album_);
    destroyTexture(outgoingAlbum_);
    destroyMesh(unitQuad_);
    destroyMesh(barRod_);
    for (auto& [key, m] : glyphCache_) destroyMesh(m);
    glyphCache_.clear();
    if (camUboMapped_) vkUnmapMemory(device_, camUboMem_);
    if (camUbo_) vkDestroyBuffer(device_, camUbo_, nullptr);
    if (camUboMem_) vkFreeMemory(device_, camUboMem_, nullptr);
    if (meshPipeline_) vkDestroyPipeline(device_, meshPipeline_, nullptr);
    if (tessPipeline_) vkDestroyPipeline(device_, tessPipeline_, nullptr);
    if (meshPipeLayout_) vkDestroyPipelineLayout(device_, meshPipeLayout_, nullptr);
    if (washShadowPipeline_) vkDestroyPipeline(device_, washShadowPipeline_, nullptr);
    if (washShadowPipeLayout_) vkDestroyPipelineLayout(device_, washShadowPipeLayout_, nullptr);
    if (washStorePool_) vkDestroyDescriptorPool(device_, washStorePool_, nullptr);
    if (washStoreLayout_) vkDestroyDescriptorSetLayout(device_, washStoreLayout_, nullptr);
    if (washSampler_) vkDestroySampler(device_, washSampler_, nullptr);
    if (camPool_) vkDestroyDescriptorPool(device_, camPool_, nullptr);
    if (camLayout_) vkDestroyDescriptorSetLayout(device_, camLayout_, nullptr);
    if (sceneInstMapped_) vkUnmapMemory(device_, sceneInstMem_);
    if (sceneTlas_) pfnDestroyAS_(device_, sceneTlas_, nullptr);
    for (VkBuffer b : {sceneTlasBuf_, sceneInstBuf_, sceneScratchBuf_}) if (b) vkDestroyBuffer(device_, b, nullptr);
    for (VkDeviceMemory m : {sceneTlasMem_, sceneInstMem_, sceneScratchMem_}) if (m) vkFreeMemory(device_, m, nullptr);
    if (sceneAsPool_) vkDestroyDescriptorPool(device_, sceneAsPool_, nullptr);
    if (sceneAsLayout_) vkDestroyDescriptorSetLayout(device_, sceneAsLayout_, nullptr);
    if (sampler_) vkDestroySampler(device_, sampler_, nullptr);
    if (dsetPool_) vkDestroyDescriptorPool(device_, dsetPool_, nullptr);
    if (dsetLayout_) vkDestroyDescriptorSetLayout(device_, dsetLayout_, nullptr);
    destroySwapchain();
    if (semAcquire_) vkDestroySemaphore(device_, semAcquire_, nullptr);
    if (semRender_) vkDestroySemaphore(device_, semRender_, nullptr);
    if (fence_) vkDestroyFence(device_, fence_, nullptr);
    if (cmdPool_) vkDestroyCommandPool(device_, cmdPool_, nullptr);
    if (device_) vkDestroyDevice(device_, nullptr);
    if (surface_) vkDestroySurfaceKHR(instance_, surface_, nullptr);
    if (instance_) vkDestroyInstance(instance_, nullptr);
    device_ = VK_NULL_HANDLE;
    platformShutdownThread();
}
