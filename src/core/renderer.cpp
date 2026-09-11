// Scene composition and per-track state: text layout, the song-change state machine, and draw().
// Vulkan plumbing lives in renderer_vk.cpp.
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

// Song-to-song swap is TRANSITION_SEC: outgoing fades out over the first half, incoming fades in
// over the second.
constexpr double TRANSITION_SEC = 3.0;
constexpr double TEXT_FADE_SEC = TRANSITION_SEC / 2.0;

// A frame-to-frame change in the true playhead larger than SEEK_JUMP_SEC, beyond what elapsed
// playback explains, is treated as a seek and glided over SEEK_GLIDE_SEC.
constexpr double SEEK_GLIDE_SEC = 1.0;
constexpr double SEEK_JUMP_SEC = 1.5;

// Ease time between the accent/PAUSED_FG colours and full/PAUSED_TIP_SCALE intensities on a
// play/pause change.
constexpr double PLAY_FADE_SEC = 0.5;

// A line too long for its column holds left-aligned for MARQUEE_HOLD_SEC (> TRANSITION_SEC, so a
// new line is solid before it moves), then drifts left at MARQUEE_EM_PER_SEC of its own em-widths
// per second, wrapping with a MARQUEE_GAP_EM blank gap.
constexpr double MARQUEE_HOLD_SEC = 4.5;
constexpr float MARQUEE_EM_PER_SEC = 0.512f;
constexpr float MARQUEE_GAP_EM = 2.0f;

// `occluder` items are projected onto the card plane for the wash-shadow pass; the flat
// receivers (card wash, album) are not. The key light includes the groove (`rod`); the tip
// light skips it (the light sits on the rod). Marquee glyphs set `clipColumn` so both the
// colour pass and the shadow pass scissor to the text column.
struct DrawItem {
    const Mesh* mesh;
    mat4 model;
    vec4 color;
    float fade;
    MeshMode mode;
    bool occluder;
    bool rod = false;         // groove: key-light shadow only
    VkDescriptorSet tex;
    bool tess = false;
    float tessLevel = 0.0f;
    float barLen = 0.0f;      // bar fill only: the rod's world length, sizing the tip in fixed world units
    float liveBar = 0.0f;     // bar fill only: >0 = live stream, glows uniformly with no playhead tip
    float washDim = 1.0f;     // wash only: per-cover darkening factor for the blurred background
    bool clipColumn = false;  // scissor the raster to the text column (a scrolling marquee line)
};

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
    // No frame command buffer exists yet; flush the staged upload synchronously.
    for (StagedUpload& su : pendingUploads_)
        submitNow([&](VkCommandBuffer cb) { recordTextureUpload(cb, su); });
    pendingUploads_.clear();
    for (StagedUpload& su : inFlightStagings_) {
        vkDestroyBuffer(device_, su.staging, nullptr);
        vkFreeMemory(device_, su.stagingMem, nullptr);
    }
    inFlightStagings_.clear();

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

    // Occ render pass before the swapchain so the first wash-shadow images can attach a framebuffer.
    createWashOccPass();
    createSwapchain();

    makeMeshPipeline();
    makeShadowPipeline();
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

void Renderer::buildCurrent(const Track& t, bool redecode) {
    for (Marquee& m : mq_) m = Marquee{};
    // The cover slot is reserved from the bytes alone so the layout stands before the decode lands;
    // an already-valid album_ (setTrack kept an identical cover) needs no decode unless the caller
    // is replacing the shown cover with new bytes (redecode).
    artWait_.reserved = !t.artPng.empty();
    if (artWait_.reserved && (redecode || !album_.valid)) {
        artWait_.seq = artDecoder_.submit(t.artPng, WIN_H);
        artWait_.pending = true;
    }
    layoutText(t);
}

void Renderer::layoutText(const Track& t) {
    titleGlyphs_.clear();
    artistGlyphs_.clear();
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
}

// Move the live line into its outgoing slot. The scroll freezes where the last drawn frame left
// it (fpsLastSteady_ is that frame's clock); an armed line was never drawn, so it froze at zero.
void Renderer::latchOutgoingLine(Line l) {
    const Marquee& m = mq(l);
    OutgoingLine& o = outgoing(l);
    o.changed = true;
    o.glyphs = std::move(glyphs(l));
    o.scrolling = m.active;
    o.period = m.width + m.gap;
    double scroll = m.armed ? 0.0 : std::max(0.0, (fpsLastSteady_ - m.start) - MARQUEE_HOLD_SEC) * m.speed;
    o.shift = o.period > 0.0f ? static_cast<float>(std::fmod(scroll, o.period)) : 0.0f;
}

// Replace the incoming song without touching the outgoing fade or the transition clock. A fast
// skip must not fade in the cover of a song that has already been left.
void Renderer::retargetIncoming(const Track& t) {
    // A line the first change left solid still shows the outgoing song's text, so it is latched
    // now if the latest song differs. A latched line whose text the latest song restores is
    // dropped: the incoming line is laid with that text and draws solid.
    for (Line l : {Line::Title, Line::Artist}) {
        const bool changed = lineText(t, l) != lineText(outgoingTrack_, l);
        if (changed && !outgoing(l).changed) latchOutgoingLine(l);
        else if (!changed) outgoing(l) = {};
    }
    coverChanged_ = t.artPng != outgoingTrack_.artPng;

    // Any decode in flight is for a song we just skipped. It is dropped so its result cannot land on
    // album_ after this retarget (submit() sequence numbers start at 1).
    artWait_.pending = false;
    artWait_.seq = 0;

    // With no separate outgoing texture, album_ is the outgoing song's cover only while the
    // skipped song shared its bytes (setTrack made no handoff). Otherwise it is the skipped
    // song's own decode and must not be shown for any other song.
    const bool albumIsOutgoing = !outgoingAlbum_.valid && currentTrack_.artPng == outgoingTrack_.artPng;

    if (!coverChanged_) {
        // Same bytes as the outgoing cover: hold that image solid, no cover fade.
        if (outgoingAlbum_.valid) {
            deferDestroyTexture(album_);
            album_ = outgoingAlbum_;
            outgoingAlbum_ = {};
            washDim_ = outgoingWashDim_;
            albumAccent_ = outgoingAccent_;
            hasAlbumAccent_ = outgoingHasAccent_;
        } else if (!albumIsOutgoing) {
            deferDestroyTexture(album_);
            album_ = {};
            hasAlbumAccent_ = false;
        }
    } else if (t.artPng != currentTrack_.artPng) {
        if (albumIsOutgoing) {
            // The shared cover fades out with the outgoing song. Its accent and wash factor were
            // latched by setTrack.
            outgoingAlbum_ = album_;
        } else {
            // Keep outgoingAlbum_ (the song still fading out). Incoming GPU cover is the skipped song.
            deferDestroyTexture(album_);
        }
        album_ = {};
        hasAlbumAccent_ = false;
    }

    havePlayhead_ = false;
    seekGliding_ = false;
    playOffset_ = 0.0;
    playFade_ = playFadeTarget_ = t.playing ? 1.0f : 0.0f;
    playFading_ = false;

    buildCurrent(t, coverChanged_ && !t.artPng.empty() && !album_.valid);
    currentTrack_ = t;
}

void Renderer::setTrack(const Track& t) {
    // A skip during the fade-out half replaces the still invisible incoming song in place, so the
    // outgoing song keeps fading and no skipped song's text or cover is ever drawn. A skip during
    // the fade-in half falls through to a fresh change: the partly visible incoming song becomes
    // the outgoing one and fades out from its current opacity. A media-cleared request is a hard
    // cut, never folded into a running fade.
    const bool transitioning = state_ == PlaybackState::Transitioning;
    const double fadeDt = transitionArmed_ ? 0.0 : fpsLastSteady_ - transitionStart_;
    if (t.valid && transitioning && fadeDt < TEXT_FADE_SEC) {
        retargetIncoming(t);
        return;
    }
    const float incomingFade = transitioning ? static_cast<float>(std::clamp((fadeDt - TEXT_FADE_SEC) / TEXT_FADE_SEC, 0.0, 1.0)) : 1.0f;

    if (!t.valid) {
        // Media cleared. The card-to-key transition is a hard cut: drop everything with no fade.
        state_ = PlaybackState::NoMedia;  transitionArmed_ = false;
        for (Marquee& m : mq_) m = Marquee{};
        for (OutgoingLine& o : outgoing_) o = {};
        deferDestroyTexture(outgoingAlbum_);
        deferDestroyTexture(album_);
        titleGlyphs_.clear();       artistGlyphs_.clear();
        hasAlbumAccent_ = false;
        artWait_ = ArtWait{};
        currentTrack_ = Track{};
        outgoingTrack_ = Track{};
        playFade_ = playFadeTarget_ = 1.0f;
        playFading_ = false;
        return;
    }

    // A changed line moves to its outgoing slot before buildCurrent lays the new text. An
    // unchanged line stays solid through the change. A line or cover still fading in from the
    // interrupted change counts as changed even when its content matches, so it fades out from
    // its current opacity instead of snapping to solid.
    for (Line l : {Line::Title, Line::Artist}) {
        const bool fadingIn = transitioning && outgoing(l).changed;
        outgoing(l) = {};
        if (fadingIn || lineText(t, l) != lineText(currentTrack_, l)) latchOutgoingLine(l);
    }
    const bool hadTrack = currentTrack_.valid;
    coverChanged_ = (transitioning && coverChanged_) || t.artPng != currentTrack_.artPng;
    fadeOutFrom_ = incomingFade;

    // Latch the outgoing fill before the incoming cover decode can replace albumAccent_.
    outgoingTrack_ = currentTrack_;
    outgoingAccent_ = albumAccent_;
    outgoingHasAccent_ = hasAlbumAccent_;
    outgoingHadAlbum_ = artWait_.reserved;
    outgoingWashDim_ = washDim_;

    // A different cover moves to outgoingAlbum_ so it can fade out; album_ is cleared so the
    // incoming decode does not overwrite the outgoing GPU image. Identical bytes stay bound.
    deferDestroyTexture(outgoingAlbum_);
    if (coverChanged_) {
        outgoingAlbum_ = album_;
        album_ = Texture{};
        hasAlbumAccent_ = false;
    }

    // The text fade runs only on a song-to-song change; a first song reached from a blank
    // card appears instantly. transitionArmed_ latches transitionStart_ on the next draw frame.
    if (hadTrack) {
        state_ = PlaybackState::Transitioning;
        transitionArmed_ = true;
    } else {
        state_ = PlaybackState::Active;
    }

    // Re-anchor the jump detector so the song change itself is not mistaken for a seek.
    havePlayhead_ = false;
    seekGliding_ = false;
    playOffset_ = 0.0;

    // Incoming play/pause mix starts at the new song; the outgoing fill uses outgoingTrack_.playing.
    playFade_ = playFadeTarget_ = t.playing ? 1.0f : 0.0f;
    playFading_ = false;

    buildCurrent(t, coverChanged_ && !t.artPng.empty());
    currentTrack_ = t;
}

void Renderer::refreshArt(const Track& t) {
    // Late art for a song that is not the incoming one is stale (already skipped). Applying it
    // would fade in the previous cover.
    if (t.identity() != currentTrack_.identity()) return;
    // A cover replacing one held on screen mid-fade joins the running cover fade as its outgoing
    // side only when there is no song-change cover already fading out.
    if (state_ == PlaybackState::Transitioning && album_.valid && !outgoingAlbum_.valid) {
        outgoingAlbum_ = album_;
        album_ = Texture{};
        outgoingWashDim_ = washDim_;
        coverChanged_ = true;
        hasAlbumAccent_ = false;
    }
    buildCurrent(t, true);
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
    vkWaitForFences(device_, 1, &fence_, VK_TRUE, UINT64_MAX);

    // The fence signals that the last submitted frame completed: safe to free textures and staging
    // buffers that were handed off mid-frame.
    for (Texture& tex : pendingDestroyTextures_) destroyTexture(tex);
    pendingDestroyTextures_.clear();
    for (StagedUpload& su : inFlightStagings_) {
        vkDestroyBuffer(device_, su.staging, nullptr);
        vkFreeMemory(device_, su.stagingMem, nullptr);
    }
    inFlightStagings_.clear();

    uint32_t idx = 0;
    VkResult acq = vkAcquireNextImageKHR(device_, swapchain_, UINT64_MAX, semAcquire_, VK_NULL_HANDLE, &idx);
    if (acq == VK_ERROR_OUT_OF_DATE_KHR) { destroySwapchain(); createSwapchain(); return; }
    if (acq != VK_SUCCESS && acq != VK_SUBOPTIMAL_KHR) vkCheck(acq);
    vkResetFences(device_, 1, &fence_);

    // Land a finished cover decode; a result whose seq is stale (superseded submit) is dropped.
    // Runs after the fence wait so a cover held on screen through the decode can be destroyed here:
    // the last frame that sampled it has completed.
    if (artWait_.pending) {
        if (auto r = artDecoder_.take(); r && r->seq == artWait_.seq) {
            artWait_.pending = false;
            const DecodedImage& img = r->img;
            if (img.w > 0) {
                destroyTexture(album_);
                album_ = createTextureRGBA(img.rgba.data(), img.w, img.h, true);  // mips: the wash mip-taps for its blur
                if (img.hasAccent) albumAccent_ = {img.ar, img.ag, img.ab, 1.0f};
                hasAlbumAccent_ = img.hasAccent;
                washDim_ = img.washDim;
            } else {
                // Undecodable cover: drop the reserved slot and re-lay the text at the coverless offset.
                destroyTexture(album_);
                hasAlbumAccent_ = false;
                artWait_.reserved = false;
                for (Marquee& m : mq_) m = Marquee{};
                layoutText(currentTrack_);
            }
        }
    }

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

    // A finished fade releases the outgoing letters and cover. Incoming was already the latest
    // song (a mid-fade skip retargets in place rather than queuing a second fade).
    if (state_ == PlaybackState::Transitioning && nowSteady - transitionStart_ >= TRANSITION_SEC) {
        state_ = PlaybackState::Active;
        destroyTexture(outgoingAlbum_);
        for (OutgoingLine& o : outgoing_) o = {};
        outgoingTrack_ = Track{};
    }

    // Song-change text fade: outgoing opacity over the first TEXT_FADE_SEC, then incoming over
    // the second. Defaults draw the incoming text solid with no transition.
    float fadeOut = 0.0f, fadeIn = 1.0f;
    if (state_ == PlaybackState::Transitioning) {
        double dt = nowSteady - transitionStart_;
        if (dt < TEXT_FADE_SEC) {
            fadeOut = fadeOutFrom_ * (1.0f - static_cast<float>(dt / TEXT_FADE_SEC));
            fadeIn = 0.0f;
        } else {
            fadeOut = 0.0f;
            fadeIn = static_cast<float>(std::clamp((dt - TEXT_FADE_SEC) / TEXT_FADE_SEC, 0.0, 1.0));
        }
    }

    const bool transitioning = state_ == PlaybackState::Transitioning;

    // Playhead smoothing reads the incoming song. t is the latest poll of that song; currentTrack_
    // is used when identity does not match so a stale poll cannot drive the fill.
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
    // oblique: world-space lighting and planar projected shadows are unaffected.
    mat4 shear;                                  // x' = x + OBLIQUE_X*z; y' = y + OBLIQUE_Y*z
    shear.at(2, 0) = OBLIQUE_X;
    shear.at(2, 1) = OBLIQUE_Y;
    CameraUBO cam;
    cam.viewProj = ortho(CARD_HALF_W, CARD_HALF_H, -1.0f, 1.0f) * shear;
    // Ortho view looks down -z; a far point on that axis gives the fragment shader a uniform view
    // vector for the specular term.
    cam.camPos = {0.0f, 0.0f, 100.0f, 0.0f};
    // params.y is the card's world half-height, used to map card-plane Y into wash-shadow UV.
    cam.params = {static_cast<float>(WIN_W) / static_cast<float>(WIN_H), CARD_HALF_H, 0.0f, 0.0f};

    const bool drawCard = transitioning ||
                          (t.valid && (!titleGlyphs_.empty() || !artistGlyphs_.empty() || artWait_.reserved));

    // Fill, tip light, and groove follow the same sequential fade as the text: outgoing playhead
    // and accent through fade-out, incoming only once fade-in starts. The incoming cover's accent
    // must not recolour the still-visible outgoing fill.
    const bool outgoingBar = transitioning && fadeOut > 0.0f && outgoingTrack_.valid;
    const Track& barTrack = outgoingBar ? outgoingTrack_ : onScreen;
    const bool barHasAlbum = outgoingBar ? outgoingHadAlbum_ : artWait_.reserved;
    const float barFade = transitioning ? (outgoingBar ? fadeOut : fadeIn) : 1.0f;
    const RGBA barColor = outgoingBar
        ? (outgoingTrack_.playing ? (outgoingHasAccent_ ? outgoingAccent_ : ACCENT) : PAUSED_FG)
        : mix(PAUSED_FG, hasAlbumAccent_ ? albumAccent_ : ACCENT, playMix);
    const double barLive = outgoingBar
        ? (outgoingTrack_.playing ? outgoingTrack_.position + (nowSteady - outgoingTrack_.posBase)
                                  : outgoingTrack_.position)
        : curLive;

    // Point light at the playhead x on the bar's row, coloured like the fill.
    cam.tipLight = {0.0f, 0.0f, 0.0f, 0.0f};
    cam.tipColor = {0.0f, 0.0f, 0.0f, 0.0f};
    if (drawCard && barTrack.valid && barTrack.duration > 0.0 && barFade > 0.0f) {
        float playScale = outgoingBar
            ? (outgoingTrack_.playing ? 1.0f : PAUSED_TIP_SCALE)
            : std::lerp(PAUSED_TIP_SCALE, 1.0f, playMix);
        float intensity = TIP_LIGHT_INTENSITY * playScale * barFade;
        if (intensity > 0.0f) {
            BarGeometry bar = barGeometry(barHasAlbum);
            float frac = static_cast<float>(std::clamp(barLive / barTrack.duration, 0.0, 1.0));
            float tipXw = bar.leftXw + (bar.rightXw - bar.leftXw) * frac;
            cam.tipLight = {tipXw, bar.cyW, TIP_LIGHT_Z, TIP_LIGHT_RANGE};
            cam.tipColor = {barColor.r, barColor.g, barColor.b, intensity};
        }
    }
    std::memcpy(camUboMapped_, &cam, sizeof(cam));

    // Collect the scene on the CPU first so the occluders can be projected into the wash-shadow
    // target before the main render pass records its draws.
    std::vector<DrawItem> items;
    auto addCard = [&](const Track& tk, Texture& album) {
        const mat4 washQuad = quadModel(0, 0, WIN_W, WIN_H, Z_CARD, 0.0f);
        const mat4 coverQuad = quadModel(0, 0, WIN_H, WIN_H, Z_ALBUM, 0.0f);
        auto addBg = [&]() {
            items.push_back({.mesh = &unitQuad_, .model = washQuad,
                             .color = {CARD_BG.r, CARD_BG.g, CARD_BG.b, 1.0f}, .fade = 1.0f,
                             .mode = MeshMode::Lit, .occluder = false, .tex = white_.dset});
        };
        auto addArt = [&](Texture& tex, float fade, float dim) {
            if (fade <= 0.0f || !tex.valid) return;
            items.push_back({.mesh = &unitQuad_, .model = washQuad, .color = {1, 1, 1, 1},
                             .fade = fade, .mode = MeshMode::Wash, .occluder = false, .tex = tex.dset, .washDim = dim});
            items.push_back({.mesh = &unitQuad_, .model = coverQuad, .color = {1, 1, 1, 1},
                             .fade = fade, .mode = MeshMode::Flat, .occluder = false, .tex = tex.dset});
        };
        const bool coverFade = transitioning && coverChanged_;
        if (coverFade) {
            // Solid card under the fading wash so the chroma key never shows through.
            addBg();
            addArt(outgoingAlbum_, fadeOut, outgoingWashDim_);
            addArt(album, fadeIn, washDim_);
        } else if (album.valid) {
            addArt(album, 1.0f, washDim_);
        } else {
            addBg();
        }
        if (tk.duration > 0.0 || tk.live) {
            // Unfilled groove: the rod's centre is lifted one radius off Z_BAR so it sits in front
            // of the card rather than sinking into it. Sized with the fill currently on screen so
            // the rod does not jump to the incoming cover inset during fade-out.
            BarGeometry bar = barGeometry(barHasAlbum);
            items.push_back({.mesh = &barRod_,
                             .model = translate({bar.leftXw, bar.cyW, Z_BAR + bar.radius}) *
                                      scale({bar.rightXw - bar.leftXw, bar.radius, bar.radius}),
                             .color = {TRACK_FG.r, TRACK_FG.g, TRACK_FG.b, 1.0f}, .fade = 1.0f, .mode = MeshMode::Lit,
                             .occluder = true, .rod = true, .tex = white_.dset, .tess = true, .tessLevel = 3.0f});
        }
    };

    // Letters cast projected shadows. A fading letter stays in the occluder set: the shadow pass
    // writes (1 - fade) so the cast shadow fades with the glyph. A marquee glyph sets clipColumn
    // so the wash-shadow pass scissors it to the column.
    auto addGlyphs = [&](const std::vector<GlyphInstance>& glyphs, float fade,
                         float penShift = 0.0f, bool clip = false) {
        for (const GlyphInstance& gi : glyphs) {
            float sx = gi.pos.x + penShift;
            items.push_back({.mesh = gi.mesh,
                             .model = translate({sx, gi.pos.y, gi.pos.z}) * scale(gi.scale),
                             .color = gi.color, .fade = fade, .mode = MeshMode::Lit, .occluder = true, .tex = white_.dset,
                             .clipColumn = clip});
        }
    };

    // Neon progress fill: the played span from the bar's left edge to the playhead, one Z step
    // nearer than the groove.
    auto addFill = [&](const Track& tk, bool hasAlbum, RGBA fillColor, float fade, double playhead) {
        if (fade <= 0.0f || (tk.duration <= 0.0 && !tk.live)) return;
        double frac = tk.live ? 1.0 : std::clamp(std::min(playhead, tk.duration) / tk.duration, 0.0, 1.0);
        if (frac <= 0.0) return;
        BarGeometry bar = barGeometry(hasAlbum);
        float tipXw = bar.leftXw + static_cast<float>((bar.rightXw - bar.leftXw) * frac);
        float fillLen = std::max(tipXw - bar.leftXw, bar.radius);
        items.push_back({.mesh = &barRod_,
                         .model = translate({bar.leftXw, bar.cyW, Z_BAR_FILL + bar.radius}) *
                                  scale({fillLen, bar.radius, bar.radius}),
                         .color = {fillColor.r, fillColor.g, fillColor.b, 1.0f}, .fade = fade, .mode = MeshMode::Bar, .occluder = false,
                         .tex = white_.dset, .tess = true, .tessLevel = 24.0f,
                         .barLen = fillLen, .liveBar = tk.live ? 1.0f : 0.0f});
    };

    if (drawCard) {
        addCard(barTrack, album_);
        // A scrolling line draws two copies a period apart so one enters from the right as the
        // other leaves left, both column-clipped; a line that fits draws one unclipped copy. The
        // live line scrolls with the clock, the outgoing copy holds its frozen offset.
        auto drawLine = [&](const std::vector<GlyphInstance>& glyphs, bool scrolling, float shift, float period, float fade) {
            if (fade <= 0.0f) return;
            if (!scrolling) { addGlyphs(glyphs, fade); return; }
            addGlyphs(glyphs, fade, -shift, true);
            addGlyphs(glyphs, fade, period - shift, true);
        };
        for (Line l : {Line::Title, Line::Artist}) {
            const Marquee& m = mq(l);
            const OutgoingLine& o = outgoing(l);
            const bool fading = transitioning && o.changed;
            double scroll = std::max(0.0, (nowSteady - m.start) - MARQUEE_HOLD_SEC) * m.speed;
            double period = m.width + m.gap;
            float shift = period > 0.0 ? static_cast<float>(std::fmod(scroll, period)) : 0.0f;
            drawLine(glyphs(l), m.active, shift, static_cast<float>(period), fading ? fadeIn : 1.0f);
            if (fading) drawLine(o.glyphs, o.scrolling, o.shift, o.period, fadeOut);
        }
        addFill(barTrack, barHasAlbum, barColor, barFade, barLive);
    }

    // Refresh the incoming song's snapshot each frame so the next change starts from its real
    // last-shown position (a seek does not update the setTrack snapshot). artPng is preserved:
    // only setTrack/refreshArt rebuild album_, so currentTrack_.artPng must keep mirroring the
    // bytes album_ was built from.
    if (t.valid && t.title == currentTrack_.title && t.artist == currentTrack_.artist) {
        currentTrack_.valid = t.valid;
        currentTrack_.playing = t.playing;
        currentTrack_.live = t.live;
        currentTrack_.duration = t.duration;
        currentTrack_.position = t.position;
        currentTrack_.posBase = t.posBase;
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
    }

    VkCommandBuffer cb = cmds_[idx];
    vkResetCommandBuffer(cb, 0);
    VkCommandBufferBeginInfo bi{VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO};
    vkBeginCommandBuffer(cb, &bi);

    for (StagedUpload& su : pendingUploads_)
        recordTextureUpload(cb, su);
    pendingUploads_.clear();

    constexpr VkShaderStageFlags pushStages = VK_SHADER_STAGE_VERTEX_BIT |
        VK_SHADER_STAGE_TESSELLATION_CONTROL_BIT | VK_SHADER_STAGE_TESSELLATION_EVALUATION_BIT |
        VK_SHADER_STAGE_FRAGMENT_BIT;

    // Only album washes sample this texture. Coverless cards and transition frames with no
    // visible wash need neither the occluder draws nor the compute filter.
    const bool drawWash = std::any_of(items.begin(), items.end(), [](const DrawItem& it) {
        return it.mode == MeshMode::Wash;
    });
    if (drawWash) {
        VkClearValue occClear{};
        occClear.color = VkClearColorValue{{1.0f, 1.0f, 1.0f, 1.0f}};
        VkRenderPassBeginInfo sbi{VK_STRUCTURE_TYPE_RENDER_PASS_BEGIN_INFO};
        sbi.renderPass = washOccPass_;
        sbi.framebuffer = washOccFb_;
        sbi.renderArea.extent = washShadowExtent_;
        sbi.clearValueCount = 1;
        sbi.pClearValues = &occClear;
        vkCmdBeginRenderPass(cb, &sbi, VK_SUBPASS_CONTENTS_INLINE);

        const float ww = static_cast<float>(washShadowExtent_.width);
        VkViewport svpt{0, 0, ww, static_cast<float>(washShadowExtent_.height), 0, 1};
        VkRect2D ssc{{0, 0}, washShadowExtent_};
        vkCmdSetViewport(cb, 0, 1, &svpt);
        vkCmdSetScissor(cb, 0, 1, &ssc);
        vkCmdBindDescriptorSets(cb, VK_PIPELINE_BIND_POINT_GRAPHICS, shadowPipeLayout_, 0, 1, &camSet_, 0, nullptr);

        // Column clip in wash UV: world x without the camera's oblique shear, compared to the
        // sheared letter edges so the cut lines up with the scissored raster.
        auto washEdgePx = [&](float worldXv) {
            return (worldXv / CARD_HALF_W * 0.5f + 0.5f) * ww;
        };
        const int sColL = std::clamp(static_cast<int>(std::floor(washEdgePx(colLeft_ + OBLIQUE_X * Z_TEXT))),
                                     0, static_cast<int>(washShadowExtent_.width));
        const int sColR = std::clamp(static_cast<int>(std::ceil(washEdgePx(colRight_ + OBLIQUE_X * Z_TEXT))),
                                     sColL, static_cast<int>(washShadowExtent_.width));
        VkRect2D sColSc{{sColL, 0}, {static_cast<uint32_t>(sColR - sColL), washShadowExtent_.height}};

        auto drawOcc = [&](int lightMode, bool includeRod) {
            bool colScissor = false;
            VkPipeline bound = VK_NULL_HANDLE;
            vkCmdSetScissor(cb, 0, 1, &ssc);
            for (const DrawItem& it : items) {
                if (!it.occluder || it.mesh->indexCount == 0 || it.fade <= 0.0f) continue;
                if (it.rod && !includeRod) continue;
                if (it.clipColumn != colScissor) {
                    vkCmdSetScissor(cb, 0, 1, it.clipColumn ? &sColSc : &ssc);
                    colScissor = it.clipColumn;
                }
                VkPipeline want = it.tess ? shadowTessPipeline_ : shadowPipeline_;
                if (want != bound) {
                    vkCmdBindPipeline(cb, VK_PIPELINE_BIND_POINT_GRAPHICS, want);
                    bound = want;
                }
                MeshPush pc{};
                pc.model = it.model;
                pc.fade = it.fade;
                pc.mode = lightMode;
                pc.tessLevel = it.tessLevel;
                vkCmdPushConstants(cb, shadowPipeLayout_, pushStages, 0, sizeof(pc), &pc);
                VkDeviceSize off = 0;
                vkCmdBindVertexBuffers(cb, 0, 1, &it.mesh->vbo, &off);
                vkCmdBindIndexBuffer(cb, it.mesh->ibo, 0, VK_INDEX_TYPE_UINT32);
                vkCmdDrawIndexed(cb, it.mesh->indexCount, 1, 0, 0, 0);
            }
        };
        drawOcc(0, true);
        if (cam.tipColor.w > 0.0f) drawOcc(1, false);
        vkCmdEndRenderPass(cb);

        vkCmdBindPipeline(cb, VK_PIPELINE_BIND_POINT_COMPUTE, washShadowPipeline_);
        std::array<VkDescriptorSet, 2> washSets{washStoreSet_, camSet_};
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
    // keys it out. The card-to-key swap is a hard cut: the text fade runs over the card and never
    // reveals the key.
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
    // set 1 (camera) is bound once: both pipelines share meshPipeLayout_, so switching between the
    // flat and tessellated pipelines leaves it in place.
    vkCmdBindDescriptorSets(cb, VK_PIPELINE_BIND_POINT_GRAPHICS, meshPipeLayout_, 1, 1, &camSet_, 0, nullptr);
    VkDescriptorSet boundTex = VK_NULL_HANDLE;
    VkPipeline boundPipe = VK_NULL_HANDLE;
    for (const DrawItem& it : items) {
        const Mesh& m = *it.mesh;
        if (m.indexCount == 0 || it.fade <= 0.0f) continue;
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
        pc.liveBar = it.liveBar;
        pc.washDim = it.washDim;
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
    for (Texture& tex : pendingDestroyTextures_) destroyTexture(tex);
    pendingDestroyTextures_.clear();
    for (StagedUpload& su : pendingUploads_) {
        vkDestroyBuffer(device_, su.staging, nullptr);
        vkFreeMemory(device_, su.stagingMem, nullptr);
    }
    pendingUploads_.clear();
    for (StagedUpload& su : inFlightStagings_) {
        vkDestroyBuffer(device_, su.staging, nullptr);
        vkFreeMemory(device_, su.stagingMem, nullptr);
    }
    inFlightStagings_.clear();
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
    if (shadowPipeline_) vkDestroyPipeline(device_, shadowPipeline_, nullptr);
    if (shadowTessPipeline_) vkDestroyPipeline(device_, shadowTessPipeline_, nullptr);
    if (shadowPipeLayout_) vkDestroyPipelineLayout(device_, shadowPipeLayout_, nullptr);
    if (washShadowPipeline_) vkDestroyPipeline(device_, washShadowPipeline_, nullptr);
    if (washShadowPipeLayout_) vkDestroyPipelineLayout(device_, washShadowPipeLayout_, nullptr);
    if (washStorePool_) vkDestroyDescriptorPool(device_, washStorePool_, nullptr);
    if (washStoreLayout_) vkDestroyDescriptorSetLayout(device_, washStoreLayout_, nullptr);
    if (washSampler_) vkDestroySampler(device_, washSampler_, nullptr);
    if (washOccSampler_) vkDestroySampler(device_, washOccSampler_, nullptr);
    if (camPool_) vkDestroyDescriptorPool(device_, camPool_, nullptr);
    if (camLayout_) vkDestroyDescriptorSetLayout(device_, camLayout_, nullptr);
    if (sampler_) vkDestroySampler(device_, sampler_, nullptr);
    if (dsetPool_) vkDestroyDescriptorPool(device_, dsetPool_, nullptr);
    if (dsetLayout_) vkDestroyDescriptorSetLayout(device_, dsetLayout_, nullptr);
    destroySwapchain();
    if (washOccPass_) vkDestroyRenderPass(device_, washOccPass_, nullptr);
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
