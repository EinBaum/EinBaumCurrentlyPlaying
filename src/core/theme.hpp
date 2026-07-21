#pragma once

// Layout reference space, not actual resolution: the renderer normalizes every position by WIN_W
// into world space, so any 5:1 on-screen window renders this same composition.
constexpr int WIN_W = 2400, WIN_H = 480;
// On-screen / swapchain window, same 5:1 aspect as the reference space.
constexpr int WINDOW_W = 1000, WINDOW_H = 200;
constexpr int EDGE_GAP = 5;
constexpr int PAD = 66;         // cover->text left offset (WIN_H + PAD) and right margin (WIN_W - PAD)
constexpr int PAD_TOP = 30;
constexpr int PAD_BOTTOM = 36;
constexpr int BAR_H = 15;
constexpr int TITLE_PX = 144;
constexpr int ARTIST_PX = 112;

struct RGBA { float r, g, b, a; };

[[nodiscard]] constexpr RGBA hex(int rr, int gg, int bb, float a = 1.0f) {
    return {rr / 255.0f, gg / 255.0f, bb / 255.0f, a};
}
[[nodiscard]] constexpr RGBA mix(RGBA a, RGBA b, float t) {
    return {a.r + (b.r - a.r) * t, a.g + (b.g - a.g) * t,
            a.b + (b.b - a.b) * t, a.a + (b.a - a.a) * t};
}

// The swapchain is non-sRGB UNORM, so a float of v/255 is written back as the byte v exactly.
constexpr RGBA KEY_COLOR  = hex(0x00, 0xFF, 0x00);
constexpr RGBA CARD_BG    = hex(0x0E, 0x0E, 0x12);
constexpr RGBA ACCENT     = hex(0xFF, 0x33, 0x55);
constexpr RGBA TITLE_FG   = hex(0xFF, 0xFF, 0xFF);
constexpr RGBA ARTIST_FG  = hex(0xF1, 0xF1, 0xFC);
constexpr RGBA TRACK_FG   = hex(0x2C, 0x2C, 0x36);
constexpr RGBA PAUSED_FG  = hex(0x6E, 0x6E, 0x78);
