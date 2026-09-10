// Fake media session for --simulate: a 10-song playlist and one generated cover PNG per track.
#include "core/simulate.hpp"
#include <array>
#include <cmath>
#include <cstdint>
#include <utility>
#include <vector>

namespace {

constexpr std::array<std::pair<const wchar_t*, const wchar_t*>, SongSwitchSim::kSongCount> kSongs{{
    {L"Midnight Static", L"Vera Coil"},
    {L"Paper Moons", L"Juniper Salt"},
    {L"Copper Hymn", L"Old Radio"},
    {L"Velvet Runway", L"Chrome Saints"},
    {L"Low Tide Letters", L"Marina Finch"},
    {L"Secondhand Light", L"Oak & Wire"},
    {L"Fallow Year", L"North Window"},
    {L"Brass Comet", L"Kite Club"},
    {L"Hollow Frequency", L"Glass Garden"},
    {L"Last Warm Map", L"Ivory Mile"},
}};

// Background / shape / footer, one triple per song so the square, wash, and accent all change.
struct Rgb { uint8_t r, g, b; };
constexpr std::array<std::array<Rgb, 3>, SongSwitchSim::kSongCount> kCover{{
    {{{0x2A, 0x14, 0x48}, {0xC9, 0x4B, 0xFF}, {0x12, 0x08, 0x22}}},
    {{{0xF2, 0xE6, 0xC2}, {0x3A, 0x5B, 0xA8}, {0xC4, 0xB4, 0x8A}}},
    {{{0xB8, 0x5A, 0x2A}, {0xF0, 0xC0, 0x78}, {0x6A, 0x2E, 0x12}}},
    {{{0x7A, 0x14, 0x48}, {0xFF, 0x6B, 0xC9}, {0x3A, 0x08, 0x22}}},
    {{{0x0E, 0x5C, 0x5C}, {0x7E, 0xE0, 0xD0}, {0x06, 0x2E, 0x32}}},
    {{{0xC4, 0x7A, 0x18}, {0xFF, 0xE0, 0x8A}, {0x6A, 0x3A, 0x08}}},
    {{{0x4A, 0x58, 0x20}, {0xC8, 0xD4, 0x6A}, {0x24, 0x2C, 0x10}}},
    {{{0xC8, 0x9A, 0x18}, {0xFF, 0xF0, 0xB0}, {0x6A, 0x48, 0x08}}},
    {{{0x1A, 0x18, 0x68}, {0x8A, 0x7C, 0xFF}, {0x0A, 0x08, 0x32}}},
    {{{0xC4, 0x5A, 0x38}, {0xF0, 0xD0, 0x98}, {0x6A, 0x24, 0x14}}},
}};

void appendU32be(std::vector<uint8_t>& o, uint32_t v) {
    o.push_back(static_cast<uint8_t>(v >> 24));
    o.push_back(static_cast<uint8_t>(v >> 16));
    o.push_back(static_cast<uint8_t>(v >> 8));
    o.push_back(static_cast<uint8_t>(v));
}

[[nodiscard]] uint32_t crc32(const uint8_t* p, size_t n) {
    uint32_t c = 0xFFFFFFFFu;
    for (size_t i = 0; i < n; ++i) {
        c ^= p[i];
        for (int b = 0; b < 8; ++b)
            c = (c & 1u) ? (c >> 1) ^ 0xEDB88320u : c >> 1;
    }
    return c ^ 0xFFFFFFFFu;
}

[[nodiscard]] uint32_t adler32(const uint8_t* p, size_t n) {
    uint32_t s1 = 1, s2 = 0;
    for (size_t i = 0; i < n; ++i) {
        s1 = (s1 + p[i]) % 65521u;
        s2 = (s2 + s1) % 65521u;
    }
    return (s2 << 16) | s1;
}

void appendPngChunk(std::vector<uint8_t>& o, const char type[4], const uint8_t* data, size_t n) {
    appendU32be(o, static_cast<uint32_t>(n));
    const size_t typeOff = o.size();
    o.insert(o.end(), type, type + 4);
    if (n) o.insert(o.end(), data, data + n);
    appendU32be(o, crc32(o.data() + typeOff, 4 + n));
}

// Uncompressed truecolor PNG: filter-none scanlines in one stored deflate block (no zlib).
[[nodiscard]] std::vector<uint8_t> coverPng(int song) {
    constexpr int kN = 96;
    const auto& pal = kCover[static_cast<size_t>(song)];
    const Rgb bg = pal[0], hi = pal[1], lo = pal[2];
    const int kind = song % 3;

    std::vector<uint8_t> raw(static_cast<size_t>(kN) * (1 + kN * 3));
    for (int y = 0; y < kN; ++y) {
        uint8_t* row = &raw[static_cast<size_t>(y) * (1 + kN * 3)];
        row[0] = 0;
        for (int x = 0; x < kN; ++x) {
            const float u = (x + 0.5f) / kN, v = (y + 0.5f) / kN;
            const float dx = u - 0.38f, dy = v - 0.40f;
            const bool disc = dx * dx + dy * dy < 0.20f * 0.20f;
            const bool diag = (u + v) > 1.05f;
            const bool stripe = u > 0.28f && u < 0.52f;
            const bool shape = kind == 0 ? disc : kind == 1 ? diag : stripe;
            const Rgb c = (v > 0.78f) ? lo : (shape ? hi : bg);
            uint8_t* px = row + 1 + x * 3;
            px[0] = c.r; px[1] = c.g; px[2] = c.b;
        }
    }

    std::vector<uint8_t> ihdr;
    appendU32be(ihdr, static_cast<uint32_t>(kN));
    appendU32be(ihdr, static_cast<uint32_t>(kN));
    ihdr.insert(ihdr.end(), {8, 2, 0, 0, 0});

    std::vector<uint8_t> z{0x78, 0x01, 0x01};
    const uint16_t len = static_cast<uint16_t>(raw.size());
    z.push_back(static_cast<uint8_t>(len));
    z.push_back(static_cast<uint8_t>(len >> 8));
    z.push_back(static_cast<uint8_t>(~len));
    z.push_back(static_cast<uint8_t>(~len >> 8));
    z.insert(z.end(), raw.begin(), raw.end());
    appendU32be(z, adler32(raw.data(), raw.size()));

    std::vector<uint8_t> png{137, 80, 78, 71, 13, 10, 26, 10};
    appendPngChunk(png, "IHDR", ihdr.data(), ihdr.size());
    appendPngChunk(png, "IDAT", z.data(), z.size());
    appendPngChunk(png, "IEND", nullptr, 0);
    return png;
}

[[nodiscard]] Track makeTrack(int i, double now) {
    Track t;
    t.valid = true;
    t.title = kSongs[static_cast<size_t>(i)].first;
    t.artist = kSongs[static_cast<size_t>(i)].second;
    t.playing = true;
    t.duration = 240.0;
    t.position = 20.0 + static_cast<double>(i) * 12.0;
    t.posBase = now;
    t.artPng = coverPng(i);
    return t;
}

}  // namespace

std::optional<Track> SongSwitchSim::poll(double nowSteady) {
    if (shown_ == 0) {
        shown_ = 1;
        nextAt_ = nowSteady + kSwitchSec;
        return makeTrack(0, nowSteady);
    }
    if (nowSteady < nextAt_) return std::nullopt;
    if (shown_ >= kSongCount) {
        done_ = true;
        return std::nullopt;
    }
    Track t = makeTrack(shown_, nowSteady);
    shown_++;
    nextAt_ += kSwitchSec;
    return t;
}

int SongSwitchSim::waitMs(double nowSteady) const {
    if (done_ || shown_ == 0) return 0;
    const double remain = nextAt_ - nowSteady;
    if (remain <= 0.0) return 0;
    return static_cast<int>(std::ceil(remain * 1000.0));
}
