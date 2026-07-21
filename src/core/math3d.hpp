// Conventions: right-handed world space (x right, y up, z toward the viewer); column-major
// storage with data[col*4 + row]; matrices multiply as M * v with column vectors. The ortho
// projection targets Vulkan clip space — depth maps to [0, 1] and the y axis is flipped inside
// the projection, so the viewport keeps its natural positive height.
#pragma once
#include <array>
#include <cmath>

struct vec2 { float x = 0, y = 0; };
struct vec3 { float x = 0, y = 0, z = 0; };
struct vec4 { float x = 0, y = 0, z = 0, w = 0; };

// Tightly packed (8 floats) so it maps directly to the
// R32G32B32 / R32G32B32 / R32G32 vertex attributes.
struct Vertex3 { vec3 pos; vec3 normal; vec2 uv; };

[[nodiscard]] inline constexpr vec3 operator*(vec3 a, float s) noexcept { return {a.x * s, a.y * s, a.z * s}; }
[[nodiscard]] inline constexpr float dot(vec3 a, vec3 b) noexcept { return a.x * b.x + a.y * b.y + a.z * b.z; }
[[nodiscard]] inline float length(vec3 a) { return std::sqrt(dot(a, a)); }
[[nodiscard]] inline vec3 normalize(vec3 a) {
    float l = length(a);
    return l > 0.0f ? a * (1.0f / l) : a;
}

struct mat4 {
    std::array<float, 16> data{1, 0, 0, 0,  0, 1, 0, 0,  0, 0, 1, 0,  0, 0, 0, 1};
    constexpr float& at(int col, int row) noexcept { return data[col * 4 + row]; }
    [[nodiscard]] constexpr float at(int col, int row) const noexcept { return data[col * 4 + row]; }
};

[[nodiscard]] inline constexpr mat4 operator*(const mat4& a, const mat4& b) noexcept {
    mat4 r;
    for (int c = 0; c < 4; ++c)
        for (int row = 0; row < 4; ++row) {
            float s = 0;
            for (int k = 0; k < 4; ++k) s += a.at(k, row) * b.at(c, k);
            r.at(c, row) = s;
        }
    return r;
}

[[nodiscard]] inline constexpr mat4 translate(vec3 t) noexcept {
    mat4 m;
    m.at(3, 0) = t.x; m.at(3, 1) = t.y; m.at(3, 2) = t.z;
    return m;
}

[[nodiscard]] inline constexpr mat4 scale(vec3 s) noexcept {
    mat4 m;
    m.at(0, 0) = s.x; m.at(1, 1) = s.y; m.at(2, 2) = s.z;
    return m;
}
[[nodiscard]] inline constexpr mat4 scale(float s) noexcept { return scale(vec3{s, s, s}); }

// z in [zLo, zHi] -> depth [0, 1] with the near, viewer-facing end (zHi, larger z) mapping to 0
// so it wins the LESS_OR_EQUAL depth test.
[[nodiscard]] inline mat4 ortho(float halfW, float halfH, float zLo, float zHi) {
    mat4 m;
    m.at(0, 0) = 1.0f / halfW;
    m.at(1, 1) = -1.0f / halfH;
    m.at(2, 2) = -1.0f / (zHi - zLo);
    m.at(3, 2) = zHi / (zHi - zLo);
    m.at(3, 3) = 1.0f;
    return m;
}
