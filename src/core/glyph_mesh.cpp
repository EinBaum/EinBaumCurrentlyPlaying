// Portable glyph tessellation: turns flattened, em-normalized outline contours into an extruded 3D
// mesh (front/back caps + side walls).
#include "core/glyph_mesh.hpp"
#include <algorithm>
#include <cmath>
#include <limits>
#include <numeric>
#include <vector>

namespace {

// Em-normalized weld tolerance, only used to skip coincident bridge endpoints during ear clipping.
constexpr float WELD_EPS = 1e-6f;

using Contour = std::vector<vec2>;

[[nodiscard]] float signedArea(const Contour& c) {
    float a = 0.0f;
    for (size_t i = 0, n = c.size(); i < n; ++i) {
        const vec2& p = c[i];
        const vec2& q = c[(i + 1) % n];
        a += p.x * q.y - q.x * p.y;
    }
    return a * 0.5f;
}

[[nodiscard]] bool pointInPolygon(vec2 pt, const Contour& c) {
    bool in = false;
    for (size_t i = 0, j = c.size() - 1; i < c.size(); j = i++)
        if (((c[i].y > pt.y) != (c[j].y > pt.y)) &&
            (pt.x < (c[j].x - c[i].x) * (pt.y - c[i].y) / (c[j].y - c[i].y) + c[i].x))
            in = !in;
    return in;
}

[[nodiscard]] float cross2(vec2 o, vec2 a, vec2 b) {
    return (a.x - o.x) * (b.y - o.y) - (a.y - o.y) * (b.x - o.x);
}

// Strict crossing only: shared endpoints and collinear touches return false (all comparisons are > 0).
[[nodiscard]] bool segCross(vec2 a, vec2 b, vec2 c, vec2 d) {
    float d1 = cross2(c, d, a), d2 = cross2(c, d, b);
    float d3 = cross2(a, b, c), d4 = cross2(a, b, d);
    return ((d1 > 0) != (d2 > 0)) && ((d3 > 0) != (d4 > 0));
}

[[nodiscard]] bool coincident(vec2 a, vec2 b) {
    return std::fabs(a.x - b.x) < WELD_EPS && std::fabs(a.y - b.y) < WELD_EPS;
}

struct OutlineSet {
    std::vector<Contour> outers;               // CCW
    std::vector<std::vector<Contour>> holes;   // CW, parallel to outers
};

[[nodiscard]] OutlineSet classify(const std::vector<Contour>& contours) {
    int n = static_cast<int>(contours.size());
    // A boundary vertex represents each contour's location: for a round glyph (o, O, d) the outer
    // ring's vertex-average centre lies inside the hole, which would misclassify it. Contours never
    // cross, so a vertex of contour i is inside contour j exactly when i nests inside j.
    std::vector<vec2> rep(n);
    std::vector<float> absArea(n);
    for (int i = 0; i < n; ++i) { rep[i] = contours[i][0]; absArea[i] = std::fabs(signedArea(contours[i])); }
    std::vector<int> depth(n, 0);
    for (int i = 0; i < n; ++i)
        for (int j = 0; j < n; ++j)
            if (i != j && pointInPolygon(rep[i], contours[j])) ++depth[i];

    OutlineSet os;
    std::vector<int> outerOf(n, -1);
    for (int i = 0; i < n; ++i)
        if (depth[i] % 2 == 0) {
            outerOf[i] = static_cast<int>(os.outers.size());
            Contour o = contours[i];
            if (signedArea(o) < 0) std::reverse(o.begin(), o.end());  // force CCW
            os.outers.push_back(std::move(o));
            os.holes.emplace_back();
        }
    for (int i = 0; i < n; ++i)
        if (depth[i] % 2 == 1) {
            int best = -1; float bestA = std::numeric_limits<float>::infinity();
            for (int j = 0; j < n; ++j)
                if (outerOf[j] >= 0 && absArea[j] < bestA && pointInPolygon(rep[i], contours[j])) {
                    best = outerOf[j]; bestA = absArea[j];
                }
            if (best < 0 && !os.outers.empty()) best = 0;
            if (best >= 0) {
                Contour h = contours[i];
                if (signedArea(h) > 0) std::reverse(h.begin(), h.end());  // force CW
                os.holes[best].push_back(std::move(h));
            }
        }
    return os;
}

// A diagonal that crosses no boundary edge stays inside the outer and outside the holes, so it is
// a valid bridge; the shortest such diagonal merges each hole into the outer as one simple polygon.
[[nodiscard]] Contour mergeHoles(const Contour& outer, const std::vector<Contour>& holes) {
    Contour loop = outer;
    for (size_t hk = 0; hk < holes.size(); ++hk) {
        const Contour& hole = holes[hk];
        if (hole.size() < 3) continue;
        auto crosses = [&](vec2 P, int li, vec2 M, int hi) {
            int ln = static_cast<int>(loop.size());
            for (int i = 0; i < ln; ++i) {
                int j = (i + 1) % ln;
                if (i == li || j == li) continue;
                if (segCross(P, M, loop[i], loop[j])) return true;
            }
            for (size_t k = hk; k < holes.size(); ++k) {
                const Contour& H = holes[k];
                int hn = static_cast<int>(H.size());
                for (int i = 0; i < hn; ++i) {
                    int j = (i + 1) % hn;
                    if (k == hk && (i == hi || j == hi)) continue;
                    if (segCross(P, M, H[i], H[j])) return true;
                }
            }
            return false;
        };
        int bestLi = -1, bestHi = -1; float bestD = std::numeric_limits<float>::infinity();
        for (int hi = 0; hi < static_cast<int>(hole.size()); ++hi)
            for (int li = 0; li < static_cast<int>(loop.size()); ++li) {
                vec2 P = loop[li], M = hole[hi];
                float dx = P.x - M.x, dy = P.y - M.y, d = dx * dx + dy * dy;
                if (d >= bestD) continue;
                if (!crosses(P, li, M, hi)) { bestD = d; bestLi = li; bestHi = hi; }
            }
        if (bestLi < 0) continue;  // no valid bridge: leave the hole unfilled rather than corrupt
        Contour merged;
        merged.reserve(loop.size() + hole.size() + 2);
        for (int i = 0; i <= bestLi; ++i) merged.push_back(loop[i]);
        int hn = static_cast<int>(hole.size());
        for (int i = 0; i <= hn; ++i) merged.push_back(hole[(bestHi + i) % hn]);  // hn+1 vertices: full hole loop from M back to M
        merged.push_back(loop[bestLi]);                                            // back to P
        for (int i = bestLi + 1; i < static_cast<int>(loop.size()); ++i) merged.push_back(loop[i]);
        loop = std::move(merged);
    }
    return loop;
}

[[nodiscard]] bool pointInTri(vec2 p, vec2 a, vec2 b, vec2 c) {
    float d1 = cross2(a, b, p), d2 = cross2(b, c, p), d3 = cross2(c, a, p);
    bool neg = (d1 < 0) || (d2 < 0) || (d3 < 0);
    bool pos = (d1 > 0) || (d2 > 0) || (d3 > 0);
    return !(neg && pos);
}

void earClip(const Contour& poly, std::vector<int>& tris) {
    int n = static_cast<int>(poly.size());
    if (n < 3) return;
    std::vector<int> idx(n);
    std::iota(idx.begin(), idx.end(), 0);
    if (signedArea(poly) < 0) std::reverse(idx.begin(), idx.end());

    int guard = 0;
    const int maxGuard = n * n + 16;
    while (idx.size() > 3 && guard++ < maxGuard) {
        int m = static_cast<int>(idx.size());
        bool clipped = false;
        for (int i = 0; i < m; ++i) {
            int i0 = idx[(i + m - 1) % m], i1 = idx[i], i2 = idx[(i + 1) % m];
            vec2 a = poly[i0], b = poly[i1], c = poly[i2];
            if (cross2(a, b, c) <= 0) continue;        // reflex/degenerate (CCW ear is convex)
            bool ok = true;
            for (int k = 0; k < m && ok; ++k) {
                int vi = idx[k];
                if (vi == i0 || vi == i1 || vi == i2) continue;
                vec2 q = poly[vi];
                if (coincident(q, a) || coincident(q, b) || coincident(q, c)) continue;  // bridge dup
                if (pointInTri(q, a, b, c)) ok = false;
            }
            if (!ok) continue;
            tris.push_back(i0); tris.push_back(i1); tris.push_back(i2);
            idx.erase(idx.begin() + i);
            clipped = true;
            break;
        }
        if (!clipped) break;  // degenerate remainder: stop rather than spin
    }
    if (idx.size() == 3) { tris.push_back(idx[0]); tris.push_back(idx[1]); tris.push_back(idx[2]); }
}

}  // namespace

GlyphMesh3 tessellateGlyph(const GlyphOutline& outline) {
    GlyphMesh3 mesh;
    mesh.advance = outline.advance;
    if (outline.contours.empty()) return mesh;   // whitespace: advance only

    OutlineSet os = classify(outline.contours);

    auto pushTri = [&](vec3 a, vec3 b, vec3 c, vec3 nrm) {
        uint32_t base = static_cast<uint32_t>(mesh.verts.size());
        mesh.verts.push_back({a, nrm, {}});
        mesh.verts.push_back({b, nrm, {}});
        mesh.verts.push_back({c, nrm, {}});
        mesh.indices.push_back(base);
        mesh.indices.push_back(base + 1);
        mesh.indices.push_back(base + 2);
    };

    // One triangulation feeds both caps; the back cap reverses winding (a,c,b) so its -z normal faces outward.
    for (size_t oi = 0; oi < os.outers.size(); ++oi) {
        Contour merged = mergeHoles(os.outers[oi], os.holes[oi]);
        std::vector<int> tris;
        earClip(merged, tris);
        for (size_t t = 0; t + 2 < tris.size(); t += 3) {
            vec2 a = merged[tris[t]], b = merged[tris[t + 1]], c = merged[tris[t + 2]];
            pushTri({a.x, a.y, GLYPH_EXTRUDE_HALF}, {b.x, b.y, GLYPH_EXTRUDE_HALF}, {c.x, c.y, GLYPH_EXTRUDE_HALF}, {0, 0, 1});
            pushTri({a.x, a.y, -GLYPH_EXTRUDE_HALF}, {c.x, c.y, -GLYPH_EXTRUDE_HALF}, {b.x, b.y, -GLYPH_EXTRUDE_HALF}, {0, 0, -1});
        }
    }

    // Side walls from the oriented contour edges (outers CCW, holes CW). For both, the outward
    // normal of edge a->b is (e.y, -e.x).
    auto wall = [&](const Contour& c) {
        int m = static_cast<int>(c.size());
        for (int i = 0; i < m; ++i) {
            vec2 a = c[i], b = c[(i + 1) % m];
            vec3 e{b.x - a.x, b.y - a.y, 0};
            vec3 nrm = normalize(vec3{e.y, -e.x, 0});
            vec3 fa{a.x, a.y, GLYPH_EXTRUDE_HALF}, fb{b.x, b.y, GLYPH_EXTRUDE_HALF};
            vec3 ba{a.x, a.y, -GLYPH_EXTRUDE_HALF}, bb{b.x, b.y, -GLYPH_EXTRUDE_HALF};
            pushTri(fa, ba, bb, nrm);
            pushTri(fa, bb, fb, nrm);
        }
    };
    for (size_t oi = 0; oi < os.outers.size(); ++oi) {
        wall(os.outers[oi]);
        for (const Contour& h : os.holes[oi]) wall(h);
    }
    return mesh;
}
