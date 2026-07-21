// The OS-agnostic application core: media polling, redraw-cadence math, and the main loop. The
// platform entry points create a PlatformWindow and call runApp.
#include "core/app.hpp"
#include "core/media.hpp"
#include "core/renderer.hpp"
#include "platform/platform.hpp"
#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <limits>
#include <string>
#include <vector>

namespace {

Renderer g_renderer;
MediaPoller g_poller;
bool g_inited = false;
Track g_track;
uint64_t g_lastSeq = std::numeric_limits<uint64_t>::max();
std::wstring g_lastIdentity = L"\x01";  // sentinel != any real identity
std::vector<uint8_t> g_lastArt;
double g_noMediaSince = 0.0;            // 0 means no no-media gap is in progress
constexpr double KEEP_LAST_SEC = 10.0;

[[nodiscard]] double steadySeconds() {
    using namespace std::chrono;
    return duration<double>(steady_clock::now().time_since_epoch()).count();
}

// Returns true when the on-screen state changed and a redraw is due now.
[[nodiscard]] bool pollMedia() {
    if (!g_inited) return false;
    Track t;
    uint64_t seq = g_poller.latest(t);
    if (seq == g_lastSeq) return false;
    g_lastSeq = seq;
    if (t.valid) {
        g_noMediaSince = 0.0;
        std::wstring id = t.identity();
        if (id != g_lastIdentity) {
            g_lastIdentity = id;
            g_lastArt = t.artPng;
            g_renderer.setTrack(t);
        } else if (!t.artPng.empty() && t.artPng != g_lastArt) {
            // same song, the cover arrived/changed after the title — swap it in (no crossfade)
            g_lastArt = t.artPng;
            g_renderer.refreshArt(t);
        }
        g_track = t;
        return true;
    }
    if (g_track.valid) {
        // The media session intermittently reports no media mid-song; hold the last track through
        // gaps shorter than KEEP_LAST_SEC so a momentary stall doesn't blank the card.
        double now = steadySeconds();
        if (g_noMediaSince == 0.0) g_noMediaSince = now;
        if (now - g_noMediaSince >= KEEP_LAST_SEC) {
            g_lastIdentity = L"";
            g_noMediaSince = 0.0;
            g_track = Track{};
            g_renderer.setTrack(Track{});
            return true;
        }
    }
    return false;
}

}  // namespace

int runApp(PlatformWindow& window, bool debug) {
    g_renderer.init(window);
    g_renderer.setShowFps(debug);
    g_inited = true;
    g_poller.start();

    window.show();

    // A frame is drawn only when the scene can have changed: a media change or the first frame
    // forces one, an animation or advancing playhead paces it via cadence(), and a static scene
    // only wakes every IDLE_POLL_SEC. An OS event wakes the wait early.
    constexpr double PLAY_FPS_MAX = 60.0;            // shortest songs
    constexpr double PLAY_FPS_MIN = 1.0;             // songs at or past PLAY_FPS_SPAN
    constexpr double PLAY_FPS_SPAN = 1200.0;         // seconds over which the rate ramps down
    constexpr double PLAY_FPS_BIAS = 0.4;            // < 1 front-loads the rate drop
    constexpr double MAX_FRAME_SEC = 1.0 / 60.0;     // animation cap; FIFO present holds min(60 fps, refresh)
    constexpr double IDLE_POLL_SEC = 1.0 / 8.0;      // static scene: poll for the next media change at 8 Hz
    constexpr double MARQUEE_FRAME_SEC = 1.0 / 40.0; // redraw rate while a long line scrolls
    // The playhead's on-screen speed is bar_width/duration, so the redraw rate falls geometrically
    // from PLAY_FPS_MAX (short songs) to PLAY_FPS_MIN past PLAY_FPS_SPAN.
    auto playbackRedrawSec = [&](double duration) {
        double t = std::pow(std::clamp(duration / PLAY_FPS_SPAN, 0.0, 1.0), PLAY_FPS_BIAS);
        double fps = PLAY_FPS_MAX * std::pow(PLAY_FPS_MIN / PLAY_FPS_MAX, t);
        return 1.0 / fps;
    };
    struct Cadence { bool live; double frameTarget; };
    auto cadence = [&]() -> Cadence {
        bool anim = g_renderer.animating();
        // Only a finite, playing track advances its playhead; a live stream's full bar is static
        // and idles like a paused scene.
        bool advancing = g_track.valid && g_track.playing && g_track.duration > 0.0;
        // A scrolling line holds 40 fps; an advancing track takes the faster of that and its
        // playback cadence.
        bool scrolling = g_renderer.textScrolling();
        double frameTarget = playbackRedrawSec(g_track.duration);
        if (scrolling) frameTarget = advancing ? std::min(frameTarget, MARQUEE_FRAME_SEC) : MARQUEE_FRAME_SEC;
        return {anim || advancing || scrolling, anim ? MAX_FRAME_SEC : frameTarget};
    };

    bool running = true;
    bool drewOnce = false;
    double lastDraw = 0.0;
    while (running) {
        if (!window.pumpEvents()) { running = false; break; }

        bool changed = pollMedia();
        if (changed) {
            if (g_track.valid && !g_lastArt.empty()) window.setTaskbarArt(g_lastArt);
            else window.clearTaskbarArt();
        }
        double now = steadySeconds();
        Cadence c = cadence();
        if (changed || !drewOnce || (c.live && now - lastDraw >= c.frameTarget)) {
            window.ensureVisible();  // a capture tool can't see a minimized/occluded window
            g_renderer.draw(g_track, now);
            lastDraw = now;
            drewOnce = true;
        }

        // draw() can arm the seek glide; re-sample so this iteration's wait uses the 60 Hz cadence,
        // not the slow playback interval that would elapse the glide's ease-in in a single sleep.
        c = cadence();

        int timeoutMs;
        if (c.live) {
            double remain = c.frameTarget - (steadySeconds() - lastDraw);
            // ceil: a sub-ms residual must round up to 1 ms, else the loop spins out the last
            // fraction of every frame.
            timeoutMs = remain <= 0.0 ? 0 : static_cast<int>(std::ceil(remain * 1000.0));
        } else {
            timeoutMs = static_cast<int>(IDLE_POLL_SEC * 1000.0);
        }
        window.waitEvents(timeoutMs);
    }
    window.clearTaskbarArt();
    g_poller.stop();
    g_renderer.shutdown();
    return 0;
}
