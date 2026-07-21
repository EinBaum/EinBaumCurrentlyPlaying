#pragma once
#include <cstdint>
#include <string>
#include <vector>

// Snapshot of the current track. position/posBase let the renderer extrapolate the playhead
// between polls: live = playing ? position + (now - posBase) : position.
struct Track {
    bool valid = false;
    std::wstring title;
    std::wstring artist;
    bool playing = false;
    bool live = false;       // unbounded stream: the bar renders full with no playhead tip
    double duration = 0.0;   // seconds; > 0 means draw the progress bar
    double position = 0.0;   // seconds, snapshot at posBase
    double posBase = 0.0;    // steady-clock seconds when position was read
    std::vector<std::uint8_t> artPng;  // raw thumbnail bytes (PNG/JPEG), empty if none

    [[nodiscard]] std::wstring identity() const { return title + L"\x1f" + artist; }
};
