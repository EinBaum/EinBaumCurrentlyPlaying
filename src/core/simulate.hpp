#pragma once
#include "core/track.hpp"
#include <optional>

// --simulate playlist: 10 synthetic songs with generated covers, switched one per second.
class SongSwitchSim {
public:
    static constexpr int kSongCount = 10;
    static constexpr double kSwitchSec = 1.0;

    // A new track when the next song is due; nullopt otherwise. After the last song has been
    // shown for kSwitchSec, further polls report done().
    [[nodiscard]] std::optional<Track> poll(double nowSteady);
    [[nodiscard]] bool done() const { return done_; }
    // Milliseconds until the next switch or exit hold. 0 if a poll is due now.
    [[nodiscard]] int waitMs(double nowSteady) const;

private:
    int shown_ = 0;
    double nextAt_ = 0.0;
    bool done_ = false;
};
