#pragma once
#include "core/track.hpp"
#include <cstdint>
#include <memory>
#include <limits>

// Polls the OS media session on a background thread and hands the main thread the latest Track.
// Implemented with WinRT SMTC (win_media.cpp) on Windows and MPRIS via sd-bus (lin_media.cpp) on
// Linux; the pimpl hides each platform's threading and session-API types.
class MediaPoller {
public:
    MediaPoller();
    ~MediaPoller();
    MediaPoller(const MediaPoller&) = delete;
    MediaPoller& operator=(const MediaPoller&) = delete;

    void start();
    void stop();
    // Returns a sequence number that increments every poll (including failed polls), so the caller
    // can detect that a new read happened.
    // If since matches the current sequence, leave out untouched and avoid copying the snapshot.
    [[nodiscard]] uint64_t latest(Track& out, uint64_t since = std::numeric_limits<uint64_t>::max());

private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};
