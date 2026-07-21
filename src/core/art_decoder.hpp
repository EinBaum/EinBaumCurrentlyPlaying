#pragma once
#include "core/image.hpp"
#include <condition_variable>
#include <cstdint>
#include <mutex>
#include <optional>
#include <stop_token>
#include <thread>
#include <vector>

// Decodes album art on its own thread so the render loop never blocks on a cover. Single-slot
// request and result mailboxes: a newer submit overwrites an unstarted request, so a fast
// skip-burst decodes only the final cover. GPU upload stays with the caller.
class ArtDecoder {
public:
    struct Result {
        DecodedImage img;
        std::uint64_t seq = 0;
    };

    ~ArtDecoder() { stop(); }

    // Queue bytes for decode; the returned sequence number tags the matching Result.
    std::uint64_t submit(const std::vector<std::uint8_t>& png, int target);
    // Newest finished decode, if one landed since the last take.
    [[nodiscard]] std::optional<Result> take();
    // Join the worker (started lazily by the first submit).
    void stop();

private:
    void run(const std::stop_token& st);

    struct Request {
        std::vector<std::uint8_t> png;
        int target = 0;
        std::uint64_t seq = 0;
    };
    std::mutex mtx_;
    std::condition_variable_any cv_;
    std::optional<Request> request_;
    std::optional<Result> result_;
    std::uint64_t nextSeq_ = 0;
    std::jthread thread_;
};
