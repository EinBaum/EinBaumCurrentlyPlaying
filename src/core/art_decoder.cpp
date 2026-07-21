#include "core/art_decoder.hpp"
#include "platform/platform.hpp"
#include <utility>

std::uint64_t ArtDecoder::submit(const std::vector<std::uint8_t>& png, int target) {
    if (!thread_.joinable())
        thread_ = std::jthread([this](std::stop_token st) { run(st); });
    std::lock_guard lk(mtx_);
    request_ = Request{png, target, ++nextSeq_};
    cv_.notify_one();
    return nextSeq_;
}

std::optional<ArtDecoder::Result> ArtDecoder::take() {
    std::lock_guard lk(mtx_);
    return std::exchange(result_, std::nullopt);
}

void ArtDecoder::stop() {
    if (!thread_.joinable()) return;
    thread_.request_stop();  // the stop_token wakes the wait, no notify needed
    thread_.join();
    thread_ = {};
}

void ArtDecoder::run(const std::stop_token& st) {
    platformInitThread();
    std::unique_lock lk(mtx_);
    while (cv_.wait(lk, st, [&] { return request_.has_value(); })) {
        Request req = std::move(*request_);
        request_.reset();
        lk.unlock();
        DecodedImage img = decodeAlbumArt(req.png, req.target);
        lk.lock();
        result_ = Result{std::move(img), req.seq};
    }
    platformShutdownThread();
}
