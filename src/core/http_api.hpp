#pragma once
#include "core/media.hpp"
#include <memory>

// Loopback-only HTTP server on a fixed port (see kApiPort in http_api.cpp). Binds 127.0.0.1 and
// ::1 so nothing off-box can reach it. Serves the current track as JSON.
class LocalHttpApi {
public:
    LocalHttpApi();
    ~LocalHttpApi();
    LocalHttpApi(const LocalHttpApi&) = delete;
    LocalHttpApi& operator=(const LocalHttpApi&) = delete;

    void start(MediaPoller& poller);
    void stop();

private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};
