// Localhost JSON API: one accept loop, GET / or GET /now, no keep-alive. JSON via vendored cJSON.
// Binds 127.0.0.1 and ::1 on kApiPort so both 127.0.0.1 and localhost work; not reachable off-box.
#include "core/http_api.hpp"
#include "cJSON.h"
#include <atomic>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <format>
#include <string>
#include <string_view>
#include <thread>

#ifdef _WIN32
#ifndef WIN32_LEAN_AND_MEAN
#define WIN32_LEAN_AND_MEAN
#endif
#include <winsock2.h>
#include <ws2tcpip.h>
#include <windows.h>
#else
#include <arpa/inet.h>
#include <cerrno>
#include <netinet/in.h>
#include <sys/select.h>
#include <sys/socket.h>
#include <unistd.h>
#endif

namespace {

constexpr int kApiPort = 18881;

#ifdef _WIN32
using Fd = SOCKET;
constexpr Fd kInvalid = INVALID_SOCKET;
void closeFd(Fd fd) { if (fd != kInvalid) closesocket(fd); }
[[nodiscard]] bool isValid(Fd fd) { return fd != kInvalid; }
constexpr int kSendFlags = 0;
#else
using Fd = int;
constexpr Fd kInvalid = -1;
void closeFd(Fd fd) { if (fd >= 0) ::close(fd); }
[[nodiscard]] bool isValid(Fd fd) { return fd >= 0; }
#ifdef MSG_NOSIGNAL
constexpr int kSendFlags = MSG_NOSIGNAL;
#else
constexpr int kSendFlags = 0;
#endif
#endif

void apiLog(std::string_view msg) {
    std::fprintf(stderr, "EinBaumCurrentlyPlaying: %.*s\n",
                 static_cast<int>(msg.size()), msg.data());
    std::fflush(stderr);
#ifdef _WIN32
    OutputDebugStringA("EinBaumCurrentlyPlaying: ");
    const std::string copy(msg);
    OutputDebugStringA(copy.c_str());
    OutputDebugStringA("\n");
#endif
}

[[nodiscard]] std::string toUtf8(std::wstring_view s) {
    std::string out;
    out.reserve(s.size() * 3);
    auto append = [&](char32_t cp) {
        if (cp > 0x10FFFF || (cp >= 0xD800 && cp <= 0xDFFF)) cp = 0xFFFD;
        if (cp < 0x80) {
            out.push_back(static_cast<char>(cp));
        } else if (cp < 0x800) {
            out.push_back(static_cast<char>(0xC0 | (cp >> 6)));
            out.push_back(static_cast<char>(0x80 | (cp & 0x3F)));
        } else if (cp < 0x10000) {
            out.push_back(static_cast<char>(0xE0 | (cp >> 12)));
            out.push_back(static_cast<char>(0x80 | ((cp >> 6) & 0x3F)));
            out.push_back(static_cast<char>(0x80 | (cp & 0x3F)));
        } else {
            out.push_back(static_cast<char>(0xF0 | (cp >> 18)));
            out.push_back(static_cast<char>(0x80 | ((cp >> 12) & 0x3F)));
            out.push_back(static_cast<char>(0x80 | ((cp >> 6) & 0x3F)));
            out.push_back(static_cast<char>(0x80 | (cp & 0x3F)));
        }
    };
    if constexpr (sizeof(wchar_t) == 2) {
        for (size_t i = 0; i < s.size(); ++i) {
            const auto c = static_cast<uint16_t>(s[i]);
            if (c >= 0xD800 && c <= 0xDBFF && i + 1 < s.size()) {
                const auto d = static_cast<uint16_t>(s[i + 1]);
                if (d >= 0xDC00 && d <= 0xDFFF) {
                    append(0x10000 + ((static_cast<char32_t>(c - 0xD800) << 10) | (d - 0xDC00)));
                    ++i;
                    continue;
                }
            }
            append(c);
        }
    } else {
        for (wchar_t c : s) append(static_cast<char32_t>(static_cast<uint32_t>(c)));
    }
    return out;
}

[[nodiscard]] std::string jsonDump(cJSON* obj) {
    char* printed = obj ? cJSON_PrintUnformatted(obj) : nullptr;
    cJSON_Delete(obj);
    if (!printed) return "{}";
    std::string out = printed;
    cJSON_free(printed);
    return out;
}

[[nodiscard]] std::string errorJson(const char* msg) {
    cJSON* o = cJSON_CreateObject();
    cJSON_AddStringToObject(o, "error", msg);
    return jsonDump(o);
}

[[nodiscard]] double finiteOrZero(double x) {
    return std::isfinite(x) ? x : 0.0;
}

[[nodiscard]] double livePosition(const Track& t, double now) {
    if (!t.valid) return 0.0;
    double pos = t.position;
    if (t.playing) pos += now - t.posBase;
    if (pos < 0.0) pos = 0.0;
    if (t.duration > 0.0 && pos > t.duration) pos = t.duration;
    return finiteOrZero(pos);
}

[[nodiscard]] std::string trackJson(const Track& t) {
    const double now = std::chrono::duration<double>(
        std::chrono::steady_clock::now().time_since_epoch()).count();
    const std::string title = toUtf8(t.title);
    const std::string artist = toUtf8(t.artist);
    cJSON* o = cJSON_CreateObject();
    cJSON_AddBoolToObject(o, "valid", t.valid);
    cJSON_AddStringToObject(o, "title", title.c_str());
    cJSON_AddStringToObject(o, "artist", artist.c_str());
    cJSON_AddBoolToObject(o, "playing", t.valid && t.playing);
    cJSON_AddBoolToObject(o, "live", t.valid && t.live);
    cJSON_AddNumberToObject(o, "duration", finiteOrZero(t.valid ? t.duration : 0.0));
    cJSON_AddNumberToObject(o, "position", livePosition(t, now));
    return jsonDump(o);
}

void sendAll(Fd fd, std::string_view s) {
    while (!s.empty()) {
#ifdef _WIN32
        const int n = send(fd, s.data(), static_cast<int>(s.size()), kSendFlags);
#else
        const int n = static_cast<int>(send(fd, s.data(), s.size(), kSendFlags));
#endif
        if (n <= 0) return;
        s.remove_prefix(static_cast<size_t>(n));
    }
}

void reply(Fd fd, int code, std::string_view reason, std::string_view body) {
    const std::string hdr = std::format(
        "HTTP/1.1 {} {}\r\n"
        "Content-Type: application/json; charset=utf-8\r\n"
        "Content-Length: {}\r\n"
        "Connection: close\r\n"
        "Cache-Control: no-store\r\n"
        "Access-Control-Allow-Origin: *\r\n"
        "\r\n",
        code, reason, body.size());
    sendAll(fd, hdr);
    sendAll(fd, body);
}

[[nodiscard]] bool peerIsLoopback(Fd fd) {
    sockaddr_storage ss{};
    socklen_t len = sizeof(ss);
    if (getpeername(fd, reinterpret_cast<sockaddr*>(&ss), &len) != 0) return false;
    if (ss.ss_family == AF_INET) {
        const auto* in = reinterpret_cast<sockaddr_in*>(&ss);
        return in->sin_addr.s_addr == htonl(INADDR_LOOPBACK);
    }
    if (ss.ss_family == AF_INET6) {
        const auto* in6 = reinterpret_cast<sockaddr_in6*>(&ss);
        return IN6_IS_ADDR_LOOPBACK(&in6->sin6_addr) != 0;
    }
    return false;
}

void setRecvTimeout(Fd fd) {
#ifdef _WIN32
    DWORD ms = 2000;
    setsockopt(fd, SOL_SOCKET, SO_RCVTIMEO, reinterpret_cast<const char*>(&ms), sizeof(ms));
#else
    timeval tv{2, 0};
    setsockopt(fd, SOL_SOCKET, SO_RCVTIMEO, &tv, sizeof(tv));
#endif
}

void handleClient(Fd fd, MediaPoller& poller) {
    if (!peerIsLoopback(fd)) return;
    setRecvTimeout(fd);

    char buf[8192];
    size_t used = 0;
    bool complete = false;
    while (used < sizeof(buf) - 1) {
#ifdef _WIN32
        const int n = recv(fd, buf + used, static_cast<int>(sizeof(buf) - 1 - used), 0);
#else
        const int n = static_cast<int>(recv(fd, buf + used, sizeof(buf) - 1 - used, 0));
#endif
        if (n <= 0) return;
        used += static_cast<size_t>(n);
        buf[used] = '\0';
        if (std::strstr(buf, "\r\n\r\n") || std::strstr(buf, "\n\n")) {
            complete = true;
            break;
        }
    }
    if (!complete) return;

    std::string_view req(buf, used);
    const size_t lineEnd = req.find('\n');
    if (lineEnd == std::string_view::npos) {
        reply(fd, 400, "Bad Request", errorJson("bad request"));
        return;
    }
    std::string_view line = req.substr(0, lineEnd);
    if (!line.empty() && line.back() == '\r') line.remove_suffix(1);

    const size_t sp1 = line.find(' ');
    if (sp1 == std::string_view::npos) {
        reply(fd, 400, "Bad Request", errorJson("bad request"));
        return;
    }
    const std::string_view method = line.substr(0, sp1);
    const size_t sp2 = line.find(' ', sp1 + 1);
    std::string_view target = sp2 == std::string_view::npos
        ? line.substr(sp1 + 1)
        : line.substr(sp1 + 1, sp2 - sp1 - 1);
    if (const size_t q = target.find('?'); q != std::string_view::npos) target = target.substr(0, q);
    while (target.size() > 1 && target.back() == '/') target.remove_suffix(1);

    if (method != "GET") {
        reply(fd, 405, "Method Not Allowed", errorJson("method not allowed"));
        return;
    }
    if (target != "/" && target != "/now") {
        reply(fd, 404, "Not Found", errorJson("not found"));
        return;
    }

    Track t;
    (void)poller.latest(t);
    reply(fd, 200, "OK", trackJson(t));
}

[[nodiscard]] Fd bindLoopback4() {
#ifdef _WIN32
    const Fd fd = socket(AF_INET, SOCK_STREAM, IPPROTO_TCP);
#else
    const Fd fd = socket(AF_INET, SOCK_STREAM | SOCK_CLOEXEC, IPPROTO_TCP);
#endif
    if (!isValid(fd)) return kInvalid;
    sockaddr_in addr{};
    addr.sin_family = AF_INET;
    addr.sin_addr.s_addr = htonl(INADDR_LOOPBACK);
    addr.sin_port = htons(static_cast<uint16_t>(kApiPort));
    if (bind(fd, reinterpret_cast<sockaddr*>(&addr), sizeof(addr)) != 0 || listen(fd, 16) != 0) {
        closeFd(fd);
        return kInvalid;
    }
    return fd;
}

[[nodiscard]] Fd bindLoopback6() {
#ifdef _WIN32
    const Fd fd = socket(AF_INET6, SOCK_STREAM, IPPROTO_TCP);
#else
    const Fd fd = socket(AF_INET6, SOCK_STREAM | SOCK_CLOEXEC, IPPROTO_TCP);
#endif
    if (!isValid(fd)) return kInvalid;
    int v6only = 1;
    setsockopt(fd, IPPROTO_IPV6, IPV6_V6ONLY, reinterpret_cast<const char*>(&v6only), sizeof(v6only));
    sockaddr_in6 addr{};
    addr.sin6_family = AF_INET6;
    addr.sin6_addr = in6addr_loopback;
    addr.sin6_port = htons(static_cast<uint16_t>(kApiPort));
    if (bind(fd, reinterpret_cast<sockaddr*>(&addr), sizeof(addr)) != 0 || listen(fd, 16) != 0) {
        closeFd(fd);
        return kInvalid;
    }
    return fd;
}

}  // namespace

struct LocalHttpApi::Impl {
    MediaPoller* poller = nullptr;
    std::thread thread;
    std::atomic<bool> stop{false};
    Fd v4 = kInvalid;
    Fd v6 = kInvalid;
#ifdef _WIN32
    bool wsa = false;
#endif

    void run() {
        while (!stop.load()) {
            fd_set rfds;
            FD_ZERO(&rfds);
            int nfds = 0;
            auto add = [&](Fd fd) {
                if (!isValid(fd)) return;
                FD_SET(fd, &rfds);
#ifdef _WIN32
                (void)nfds;
#else
                if (fd + 1 > nfds) nfds = fd + 1;
#endif
            };
            add(v4);
            add(v6);
            timeval tv{0, 200000};  // 200 ms: stop_ is checked without closing fds from the other thread
#ifdef _WIN32
            const int n = select(0, &rfds, nullptr, nullptr, &tv);
#else
            const int n = select(nfds, &rfds, nullptr, nullptr, &tv);
            if (n < 0 && errno == EINTR) continue;
#endif
            if (n <= 0) continue;

            auto acceptOne = [&](Fd listenFd) {
                if (!isValid(listenFd) || !FD_ISSET(listenFd, &rfds)) return;
                sockaddr_storage ss{};
                socklen_t len = sizeof(ss);
                const Fd c = accept(listenFd, reinterpret_cast<sockaddr*>(&ss), &len);
                if (!isValid(c)) return;
                handleClient(c, *poller);
                closeFd(c);
            };
            acceptOne(v4);
            acceptOne(v6);
        }
    }
};

LocalHttpApi::LocalHttpApi() : impl_(std::make_unique<Impl>()) {}
LocalHttpApi::~LocalHttpApi() { stop(); }

void LocalHttpApi::start(MediaPoller& poller) {
    if (impl_->thread.joinable()) return;

#ifdef _WIN32
    WSADATA wsa{};
    if (WSAStartup(MAKEWORD(2, 2), &wsa) != 0) {
        apiLog("song API: WSAStartup failed");
        return;
    }
    impl_->wsa = true;
#endif

    impl_->v4 = bindLoopback4();
    if (!isValid(impl_->v4)) {
        apiLog(std::format("song API: could not bind 127.0.0.1:{}", kApiPort));
        stop();
        return;
    }
    impl_->v6 = bindLoopback6();  // optional: localhost may resolve to ::1

    impl_->poller = &poller;
    impl_->stop.store(false);
    impl_->thread = std::thread([this] { impl_->run(); });
}

void LocalHttpApi::stop() {
    impl_->stop.store(true);
    if (impl_->thread.joinable()) impl_->thread.join();
    closeFd(impl_->v4);
    closeFd(impl_->v6);
    impl_->v4 = kInvalid;
    impl_->v6 = kInvalid;
    impl_->poller = nullptr;
#ifdef _WIN32
    if (impl_->wsa) {
        WSACleanup();
        impl_->wsa = false;
    }
#endif
}
