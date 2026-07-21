// Linux media backend: reads the active MPRIS session over the session D-Bus bus via sd-bus.
// Players expose, on /org/mpris/MediaPlayer2, the org.mpris.MediaPlayer2.Player interface:
// PlaybackStatus (s), Position (x, microseconds), and Metadata (a{sv}). The Track contract and
// poll cadence match win_media.cpp.
#include "core/media.hpp"
#include "core/track_text.hpp"
#include <systemd/sd-bus.h>
#include <atomic>
#include <chrono>
#include <condition_variable>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <mutex>
#include <string>
#include <string_view>
#include <thread>
#include <vector>

namespace {

[[nodiscard]] double steadySeconds() {
    using namespace std::chrono;
    return duration<double>(steady_clock::now().time_since_epoch()).count();
}

// The bus carries UTF-8; the core keeps std::wstring. Malformed bytes become U+FFFD so a bad tag
// never aborts the read.
[[nodiscard]] std::wstring fromUtf8(std::string_view s) {
    std::wstring out;
    out.reserve(s.size());
    size_t i = 0;
    const size_t n = s.size();
    while (i < n) {
        unsigned char c = static_cast<unsigned char>(s[i]);
        char32_t cp;
        int extra;
        if (c < 0x80)        { cp = c;        extra = 0; }
        else if (c < 0xC2)   { out.push_back(L'�'); ++i; continue; }   // continuation byte or overlong 2-byte lead
        else if (c < 0xE0)   { cp = c & 0x1F; extra = 1; }
        else if (c < 0xF0)   { cp = c & 0x0F; extra = 2; }
        else if (c < 0xF8)   { cp = c & 0x07; extra = 3; }
        else                 { out.push_back(L'�'); ++i; continue; }
        if (i + extra >= n)  { out.push_back(L'�'); break; }
        bool ok = true;
        for (int k = 1; k <= extra; ++k) {
            unsigned char cc = static_cast<unsigned char>(s[i + k]);
            if ((cc & 0xC0) != 0x80) { ok = false; break; }
            cp = (cp << 6) | (cc & 0x3F);
        }
        if (!ok) { out.push_back(L'�'); ++i; continue; }
        out.push_back(static_cast<wchar_t>(cp));
        i += extra + 1;
    }
    return out;
}

// Private/incognito windows surface a fixed placeholder instead of the page title; same strings as
// win_media.cpp.
[[nodiscard]] bool isPrivatePlaceholder(std::wstring_view title) {
    return title == L"A site is playing media" || title == L"Firefox is playing media";
}

// Percent-decode a file:// URL to a filesystem path. Non-file:// (e.g. http) returns empty.
[[nodiscard]] std::string fileUrlToPath(std::string_view url) {
    constexpr std::string_view kPrefix = "file://";
    if (url.substr(0, kPrefix.size()) != kPrefix) return {};
    std::string_view rest = url.substr(kPrefix.size());
    // Skip the optional host (empty for local files); the path starts at the first '/'.
    if (size_t slash = rest.find('/'); slash != std::string_view::npos) rest = rest.substr(slash);
    else return {};
    std::string path;
    path.reserve(rest.size());
    for (size_t i = 0; i < rest.size(); ++i) {
        if (rest[i] == '%' && i + 2 < rest.size()) {
            auto hex = [](char c) -> int {
                if (c >= '0' && c <= '9') return c - '0';
                if (c >= 'a' && c <= 'f') return c - 'a' + 10;
                if (c >= 'A' && c <= 'F') return c - 'A' + 10;
                return -1;
            };
            int hi = hex(rest[i + 1]), lo = hex(rest[i + 2]);
            if (hi >= 0 && lo >= 0) { path.push_back(static_cast<char>(hi << 4 | lo)); i += 2; continue; }
        }
        path.push_back(rest[i]);
    }
    return path;
}

// Read the local album-art file artUrl points at, capped so a bogus size cannot exhaust memory.
[[nodiscard]] std::vector<uint8_t> readArt(std::string_view artUrl) {
    std::vector<uint8_t> bytes;
    std::string path = fileUrlToPath(artUrl);
    if (path.empty()) return bytes;
    FILE* f = std::fopen(path.c_str(), "rb");
    if (!f) return bytes;
    std::fseek(f, 0, SEEK_END);
    long sz = std::ftell(f);
    std::fseek(f, 0, SEEK_SET);
    if (sz > 0 && sz < (50L << 20)) {
        bytes.resize(static_cast<size_t>(sz));
        size_t got = std::fread(bytes.data(), 1, bytes.size(), f);
        bytes.resize(got);
    }
    std::fclose(f);
    return bytes;
}

bool getStringProp(sd_bus* bus, const char* svc, const char* iface, const char* prop, std::string& out) {
    char* val = nullptr;
    sd_bus_error err = SD_BUS_ERROR_NULL;
    int r = sd_bus_get_property_string(bus, svc, "/org/mpris/MediaPlayer2", iface, prop, &err, &val);
    sd_bus_error_free(&err);
    if (r < 0 || !val) { if (val) free(val); return false; }
    out = val;
    free(val);
    return true;
}

struct Meta {
    std::string title, artist, album, albumArtist, artUrl;
    int64_t lengthUs = 0;
    bool any = false;
};

// Read a string variant; for an "as" array take the first element (the MPRIS convention for artist).
bool readVariantString(sd_bus_message* m, std::string& out) {
    const char* contents = nullptr;
    char type = 0;
    if (sd_bus_message_peek_type(m, &type, &contents) < 0 || type != 'v') return false;
    if (sd_bus_message_enter_container(m, 'v', contents) < 0) return false;
    bool ok = false;
    char vtype = 0;
    const char* vcontents = nullptr;
    if (sd_bus_message_peek_type(m, &vtype, &vcontents) >= 0) {
        if (vtype == 's') {
            const char* s = nullptr;
            if (sd_bus_message_read_basic(m, 's', &s) > 0 && s) { out = s; ok = true; }
        } else if (vtype == 'a' && vcontents && vcontents[0] == 's') {
            if (sd_bus_message_enter_container(m, 'a', "s") >= 0) {
                const char* s = nullptr;
                if (sd_bus_message_read_basic(m, 's', &s) > 0 && s) { out = s; ok = true; }
                while (sd_bus_message_read_basic(m, 's', &s) > 0) {}  // consume the rest
                sd_bus_message_exit_container(m);
            }
        } else {
            sd_bus_message_skip(m, nullptr);
        }
    }
    sd_bus_message_exit_container(m);
    return ok;
}

bool readVariantInt64(sd_bus_message* m, int64_t& out) {
    const char* contents = nullptr;
    char type = 0;
    if (sd_bus_message_peek_type(m, &type, &contents) < 0 || type != 'v') return false;
    if (sd_bus_message_enter_container(m, 'v', contents) < 0) return false;
    bool ok = false;
    char vtype = 0;
    if (sd_bus_message_peek_type(m, &vtype, nullptr) >= 0 && (vtype == 'x' || vtype == 't')) {
        int64_t v = 0;
        if (sd_bus_message_read_basic(m, vtype, &v) >= 0) { out = v; ok = true; }
    } else {
        sd_bus_message_skip(m, nullptr);
    }
    sd_bus_message_exit_container(m);
    return ok;
}

bool readMetadata(sd_bus* bus, const char* svc, Meta& meta) {
    sd_bus_error err = SD_BUS_ERROR_NULL;
    sd_bus_message* reply = nullptr;
    int r = sd_bus_get_property(bus, svc, "/org/mpris/MediaPlayer2",
                                "org.mpris.MediaPlayer2.Player", "Metadata", &err, &reply, "a{sv}");
    if (r < 0 || !reply) { sd_bus_error_free(&err); if (reply) sd_bus_message_unref(reply); return false; }

    bool ok = false;
    if (sd_bus_message_enter_container(reply, 'a', "{sv}") >= 0) {
        while (sd_bus_message_enter_container(reply, 'e', "sv") > 0) {
            const char* key = nullptr;
            if (sd_bus_message_read_basic(reply, 's', &key) >= 0 && key) {
                std::string_view k = key;
                std::string sval;
                int64_t ival = 0;
                if (k == "xesam:title"       && readVariantString(reply, sval)) { meta.title = sval; meta.any = true; }
                else if (k == "xesam:artist" && readVariantString(reply, sval)) { meta.artist = sval; meta.any = true; }
                else if (k == "xesam:album"  && readVariantString(reply, sval)) { meta.album = sval; meta.any = true; }
                else if (k == "xesam:albumArtist" && readVariantString(reply, sval)) { meta.albumArtist = sval; meta.any = true; }
                else if (k == "mpris:artUrl" && readVariantString(reply, sval)) { meta.artUrl = sval; }
                else if (k == "mpris:length" && readVariantInt64(reply, ival))  { meta.lengthUs = ival; }
                else sd_bus_message_skip(reply, "v");  // unhandled key: skip its variant value
            }
            sd_bus_message_exit_container(reply);
        }
        sd_bus_message_exit_container(reply);
        ok = true;
    }
    sd_bus_message_unref(reply);
    sd_bus_error_free(&err);
    return ok;
}

bool readPositionUs(sd_bus* bus, const char* svc, int64_t& posUs) {
    sd_bus_error err = SD_BUS_ERROR_NULL;
    int r = sd_bus_get_property_trivial(bus, svc, "/org/mpris/MediaPlayer2",
                                        "org.mpris.MediaPlayer2.Player", "Position", &err, 'x', &posUs);
    sd_bus_error_free(&err);
    return r >= 0;
}

// playerctld is an aggregator that duplicates whichever player is active, so it is kept separate
// and used only when no concrete player is present.
void listPlayers(sd_bus* bus, std::vector<std::string>& players, std::string& playerctld) {
    char** names = nullptr;
    if (sd_bus_list_names(bus, &names, nullptr) < 0 || !names) return;
    for (char** p = names; *p; ++p) {
        std::string_view n = *p;
        if (n.substr(0, 23) != "org.mpris.MediaPlayer2.") continue;
        if (n == "org.mpris.MediaPlayer2.playerctld") playerctld = *p;
        else players.emplace_back(*p);
    }
    for (char** p = names; *p; ++p) free(*p);
    free(names);
}

// Pick a player (preferring one that is Playing) and fill out; true when a valid track was read.
bool readOnce(sd_bus* bus, Track& out) {
    std::vector<std::string> players;
    std::string playerctld;
    listPlayers(bus, players, playerctld);
    if (players.empty()) {
        if (playerctld.empty()) return false;
        players.push_back(playerctld);  // nothing concrete; fall back to the aggregator
    }

    // Choose: first a Playing player, else the first that yields any metadata.
    std::string chosen;
    std::string chosenStatus;
    for (const std::string& svc : players) {
        std::string status;
        getStringProp(bus, svc.c_str(), "org.mpris.MediaPlayer2.Player", "PlaybackStatus", status);
        if (status == "Playing") { chosen = svc; chosenStatus = status; break; }
        if (chosen.empty())      { chosen = svc; chosenStatus = status; }
    }
    if (chosen.empty()) return false;

    Meta meta;
    if (!readMetadata(bus, chosen.c_str(), meta) || !meta.any) return false;

    out = Track{};
    out.playing = (chosenStatus == "Playing");
    out.duration = meta.lengthUs > 0 ? static_cast<double>(meta.lengthUs) / 1e6 : 0.0;
    out.live = out.duration <= 0.0;  // no mpris:length -> unbounded stream

    out.title = fromUtf8(meta.title);
    out.artist = fromUtf8(meta.artist);
    if (isPrivatePlaceholder(out.title)) return false;

    // With no artist/album the title is a bare filename, so reparse it; else clean the tagged pair.
    const bool filenameOnly = meta.artist.empty() && meta.album.empty() && meta.albumArtist.empty();
    TrackText cleaned = filenameOnly ? titleFromFilename(std::move(out.title))
                                     : cleanTrackText(std::move(out.title), std::move(out.artist));
    out.title = std::move(cleaned.title);
    out.artist = std::move(cleaned.artist);

    if (!meta.artUrl.empty()) out.artPng = readArt(meta.artUrl);

    // MPRIS Position is read live (microseconds), so it is current as of this call; anchor posBase
    // to the same instant and the renderer extrapolates position + (now - posBase) while playing.
    int64_t posUs = 0;
    double pos = readPositionUs(bus, chosen.c_str(), posUs) ? static_cast<double>(posUs) / 1e6 : 0.0;
    out.posBase = steadySeconds();
    if (out.duration > 0 && pos > out.duration) pos = out.duration;
    if (pos < 0) pos = 0;
    out.position = pos;

    out.valid = !(out.title.empty() && out.artist.empty());
    return out.valid;
}

}  // namespace

struct MediaPoller::Impl {
    std::thread thread_;
    std::atomic<bool> stop_{false};
    std::mutex mtx_;
    std::condition_variable cv_;
    Track track_;
    uint64_t seq_ = 0;

    void run() {
        sd_bus* bus = nullptr;
        // No session bus (e.g. a bare TTY): seq_ still advances on cadence with an empty Track.
        sd_bus_open_user(&bus);
        while (!stop_.load()) {
            Track t;
            bool ok = bus && readOnce(bus, t);
            {
                std::lock_guard<std::mutex> lk(mtx_);
                track_ = ok ? std::move(t) : Track{};
                ++seq_;
            }
            std::unique_lock<std::mutex> lk(mtx_);
            cv_.wait_for(lk, std::chrono::milliseconds(500), [this] { return stop_.load(); });
        }
        if (bus) sd_bus_flush_close_unref(bus);
    }
};

MediaPoller::MediaPoller() : impl_(std::make_unique<Impl>()) {}
MediaPoller::~MediaPoller() { stop(); }

void MediaPoller::start() {
    impl_->stop_.store(false);
    impl_->thread_ = std::thread([this] { impl_->run(); });
}

void MediaPoller::stop() {
    impl_->stop_.store(true);
    impl_->cv_.notify_all();
    if (impl_->thread_.joinable()) impl_->thread_.join();
}

uint64_t MediaPoller::latest(Track& out) {
    std::lock_guard<std::mutex> lk(impl_->mtx_);
    out = impl_->track_;
    return impl_->seq_;
}
