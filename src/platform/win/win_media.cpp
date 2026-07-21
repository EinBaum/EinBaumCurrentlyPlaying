// Windows media backend: reads the system media session (SMTC) on a background thread via the
// hand-declared Windows.Media.Control ABI. Uses Win32 threading (CreateThread + CRITICAL_SECTION)
// so the statically-linked exe carries no libwinpthread dependency.
#include "core/media.hpp"
#include "core/fatal.hpp"
#include "core/track_text.hpp"
#include "platform/win/media_abi.hpp"
#include <objidl.h>
#include <atomic>
#include <chrono>
#include <cstdint>
#include <cwchar>
#include <cwctype>
#include <format>
#include <string_view>

struct MediaPoller::Impl {
    void run();
    static DWORD WINAPI threadProc(LPVOID self);
    HANDLE thread_ = nullptr;
    std::atomic<bool> stop_{false};
    CRITICAL_SECTION cs_;
    Track track_;
    uint64_t seq_ = 0;
};

namespace {

[[nodiscard]] double steadySeconds() {
    using namespace std::chrono;
    return duration<double>(steady_clock::now().time_since_epoch()).count();
}

// Wall-clock seconds in the FILETIME epoch (100-ns ticks since 1601 UTC), to compare against the
// timeline's LastUpdatedTime (a DateTime in the same epoch).
[[nodiscard]] double wallSeconds() {
    FILETIME ft;
    GetSystemTimePreciseAsFileTime(&ft);
    ULARGE_INTEGER u;
    u.LowPart = ft.dwLowDateTime;
    u.HighPart = ft.dwHighDateTime;
    return static_cast<double>(u.QuadPart) / 1e7;
}

template <class T> void rel(T* p) { if (p) p->v->Release(p); }

// Busy-polls get_Status instead of registering a completion handler; valid because the MTA runs
// the async op on a pool thread, so this thread can block without stalling completion.
[[nodiscard]] void* await_op(IAsyncOp* op) {
    if (!op) return nullptr;
    IAsyncInfo* info = nullptr;
    if (FAILED(op->v->QueryInterface(op, &IID_IAsyncInfo, reinterpret_cast<void**>(&info))) || !info) return nullptr;
    for (int i = 0; i < 2000; ++i) {
        int status = 0;
        info->v->get_Status(info, &status);
        if (status != 0) break;  // 0 = Started
        Sleep(5);
    }
    rel(info);
    void* result = nullptr;
    if (FAILED(op->v->GetResults(op, &result))) return nullptr;
    return result;
}

[[nodiscard]] std::wstring hstr(HSTRING h) {
    UINT32 len = 0;
    const wchar_t* buf = WindowsGetStringRawBuffer(h, &len);
    return buf ? std::wstring(buf, len) : std::wstring();
}

// A private/incognito window withholds the page title from SMTC and surfaces a fixed placeholder
// with no artist; these are the en-US strings for Chrome and Firefox.
[[nodiscard]] bool isPrivatePlaceholder(std::wstring_view title) {
    return title == L"A site is playing media" || title == L"Firefox is playing media";
}

using PFN_CreateStreamOverRAS = HRESULT (WINAPI*)(IUnknown*, REFIID, void**);

[[nodiscard]] std::vector<uint8_t> readThumbnail(IStreamRef* ref) {
    std::vector<uint8_t> bytes;
    if (!ref) return bytes;
    IAsyncOp* op = nullptr;
    if (FAILED(ref->v->OpenReadAsync(ref, &op)) || !op) return bytes;
    IUnknown* ras = static_cast<IUnknown*>(await_op(op));  // IRandomAccessStreamWithContentType
    rel(op);
    if (!ras) return bytes;

    static PFN_CreateStreamOverRAS create = nullptr;
    if (!create) {
        HMODULE sh = LoadLibraryW(L"shcore.dll");
        if (sh) create = reinterpret_cast<PFN_CreateStreamOverRAS>(GetProcAddress(sh, "CreateStreamOverRandomAccessStream"));
    }
    IStream* s = nullptr;
    if (create && SUCCEEDED(create(ras, IID_IStream, reinterpret_cast<void**>(&s))) && s) {
        STATSTG st{};
        if (SUCCEEDED(s->Stat(&st, STATFLAG_NONAME))) {
            ULONGLONG size = st.cbSize.QuadPart;
            if (size > 0 && size < (50ull << 20)) {
                LARGE_INTEGER zero{};
                s->Seek(zero, STREAM_SEEK_SET, nullptr);
                bytes.resize(static_cast<size_t>(size));
                ULONG got = 0;
                s->Read(bytes.data(), static_cast<ULONG>(bytes.size()), &got);
                bytes.resize(got);
            }
        }
        s->Release();
    }
    ras->Release();
    return bytes;
}

bool readOnce(IMgrStatics* statics, Track& out) {
    IAsyncOp* op = nullptr;
    if (FAILED(statics->v->RequestAsync(statics, &op))) return false;
    IMgr* mgr = static_cast<IMgr*>(await_op(op));
    rel(op);
    if (!mgr) return false;

    IVVSession* sessions = nullptr;
    bool found = false;
    if (SUCCEEDED(mgr->v->GetSessions(mgr, &sessions)) && sessions) {
        unsigned n = 0;
        sessions->v->get_Size(sessions, &n);
        for (unsigned i = 0; i < n && !found; ++i) {
            ISession* s = nullptr;
            if (FAILED(sessions->v->GetAt(sessions, i, &s)) || !s) continue;

            ITimeline* tl = nullptr;
            double duration = 0, position = 0, lastUpdated = 0;
            if (SUCCEEDED(s->v->GetTimelineProperties(s, &tl)) && tl) {
                TimeSpan end{}, pos{};
                DateTime upd{};
                tl->v->get_EndTime(tl, &end);
                tl->v->get_Position(tl, &pos);
                tl->v->get_LastUpdatedTime(tl, &upd);
                duration = end.Duration / 1e7;
                position = pos.Duration / 1e7;
                lastUpdated = upd.UniversalTime / 1e7;  // wall-clock seconds (same epoch as wallSeconds)
                rel(tl);
            }

            out = Track{};
            out.duration = duration;
            out.live = duration <= 0.0;  // live streams report a non-positive EndTime

            IMediaProps* mp = nullptr;
            IAsyncOp* mpop = nullptr;
            s->v->TryGetMediaPropertiesAsync(s, &mpop);
            mp = static_cast<IMediaProps*>(await_op(mpop));
            rel(mpop);
            if (mp) {
                HSTRING t = nullptr, a = nullptr, sub = nullptr, alb = nullptr, albArt = nullptr;
                mp->v->get_Title(mp, &t);
                mp->v->get_Artist(mp, &a);
                mp->v->get_Subtitle(mp, &sub);
                mp->v->get_AlbumTitle(mp, &alb);
                mp->v->get_AlbumArtist(mp, &albArt);
                out.title = hstr(t);
                out.artist = hstr(a);
                // Reparse Title as a filename only when no other metadata is set, so tagged sources are untouched.
                const bool filenameOnly = out.artist.empty() && hstr(sub).empty() &&
                                          hstr(alb).empty() && hstr(albArt).empty();
                WindowsDeleteString(t);
                WindowsDeleteString(a);
                WindowsDeleteString(sub);
                WindowsDeleteString(alb);
                WindowsDeleteString(albArt);
                if (isPrivatePlaceholder(out.title)) { rel(mp); rel(s); continue; }
                TrackText cleaned = filenameOnly
                    ? titleFromFilename(std::move(out.title))
                    : cleanTrackText(std::move(out.title), std::move(out.artist));
                out.title = std::move(cleaned.title);
                out.artist = std::move(cleaned.artist);
                IStreamRef* thumb = nullptr;
                if (SUCCEEDED(mp->v->get_Thumbnail(mp, &thumb)) && thumb) {
                    out.artPng = readThumbnail(thumb);
                    rel(thumb);
                }
                rel(mp);
            }
            IPlayback* pb = nullptr;
            if (SUCCEEDED(s->v->GetPlaybackInfo(s, &pb)) && pb) {
                int st = -1;
                pb->v->get_PlaybackStatus(pb, &st);
                out.playing = (st == 4);  // 4 = Playing
                rel(pb);
            }

            // SMTC reports Position as a snapshot taken at LastUpdatedTime, which the source app
            // refreshes only intermittently; while playing, the true current position is
            // Position + (wallSeconds() - LastUpdatedTime). posBase (steady clock) is read
            // alongside the wall-clock reference so the renderer's later extrapolation stays
            // continuous.
            double pos = position;
            out.posBase = steadySeconds();
            if (out.playing && lastUpdated > 0) {
                double elapsed = wallSeconds() - lastUpdated;
                if (elapsed > 0) pos += elapsed;
            }
            if (out.duration > 0 && pos > out.duration) pos = out.duration;
            if (pos < 0) pos = 0;
            out.position = pos;

            out.valid = !(out.title.empty() && out.artist.empty());
            found = out.valid;
            rel(s);
        }
        rel(sessions);
    }
    rel(mgr);
    return found;
}

}  // namespace

void MediaPoller::Impl::run() {
    const HRESULT hrInit = RoInitialize(RO_INIT_MULTITHREADED);
    if (FAILED(hrInit) && hrInit != RPC_E_CHANGED_MODE)
        fatal(std::format("RoInitialize failed (0x{:08X}).", static_cast<uint32_t>(hrInit)));
    IMgrStatics* statics = nullptr;
    const wchar_t* cls = L"Windows.Media.Control.GlobalSystemMediaTransportControlsSessionManager";
    HSTRING hcls = nullptr;
    WindowsCreateString(cls, static_cast<UINT32>(std::wcslen(cls)), &hcls);
    const HRESULT hrFac = RoGetActivationFactory(hcls, IID_IMgrStatics, reinterpret_cast<void**>(&statics));
    if (FAILED(hrFac) || !statics)
        fatal(std::format("Cannot reach the Windows media session API (RoGetActivationFactory 0x{:08X}); "
                          "needs Windows 10 1809 or newer.", static_cast<uint32_t>(hrFac)));

    while (!stop_.load()) {
        Track t;
        bool ok = readOnce(statics, t);
        EnterCriticalSection(&cs_);
        track_ = ok ? t : Track{};
        seq_++;
        LeaveCriticalSection(&cs_);
        for (int i = 0; i < 5 && !stop_.load(); ++i) Sleep(100);   // 500 ms total, chunked so stop_ is checked every 100 ms
    }

    rel(statics);
    WindowsDeleteString(hcls);
    RoUninitialize();
}

DWORD WINAPI MediaPoller::Impl::threadProc(LPVOID self) {
    static_cast<MediaPoller::Impl*>(self)->run();
    return 0;
}

MediaPoller::MediaPoller() : impl_(std::make_unique<Impl>()) {}
MediaPoller::~MediaPoller() { stop(); }

void MediaPoller::start() {
    InitializeCriticalSection(&impl_->cs_);
    impl_->thread_ = CreateThread(nullptr, 0, &MediaPoller::Impl::threadProc, impl_.get(), 0, nullptr);
}

void MediaPoller::stop() {
    impl_->stop_.store(true);
    if (impl_->thread_) {
        WaitForSingleObject(impl_->thread_, INFINITE);
        CloseHandle(impl_->thread_);
        impl_->thread_ = nullptr;
        DeleteCriticalSection(&impl_->cs_);
    }
}

uint64_t MediaPoller::latest(Track& out) {
    EnterCriticalSection(&impl_->cs_);
    out = impl_->track_;
    uint64_t s = impl_->seq_;
    LeaveCriticalSection(&impl_->cs_);
    return s;
}
