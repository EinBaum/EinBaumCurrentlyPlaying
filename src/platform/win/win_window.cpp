// Windows window backend: a borderless WS_POPUP overlay at the screen's bottom-right, its Vulkan
// surface via VK_KHR_win32_surface, and a taskbar icon built from the album art. Also provides the
// Win32 per-thread decode hooks (COM init for WIC).
#include "platform/platform.hpp"
#include "platform/win/win_fatal.hpp"
#include "core/image.hpp"
#include "core/theme.hpp"
#include <windows.h>
#define VK_USE_PLATFORM_WIN32_KHR
#include <vulkan/vulkan_win32.h>
#include <cstdint>
#include <vector>

namespace {

constexpr const wchar_t* WINDOW_TITLE = L"EinBaumCurrentlyPlaying";
constexpr const wchar_t* WINDOW_CLASS = L"EinBaumCurrentlyPlayingWindow";

LRESULT CALLBACK wndProc(HWND hwnd, UINT msg, WPARAM wp, LPARAM lp) {
    switch (msg) {
    case WM_SYSCOMMAND:
        // Pinned overlay: block minimize (OBS can't capture a minimized window) and move.
        if ((wp & 0xFFF0) == SC_MINIMIZE || (wp & 0xFFF0) == SC_MOVE) return 0;
        break;
    case WM_PAINT:
        ValidateRect(hwnd, nullptr);  // the render loop drives drawing
        return 0;
    case WM_DESTROY:
        PostQuitMessage(0);
        return 0;
    }
    return DefWindowProcW(hwnd, msg, wp, lp);
}

void enableDpiAwareness() {
    HMODULE u = GetModuleHandleW(L"user32.dll");
    using PFN = BOOL (WINAPI*)(HANDLE);
    auto p = reinterpret_cast<PFN>(GetProcAddress(u, "SetProcessDpiAwarenessContext"));
    if (p && p(reinterpret_cast<HANDLE>(static_cast<INT_PTR>(-4)))) return;  // PER_MONITOR_AWARE_V2
    SetProcessDPIAware();
}

// Build an HICON from album art at the given size: decode -> DIB (BGRA) -> CreateIconIndirect.
[[nodiscard]] HICON iconFromAlbumArt(const std::vector<std::uint8_t>& png, int size) {
    DecodedImage img = decodeAlbumArt(png, size);
    if (img.rgba.empty()) return nullptr;

    BITMAPINFO bmi{};
    bmi.bmiHeader.biSize = sizeof(BITMAPINFOHEADER);
    bmi.bmiHeader.biWidth = size;
    bmi.bmiHeader.biHeight = -size;   // top-down
    bmi.bmiHeader.biPlanes = 1;
    bmi.bmiHeader.biBitCount = 32;
    bmi.bmiHeader.biCompression = BI_RGB;
    void* bits = nullptr;
    HBITMAP color = CreateDIBSection(nullptr, &bmi, DIB_RGB_COLORS, &bits, nullptr, 0);
    if (!color) return nullptr;
    auto* dst = static_cast<std::uint8_t*>(bits);
    for (size_t i = 0; i < img.rgba.size(); i += 4) {   // RGBA -> DIB BGRA
        dst[i + 0] = img.rgba[i + 2];
        dst[i + 1] = img.rgba[i + 1];
        dst[i + 2] = img.rgba[i + 0];
        dst[i + 3] = img.rgba[i + 3];
    }

    const int maskStride = ((size + 15) / 16) * 2;   // WORD-aligned monochrome scanlines
    std::vector<std::uint8_t> maskBits(static_cast<size_t>(maskStride) * size, 0);
    HBITMAP mask = CreateBitmap(size, size, 1, 1, maskBits.data());

    ICONINFO ii{};
    ii.fIcon = TRUE;
    ii.hbmColor = color;
    ii.hbmMask = mask;
    HICON icon = CreateIconIndirect(&ii);

    DeleteObject(color);
    DeleteObject(mask);
    return icon;
}

class WinWindow final : public PlatformWindow {
public:
    WinWindow(int width, int height) : width_(width), height_(height) {
        enableDpiAwareness();
        hinst_ = GetModuleHandleW(nullptr);

        WNDCLASSEXW wc{sizeof(wc)};
        wc.lpfnWndProc = wndProc;
        wc.hInstance = hinst_;
        wc.hCursor = LoadCursor(nullptr, IDC_ARROW);
        wc.lpszClassName = WINDOW_CLASS;
        if (!RegisterClassExW(&wc)) fatalWin32("RegisterClassExW failed");

        RECT wa{};
        SystemParametersInfoW(SPI_GETWORKAREA, 0, &wa, 0);
        int x = wa.right - width - EDGE_GAP;
        int y = wa.bottom - height - EDGE_GAP;

        hwnd_ = CreateWindowExW(
            WS_EX_APPWINDOW, WINDOW_CLASS, WINDOW_TITLE,
            WS_POPUP | WS_SYSMENU,
            x, y, width, height, nullptr, nullptr, hinst_, nullptr);
        if (!hwnd_) fatalWin32("CreateWindowExW failed");
    }

    ~WinWindow() override {
        clearTaskbarArt();
        if (hwnd_) DestroyWindow(hwnd_);
    }

    [[nodiscard]] std::vector<const char*> requiredInstanceExtensions() const override {
        return {VK_KHR_SURFACE_EXTENSION_NAME, VK_KHR_WIN32_SURFACE_EXTENSION_NAME};
    }

    [[nodiscard]] VkSurfaceKHR createSurface(VkInstance instance) const override {
        VkWin32SurfaceCreateInfoKHR sci{VK_STRUCTURE_TYPE_WIN32_SURFACE_CREATE_INFO_KHR};
        sci.hinstance = hinst_;
        sci.hwnd = hwnd_;
        VkSurfaceKHR surface = VK_NULL_HANDLE;
        if (vkCreateWin32SurfaceKHR(instance, &sci, nullptr, &surface) != VK_SUCCESS)
            fatalWin32("vkCreateWin32SurfaceKHR failed");
        return surface;
    }

    [[nodiscard]] VkExtent2D defaultExtent() const override {
        return {static_cast<uint32_t>(width_), static_cast<uint32_t>(height_)};
    }

    void show() override {
        ShowWindow(hwnd_, SW_SHOW);
        UpdateWindow(hwnd_);
    }

    [[nodiscard]] bool pumpEvents() override {
        MSG m;
        while (PeekMessageW(&m, nullptr, 0, 0, PM_REMOVE)) {
            if (m.message == WM_QUIT) return false;
            TranslateMessage(&m);
            DispatchMessageW(&m);
        }
        return true;
    }

    void waitEvents(int timeoutMs) override {
        MsgWaitForMultipleObjects(0, nullptr, FALSE, static_cast<DWORD>(timeoutMs), QS_ALLINPUT);
    }

    void ensureVisible() override {
        if (IsIconic(hwnd_)) ShowWindow(hwnd_, SW_SHOWNOACTIVATE);
    }

    void setTaskbarArt(const std::vector<std::uint8_t>& png) override {
        if (png.empty() || png == shownArt_) return;
        constexpr int BIG = 64, SMALL = 32;   // larger than on-screen size so Windows downscales crisply
        HICON newBig = iconFromAlbumArt(png, BIG);
        HICON newSmall = iconFromAlbumArt(png, SMALL);
        if (!newBig || !newSmall) {
            if (newBig) DestroyIcon(newBig);
            if (newSmall) DestroyIcon(newSmall);
            return;
        }
        SendMessageW(hwnd_, WM_SETICON, ICON_BIG, reinterpret_cast<LPARAM>(newBig));
        SendMessageW(hwnd_, WM_SETICON, ICON_SMALL, reinterpret_cast<LPARAM>(newSmall));
        if (bigIcon_) DestroyIcon(bigIcon_);     // safe only now the replacements are installed
        if (smallIcon_) DestroyIcon(smallIcon_);
        bigIcon_ = newBig;
        smallIcon_ = newSmall;
        shownArt_ = png;
    }

    void clearTaskbarArt() override {
        if (shownArt_.empty() && !bigIcon_ && !smallIcon_) return;
        if (hwnd_) {
            SendMessageW(hwnd_, WM_SETICON, ICON_BIG, 0);
            SendMessageW(hwnd_, WM_SETICON, ICON_SMALL, 0);
        }
        if (bigIcon_) DestroyIcon(bigIcon_);
        if (smallIcon_) DestroyIcon(smallIcon_);
        bigIcon_ = smallIcon_ = nullptr;
        shownArt_.clear();
    }

private:
    int width_ = 0, height_ = 0;
    HINSTANCE hinst_ = nullptr;
    HWND hwnd_ = nullptr;
    HICON bigIcon_ = nullptr;
    HICON smallIcon_ = nullptr;
    std::vector<std::uint8_t> shownArt_;
};

}  // namespace

std::unique_ptr<PlatformWindow> createPlatformWindow(int width, int height) {
    return std::make_unique<WinWindow>(width, height);
}

void platformInitThread() {
    CoInitializeEx(nullptr, COINIT_APARTMENTTHREADED);  // WIC album-art decode (this thread)
}

void platformShutdownThread() {
    CoUninitialize();
}
