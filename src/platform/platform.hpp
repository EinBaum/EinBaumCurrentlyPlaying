#pragma once
#include <vulkan/vulkan.h>
#include <cstdint>
#include <memory>
#include <vector>

// The OS window plus its Vulkan surface, behind one interface the core renderer and run loop use.
// Windows backs it with an HWND (win_window.cpp); Linux with a Wayland wl_surface + xdg-shell
// toplevel (lin_window.cpp). The window is created sized for the requested resolution and placed at
// the screen's bottom-right (where the compositor allows it); see createPlatformWindow.
class PlatformWindow {
public:
    virtual ~PlatformWindow() = default;

    // Instance extensions this surface needs, e.g. {VK_KHR_surface, VK_KHR_win32_surface} or
    // {VK_KHR_surface, VK_KHR_wayland_surface}. Passed to vkCreateInstance by the renderer.
    [[nodiscard]] virtual std::vector<const char*> requiredInstanceExtensions() const = 0;

    // Create the VkSurfaceKHR for this window against an already-created instance.
    [[nodiscard]] virtual VkSurfaceKHR createSurface(VkInstance instance) const = 0;

    // The client size the window was created with, used as the swapchain extent when the surface
    // reports no fixed size (currentExtent == 0xFFFFFFFF, as Wayland does).
    [[nodiscard]] virtual VkExtent2D defaultExtent() const = 0;

    // Map the window and start presenting. Called once after the renderer is initialized.
    virtual void show() = 0;

    // Drain the OS event queue without blocking. Returns false once the window has been closed,
    // which ends the run loop.
    [[nodiscard]] virtual bool pumpEvents() = 0;

    // Block until an OS event arrives or `timeoutMs` elapses, then return (the caller pumps after).
    // Replaces the Win32 MsgWaitForMultipleObjects wait. timeoutMs == 0 returns immediately.
    virtual void waitEvents(int timeoutMs) = 0;

    // Undo any minimize/occlusion so a capture tool keeps seeing the window. No-op where the
    // platform never minimizes an override/overlay surface (Wayland).
    virtual void ensureVisible() = 0;

    // Set/clear the window's taskbar icon from raw album-art bytes (PNG/JPEG). A window attribute, so
    // it lives here rather than in the core loop. No-op on platforms without a per-window taskbar icon
    // (Wayland); the default no-ops let such backends ignore it entirely.
    virtual void setTaskbarArt(const std::vector<std::uint8_t>& /*art*/) {}
    virtual void clearTaskbarArt() {}
};

// Create the single overlay window at the given client size. Implemented once per platform.
[[nodiscard]] std::unique_ptr<PlatformWindow> createPlatformWindow(int width, int height);

// Initialize whatever per-thread state the platform's image decode needs on the thread that calls
// decodeAlbumArt (Windows: CoInitializeEx for WIC). Paired with platformShutdownThread(). Linux: no-op.
void platformInitThread();
void platformShutdownThread();
