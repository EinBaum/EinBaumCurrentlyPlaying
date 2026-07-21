// Linux window backend: a Wayland wl_surface promoted to an xdg-shell toplevel, with its Vulkan
// surface via VK_KHR_wayland_surface. Wayland gives the client no control over position or
// decorations, so there is no bottom-right placement and no taskbar icon; ensureVisible is a no-op.
#include "platform/platform.hpp"
#include "core/fatal.hpp"
#include <wayland-client.h>
#define VK_USE_PLATFORM_WAYLAND_KHR
#include <vulkan/vulkan_wayland.h>
#include "xdg-shell-client-protocol.h"
#include <chrono>
#include <cstring>
#include <poll.h>

namespace {

constexpr const char* WINDOW_TITLE = "EinBaumCurrentlyPlaying";

class WaylandWindow final : public PlatformWindow {
public:
    WaylandWindow(int width, int height) : width_(width), height_(height) {
        display_ = wl_display_connect(nullptr);
        if (!display_) fatal("Wayland: cannot connect to the compositor (is WAYLAND_DISPLAY set?).");

        wl_registry* registry = wl_display_get_registry(display_);
        wl_registry_add_listener(registry, &kRegistryListener, this);
        // Two roundtrips: the first delivers the global advertisements, the second any dependent events.
        wl_display_roundtrip(display_);
        wl_display_roundtrip(display_);
        if (!compositor_) fatal("Wayland: compositor did not advertise wl_compositor.");
        if (!wmBase_) fatal("Wayland: compositor did not advertise xdg_wm_base (no xdg-shell support).");

        surface_ = wl_compositor_create_surface(compositor_);
        xdgSurface_ = xdg_wm_base_get_xdg_surface(wmBase_, surface_);
        xdg_surface_add_listener(xdgSurface_, &kXdgSurfaceListener, this);
        xdgToplevel_ = xdg_surface_get_toplevel(xdgSurface_);
        xdg_toplevel_add_listener(xdgToplevel_, &kXdgToplevelListener, this);
        xdg_toplevel_set_title(xdgToplevel_, WINDOW_TITLE);
        xdg_toplevel_set_app_id(xdgToplevel_, WINDOW_TITLE);

        // Commit the role assignment, then roundtrip to ack the first configure so the surface is
        // in a configured state before Vulkan creates the swapchain against it.
        wl_surface_commit(surface_);
        wl_display_roundtrip(display_);
    }

    ~WaylandWindow() override {
        if (xdgToplevel_) xdg_toplevel_destroy(xdgToplevel_);
        if (xdgSurface_) xdg_surface_destroy(xdgSurface_);
        if (surface_) wl_surface_destroy(surface_);
        if (wmBase_) xdg_wm_base_destroy(wmBase_);
        if (compositor_) wl_compositor_destroy(compositor_);
        if (display_) wl_display_disconnect(display_);
    }

    [[nodiscard]] std::vector<const char*> requiredInstanceExtensions() const override {
        return {VK_KHR_SURFACE_EXTENSION_NAME, VK_KHR_WAYLAND_SURFACE_EXTENSION_NAME};
    }

    [[nodiscard]] VkSurfaceKHR createSurface(VkInstance instance) const override {
        VkWaylandSurfaceCreateInfoKHR sci{VK_STRUCTURE_TYPE_WAYLAND_SURFACE_CREATE_INFO_KHR};
        sci.display = display_;
        sci.surface = surface_;
        VkSurfaceKHR surface = VK_NULL_HANDLE;
        if (vkCreateWaylandSurfaceKHR(instance, &sci, nullptr, &surface) != VK_SUCCESS)
            fatal("Wayland: vkCreateWaylandSurfaceKHR failed.");
        return surface;
    }

    [[nodiscard]] VkExtent2D defaultExtent() const override {
        return {static_cast<uint32_t>(width_), static_cast<uint32_t>(height_)};
    }

    void show() override {
        // The xdg toplevel is already committed; the first buffer the swapchain presents maps it.
        wl_surface_commit(surface_);
        wl_display_flush(display_);
    }

    [[nodiscard]] bool pumpEvents() override {
        wl_display_dispatch_pending(display_);
        wl_display_flush(display_);
        return !closed_;
    }

    void waitEvents(int timeoutMs) override {
        // poll() owns the timeout via the prepare_read/read_events idiom so libwayland never
        // blocks; bare wl_display_dispatch would sleep until the next event, stalling redraws.
        using namespace std::chrono;
        const auto deadline = steady_clock::now() + milliseconds(timeoutMs <= 0 ? 0 : timeoutMs);
        pollfd pfd{wl_display_get_fd(display_), POLLIN, 0};
        do {
            while (wl_display_prepare_read(display_) != 0)
                wl_display_dispatch_pending(display_);  // queue had events: dispatch, then retry the read lock
            wl_display_flush(display_);
            int remainMs = static_cast<int>(duration_cast<milliseconds>(deadline - steady_clock::now()).count());
            if (remainMs < 0) remainMs = 0;
            if (poll(&pfd, 1, remainMs) > 0 && (pfd.revents & POLLIN)) wl_display_read_events(display_);
            else wl_display_cancel_read(display_);      // timeout/error: release the read lock, don't read
            wl_display_dispatch_pending(display_);
        } while (steady_clock::now() < deadline && !closed_);
    }

    void ensureVisible() override {}  // Wayland never minimizes an overlay toplevel

private:
    static void onGlobal(void* data, wl_registry* reg, uint32_t name,
                         const char* iface, uint32_t version) {
        auto* self = static_cast<WaylandWindow*>(data);
        if (std::strcmp(iface, wl_compositor_interface.name) == 0) {
            self->compositor_ = static_cast<wl_compositor*>(
                wl_registry_bind(reg, name, &wl_compositor_interface, version < 4 ? version : 4));
        } else if (std::strcmp(iface, xdg_wm_base_interface.name) == 0) {
            self->wmBase_ = static_cast<xdg_wm_base*>(
                wl_registry_bind(reg, name, &xdg_wm_base_interface, 1));
            xdg_wm_base_add_listener(self->wmBase_, &kWmBaseListener, self);
        }
    }
    static void onGlobalRemove(void*, wl_registry*, uint32_t) {}

    static void onPing(void*, xdg_wm_base* wm, uint32_t serial) { xdg_wm_base_pong(wm, serial); }

    static void onSurfaceConfigure(void*, xdg_surface* xs, uint32_t serial) {
        xdg_surface_ack_configure(xs, serial);
    }

    static void onToplevelConfigure(void*, xdg_toplevel*, int32_t, int32_t, wl_array*) {}
    static void onToplevelClose(void* data, xdg_toplevel*) {
        static_cast<WaylandWindow*>(data)->closed_ = true;
    }
    static void onToplevelConfigureBounds(void*, xdg_toplevel*, int32_t, int32_t) {}
    static void onToplevelWmCapabilities(void*, xdg_toplevel*, wl_array*) {}

    static const wl_registry_listener kRegistryListener;
    static const xdg_wm_base_listener kWmBaseListener;
    static const xdg_surface_listener kXdgSurfaceListener;
    static const xdg_toplevel_listener kXdgToplevelListener;

    int width_ = 0, height_ = 0;
    bool closed_ = false;
    wl_display* display_ = nullptr;
    wl_compositor* compositor_ = nullptr;
    xdg_wm_base* wmBase_ = nullptr;
    wl_surface* surface_ = nullptr;
    xdg_surface* xdgSurface_ = nullptr;
    xdg_toplevel* xdgToplevel_ = nullptr;
};

const wl_registry_listener WaylandWindow::kRegistryListener = {onGlobal, onGlobalRemove};
const xdg_wm_base_listener WaylandWindow::kWmBaseListener = {onPing};
const xdg_surface_listener WaylandWindow::kXdgSurfaceListener = {onSurfaceConfigure};
const xdg_toplevel_listener WaylandWindow::kXdgToplevelListener = {
    onToplevelConfigure, onToplevelClose, onToplevelConfigureBounds, onToplevelWmCapabilities};

}  // namespace

std::unique_ptr<PlatformWindow> createPlatformWindow(int width, int height) {
    return std::make_unique<WaylandWindow>(width, height);
}

void platformInitThread() {}      // Linux image decode (stb) needs no per-thread setup
void platformShutdownThread() {}
