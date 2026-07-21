// Linux error-exit path: write the message to stderr, then terminate (no modal dialog; this is an
// overlay launched from a terminal/OBS).
#include "core/fatal.hpp"
#include <cstdio>
#include <cstdlib>
#include <exception>
#include <format>
#include <string>

namespace {
[[noreturn]] void onTerminate() {
    // std::terminate runs with the offending exception still active; rethrow to recover its message.
    std::string msg = "Fatal error: terminate called.";
    if (std::exception_ptr e = std::current_exception()) {
        try { std::rethrow_exception(e); }
        catch (const std::exception& ex) { msg = std::format("Unhandled exception: {}", ex.what()); }
        catch (...) { msg = "Unhandled non-standard exception."; }
    }
    fatal(msg);
}
}  // namespace

void fatal(std::string_view message) {
    std::fprintf(stderr, "EinBaumCurrentlyPlaying: fatal: %.*s\n",
                 static_cast<int>(message.size()), message.data());
    std::fflush(stderr);
    std::_Exit(1);  // skip static destructors; the Vulkan/Wayland state is being torn down anyway
}

void installCrashHandler() {
    std::set_terminate(onTerminate);
}
