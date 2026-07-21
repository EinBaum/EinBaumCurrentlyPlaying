// Windows error-exit path: a modal message box, then process exit.
#include "core/fatal.hpp"
#include "platform/win/win_fatal.hpp"
#include <windows.h>
#include <exception>
#include <format>
#include <string>

namespace {
constexpr const char* TITLE = "EinBaumCurrentlyPlaying";

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
    const std::string text(message);
    MessageBoxA(nullptr, text.c_str(), TITLE, MB_OK | MB_ICONERROR | MB_SETFOREGROUND | MB_TOPMOST);
    ExitProcess(1);
}

void fatalWin32(std::string_view context) {
    const DWORD err = GetLastError();
    char* sys = nullptr;
    FormatMessageA(FORMAT_MESSAGE_ALLOCATE_BUFFER | FORMAT_MESSAGE_FROM_SYSTEM | FORMAT_MESSAGE_IGNORE_INSERTS,
                   nullptr, err, 0, reinterpret_cast<char*>(&sys), 0, nullptr);
    const std::string detail = sys ? sys : "";
    if (sys) LocalFree(sys);
    fatal(std::format("{} (error 0x{:08X}: {})", context, err, detail));
}

void installCrashHandler() {
    std::set_terminate(onTerminate);
}
