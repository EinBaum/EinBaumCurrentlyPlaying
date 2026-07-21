// Windows entry point: parses --debug, creates the overlay window, and hands control to runApp.
#include "core/app.hpp"
#include "core/fatal.hpp"
#include "core/theme.hpp"
#include "platform/platform.hpp"
#include <windows.h>
#include <string>

int WINAPI wWinMain(HINSTANCE, HINSTANCE, LPWSTR lpCmdLine, int) {
    installCrashHandler();

    // --debug only shows the fps readout.
    const std::wstring cmdLine = lpCmdLine ? lpCmdLine : L"";
    const bool debug = cmdLine.find(L"--debug") != std::wstring::npos;

    std::unique_ptr<PlatformWindow> window = createPlatformWindow(WINDOW_W, WINDOW_H);
    return runApp(*window, debug);
}
