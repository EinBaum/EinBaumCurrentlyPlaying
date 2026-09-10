// Linux entry point: parses --debug/--http/--simulate, creates the Wayland overlay window, and
// hands control to runApp.
#include "core/app.hpp"
#include "core/fatal.hpp"
#include "core/theme.hpp"
#include "platform/platform.hpp"
#include <memory>
#include <string_view>

int main(int argc, char** argv) {
    installCrashHandler();

    bool debug = false;
    bool http = false;
    bool simulate = false;
    for (int i = 1; i < argc; ++i) {
        const std::string_view a = argv[i];
        if (a == "--debug") debug = true;
        else if (a == "--http") http = true;
        else if (a == "--simulate") simulate = true;
    }
    std::unique_ptr<PlatformWindow> window = createPlatformWindow(WINDOW_W, WINDOW_H);
    return runApp(*window, debug, http, simulate);
}
