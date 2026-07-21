// Linux entry point: parses --debug, creates the Wayland overlay window, and hands control to runApp.
#include "core/app.hpp"
#include "core/fatal.hpp"
#include "core/theme.hpp"
#include "platform/platform.hpp"
#include <memory>
#include <string_view>

int main(int argc, char** argv) {
    installCrashHandler();

    bool debug = false;
    for (int i = 1; i < argc; ++i)
        if (std::string_view(argv[i]) == "--debug") debug = true;
    std::unique_ptr<PlatformWindow> window = createPlatformWindow(WINDOW_W, WINDOW_H);
    return runApp(*window, debug);
}
