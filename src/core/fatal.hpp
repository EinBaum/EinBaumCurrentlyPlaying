#pragma once
#include <string_view>

// Single error-exit path: present the message as the platform allows (Windows: modal dialog;
// Linux: stderr), then terminate the process. Callable from any thread.
[[noreturn]] void fatal(std::string_view message);

// Route std::terminate (any uncaught exception, on any thread) through fatal(). Call once at startup.
void installCrashHandler();
