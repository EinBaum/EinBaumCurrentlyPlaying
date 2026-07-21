#pragma once
#include <string_view>

// fatal() with the calling thread's GetLastError code and its system text appended to `context`.
// Windows-only; the portable error path is fatal() in fatal.hpp.
[[noreturn]] void fatalWin32(std::string_view context);
