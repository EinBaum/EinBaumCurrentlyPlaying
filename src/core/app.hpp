#pragma once
class PlatformWindow;

// The OS-agnostic application: owns the renderer, the media poller, and the redraw loop. The two
// platform entry points create a PlatformWindow, then hand it here. Returns the process exit code.
int runApp(PlatformWindow& window, bool debug);
