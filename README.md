# EinBaumCurrentlyPlaying

Borderless "now playing" overlay for OBS, drawn with Vulkan. Shows album cover, title, artist, and
progress bar on a green chroma-key card, bottom-right. Finite tracks get a playhead; streams get a
full bar.

![screenshot](screenshot.jpg)

Reads the OS media session: SMTC on Windows, MPRIS over D-Bus on Linux. No network access.

## Requirements

- GPU with hardware ray tracing + Vulkan 1.2 driver (NVIDIA RTX 20xx+, AMD RX 6000+/890M, Intel Arc).
- Windows 10/11 x64, or Linux + Wayland.

## Run

`--debug` = show an fps readout.

```
EinBaumCurrentlyPlaying --debug
```

## OBS

Add a Window/Game Capture source, then a Chroma Key filter, Key Color green (`#00FF00`).

## Build

CMake (Ninja) on both platforms.

**Windows** — [MSYS2](https://www.msys2.org/) UCRT64:

```
pacman -S mingw-w64-ucrt-x86_64-{gcc,cmake,ninja,shaderc,vulkan-headers}
cmake -S . -B build-windows -G Ninja -DCMAKE_BUILD_TYPE=Release
cmake --build build-windows   # -> build-windows\EinBaumCurrentlyPlaying.exe
```

**Linux** — needs `glslc`, `wayland-scanner`, wayland-protocols, vulkan, freetype2, fontconfig, libsystemd:

```
cmake -S . -B build-linux -G Ninja -DCMAKE_BUILD_TYPE=Release
cmake --build build-linux     # -> build-linux/EinBaumCurrentlyPlaying
```
