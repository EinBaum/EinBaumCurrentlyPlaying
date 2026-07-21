cmake_minimum_required(VERSION 3.20)
# C is enabled for the wayland-scanner-generated xdg-shell-protocol.c (Linux); the app itself is C++.
project(EinBaumCurrentlyPlaying CXX C)

# C++23 across the board (std::format, std::source_location, designated initializers, std::lerp).
set(CMAKE_CXX_STANDARD 23)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release)
endif()

set(SRC ${CMAKE_SOURCE_DIR}/src)

# --- Shaders: compile to embeddable SPIR-V C arrays (the binary carries no .spv files) ---------------
# The fragment and the half-res shadow compute shaders cast ray queries against the scene
# acceleration structure, so the stages target Vulkan 1.2 / SPIR-V 1.4+. -I resolves the
# shadow_common.glsl include. Output goes to ${CMAKE_BINARY_DIR}/shaders/*.inl, which the renderer
# includes as "shaders/<name>.inl" (so ${CMAKE_BINARY_DIR} is on the include path below).
find_program(GLSLC glslc REQUIRED)
set(SHADER_DIR ${SRC}/shaders)
set(SHADER_OUT ${CMAKE_BINARY_DIR}/shaders)
file(MAKE_DIRECTORY ${SHADER_OUT})

# stage_source -> output_basename
set(SHADERS
    mesh.vert:mesh_vert
    mesh.frag:mesh_frag
    mesh.tesc:mesh_tesc
    mesh.tese:mesh_tese
    wash_shadow.comp:wash_shadow_comp
)
set(SHADER_INLS "")
foreach(entry ${SHADERS})
    string(REPLACE ":" ";" parts ${entry})
    list(GET parts 0 stage_src)
    list(GET parts 1 out_base)
    set(out_inl ${SHADER_OUT}/${out_base}.inl)
    add_custom_command(
        OUTPUT ${out_inl}
        COMMAND ${GLSLC} --target-env=vulkan1.2 -I ${SHADER_DIR} -mfmt=c
                -o ${out_inl} ${SHADER_DIR}/${stage_src}
        DEPENDS ${SHADER_DIR}/${stage_src} ${SHADER_DIR}/shadow_common.glsl
        COMMENT "glslc ${stage_src} -> ${out_base}.inl"
        VERBATIM)
    list(APPEND SHADER_INLS ${out_inl})
endforeach()
add_custom_target(shaders DEPENDS ${SHADER_INLS})

# --- Portable core (src/core: Vulkan renderer + math + layout + glyph tessellation + app loop) -------
add_library(core STATIC
    ${SRC}/core/app.cpp
    ${SRC}/core/art_decoder.cpp
    ${SRC}/core/glyph_mesh.cpp
    ${SRC}/core/image.cpp
    ${SRC}/core/renderer.cpp
    ${SRC}/core/renderer_raytracing.cpp
    ${SRC}/core/renderer_vk.cpp
    ${SRC}/core/track_text.cpp
)
add_dependencies(core shaders)
target_include_directories(core PUBLIC ${SRC} ${CMAKE_BINARY_DIR})

find_package(Vulkan REQUIRED)
target_link_libraries(core PUBLIC Vulkan::Vulkan)

# --- Platform layer + entry point --------------------------------------------------------------------
if(WIN32)
    add_executable(EinBaumCurrentlyPlaying WIN32
        ${SRC}/platform/win/win_main.cpp
        ${SRC}/platform/win/win_window.cpp
        ${SRC}/platform/win/win_font.cpp
        ${SRC}/platform/win/win_image.cpp
        ${SRC}/platform/win/win_media.cpp
        ${SRC}/platform/win/win_fatal.cpp
    )
    target_link_libraries(EinBaumCurrentlyPlaying PRIVATE
        core gdi32 ole32 uuid windowscodecs runtimeobject)

    # MSYS2 UCRT64 (GCC): -municode for the Unicode wWinMain entry; -static links the MinGW runtime
    # in so the exe depends on OS DLLs only (their import libs stay dynamic); --gc-sections drops dead
    # code; -s strips symbols.
    if(MINGW)
        target_compile_options(core PRIVATE -municode -ffunction-sections -fdata-sections)
        target_compile_options(EinBaumCurrentlyPlaying PRIVATE
            -municode -ffunction-sections -fdata-sections)
        target_link_options(EinBaumCurrentlyPlaying PRIVATE
            -municode -mwindows -static -s -Wl,--gc-sections)
    endif()
else()
    # Generate the xdg-shell client protocol glue with wayland-scanner.
    find_program(WAYLAND_SCANNER wayland-scanner REQUIRED)
    find_package(PkgConfig REQUIRED)
    pkg_get_variable(WAYLAND_PROTOCOLS_DIR wayland-protocols pkgdatadir)
    set(XDG_SHELL_XML ${WAYLAND_PROTOCOLS_DIR}/stable/xdg-shell/xdg-shell.xml)
    set(XDG_HEADER ${CMAKE_BINARY_DIR}/xdg-shell-client-protocol.h)
    set(XDG_CODE ${CMAKE_BINARY_DIR}/xdg-shell-protocol.c)
    add_custom_command(OUTPUT ${XDG_HEADER}
        COMMAND ${WAYLAND_SCANNER} client-header ${XDG_SHELL_XML} ${XDG_HEADER}
        DEPENDS ${XDG_SHELL_XML} VERBATIM)
    add_custom_command(OUTPUT ${XDG_CODE}
        COMMAND ${WAYLAND_SCANNER} private-code ${XDG_SHELL_XML} ${XDG_CODE}
        DEPENDS ${XDG_SHELL_XML} VERBATIM)
    add_custom_target(xdg_shell_protocol DEPENDS ${XDG_HEADER} ${XDG_CODE})

    pkg_check_modules(WAYLAND REQUIRED wayland-client)
    pkg_check_modules(FREETYPE REQUIRED freetype2)
    pkg_check_modules(FONTCONFIG REQUIRED fontconfig)
    # lin_media.cpp reads the active MPRIS session over the session D-Bus bus via sd-bus.
    pkg_check_modules(LIBSYSTEMD REQUIRED libsystemd)

    add_executable(EinBaumCurrentlyPlaying
        ${SRC}/platform/linux/lin_main.cpp
        ${SRC}/platform/linux/lin_window.cpp
        ${SRC}/platform/linux/lin_font.cpp
        ${SRC}/platform/linux/lin_image.cpp
        ${SRC}/platform/linux/lin_media.cpp
        ${SRC}/platform/linux/lin_fatal.cpp
        ${XDG_CODE}
    )
    add_dependencies(EinBaumCurrentlyPlaying xdg_shell_protocol)
    # lin_window.cpp includes the generated header by name; ${CMAKE_BINARY_DIR} is already on core's
    # PUBLIC include path. lin_image.cpp vendors stb_image.h from third_party/.
    target_include_directories(EinBaumCurrentlyPlaying PRIVATE
        ${CMAKE_SOURCE_DIR}/third_party
        ${FREETYPE_INCLUDE_DIRS}
        ${FONTCONFIG_INCLUDE_DIRS}
        ${WAYLAND_INCLUDE_DIRS}
        ${LIBSYSTEMD_INCLUDE_DIRS})
    target_link_libraries(EinBaumCurrentlyPlaying PRIVATE
        core
        ${WAYLAND_LIBRARIES}
        ${FREETYPE_LIBRARIES}
        ${FONTCONFIG_LIBRARIES}
        ${LIBSYSTEMD_LIBRARIES})
endif()
