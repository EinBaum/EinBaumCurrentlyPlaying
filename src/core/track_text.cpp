#include "core/track_text.hpp"
#include <algorithm>
#include <array>
#include <cwctype>
#include <string_view>

namespace {

[[nodiscard]] std::wstring_view trim(std::wstring_view s) {
    while (!s.empty() && std::iswspace(s.front())) s.remove_prefix(1);
    while (!s.empty() && std::iswspace(s.back())) s.remove_suffix(1);
    return s;
}

constexpr auto MEDIA_EXTENSIONS = std::to_array<std::wstring_view>({
    L"mp3", L"flac", L"m4a", L"aac", L"ogg", L"opus", L"wav", L"wma", L"aiff", L"aif", L"alac", L"ape", L"mka",
    L"mp4", L"mkv", L"webm", L"avi", L"mov", L"flv", L"wmv", L"m4v", L"ts", L"mpg", L"mpeg", L"3gp",
});

[[nodiscard]] bool equalAsciiNoCase(std::wstring_view a, std::wstring_view b) {
    return std::ranges::equal(a, b, [](wchar_t x, wchar_t y) { return std::towlower(x) == std::towlower(y); });
}

void stripMediaExtension(std::wstring& s) {
    const auto dot = s.rfind(L'.');
    if (dot == std::wstring::npos || dot == 0) return;
    const std::wstring_view ext = std::wstring_view{s}.substr(dot + 1);
    if (std::ranges::any_of(MEDIA_EXTENSIONS, [&](std::wstring_view e) { return equalAsciiNoCase(ext, e); }))
        s.erase(dot);
}

}  // namespace

TrackText cleanTrackText(std::wstring title, std::wstring artist) {
    constexpr std::wstring_view TOPIC = L" - Topic";
    if (artist.ends_with(TOPIC)) {
        artist.erase(artist.size() - TOPIC.size());
    }

    if (!artist.empty()) {
        const std::wstring prefix = artist + L" - ";
        if (title.starts_with(prefix)) {
            title.erase(0, prefix.size());
        }
    }

    return {std::wstring{trim(title)}, std::wstring{trim(artist)}};
}

TrackText titleFromFilename(std::wstring title) {
    stripMediaExtension(title);

    const std::wstring_view sv = trim(title);
    constexpr std::wstring_view SEP = L" - ";
    if (const auto at = sv.find(SEP); at != std::wstring_view::npos) {
        const std::wstring_view artist = trim(sv.substr(0, at));
        const std::wstring_view rest = trim(sv.substr(at + SEP.size()));
        if (!artist.empty() && !rest.empty())
            return {std::wstring{rest}, std::wstring{artist}};
    }
    return {std::wstring{sv}, std::wstring{}};
}

std::optional<std::size_t> bracketSuffixStart(const std::wstring& s) {
    auto trimEnd = [&](int e) { while (e > 0 && std::iswspace(s[e - 1])) --e; return e; };
    int cur = trimEnd(static_cast<int>(s.size()));
    // Peel balanced trailing groups (each optionally preceded by whitespace) off the end until a
    // non-bracket character or an unbalanced group stops the run, e.g. "(Sail Away) [Single Version]".
    int suffix = -1;
    while (cur > 0) {
        wchar_t close = s[cur - 1], open;
        if (close == L')') open = L'(';
        else if (close == L']') open = L'[';
        else if (close == L'}') open = L'{';
        else break;
        int depth = 0, i = cur - 1;
        for (; i >= 0; --i) {
            if (s[i] == close) ++depth;
            else if (s[i] == open && --depth == 0) break;
        }
        if (i < 0 || depth != 0) break;             // unbalanced: keep the groups found so far
        suffix = i;
        cur = trimEnd(i);
    }
    if (suffix < 0) return std::nullopt;
    return trimEnd(suffix) > 0 ? std::optional<std::size_t>(suffix) : std::nullopt;  // need real text before the run
}
