#pragma once
#include <cstddef>
#include <optional>
#include <string>

struct TrackText {
    std::wstring title;
    std::wstring artist;
};

// Strips a case-sensitive " - Topic" suffix from the artist (YouTube auto-generated "Topic"
// channels), then drops a leading "<cleaned artist> - " prefix from the title. Case-sensitivity is
// deliberate so a real artist differing only in case (" - topic") is left intact.
[[nodiscard]] TrackText cleanTrackText(std::wstring title, std::wstring artist);

// A local-file player reports only a filename as the title with no artist; drop a known media
// extension, then split a leading "Artist - " off.
[[nodiscard]] TrackText titleFromFilename(std::wstring title);

// If the title ends with one or more consecutive bracketed groups — "(...)", "[...]" or "{...}" —
// return the index where the first such group opens, so the whole trailing run can be drawn at half
// size; std::nullopt if there is none or the title is nothing but bracket groups.
[[nodiscard]] std::optional<std::size_t> bracketSuffixStart(const std::wstring& s);
