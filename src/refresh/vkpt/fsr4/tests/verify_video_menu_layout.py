#!/usr/bin/env python3
"""Keep the RTX Video menu readable at low display heights."""

from __future__ import annotations

import argparse
from pathlib import Path
import re
import sys


MAX_COMPACT_ITEMS = 15
STATUS_WRAP_COLUMNS = 960 // 8
MAX_STATUS_LINES = 8


def menu_block(menu: str, name: str) -> list[str]:
    marker = f"begin {name}"
    start = menu.find(marker)
    if start < 0:
        raise ValueError(f"missing {marker}")
    end = menu.find("\nend", start)
    if end < 0:
        raise ValueError(f"missing end for {name}")
    return [line.strip() for line in menu[start:end].splitlines()]


def visible_item_count(lines: list[str]) -> int:
    prefixes = ("action ", "pairs ", "range ", "toggle ", "values ", "static ")
    return sum(line.startswith(prefixes) for line in lines)


def require(lines: list[str], text: str) -> None:
    if not any(text in line for line in lines):
        raise ValueError(f"missing menu control: {text}")


def status_line_count(text: str, columns: int) -> int:
    """Match Menu_DrawStatus's bounded word-wrap calculation."""
    count = 0
    used = 0
    position = 0
    while position < len(text):
        word_end = position
        while word_end < len(text) and ord(text[word_end]) > 32:
            word_end += 1
        word_length = word_end - position
        if (word_length < columns and used + word_length > columns) or used == columns:
            if count == MAX_STATUS_LINES - 1:
                break
            count += 1
            used = 0
        position += 1
        used += 1
    return count + 1 if text else 0


def verify_status_space(lines: list[str], name: str) -> None:
    statuses = re.findall(r'--status "([^"]*)"', "\n".join(lines))
    if not statuses:
        return
    longest = max(status_line_count(status, STATUS_WRAP_COLUMNS) for status in statuses)
    if longest > MAX_STATUS_LINES:
        raise ValueError(f"{name} has help text exceeding the renderer's {MAX_STATUS_LINES}-line limit")


def verify(menu_path: Path) -> None:
    menu = menu_path.read_text(encoding="utf-8")
    video = menu_block(menu, "video")
    video_rtx_start = video.index("ifeq vid_rtx 1")
    video_rtx_end = video.index("endif", video_rtx_start)
    navigation = video[video_rtx_start + 1:video_rtx_end]
    expected = (
        "resolution scaling options...",
        "HDR options...",
        "image tuning...",
        "temporal upscaling...",
        "ray tracing features...",
    )
    if tuple(line.split('"')[1] for line in navigation if line.startswith("action ")) != expected:
        raise ValueError("RTX Video page navigation changed or contains crowding controls")

    pages = {
        "image_tuning": ("fallback anti-aliasing",),
        "temporal_settings": (
            "temporal upscaler",
            "FSR3 frame generation",
            "temporal diagnostics...",
        ),
        "ray_tracing_settings": ("global illumination", "GPU profiler"),
    }
    for name, controls in pages.items():
        lines = menu_block(menu, name)
        count = visible_item_count(lines)
        if count > MAX_COMPACT_ITEMS:
            raise ValueError(f"{name} has {count} controls; maximum is {MAX_COMPACT_ITEMS}")
        for control in controls:
            require(lines, control)
        verify_status_space(lines, name)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("menu", type=Path)
    parser.add_argument("--menu-source", type=Path,
                        help="menu.c implementation; verifies compact help reserves its wrapped height")
    args = parser.parse_args()
    try:
        verify(args.menu)
        if args.menu_source:
            source = args.menu_source.read_text(encoding="utf-8")
            if ("Menu_StatusLineCount" not in source
                    or "status_lines * CHAR_HEIGHT" not in source):
                raise ValueError("compact menu help does not reserve its measured wrapped height")
    except (OSError, ValueError) as error:
        print(f"Video-menu layout verification failed: {error}", file=sys.stderr)
        return 1
    print("Video-menu layout verification passed: compact RTX navigation, bounded pages, and reserved help text")
    return 0


if __name__ == "__main__":
    sys.exit(main())
