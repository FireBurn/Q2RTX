#!/usr/bin/env python3
"""Keep the RTX Video menu readable at low display heights."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys


MAX_COMPACT_ITEMS = 15


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


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("menu", type=Path)
    args = parser.parse_args()
    try:
        verify(args.menu)
    except (OSError, ValueError) as error:
        print(f"Video-menu layout verification failed: {error}", file=sys.stderr)
        return 1
    print("Video-menu layout verification passed: compact RTX navigation and three bounded pages")
    return 0


if __name__ == "__main__":
    sys.exit(main())
