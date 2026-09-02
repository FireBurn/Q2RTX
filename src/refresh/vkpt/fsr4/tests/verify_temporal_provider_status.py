#!/usr/bin/env python3
"""Verify that Q2RTX exposes the official-provider boundary read-only."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys


REQUIRED_ROWS = (
    ("official FSR4.1.1", "flt_fsr4_official_provider_reason",
     "unavailable: signed DX12 provider (no native Vulkan)"),
    ("official Ray Regeneration", "flt_ray_regeneration_provider_reason",
     "unavailable: signed DX12 provider (RX 9000+)"),
    ("official ML Frame Generation", "flt_ml_frame_generation_provider_reason",
     "unavailable: signed DX12 provider (RX 9000+)"),
)


def diagnostics_block(menu: str) -> str:
    start = menu.find("begin temporal_diagnostics")
    if start < 0:
        raise ValueError("temporal diagnostics menu is missing")
    end = menu.find("\nend", start)
    if end < 0:
        raise ValueError("temporal diagnostics menu is unterminated")
    return menu[start:end]


def verify(menu_path: Path, source_path: Path) -> None:
    menu = diagnostics_block(menu_path.read_text(encoding="utf-8"))
    source = source_path.read_text(encoding="utf-8")
    previous = -1
    for label, cvar, default in REQUIRED_ROWS:
        row = f'"{label}" {cvar}'
        position = menu.find(row)
        if position < 0:
            raise ValueError(f"diagnostics row missing: {label}")
        if position <= previous:
            raise ValueError(f"diagnostics row order is not stable: {label}")
        previous = position
        if f'"{cvar}"' not in source:
            raise ValueError(f"read-only cvar missing from fsr.c: {cvar}")
        if f'"{default}"' not in source:
            raise ValueError(f"concise cvar default missing: {cvar}")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("menu", type=Path)
    parser.add_argument("fsr_source", type=Path)
    args = parser.parse_args()
    try:
        verify(args.menu, args.fsr_source)
    except (OSError, ValueError) as error:
        print(f"Temporal provider-status verification failed: {error}", file=sys.stderr)
        return 1
    print("Temporal provider-status verification passed: 3 official-provider rows")
    return 0


if __name__ == "__main__":
    sys.exit(main())
