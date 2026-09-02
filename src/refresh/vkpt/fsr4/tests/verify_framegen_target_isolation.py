#!/usr/bin/env python3
"""Keep analytical frame-generation output separate from FSR4 recurrence."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys


def require(text: str, needle: str, source: Path) -> None:
    if needle not in text:
        raise ValueError(f"missing {needle!r} in {source}")


def verify(texture_header: Path, fsr_source: Path, main_source: Path) -> None:
    textures = texture_header.read_text(encoding="utf-8")
    fsr = fsr_source.read_text(encoding="utf-8")
    main = main_source.read_text(encoding="utf-8")

    require(textures, "IMG_DO(FSR_RCAS_OUTPUT,", texture_header)
    require(textures, "IMG_DO(FSR_FRAMEGEN_OUTPUT,", texture_header)
    if "FSR_RCAS_OUTPUT,           36" not in textures:
        raise ValueError("FSR4 recurrent image must retain its stable slot")
    if "FSR_FRAMEGEN_OUTPUT, 62" not in textures:
        raise ValueError("frame-generation image must have its own stable slot")

    require(fsr, "dispatch.output = fsr3_316_image(VKPT_IMG_FSR_FRAMEGEN_OUTPUT,",
            fsr_source)
    require(fsr, "dispatch.output = fsr3_screen_image(VKPT_IMG_FSR_FRAMEGEN_OUTPUT,",
            fsr_source)
    require(fsr, ".image = qvk.images[VKPT_IMG_FSR_FRAMEGEN_OUTPUT],",
            fsr_source)
    if "dispatch.output = fsr3_316_image(VKPT_IMG_FSR_RCAS_OUTPUT," in fsr or \
       "dispatch.output = fsr3_screen_image(VKPT_IMG_FSR_RCAS_OUTPUT," in fsr:
        raise ValueError("frame generation aliases FSR4 recurrent storage")

    require(main, "? VKPT_IMG_FSR_FRAMEGEN_OUTPUT : VKPT_IMG_TAA_OUTPUT,",
            main_source)
    require(main, "? VKPT_IMG_TAA_OUTPUT : VKPT_IMG_FSR_FRAMEGEN_OUTPUT;",
            main_source)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("texture_header", type=Path)
    parser.add_argument("fsr_source", type=Path)
    parser.add_argument("main_source", type=Path)
    args = parser.parse_args()
    try:
        verify(args.texture_header, args.fsr_source, args.main_source)
    except (OSError, ValueError) as error:
        print(f"Frame-generation target-isolation verification failed: {error}",
              file=sys.stderr)
        return 1
    print("Frame-generation target isolation verification passed")
    return 0


if __name__ == "__main__":
    sys.exit(main())
