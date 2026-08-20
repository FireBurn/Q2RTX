#!/usr/bin/env python3
"""Verify the tracked source-v07 FSR4 model binaries against their manifests."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys


MODELS = ("native", "quality", "balanced", "performance", "ultraperf", "drs")


def digest(path: Path) -> str:
    hasher = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(65536), b""):
            hasher.update(block)
    return hasher.hexdigest()


def verify_asset(directory: Path, name: str, record: object) -> None:
    if not isinstance(record, dict):
        raise ValueError(f"{name}: manifest record is not an object")
    expected_size = record.get("size")
    expected_digest = record.get("sha256")
    if not isinstance(expected_size, int) or not isinstance(expected_digest, str):
        raise ValueError(f"{name}: manifest has no size/SHA-256")
    path = directory / name
    if not path.is_file():
        raise ValueError(f"{name}: file is missing")
    actual_size = path.stat().st_size
    if actual_size != expected_size:
        raise ValueError(f"{name}: size {actual_size}, expected {expected_size}")
    actual_digest = digest(path)
    if actual_digest != expected_digest:
        raise ValueError(f"{name}: SHA-256 {actual_digest}, expected {expected_digest}")


def verify_model(directory: Path, model: str) -> None:
    prefix = f"fsr4_model_v07_i8_{model}"
    manifest_path = directory / f"{prefix}_shader_manifest.json"
    if not manifest_path.is_file():
        raise ValueError(f"{model}: manifest is missing")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    artifacts = manifest.get("artifacts")
    if not isinstance(artifacts, dict):
        raise ValueError(f"{manifest_path.name}: artifacts object is missing")
    expected = (f"{prefix}_initializers.bin", f"{prefix}_pre_weights.bin")
    manifest_bins = sorted(name for name in artifacts if name.endswith(".bin"))
    if manifest_bins != sorted(expected):
        raise ValueError(
            f"{manifest_path.name}: binary set {manifest_bins}, expected {list(expected)}"
        )
    for asset in expected:
        verify_asset(directory, asset, artifacts[asset])


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("shader_dir", type=Path)
    args = parser.parse_args()
    try:
        for model in MODELS:
            verify_model(args.shader_dir, model)
    except (OSError, ValueError, json.JSONDecodeError) as error:
        print(f"FSR4 v07 asset verification failed: {error}", file=sys.stderr)
        return 1
    print(f"FSR4 v07 asset verification passed: {len(MODELS)} model pairs")
    return 0


if __name__ == "__main__":
    sys.exit(main())
