#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""Package local recovered payloads; do not commit or redistribute the bundle."""
import argparse
import hashlib
import json
from pathlib import Path


def pack(replay, shaders, output):
    output.mkdir(parents=True, exist_ok=False)
    assets = {}

    def asset(path):
        data = Path(path).read_bytes()
        digest = hashlib.sha256(data).hexdigest()
        name = digest + Path(path).suffix
        (output / name).write_bytes(data)
        assets[name] = digest
        return name

    def rewrite(value):
        if isinstance(value, dict):
            return {k: asset(v) if k in ("file", "shader") else rewrite(v)
                    for k,v in value.items()}
        if isinstance(value, list):
            return [rewrite(v) for v in value]
        return value

    bundle = {"schema": "xess.native-bundle", "version": 1,
        "profile": "753x424-1280x720-unit-exposure",
        "prepare": asset(shaders / "40553fb1ea6dd6f3.spv"),
        "convolutions": rewrite(json.loads((replay / "convolutions/manifest.json").read_text())),
        "resolve": rewrite(json.loads((replay / "frame-0/resolve-data/manifest.json").read_text())),
        "assets": assets}
    (output / "bundle.json").write_text(json.dumps(bundle, indent=2)+"\n")


def load(directory):
    directory = directory.resolve()
    bundle = json.loads((directory / "bundle.json").read_text())
    if (bundle["schema"],bundle["version"],bundle["profile"]) != (
            "xess.native-bundle",1,"753x424-1280x720-unit-exposure"):
        raise ValueError("Unsupported native bundle")
    def path(name):
        target = (directory / name).resolve()
        if target.parent != directory or name not in bundle["assets"]:
            raise ValueError("Invalid bundle asset path")
        return target
    for name, digest in bundle["assets"].items():
        if hashlib.sha256(path(name).read_bytes()).hexdigest() != digest:
            raise ValueError("Bundle asset checksum mismatch: " + name)
    def rewrite(value):
        if isinstance(value,dict):
            return {k:str(path(v)) if k in ("file","shader") else rewrite(v) for k,v in value.items()}
        if isinstance(value,list): return [rewrite(v) for v in value]
        return value
    bundle["prepare"] = str(path(bundle["prepare"]))
    bundle["convolutions"] = rewrite(bundle["convolutions"])
    bundle["resolve"] = rewrite(bundle["resolve"])
    return bundle


if __name__ == "__main__":
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("replay",type=Path)
    parser.add_argument("shaders",type=Path)
    parser.add_argument("output",type=Path)
    args=parser.parse_args()
    pack(args.replay,args.shaders,args.output)
