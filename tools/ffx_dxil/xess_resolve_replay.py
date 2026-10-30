#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""Materialize first-frame XeSS resolve constants, explicitly marking CBV padding."""
import argparse
import hashlib
import json
from pathlib import Path
import re
import struct
from xess_conv_replay import copied_buffers, materialize, require


def build(trace, shaders, output, padding_byte=0, frame=0):
    require(0 <= padding_byte <= 255, "Padding byte outside range")
    require(0 <= frame < 4, "Frame outside captured sequence")
    dispatch = json.loads((trace / "dispatch-index.json").read_text())["dispatches"][frame*14+13]
    require(dispatch["shader"] == "065ebeb5c40e7a3d", "Unsupported resolve profile")
    require(set(dispatch["constants"]) == {"3"}, "Unexpected root constants")
    constants = dispatch["constants"]["3"]
    require(set(constants) == {str(i) for i in range(20)}, "Incomplete constants")
    require(set(dispatch["resource_views"]) == {"0", "1", "2"}, "Unexpected descriptor tables")
    expected = [(753,424,10),(753,424,34),(1280,720,10),(753,424,41)]
    views = dispatch["resource_views"]["0"]
    require(len(views) == len(expected), "Unexpected external resources")
    for view, shape in zip(views, expected):
        desc = view["view"]["resource_description"]
        require((desc["width"], desc["height"], desc["format"]) == shape, "External resource profile changed")
    addresses = {}
    for number, line in enumerate((trace / "vkd3d.log").read_text().splitlines(), 1):
        if number >= dispatch["line"]:
            break
        match = re.search(r"GetGPUVirtualAddress: iface (\w+), va #([0-9a-f]+)", line)
        if match:
            addresses[int(match[2], 16)] = match[1]
    copies = copied_buffers(trace, dispatch["line"])
    shader = shaders / (dispatch["shader"] + ".spv")
    mapping = json.loads(shader.with_suffix(".json").read_text())
    require(hashlib.sha256(shader.read_bytes()).hexdigest() == mapping["output_sha256"], "Shader changed")
    require([b["new_binding"] for b in mapping["bindings"]] == list(range(5)), "Unexpected binding map")
    output.mkdir(parents=True, exist_ok=False)
    buffers = []
    for index, descriptor in enumerate(dispatch["resource_views"]["2"]):
        view = descriptor["view"]
        require(descriptor["register"] == index and view["kind"] == "CBV", "Unexpected CBV order")
        rid = addresses[int(view["address"], 16)]
        writes = copies[rid]
        known_size = max(w["offset"] + len(w["bytes"]) for w in writes)
        require(known_size <= view["size"] <= 65536, "Invalid CBV bounds")
        known = materialize(known_size, writes)
        blob = known + bytes([padding_byte]) * (view["size"] - known_size)
        path = output / f"cbv-{index}.bin"
        path.write_bytes(blob)
        buffers.append({"file": str(path.resolve()), "captured_bytes": known_size,
            "view_bytes": view["size"], "uncaptured_padding_bytes": view["size"] - known_size,
            "padding_fill": padding_byte, "sha256": hashlib.sha256(blob).hexdigest(), "resource": rid})
    push = [int(constants[str(i)], 16) for i in range(20)] + [0, 4, 9]
    require([b["captured_bytes"] for b in buffers] == [320, 80, 80], "Unexpected model footprint")
    require(dispatch["groups"] == [160, 90, 1], "Unexpected dispatch shape")
    result = {"schema": "xess.native-resolve-replay", "version": 1,
        "scope": "fixed 1280x720 checkerboard resolve; padding is an explicit experiment", "frame": frame,
        "shader": str(shader.resolve()), "shader_sha256": mapping["output_sha256"],
        "push": push, "groups": dispatch["groups"], "buffers": buffers,
        "reference_match": "unverified"}
    (output / "push.bin").write_bytes(struct.pack("<23I", *push))
    (output / "manifest.json").write_text(json.dumps(result, indent=2) + "\n")
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("trace", type=Path)
    parser.add_argument("normalized_shaders", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--padding-byte", type=lambda s: int(s, 0), default=0)
    parser.add_argument("--frame", type=int, default=0)
    args = parser.parse_args()
    result = build(args.trace, args.normalized_shaders, args.output, args.padding_byte, args.frame)
    print("Prepared resolve with 480 captured bytes and", sum(b["uncaptured_padding_bytes"] for b in result["buffers"]), "explicit padding bytes")
