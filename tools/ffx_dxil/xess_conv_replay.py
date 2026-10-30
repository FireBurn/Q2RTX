#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""Build a local native replay manifest for the captured XeSS convolution stack.

Consumes the serialized reference probe's trace, not arbitrary concurrent
captures. Includes passes 1..12; prepare and final resolve remain separate.
Extracted model/shader payloads stay in caller-selected ignored directories.
"""
import argparse
import hashlib
import json
from pathlib import Path
import re
import struct


def require(condition, message):
    if not condition:
        raise ValueError(message)


def copied_buffers(trace, before):
    uploads, copies = {}, {}
    for number, line in enumerate((trace / "vkd3d.log").read_text().splitlines(), 1):
        if number >= before:
            break
        match = re.search(r"REFERENCE_UPLOAD resource=(\w+) va=(\w+) size=(\d+) file=(.+)", line)
        if match:
            resource, _, size, name = match.groups()
            path = trace / "uploads" / name.replace("\\", "/").split("/")[-1]
            data = path.read_bytes()
            require(len(data) == int(size), "Upload size mismatch")
            uploads[resource] = (data, number, path.name)
        match = re.search(r"CopyBufferRegion: iface \w+, dst_resource (\w+), dst_offset (\w+), src_resource (\w+), src_offset (\w+), byte_count (\w+)", line)
        if match:
            destination, dst_offset, source, src_offset, size = match.groups()
            dst_offset, src_offset, size = (int(v, 0) for v in (dst_offset, src_offset, size))
            require(source in uploads, "Missing copied upload " + source)
            data, upload_line, name = uploads[source]
            require(src_offset + size <= len(data), "Copy exceeds upload")
            copies.setdefault(destination, []).append({
                "offset": dst_offset, "bytes": data[src_offset:src_offset + size],
                "copy_line": number, "upload_line": upload_line, "upload_file": name,
            })
    return copies


def materialize(size, copies):
    require(0 < size <= 16 * 1024 * 1024, "Unbounded model buffer")
    data, covered = bytearray(size), bytearray(size)
    for copy in copies:
        begin, payload = copy["offset"], copy["bytes"]
        end = begin + len(payload)
        require(0 <= begin <= end <= size, "Copy exceeds resource")
        data[begin:end] = payload
        covered[begin:end] = b"\1" * len(payload)
    require(all(covered), "Uncaptured model bytes; refusing implicit zero padding")
    return bytes(data)


def verify_shader(data, mapping):
    require(hashlib.sha256(data).hexdigest() == mapping["output_sha256"], "Shader hash mismatch")
    require(len(data) >= 20 and len(data) % 4 == 0, "Invalid SPIR-V size")
    words = struct.unpack(f"<{len(data)//4}I", data)
    require(words[0] == 0x07230203, "Not SPIR-V")
    types, variables, decorations = {}, {}, {}
    position = 5
    while position < len(words):
        size, op = words[position] >> 16, words[position] & 65535
        require(size > 0 and position + size <= len(words), "Invalid instruction")
        args = words[position + 1:position + size]
        if 19 <= op <= 39:
            types[args[0]] = (op, args[1:])
        elif op == 59:
            variables[args[1]] = (args[0], args[2])
        elif op == 71 and args[1] in (33, 34):
            decorations.setdefault(args[0], {})[args[1]] = args[2]
        position += size
    expected = ["sampled_uint_array", "storage_buffer", "storage_uint_array"]
    actual = [None] * 3
    for variable, deco in decorations.items():
        require(deco.get(34) == 0 and deco.get(33, 99) < 3, "Unexpected descriptor layout")
        pointer, storage = variables[variable]
        op, pointer_args = types[pointer]
        require(op == 32, "Expected descriptor pointer")
        op, array_args = types[pointer_args[1]]
        require(op == 29, "Expected runtime descriptor array")
        op, element = types[array_args[0]]
        if storage == 12 and op == 30:
            kind = "storage_buffer"
        else:
            require(storage == 0 and op == 25, "Unexpected descriptor element")
            require(types[element[0]] == (21, (32, 0)), "Expected unsigned image")
            require(element[1:5] == (1, 0, 1, 0), "Expected non-MS 2D array image")
            kind = {1: "sampled_uint_array", 2: "storage_uint_array"}[element[5]]
        require(actual[deco[33]] is None, "Aliased bindings")
        actual[deco[33]] = kind
    require(actual == expected, "Unexpected convolution shader bindings")
    pushes = [types[types[pointer][1][1]] for pointer, storage in variables.values() if storage == 9]
    require(len(pushes) == 1 and pushes[0][0] == 30 and len(pushes[0][1]) == 6,
            "Expected six-word push constants")
    require(all(types[t] == (21, (32, 0)) for t in pushes[0][1]), "Unexpected push type")


def build(trace, shaders, output):
    dispatches = json.loads((trace / "dispatch-index.json").read_text())["dispatches"]
    require(len(dispatches) >= 14, "Incomplete first frame")
    selected = dispatches[1:13]
    copies = copied_buffers(trace, selected[0]["line"])
    output.mkdir(parents=True, exist_ok=False)
    images, buffers, passes = {}, {}, []
    for dispatch in selected:
        shader = shaders / (dispatch["shader"] + ".spv")
        mapping = json.loads(shader.with_suffix(".json").read_text())
        verify_shader(shader.read_bytes(), mapping)
        require(set(dispatch["resource_views"]) == {"0", "1"}, "Unexpected root tables")
        require(set(dispatch["constants"]) == {"2"}, "Unexpected constants")
        constants = dispatch["constants"]["2"]
        require(set(constants) == {"0", "1", "2", "3"}, "Missing push words")
        entry = {"shader": str(shader.resolve()), "shader_sha256": mapping["output_sha256"],
                 "groups": dispatch["groups"], "push": [int(constants[str(i)], 16) for i in range(4)] + [0, 3],
                 "images": [], "buffers": [], "trace_line": dispatch["line"]}
        for table in ("0", "1"):
            views = dispatch["resource_views"][table]
            require(len(views) == 3, "Unexpected descriptor table size")
            base = int(dispatch["table_bases"][table]["cpu"], 16)
            stride = dispatch["table_bases"][table]["stride"]
            slots = {}
            for view in views:
                delta = int(view["cpu"], 16) - base
                require(delta >= 0 and delta % stride == 0, "Misaligned descriptor")
                slots[delta // stride] = view["view"]
            require(set(slots) == {0, 1, 2}, "Missing descriptor slot")
            for slot in range(3):
                view = slots[slot]
                rid, desc = view["resource"], view["resource_description"]
                require(desc["mips"] == 1 and desc["samples"] == 1, "Unsupported resource shape")
                if table == "0":
                    require(desc["dimension"] == 3 and desc["format"] == 3, "Expected RGBA32UI image")
                    require(view["kind"] == ("UAV" if slot == 2 else "SRV"), "Unexpected image access")
                    shape = [desc["width"], desc["height"], desc["depth_or_layers"]]
                    require(rid not in images or images[rid] == shape, "Resource identity reused")
                    images[rid] = shape
                    entry["images"].append(rid)
                else:
                    require(desc["dimension"] == 1 and view["kind"] == "SRV", "Expected model SSBO")
                    words = view["words"]
                    first = int(words["16"], 16) | int(words["20"], 16) << 32
                    length = int(words["24"], 16) * int(words["28"], 16)
                    require(first == 0 and length == desc["width"], "Partial model view not supported")
                    if rid not in buffers:
                        resource_copies = [c for c in copies.get(rid, [])
                            if c["copy_line"] >= desc["declaration_line"]]
                        payload = materialize(desc["width"], resource_copies)
                        path = output / (rid + ".bin")
                        path.write_bytes(payload)
                        buffers[rid] = {"file": str(path.resolve()), "size": len(payload),
                            "sha256": hashlib.sha256(payload).hexdigest(),
                            "copies": [{k: v for k, v in c.items() if k != "bytes"} for c in resource_copies]}
                    entry["buffers"].append(rid)
        passes.append(entry)
    produced = {passes[0]["images"][0]}
    for entry in passes:
        require(all(rid in produced for rid in entry["images"][:2]), "Uninitialized convolution input")
        require(entry["images"][2] not in produced, "In-place convolution not supported")
        produced.add(entry["images"][2])
    manifest = {"schema": "xess.native-convolution-replay", "version": 1,
        "scope": "first-frame passes 1..12 only; excludes prepare/final resolve and history",
        "reference_match": "unverified", "images": images, "buffers": buffers, "passes": passes,
        "input": passes[0]["images"][0], "output": passes[-1]["images"][2]}
    (output / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    return manifest


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("trace", type=Path)
    parser.add_argument("normalized_shaders", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    result = build(args.trace, args.normalized_shaders, args.output)
    print(f"Prepared {len(result['passes'])} convolution passes, {len(result['buffers'])} fully captured model buffers")
