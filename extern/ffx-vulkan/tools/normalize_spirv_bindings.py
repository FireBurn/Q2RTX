#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""Give each SPIR-V descriptor variable its own native Vulkan binding.

Preserves array indices, push constants, types and executable instructions.
The caller must populate the new bindings using the emitted mapping.
This does not supply resource contents or a runnable effect graph.
"""
import argparse
import hashlib
import json
from pathlib import Path
import struct
import subprocess
import tempfile


def instructions(data):
    if len(data) < 20 or len(data) % 4:
        raise ValueError("Invalid SPIR-V size")
    words = list(struct.unpack(f"<{len(data) // 4}I", data))
    if words[0] != 0x07230203:
        raise ValueError("Expected little-endian SPIR-V")
    result, offset = [], 5
    while offset < len(words):
        count, opcode = words[offset] >> 16, words[offset] & 0xffff
        if not count or count > len(words) - offset:
            raise ValueError("Truncated or zero-length SPIR-V instruction")
        result.append((opcode, words[offset:offset + count]))
        offset += count
    return words[:5], result


def normalize(data):
    header, ops = instructions(data)
    decorations, variables = {}, set()
    for opcode, words in ops:
        if opcode in (73, 74, 75):
            raise ValueError("Grouped decorations are not supported")
        if opcode == 59:  # OpVariable
            if len(words) not in (4, 5):
                raise ValueError("Malformed variable instruction")
            variables.add(words[2])
        if opcode == 71 and len(words) < 3:
            raise ValueError("Malformed decoration instruction")
        if opcode == 71 and len(words) >= 3 and words[2] in (33, 34):
            if len(words) != 4:
                raise ValueError("Malformed descriptor decoration")
            target = decorations.setdefault(words[1], {})
            if words[2] in target:
                raise ValueError("Duplicate descriptor decoration")
            target[words[2]] = words[3]
    if not decorations:
        raise ValueError("No descriptor decorations found")
    if any(set(d) != {33, 34} or identity not in variables for identity, d in decorations.items()):
        raise ValueError("Expected set and binding decorations on descriptor variables")
    mapping = [dict(variable_id=identity, old_set=decorations[identity][34],
        old_binding=decorations[identity][33], new_set=0, new_binding=index)
        for index, identity in enumerate(sorted(decorations))]
    bindings = {m["variable_id"]:m["new_binding"] for m in mapping}
    output = header[:]
    for opcode, words in ops:
        words = words[:]
        if opcode == 71 and words[1] in bindings and words[2] in (33, 34):
            words[3] = bindings[words[1]] if words[2] == 33 else 0
        output.extend(words)
    return struct.pack(f"<{len(output)}I", *output), mapping


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--manifest", required=True, type=Path)
    parser.add_argument("--validator", default="spirv-val")
    parser.add_argument("--uniform-buffer-standard-layout", action="store_true")
    args = parser.parse_args()
    if args.output.exists() or args.manifest.exists():
        parser.error("Output and manifest must not already exist")
    data = args.input.read_bytes()
    converted, mapping = normalize(data)
    command = [args.validator, "--target-env", "vulkan1.3"]
    if args.uniform_buffer_standard_layout:
        command.append("--uniform-buffer-standard-layout")
    subprocess.run(command + [str(args.input)], check=True)
    with tempfile.TemporaryDirectory() as temporary:
        path = Path(temporary) / "converted.spv"
        path.write_bytes(converted)
        subprocess.run(command + [str(path)], check=True)
    manifest = dict(schema="vulkan.descriptor-binding-map", schema_version=1,
        source_sha256=hashlib.sha256(data).hexdigest(), output_sha256=hashlib.sha256(converted).hexdigest(),
        bindings=mapping, runtime_arrays_preserved=True, executable_instructions_unchanged=True,
        uniform_buffer_standard_layout=args.uniform_buffer_standard_layout,
        execution_verified=False)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.manifest.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("xb") as f:
        f.write(converted)
    with args.manifest.open("x") as f:
        json.dump(manifest, f, indent=2)
        f.write("\n")


if __name__ == "__main__":
    main()
