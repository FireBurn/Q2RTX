#!/usr/bin/env python3
"""Reject SPIR-V built against a stale Q2RTX global-image table.

`global_textures.h` places sampled framebuffer textures immediately after the
storage-image table.  Adding an image therefore shifts every TEX_* binding.
The host and shader table must always be rebuilt as one ABI.
"""

from __future__ import annotations

import argparse
from pathlib import Path
import re
import struct
import sys


OP_NAME = 5
OP_DECORATE = 71
DECORATION_BINDING = 33
DECORATION_DESCRIPTOR_SET = 34


def macro_block(source: str, name: str, end_marker: str) -> str:
    start = source.find(f"#define {name}")
    end = source.find(end_marker, start)
    if start < 0 or end < 0:
        raise ValueError(f"could not find {name} in global-texture header")
    return source[start:end]


def parse_binding(expression: str, image_base: int) -> int:
    expression = expression.strip()
    if not re.fullmatch(r"[0-9A-Z_+() \t]+", expression):
        raise ValueError(f"unsafe image-binding expression: {expression!r}")
    try:
        value = eval(expression, {"__builtins__": {}},
                     {"NUM_IMAGES_BASE": image_base})
    except (NameError, SyntaxError) as error:
        raise ValueError(f"invalid image-binding expression {expression!r}") from error
    if not isinstance(value, int) or value < 0:
        raise ValueError(f"invalid image-binding value {expression!r}")
    return value


def parse_image_bindings(header: Path) -> dict[str, int]:
    source = header.read_text(encoding="utf-8")
    base_match = re.search(r"^#define\s+NUM_IMAGES_BASE\s+(\d+)", source,
                           re.MULTILINE)
    if not base_match:
        raise ValueError("NUM_IMAGES_BASE is missing")
    image_base = int(base_match.group(1))
    blocks = (
        macro_block(source, "LIST_IMAGES", "#define NUM_IMAGES_BASE"),
        macro_block(source, "LIST_IMAGES_A_B", "#define LIST_IMAGES_B_A"),
    )
    bindings: dict[str, int] = {}
    for block in blocks:
        for name, expression in re.findall(
                r"IMG_DO\(\s*([A-Z0-9_]+)\s*,\s*([^,]+),", block):
            binding = parse_binding(expression, image_base)
            if name in bindings:
                raise ValueError(f"duplicate global image name: {name}")
            bindings[name] = binding

    expected_count = image_base + 30
    if len(bindings) != expected_count or set(bindings.values()) != set(range(expected_count)):
        raise ValueError("global-image bindings are not a contiguous ABI table")
    return bindings


def decode_string(words: tuple[int, ...]) -> str:
    raw = b"".join(struct.pack("<I", word) for word in words)
    return raw.split(b"\0", 1)[0].decode("utf-8", errors="strict")


def reflect_bindings(module: Path) -> dict[str, tuple[int, int]]:
    payload = module.read_bytes()
    if len(payload) < 20 or len(payload) % 4:
        raise ValueError(f"invalid SPIR-V module: {module}")
    words = struct.unpack(f"<{len(payload) // 4}I", payload)
    if words[0] != 0x07230203:
        raise ValueError(f"not SPIR-V: {module}")

    names: dict[int, str] = {}
    decorations: dict[int, dict[int, int]] = {}
    index = 5
    while index < len(words):
        instruction = words[index]
        word_count = instruction >> 16
        opcode = instruction & 0xffff
        if word_count == 0 or index + word_count > len(words):
            raise ValueError(f"malformed SPIR-V instruction in {module}")
        operands = words[index + 1:index + word_count]
        if opcode == OP_NAME and len(operands) >= 2:
            names[operands[0]] = decode_string(operands[1:])
        elif opcode == OP_DECORATE and len(operands) >= 3:
            target, decoration, value = operands[:3]
            decorations.setdefault(target, {})[decoration] = value
        index += word_count

    result: dict[str, tuple[int, int]] = {}
    for target, name in names.items():
        if target not in decorations:
            continue
        decoration = decorations[target]
        if DECORATION_BINDING in decoration and DECORATION_DESCRIPTOR_SET in decoration:
            result[name] = (decoration[DECORATION_DESCRIPTOR_SET],
                            decoration[DECORATION_BINDING])
    return result


def verify(header: Path, shader_dir: Path) -> tuple[int, int]:
    images = parse_image_bindings(header)
    expected: dict[str, int] = {}
    image_count = len(images)
    for name, binding in images.items():
        expected[f"IMG_{name}"] = 1 + binding
        expected[f"TEX_{name}"] = 1 + image_count + binding

    modules = sorted(shader_dir.glob("*.spv"))
    if not modules:
        raise ValueError(f"no compiled Q2RTX shaders in {shader_dir}")

    checked = 0
    for module in modules:
        module_set: int | None = None
        for name, (descriptor_set, binding) in reflect_bindings(module).items():
            required = expected.get(name)
            if required is None:
                continue
            checked += 1
            if module_set is None:
                module_set = descriptor_set
            if descriptor_set != module_set or binding != required:
                raise ValueError(
                    f"{module.name}: {name} has set={descriptor_set}, binding={binding}; "
                    f"expected set={module_set}, binding={required}. "
                    "Rebuild shader_vkpt after changing global_textures.h.")
    if not checked:
        raise ValueError("no global image bindings were reflected from shader modules")
    return len(modules), checked


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("texture_header", type=Path)
    parser.add_argument("shader_dir", type=Path)
    args = parser.parse_args()
    try:
        modules, bindings = verify(args.texture_header, args.shader_dir)
    except (OSError, UnicodeError, ValueError) as error:
        print(f"Global-texture shader ABI verification failed: {error}", file=sys.stderr)
        return 1
    print(f"Global-texture shader ABI verification passed: {modules} modules, "
          f"{bindings} bindings")
    return 0


if __name__ == "__main__":
    sys.exit(main())
