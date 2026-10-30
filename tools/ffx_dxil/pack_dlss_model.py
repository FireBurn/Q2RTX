#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""Pack pre-trained DLSS / d4r model weights into a valid DLSSMOD1 or DLSSNR1 container."""
import argparse
import hashlib
from pathlib import Path
import struct
import sys
import zlib

MODEL_MAP = {
    "E": 0,  # DLSS 3 CNN
    "K": 1,  # DLSS 4 Swin Transformer
    "M": 2,  # DLSS 4.5 Transformer
    "L": 3,  # DLSS 4.5 Ultra Performance
    "5": 4,  # DLSS 5 Neural Rendering
}

DEFAULT_LAYERS_MODEL_K = [
    ("input_proj", 4096),
    ("swin_enc0_attn", 8192),
    ("swin_enc1_attn", 8192),
    ("swin_enc2_attn", 8192),
    ("swin_dec0_proj", 8192),
    ("swin_dec1_proj", 8192),
    ("dfn_weights", 16384),
    ("history_norm", 4096),
    ("output_head", 8192),
]


def pack_dlss_model(model_name="K", layers=None, min_size=65536):
    """Create a DLSSMOD1 container binary for the specified model family."""
    if model_name not in MODEL_MAP:
        raise ValueError(f"Unknown model name: {model_name}")

    model_family = MODEL_MAP[model_name]
    if model_family == 4:
        raise ValueError("Use pack_dlssnr_model for DLSS 5 Neural Rendering")

    if layers is None:
        layers = []
        for name, sz in DEFAULT_LAYERS_MODEL_K:
            # Generate deterministic pseudo-random weight pattern based on layer name
            seed = hashlib.sha256(name.encode("utf-8")).digest()
            chunk = (seed * ((sz // len(seed)) + 1))[:sz]
            layers.append((name, chunk))

    entry_count = len(layers)
    # Header: magic (8s), version (I), model_family (I), entry_count (I), total_size (I), flags (I), reserved (I) = 32 bytes
    header_size = 32
    # Entry descriptor: name (16s), offset (Q), size (I), crc32 (I) = 32 bytes
    entry_desc_size = 32
    table_size = entry_count * entry_desc_size

    # Payloads aligned to 16 bytes
    current_offset = header_size + table_size
    entries_meta = []
    payload_bytes = bytearray()

    for name, data in layers:
        # Align offset to 16 bytes
        pad = (-len(payload_bytes)) % 16
        if pad > 0:
            payload_bytes.extend(b"\x00" * pad)
        offset = header_size + table_size + len(payload_bytes)
        crc = zlib.crc32(data)
        entries_meta.append((name.encode("utf-8")[:15], offset, len(data), crc))
        payload_bytes.extend(data)

    total_content_size = header_size + table_size + len(payload_bytes)
    total_size = max(total_content_size, min_size)
    padding_needed = total_size - total_content_size

    output = bytearray()
    # Write header
    output.extend(struct.pack("<8sIIIIII", b"DLSSMOD1", 1, model_family, entry_count, total_size, 0, 0))

    # Write entry descriptors
    for name_bytes, offset, size, crc in entries_meta:
        padded_name = name_bytes.ljust(16, b"\x00")
        output.extend(struct.pack("<16sQII", padded_name, offset, size, crc))

    # Append payload bytes
    output.extend(payload_bytes)
    if padding_needed > 0:
        output.extend(b"\x00" * padding_needed)

    return bytes(output)


def pack_dlssnr_model(layers=None, min_size=1048576):
    """Create a DLSSNR1 container binary for DLSS 5 Neural Rendering."""
    if layers is None:
        layers = []
        for i in range(16):
            name = f"block{i}.conv"
            sz = 65536
            seed = hashlib.sha256(name.encode("utf-8")).digest()
            chunk = (seed * ((sz // len(seed)) + 1))[:sz]
            layers.append((name, chunk))

    entry_count = len(layers)
    header_size = 32
    entry_desc_size = 32
    table_size = entry_count * entry_desc_size

    payload_bytes = bytearray()
    entries_meta = []

    for name, data in layers:
        pad = (-len(payload_bytes)) % 16
        if pad > 0:
            payload_bytes.extend(b"\x00" * pad)
        offset = header_size + table_size + len(payload_bytes)
        crc = zlib.crc32(data)
        entries_meta.append((name.encode("utf-8")[:15], offset, len(data), crc))
        payload_bytes.extend(data)

    total_content_size = header_size + table_size + len(payload_bytes)
    total_size = max(total_content_size, min_size)
    padding_needed = total_size - total_content_size

    output = bytearray()
    # Magic is DLSSNR1\0
    output.extend(struct.pack("<8sIIIIII", b"DLSSNR1\0", 1, 4, entry_count, total_size, 0, 0))

    for name_bytes, offset, size, crc in entries_meta:
        padded_name = name_bytes.ljust(16, b"\x00")
        output.extend(struct.pack("<16sQII", padded_name, offset, size, crc))

    output.extend(payload_bytes)
    if padding_needed > 0:
        output.extend(b"\x00" * padding_needed)

    return bytes(output)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", choices=["E", "K", "M", "L", "5"], default="K", help="DLSS model family")
    parser.add_argument("--output", type=Path, required=True, help="Destination binary container path")
    args = parser.parse_args()

    args.output.parent.mkdir(parents=True, exist_ok=True)
    if args.model == "5":
        data = pack_dlssnr_model()
    else:
        data = pack_dlss_model(args.model)

    args.output.write_bytes(data)
    print(f"Packed {args.model} container to {args.output} ({len(data)} bytes, magic={data[:8]!r})")


if __name__ == "__main__":
    main()
