#!/usr/bin/env python3
"""Tool for parsing, validating, and bundling XeSS INT8 neural upscaler models for Vulkan."""
import argparse
import hashlib
import json
from pathlib import Path
import struct
import sys

# XeSS 2.0.2 / 3.0 U-Net topology: 14 dispatches, 13 neural weight layers.
XESS_UNET_LAYERS = [
    {"name": "XeSS_i8", "dispatch": 13, "weights_bytes": 320, "scale_bytes": 80, "bias_bytes": 80},
    {"name": "XeSS_i10", "dispatch": 1, "weights_bytes": 2304, "scale_bytes": 64, "bias_bytes": 64},
    {"name": "XeSS_i12", "dispatch": 2, "weights_bytes": 4608, "scale_bytes": 128, "bias_bytes": 128},
    {"name": "XeSS_i13", "dispatch": 3, "weights_bytes": 18432, "scale_bytes": 256, "bias_bytes": 256},
    {"name": "XeSS_i14", "dispatch": 4, "weights_bytes": 73728, "scale_bytes": 512, "bias_bytes": 512},
    {"name": "XeSS_i15", "dispatch": 5, "weights_bytes": 73728, "scale_bytes": 256, "bias_bytes": 256},
    {"name": "XeSS_i16", "dispatch": 6, "weights_bytes": 36864, "scale_bytes": 256, "bias_bytes": 256},
    {"name": "XeSS_i17", "dispatch": 7, "weights_bytes": 18432, "scale_bytes": 128, "bias_bytes": 128},
    {"name": "XeSS_i18", "dispatch": 8, "weights_bytes": 9216, "scale_bytes": 128, "bias_bytes": 128},
    {"name": "XeSS_i19", "dispatch": 9, "weights_bytes": 4608, "scale_bytes": 64, "bias_bytes": 64},
    {"name": "XeSS_i20", "dispatch": 10, "weights_bytes": 2304, "scale_bytes": 64, "bias_bytes": 64},
    {"name": "XeSS_i21", "dispatch": 11, "weights_bytes": 2304, "scale_bytes": 64, "bias_bytes": 64},
    {"name": "XeSS_i23", "dispatch": 12, "weights_bytes": 2304, "scale_bytes": 64, "bias_bytes": 64},
]

XESS_DISPATCH_TOPOLOGY = [
    {"dispatch": 0, "name": "prepare", "groups": [80, 23, 1], "in_res": "753x424", "out_res": "640x360x1"},
    {"dispatch": 1, "name": "down_1", "groups": [20, 12, 1], "in_res": "640x360x1", "out_res": "320x180x1"},
    {"dispatch": 2, "name": "down_2", "groups": [40, 6, 1], "in_res": "320x180x1", "out_res": "160x90x2"},
    {"dispatch": 3, "name": "down_3", "groups": [20, 6, 1], "in_res": "160x90x2", "out_res": "80x45x4"},
    {"dispatch": 4, "name": "down_4", "groups": [5, 6, 1], "in_res": "80x45x4", "out_res": "40x23x8"},
    {"dispatch": 5, "name": "bottleneck", "groups": [3, 2, 1], "in_res": "40x23x8", "out_res": "40x23x4"},
    {"dispatch": 6, "name": "up_4_skip_3", "groups": [10, 3, 1], "in_res": "40x23x4+80x45x4", "out_res": "80x45x4"},
    {"dispatch": 7, "name": "up_3a", "groups": [5, 6, 1], "in_res": "80x45x4", "out_res": "80x45x2"},
    {"dispatch": 8, "name": "up_3_skip_2", "groups": [5, 12, 1], "in_res": "80x45x2+160x90x2", "out_res": "160x90x2"},
    {"dispatch": 9, "name": "up_2a", "groups": [20, 12, 1], "in_res": "160x90x2", "out_res": "160x90x1"},
    {"dispatch": 10, "name": "up_2_skip_1", "groups": [20, 6, 1], "in_res": "160x90x1+320x180x1", "out_res": "320x180x1"},
    {"dispatch": 11, "name": "up_1a", "groups": [40, 23, 1], "in_res": "320x180x1", "out_res": "320x180x1"},
    {"dispatch": 12, "name": "up_1_skip_0", "groups": [40, 12, 1], "in_res": "320x180x1+640x360x1", "out_res": "640x360x1"},
    {"dispatch": 13, "name": "resolve", "groups": [160, 90, 1], "in_res": "640x360x1+inputs", "out_res": "1280x720x1"},
]

MAGIC = b"XESSMOD2"  # XeSS Model Container v2


def validate_unet_topology():
    """Verify topological integrity of the XeSS U-Net pipeline."""
    assert len(XESS_DISPATCH_TOPOLOGY) == 14
    assert len(XESS_UNET_LAYERS) == 13
    total_weights = sum(l["weights_bytes"] for l in XESS_UNET_LAYERS)
    total_scale = sum(l["scale_bytes"] for l in XESS_UNET_LAYERS)
    total_bias = sum(l["bias_bytes"] for l in XESS_UNET_LAYERS)
    total_bytes = total_weights + total_scale + total_bias
    assert total_bytes == 253280, f"Expected 253280 total weight bytes, got {total_bytes}"
    return {
        "dispatches": 14,
        "layers": 13,
        "weights_bytes": total_weights,
        "scale_bytes": total_scale,
        "bias_bytes": total_bias,
        "total_bytes": total_bytes
    }


def pack_model(layer_data_map):
    """Pack verified layer data into an aligned standalone binary bundle.

    layer_data_map is a dict of:
      { layer_name: {"weights": bytes, "scale": bytes, "bias": bytes} }
    """
    header = bytearray(MAGIC)
    num_layers = len(XESS_UNET_LAYERS)
    header += struct.pack("<II", 2, num_layers)  # version=2, layer_count=13

    # Directory entries: 32 bytes per layer:
    #   name (16 bytes, ascii padded with 0)
    #   dispatch_id (u16), pad (u16)
    #   weights_offset (u32), weights_size (u32)
    #   scale_offset (u32), scale_size (u32)
    #   bias_offset (u32), bias_size (u32)
    dir_entry_size = 44
    data_start = len(header) + num_layers * dir_entry_size
    # Align data_start to 256 bytes for Vulkan uniform/storage buffer alignment
    data_start = (data_start + 255) & ~255

    dir_entries = bytearray()
    payload = bytearray()
    current_offset = data_start

    for l in XESS_UNET_LAYERS:
        name = l["name"]
        if name not in layer_data_map:
            raise ValueError(f"Missing required layer '{name}' in layer_data_map")
        ld = layer_data_map[name]
        wb, sb, bb = ld["weights"], ld["scale"], ld["bias"]
        if len(wb) != l["weights_bytes"]:
            raise ValueError(f"Layer {name} weights size {len(wb)} != expected {l['weights_bytes']}")
        if len(sb) != l["scale_bytes"]:
            raise ValueError(f"Layer {name} scale size {len(sb)} != expected {l['scale_bytes']}")
        if len(bb) != l["bias_bytes"]:
            raise ValueError(f"Layer {name} bias size {len(bb)} != expected {l['bias_bytes']}")

        # Weights chunk
        w_off = current_offset
        payload += wb
        current_offset += len(wb)
        # Pad to 64 bytes
        w_pad = (64 - (current_offset % 64)) % 64
        payload += b"\x00" * w_pad
        current_offset += w_pad

        # Scale chunk
        s_off = current_offset
        payload += sb
        current_offset += len(sb)
        s_pad = (64 - (current_offset % 64)) % 64
        payload += b"\x00" * s_pad
        current_offset += s_pad

        # Bias chunk
        b_off = current_offset
        payload += bb
        current_offset += len(bb)
        b_pad = (64 - (current_offset % 64)) % 64
        payload += b"\x00" * b_pad
        current_offset += b_pad

        name_encoded = name.encode("ascii")[:15].ljust(16, b"\x00")
        dir_entries += struct.pack(
            "<16sHHIIIIII",
            name_encoded,
            l["dispatch"],
            0,
            w_off,
            len(wb),
            s_off,
            len(sb),
            b_off,
            len(bb)
        )

    # Pad header+dir to data_start
    header_block = header + dir_entries
    header_pad = data_start - len(header_block)
    header_block += b"\x00" * header_pad

    return bytes(header_block + payload)


def unpack_model(data):
    """Unpack and validate a standalone XeSS binary bundle."""
    if len(data) < len(MAGIC) + 8:
        raise ValueError("Data too short for XeSS model header")
    if data[:8] != MAGIC:
        raise ValueError(f"Invalid magic: {data[:8]!r}")
    version, num_layers = struct.unpack_from("<II", data, 8)
    if version != 2:
        raise ValueError(f"Unsupported model version: {version}")
    if num_layers != len(XESS_UNET_LAYERS):
        raise ValueError(f"Unexpected layer count {num_layers} != {len(XESS_UNET_LAYERS)}")

    dir_entry_size = 44
    offset = 16
    layers = {}
    for i in range(num_layers):
        name_bytes, dispatch_id, _, w_off, w_sz, s_off, s_sz, b_off, b_sz = struct.unpack_from(
            "<16sHHIIIIII", data, offset
        )
        offset += dir_entry_size
        name = name_bytes.rstrip(b"\x00").decode("ascii")
        expected_meta = XESS_UNET_LAYERS[i]
        if name != expected_meta["name"]:
            raise ValueError(f"Layer {i} name '{name}' != expected '{expected_meta['name']}'")
        if dispatch_id != expected_meta["dispatch"]:
            raise ValueError(f"Layer {name} dispatch {dispatch_id} != expected {expected_meta['dispatch']}")
        if w_sz != expected_meta["weights_bytes"]:
            raise ValueError(f"Layer {name} weights size {w_sz} != expected {expected_meta['weights_bytes']}")
        if s_sz != expected_meta["scale_bytes"]:
            raise ValueError(f"Layer {name} scale size {s_sz} != expected {expected_meta['scale_bytes']}")
        if b_sz != expected_meta["bias_bytes"]:
            raise ValueError(f"Layer {name} bias size {b_sz} != expected {expected_meta['bias_bytes']}")

        wb = data[w_off : w_off + w_sz]
        sb = data[s_off : s_off + s_sz]
        bb = data[b_off : b_off + b_sz]
        if len(wb) != w_sz or len(sb) != s_sz or len(bb) != b_sz:
            raise ValueError(f"Layer {name} payload truncated in file")

        layers[name] = {
            "dispatch": dispatch_id,
            "weights": wb,
            "scale": sb,
            "bias": bb
        }

    return layers


def extract_from_provenance(provenance_path, uploads_dir):
    """Extract all 13 layers from a trace upload provenance record."""
    with open(provenance_path, "r") as f:
        data = json.load(f)

    layer_map = {}
    for entry in data.get("copies", []):
        dst = entry.get("destination_name")
        if not dst or not dst.startswith("XeSS_"):
            continue
        parts = dst.split("_")
        if len(parts) != 3:
            continue
        layer_name = f"{parts[0]}_{parts[1]}"
        component = parts[2]  # weights, scale, bias
        if layer_name not in layer_map:
            layer_map[layer_name] = {}
        src_file = uploads_dir / entry["file"]
        if not src_file.exists():
            raise FileNotFoundError(f"Missing upload file: {src_file}")
        with open(src_file, "rb") as f:
            src_bytes = f.read()
        so = entry["source_offset"]
        sz = entry["size"]
        layer_map[layer_name][component] = src_bytes[so : so + sz]

    return layer_map


def main():
    parser = argparse.ArgumentParser(description="XeSS U-Net neural model tool for Vulkan.")
    subparsers = parser.add_subparsers(dest="command")

    topo_p = subparsers.add_parser("topology", help="Print verified U-Net topology summary.")

    pack_p = subparsers.add_parser("pack", help="Pack layer files or provenance into binary bundle.")
    pack_p.add_argument("--provenance", type=Path, required=True, help="Path to upload-copy-provenance.json")
    pack_p.add_argument("--uploads", type=Path, required=True, help="Path to uploads directory")
    pack_p.add_argument("--out", type=Path, required=True, help="Output .bin file")

    inspect_p = subparsers.add_parser("inspect", help="Inspect binary model bundle.")
    inspect_p.add_argument("model_bin", type=Path, help="Path to packed .bin")

    args = parser.parse_args()
    if args.command == "topology":
        summary = validate_unet_topology()
        print(json.dumps(summary, indent=2))
        for t in XESS_DISPATCH_TOPOLOGY:
            print(f"Dispatch {t['dispatch']:02d}: {t['name']:15s} groups={t['groups']} {t['in_res']:25s} -> {t['out_res']}")
    elif args.command == "pack":
        layer_data = extract_from_provenance(args.provenance, args.uploads)
        packed = pack_model(layer_data)
        with open(args.out, "wb") as f:
            f.write(packed)
        print(f"Successfully packed {len(layer_data)} layers ({len(packed)} bytes) into {args.out}")
    elif args.command == "inspect":
        with open(args.model_bin, "rb") as f:
            data = f.read()
        layers = unpack_model(data)
        print(f"XeSS Model Container: {len(layers)} layers, total payload {len(data)} bytes")
        for name, l in layers.items():
            print(f"  {name:12s} (dispatch {l['dispatch']:02d}): weights={len(l['weights'])} B, scale={len(l['scale'])} B, bias={len(l['bias'])} B")
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
