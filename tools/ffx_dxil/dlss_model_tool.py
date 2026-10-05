#!/usr/bin/env python3
"""Tool for inspecting, validating, and scheduling DLSS / d4r and DLSS 5 models for Vulkan."""
import argparse
import hashlib
import json
from pathlib import Path
import struct
import sys

KNOWN_D4R_ARCHS = {
    "gfx1100", "gfx1101", "gfx1102", "gfx1103",
    "gfx1200", "gfx1201",
    "sm_80", "sm_89", "sm_120"
}

KNOWN_MODELS = {
    "E": "DLSS 3 CNN (Model E)",
    "K": "DLSS 4 Swin Transformer (Model K)",
    "M": "DLSS 4.5 Transformer (Model M)",
    "L": "DLSS 4.5 Ultra Performance (Model L)",
    "5": "DLSS 5 Neural Rendering (DLSSNR)"
}


def parse_d4r_manifest(lines):
    """Parse a d4r-kernels.txt manifest into structured records."""
    entries = []
    seen_hashes = set()
    for line_num, line in enumerate(lines, 1):
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        parts = line.split()
        if len(parts) < 3:
            raise ValueError(f"Line {line_num}: invalid d4r kernel manifest format: '{line}'")
        ptx_hash, kernel_name, arch = parts[0], parts[1], parts[2]
        flags = parts[3:] if len(parts) > 3 else []
        if len(ptx_hash) != 64 or not all(c in "0123456789abcdefABCDEF" for c in ptx_hash):
            raise ValueError(f"Line {line_num}: invalid 64-character SHA256 hex hash: '{ptx_hash}'")
        ptx_hash = ptx_hash.lower()
        if arch not in KNOWN_D4R_ARCHS:
            raise ValueError(f"Line {line_num}: unknown architecture '{arch}'")
        key = (ptx_hash, arch)
        if key in seen_hashes:
            raise ValueError(f"Line {line_num}: duplicate entry for hash {ptx_hash} and arch {arch}")
        seen_hashes.add(key)
        entries.append({
            "ptx_hash": ptx_hash,
            "kernel_name": kernel_name,
            "gpu_arch": arch,
            "flags": flags
        })
    return entries


def decode_dlss_nr_weights(data):
    """Bounds-check and index the DLSS 5 WEIGHTS_HT resource serialization."""
    offset = 0

    def u64():
        nonlocal offset
        if offset + 8 > len(data):
            raise ValueError("Truncated 64-bit integer in weights serialization")
        val = struct.unpack_from("<Q", data, offset)[0]
        offset += 8
        return val

    total_len = u64()
    if total_len != len(data):
        raise ValueError(f"Resource length mismatch: header says {total_len}, got {len(data)}")

    blobs, names = [], set()
    layer_counts = {}
    while offset < len(data):
        name_len = u64()
        if not (0 < name_len <= 4096) or name_len > len(data) - offset:
            raise ValueError(f"Invalid layer name length: {name_len}")
        name = data[offset:offset + name_len].decode("utf-8", errors="strict")
        offset += name_len
        if name in names or "\0" in name:
            raise ValueError(f"Duplicate or malformed layer name: '{name}'")
        names.add(name)

        blob_size = u64()
        if blob_size == 0 or blob_size > len(data) - offset:
            raise ValueError(f"Empty or truncated blob for layer '{name}'")

        blob_data = data[offset:offset + blob_size]
        blob_sha = hashlib.sha256(blob_data).hexdigest()

        # Classify layer prefix (e.g. block0, encoder, head)
        prefix = name.split(".")[0] if "." in name else "root"
        layer_counts[prefix] = layer_counts.get(prefix, 0) + 1

        blobs.append({
            "name": name,
            "offset": offset,
            "size": blob_size,
            "sha256": blob_sha,
            "prefix": prefix
        })
        offset += blob_size

    return {
        "total_size": len(data),
        "blob_count": len(blobs),
        "blobs": blobs,
        "layer_prefixes": layer_counts
    }


def generate_vulkan_schedule(model_id, render_w, render_h, display_w, display_h, gpu_arch):
    """Generate a Vulkan allocation and descriptor binding schedule for the model."""
    if model_id not in KNOWN_MODELS:
        raise ValueError(f"Unknown DLSS model ID: {model_id}")

    is_neural = (model_id == "5")
    # Base bindings for DLSS super resolution
    bindings = [
        {"binding": 0, "name": "ColorInput", "type": "VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE", "format": "R16G16B16A16_SFLOAT", "extent": [render_w, render_h]},
        {"binding": 1, "name": "DepthInput", "type": "VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE", "format": "D32_SFLOAT", "extent": [render_w, render_h]},
        {"binding": 2, "name": "MotionVectors", "type": "VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE", "format": "R16G16_SFLOAT", "extent": [render_w, render_h]},
        {"binding": 3, "name": "Exposure", "type": "VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE", "format": "R32_SFLOAT", "extent": [1, 1]},
        {"binding": 4, "name": "ReactiveMask", "type": "VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE", "format": "R8_UNORM", "extent": [render_w, render_h]},
        {"binding": 5, "name": "UpscaledOutput", "type": "VK_DESCRIPTOR_TYPE_STORAGE_IMAGE", "format": "R16G16B16A16_SFLOAT", "extent": [display_w, display_h]},
        {"binding": 6, "name": "ConstantsUBO", "type": "VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER", "size": 256}
    ]

    if is_neural:
        # Extended neural rendering inputs
        neural_bindings = [
            {"binding": 7, "name": "NormalsRoughnessMaterial", "type": "VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE", "format": "R8G8B8A8_UNORM", "extent": [render_w, render_h]},
            {"binding": 8, "name": "DiffuseAlbedo", "type": "VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE", "format": "R8G8B8A8_UNORM", "extent": [render_w, render_h]},
            {"binding": 9, "name": "SpecularAlbedo", "type": "VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE", "format": "R8G8B8A8_UNORM", "extent": [render_w, render_h]},
            {"binding": 10, "name": "DirectDiffuseRadiance", "type": "VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE", "format": "R16G16B16A16_SFLOAT", "extent": [render_w, render_h]},
            {"binding": 11, "name": "DirectSpecularRadiance", "type": "VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE", "format": "R16G16B16A16_SFLOAT", "extent": [render_w, render_h]},
            {"binding": 12, "name": "IndirectDiffuseRadiance", "type": "VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE", "format": "R16G16B16A16_SFLOAT", "extent": [render_w, render_h]},
            {"binding": 13, "name": "IndirectSpecularRadiance", "type": "VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE", "format": "R16G16B16A16_SFLOAT", "extent": [render_w, render_h]},
            {"binding": 14, "name": "DominantLightBlocker", "type": "VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE", "format": "R16_SFLOAT", "extent": [render_w, render_h]},
            {"binding": 15, "name": "WeightsTensorBuffer", "type": "VK_DESCRIPTOR_TYPE_STORAGE_BUFFER", "size": 147689898}
        ]
        bindings.extend(neural_bindings)

    return {
        "schema": "q2rtx.vulkan-dlss-schedule",
        "schema_version": 1,
        "model_id": model_id,
        "model_name": KNOWN_MODELS[model_id],
        "gpu_arch": gpu_arch,
        "render_extent": [render_w, render_h],
        "display_extent": [display_w, display_h],
        "scale_factor": round(display_w / render_w, 2),
        "is_neural_rendering": is_neural,
        "descriptor_bindings": bindings
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    # d4r manifest parser
    p_d4r = subparsers.add_parser("d4r-manifest", help="Parse and validate d4r-kernels.txt")
    p_d4r.add_argument("manifest_file", type=Path)
    p_d4r.add_argument("--json", action="store_true", help="Output JSON")

    # DLSS NR weights inspector
    p_weights = subparsers.add_parser("nr-weights", help="Inspect and index DLSS 5 WEIGHTS_HT")
    p_weights.add_argument("weights_file", type=Path)
    p_weights.add_argument("--manifest", type=Path, help="Write output manifest JSON")

    # Vulkan schedule generator
    p_sched = subparsers.add_parser("vulkan-schedule", help="Generate Vulkan binding schedule")
    p_sched.add_argument("--model", choices=list(KNOWN_MODELS.keys()), default="K")
    p_sched.add_argument("--render-size", nargs=2, type=int, default=[1280, 720])
    p_sched.add_argument("--display-size", nargs=2, type=int, default=[2560, 1440])
    p_sched.add_argument("--arch", choices=list(KNOWN_D4R_ARCHS), default="gfx1100")
    p_sched.add_argument("--output", type=Path, help="Write JSON to file")

    args = parser.parse_args()

    if args.command == "d4r-manifest":
        lines = args.manifest_file.read_text(encoding="utf-8").splitlines()
        entries = parse_d4r_manifest(lines)
        if args.json:
            print(json.dumps(entries, indent=2))
        else:
            print(f"Validated {len(entries)} d4r kernel mappings across architectures.")

    elif args.command == "nr-weights":
        data = args.weights_file.read_bytes()
        res = decode_dlss_nr_weights(data)
        print(f"DLSS 5 Neural Weights: {res['blob_count']} tensor blobs ({res['total_size']:,} bytes)")
        for prefix, count in sorted(res["layer_prefixes"].items()):
            print(f"  {prefix:12s}: {count:3d} blobs")
        if args.manifest:
            args.manifest.write_text(json.dumps(res, indent=2) + "\n")

    elif args.command == "vulkan-schedule":
        sched = generate_vulkan_schedule(
            args.model,
            args.render_size[0], args.render_size[1],
            args.display_size[0], args.display_size[1],
            args.arch
        )
        out_str = json.dumps(sched, indent=2)
        if args.output:
            args.output.write_text(out_str + "\n")
            print(f"Wrote Vulkan schedule to {args.output}")
        else:
            print(out_str)


if __name__ == "__main__":
    main()
