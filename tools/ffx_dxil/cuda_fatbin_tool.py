#!/usr/bin/env python3
"""Inventory bounded CUDA fatbinary envelopes; payload validation needs cuobjdump."""
import argparse
import hashlib
import json
from pathlib import Path
import struct

MAGIC = bytes.fromhex("50ed55ba")
HEADER = struct.Struct("<IHHQ")


def scan(data):
    """Return envelope metadata. Never mistake a marker for validated GPU code."""
    bundles, rejected = [], []
    cursor = 0
    while (offset := data.find(MAGIC, cursor)) >= 0:
        cursor = offset + 4
        reason = None
        if len(data) - offset < HEADER.size:
            reason = "truncated header"
        else:
            _, version, header_size, payload_size = HEADER.unpack_from(data, offset)
            if version != 1 or header_size != HEADER.size:
                reason = "unsupported envelope version/header size"
            elif not payload_size or payload_size > len(data) - offset - header_size:
                reason = "empty or out-of-bounds payload"
        if reason:
            rejected.append(dict(offset=offset, reason=reason))
            continue
        size = header_size + payload_size
        bundles.append(dict(offset=offset, size=size, version=version,
            sha256=hashlib.sha256(data[offset:offset + size]).hexdigest()))
        # An embedded magic in a payload is not another top-level envelope.
        cursor = offset + size
    return dict(envelopes=bundles, rejected_markers=rejected,
        payload_validation="not performed; use NVIDIA cuobjdump")


def extract(data, inventory, directory):
    directory.mkdir(parents=True, exist_ok=True)
    for bundle in inventory["envelopes"]:
        payload = data[bundle["offset"]:bundle["offset"] + bundle["size"]]
        path = directory / (bundle["sha256"] + ".fatbin")
        if path.exists():
            if path.read_bytes() != payload:
                raise ValueError(f"Refusing to overwrite differing file: {path}")
        else:
            with path.open("xb") as output:
                output.write(payload)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("inputs", nargs="+", type=Path)
    parser.add_argument("--manifest", required=True, type=Path)
    parser.add_argument("--extract-dir", type=Path,
        help="opt-in local extraction; retains source licensing restrictions")
    args = parser.parse_args()
    sources = []
    for source in args.inputs:
        data = source.read_bytes()
        inventory = scan(data)
        inventory.update(name=source.name, size=len(data), sha256=hashlib.sha256(data).hexdigest())
        if args.extract_dir:
            extract(data, inventory, args.extract_dir)
        sources.append(inventory)
    args.manifest.parent.mkdir(parents=True, exist_ok=True)
    args.manifest.write_text(json.dumps(dict(schema="q2rtx.cuda-fatbin-envelopes",
        schema_version=1, sources=sources), indent=2) + "\n")


if __name__ == "__main__":
    main()
