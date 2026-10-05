#!/usr/bin/env python3
"""Index the observed DLSSNR 310.8.0 WEIGHTS_HT resource (not a tensor decoder)."""
import argparse
import hashlib
import json
from pathlib import Path
import struct


def index_weights(data):
    """Bounds-check the observed total-size / name / blob serialization."""
    offset = 0

    def u64():
        nonlocal offset
        if offset + 8 > len(data):
            raise ValueError("Truncated integer")
        value = struct.unpack_from("<Q", data, offset)[0]
        offset += 8
        return value

    if u64() != len(data):
        raise ValueError("Resource length does not match its declared size")
    entries, names = [], set()
    while offset < len(data):
        length = u64()
        if not 0 < length <= 4096 or length > len(data) - offset:
            raise ValueError("Invalid name length")
        name = data[offset:offset + length].decode("utf-8", errors="strict")
        offset += length
        if name in names or "\0" in name:
            raise ValueError("Duplicate or NUL-containing name")
        names.add(name)
        size = u64()
        if not size or size > len(data) - offset:
            raise ValueError("Empty or truncated blob")
        blob = data[offset:offset + size]
        entries.append(dict(name=name, offset=offset, size=size,
            sha256=hashlib.sha256(blob).hexdigest()))
        offset += size
    return entries


def write_exclusive(path, data):
    if path.exists():
        if path.read_bytes() != data:
            raise ValueError(f"Refusing to overwrite differing file: {path}")
    else:
        with path.open("xb") as f:
            f.write(data)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("dll", type=Path)
    parser.add_argument("--manifest", required=True, type=Path)
    parser.add_argument("--extract-dir", type=Path)
    args = parser.parse_args()
    import pefile  # Only needed for PE resource lookup, not the serialization parser.
    source = args.dll.read_bytes()
    pe = pefile.PE(data=source)
    matches = [language.data.struct
        for kind in pe.DIRECTORY_ENTRY_RESOURCE.entries if kind.id == 10
        for name in kind.directory.entries if str(name.name) == "WEIGHTS_HT"
        for language in name.directory.entries]
    if len(matches) != 1:
        raise ValueError("Expected one unambiguous WEIGHTS_HT resource")
    resource = matches[0]
    offset = pe.get_offset_from_rva(resource.OffsetToData)
    if offset < 0 or offset + resource.Size > len(source):
        raise ValueError("Resource extends outside the DLL")
    data = source[offset:offset + resource.Size]
    entries = index_weights(data)
    if args.extract_dir:
        args.extract_dir.mkdir(parents=True, exist_ok=True)
        write_exclusive(args.extract_dir / "WEIGHTS_HT.bin", data)
        for entry in entries:
            write_exclusive(args.extract_dir / (entry["sha256"] + ".bin"),
                data[entry["offset"]:entry["offset"] + entry["size"]])
    manifest = dict(schema="q2rtx.dlssnr-weight-envelope", schema_version=1,
        source=dict(name=args.dll.name, size=len(source), sha256=hashlib.sha256(source).hexdigest()),
        resource=dict(name="WEIGHTS_HT", file_offset=offset, size=len(data), sha256=hashlib.sha256(data).hexdigest()),
        entries=entries, tensor_layout="unknown", graph="unknown")
    args.manifest.parent.mkdir(parents=True, exist_ok=True)
    args.manifest.write_text(json.dumps(manifest, indent=2) + "\n")
    print(f"Indexed {len(entries)} named blobs; tensor types/shapes and graph are unverified")


if __name__ == "__main__":
    main()
