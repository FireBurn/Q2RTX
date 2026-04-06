#!/usr/bin/env python3
"""Inspect embedded DXBC/DXIL containers and vkd3d-proton shader dumps.

The default commands only emit metadata.  Raw containers are written only when
``scan --extract-dir`` is explicitly requested.
"""

from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
import os
from pathlib import Path
import re
import struct
import sys
import tempfile
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Set, Tuple


DXBC_MAGIC = b"DXBC"
DXIL_MAGIC = b"DXIL"
LLVM_BITCODE_MAGIC = b"BC\xc0\xde"
SPIRV_MAGIC = 0x07230203

DXBC_HEADER = struct.Struct("<4s16sIII")
DXBC_PART_HEADER = struct.Struct("<4sI")
DXIL_PROGRAM_HEADER = struct.Struct("<6I")
SPIRV_HEADER = struct.Struct("<5I")

MANIFEST_SCHEMA = "q2rtx.ffx-dxbc-manifest"
CAPTURE_SCHEMA = "q2rtx.vkd3d-shader-capture"
SCHEMA_VERSION = 1

VKD3D_DUMP_NAME = re.compile(r"^(?P<hash>[0-9a-fA-F]{16})\.(?P<tag>.+)$")

SHADER_KINDS = {
    0: "pixel",
    1: "vertex",
    2: "geometry",
    3: "hull",
    4: "domain",
    5: "compute",
    6: "library",
    7: "ray-generation",
    8: "intersection",
    9: "any-hit",
    10: "closest-hit",
    11: "miss",
    12: "callable",
    13: "mesh",
    14: "amplification",
    15: "node",
}


class ValidationError(ValueError):
    """A byte sequence is not a structurally valid supported container."""


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _version_string(raw_version: int) -> str:
    return "{}.{}".format((raw_version >> 8) & 0xFF, raw_version & 0xFF)


def _parse_dxil_program(payload: bytes, part_index: int) -> Dict[str, Any]:
    if len(payload) < DXIL_PROGRAM_HEADER.size:
        raise ValidationError("DXIL part {} is shorter than its program header".format(part_index))

    (
        program_version,
        size_in_uint32,
        dxil_magic,
        dxil_version,
        bitcode_offset,
        bitcode_size,
    ) = DXIL_PROGRAM_HEADER.unpack_from(payload)

    if dxil_magic != struct.unpack("<I", DXIL_MAGIC)[0]:
        raise ValidationError("DXIL part {} has no DXIL program magic".format(part_index))
    if size_in_uint32 * 4 != len(payload):
        raise ValidationError(
            "DXIL part {} declares {} bytes but contains {}".format(
                part_index, size_in_uint32 * 4, len(payload)
            )
        )
    if bitcode_offset < 16:
        raise ValidationError("DXIL part {} has an invalid bitcode offset".format(part_index))

    # BitcodeOffset is relative to the DXIL magic, eight bytes into the payload.
    bitcode_start = 8 + bitcode_offset
    bitcode_end = bitcode_start + bitcode_size
    if bitcode_end > len(payload):
        raise ValidationError("DXIL part {} bitcode extends past the part".format(part_index))

    bitcode = payload[bitcode_start:bitcode_end]
    if not bitcode.startswith(LLVM_BITCODE_MAGIC):
        raise ValidationError("DXIL part {} has no LLVM bitcode magic".format(part_index))

    shader_kind = program_version >> 16
    shader_model_major = (program_version >> 4) & 0xF
    shader_model_minor = program_version & 0xF
    return {
        "part_index": part_index,
        "program_version_raw": "0x{:08x}".format(program_version),
        "shader_kind": SHADER_KINDS.get(shader_kind, "unknown-{}".format(shader_kind)),
        "shader_model": "{}.{}".format(shader_model_major, shader_model_minor),
        "dxil_version": _version_string(dxil_version),
        "bitcode_offset": bitcode_start,
        "bitcode_size": bitcode_size,
        "bitcode_sha256": sha256_bytes(bitcode),
    }


def parse_dxbc_container(blob: bytes, file_offset: int = 0) -> Tuple[Dict[str, Any], bytes]:
    """Parse one DXBC container beginning at *file_offset*.

    Validation covers the container and part bounds, alignment, overlap, and
    the internal header and LLVM bitcode bounds of every DXIL part.
    """

    remaining = len(blob) - file_offset
    if file_offset < 0 or remaining < DXBC_HEADER.size:
        raise ValidationError("truncated DXBC header")

    magic, header_digest, version, container_size, part_count = DXBC_HEADER.unpack_from(
        blob, file_offset
    )
    if magic != DXBC_MAGIC:
        raise ValidationError("missing DXBC magic")
    if container_size < DXBC_HEADER.size:
        raise ValidationError("container size is smaller than its header")
    if container_size > remaining:
        raise ValidationError("container extends past the source file")
    if container_size % 4:
        raise ValidationError("container size is not four-byte aligned")
    if part_count == 0:
        raise ValidationError("container has no parts")

    offsets_size = part_count * 4
    header_size = DXBC_HEADER.size + offsets_size
    if header_size > container_size:
        raise ValidationError("part offset table extends past the container")

    container = blob[file_offset : file_offset + container_size]
    part_offsets = struct.unpack_from("<{}I".format(part_count), container, DXBC_HEADER.size)
    if len(set(part_offsets)) != len(part_offsets):
        raise ValidationError("container has duplicate part offsets")

    parts: List[Dict[str, Any]] = []
    ranges: List[Tuple[int, int, int]] = []
    dxil_programs: List[Dict[str, Any]] = []
    for part_index, part_offset in enumerate(part_offsets):
        if part_offset % 4:
            raise ValidationError("part {} offset is not four-byte aligned".format(part_index))
        if part_offset < header_size:
            raise ValidationError("part {} overlaps the container header".format(part_index))
        if part_offset + DXBC_PART_HEADER.size > container_size:
            raise ValidationError("part {} header extends past the container".format(part_index))

        fourcc_bytes, payload_size = DXBC_PART_HEADER.unpack_from(container, part_offset)
        if any(value < 0x20 or value > 0x7E for value in fourcc_bytes):
            raise ValidationError("part {} has a non-printable FourCC".format(part_index))
        part_end = part_offset + DXBC_PART_HEADER.size + payload_size
        if part_end > container_size:
            raise ValidationError("part {} payload extends past the container".format(part_index))

        fourcc = fourcc_bytes.decode("ascii")
        payload_start = part_offset + DXBC_PART_HEADER.size
        payload = container[payload_start:part_end]
        parts.append(
            {
                "index": part_index,
                "fourcc": fourcc,
                "container_offset": part_offset,
                "payload_size": payload_size,
                "payload_sha256": sha256_bytes(payload),
            }
        )
        ranges.append((part_offset, part_end, part_index))
        if fourcc_bytes == DXIL_MAGIC:
            dxil_programs.append(_parse_dxil_program(payload, part_index))

    ranges.sort()
    for previous, current in zip(ranges, ranges[1:]):
        if previous[1] > current[0]:
            raise ValidationError(
                "parts {} and {} overlap".format(previous[2], current[2])
            )

    metadata: Dict[str, Any] = {
        "sha256": sha256_bytes(container),
        "size": container_size,
        "container_version": version,
        "dxbc_header_digest": header_digest.hex(),
        "part_count": part_count,
        "parts": parts,
        "has_dxil": bool(dxil_programs),
        "dxil_programs": dxil_programs,
    }
    return metadata, container


def _marker_offsets(blob: bytes) -> Iterable[int]:
    position = 0
    while True:
        position = blob.find(DXBC_MAGIC, position)
        if position < 0:
            return
        yield position
        position += 1


def scan_bytes(blob: bytes, source_name: str) -> Tuple[Dict[str, Any], Dict[str, bytes]]:
    """Scan arbitrary bytes for valid embedded DXBC containers."""

    marker_count = 0
    valid_occurrences = 0
    invalid_markers: List[Dict[str, Any]] = []
    unique: Dict[str, Dict[str, Any]] = {}
    container_bytes: Dict[str, bytes] = {}

    for marker_offset in _marker_offsets(blob):
        marker_count += 1
        try:
            metadata, container = parse_dxbc_container(blob, marker_offset)
        except (ValidationError, struct.error) as error:
            invalid_markers.append({"file_offset": marker_offset, "reason": str(error)})
            continue

        valid_occurrences += 1
        digest = metadata["sha256"]
        if digest not in unique:
            metadata["occurrences"] = []
            unique[digest] = metadata
            container_bytes[digest] = container
        elif container_bytes[digest] != container:
            raise RuntimeError("SHA-256 collision while scanning {}".format(source_name))
        unique[digest]["occurrences"].append({"file_offset": marker_offset})

    containers = [unique[digest] for digest in sorted(unique)]
    part_fourccs = Counter(
        part["fourcc"] for container in containers for part in container["parts"]
    )
    shader_models = Counter(
        program["shader_model"]
        for container in containers
        for program in container["dxil_programs"]
    )
    dxil_versions = Counter(
        program["dxil_version"]
        for container in containers
        for program in container["dxil_programs"]
    )
    source = {
        "name": source_name,
        "size": len(blob),
        "sha256": sha256_bytes(blob),
        "dxbc_marker_count": marker_count,
        "valid_container_occurrence_count": valid_occurrences,
        "unique_container_count": len(containers),
        "unique_dxil_container_count": sum(1 for item in containers if item["has_dxil"]),
        "unique_container_bytes": sum(item["size"] for item in containers),
        "part_fourcc_counts": dict(sorted(part_fourccs.items())),
        "shader_model_counts": dict(sorted(shader_models.items())),
        "dxil_version_counts": dict(sorted(dxil_versions.items())),
        "invalid_marker_count": len(invalid_markers),
        "invalid_markers": invalid_markers,
        "containers": containers,
    }
    return source, container_bytes


def scan_file(path: Path) -> Tuple[Dict[str, Any], Dict[str, bytes]]:
    with path.open("rb") as source:
        return scan_bytes(source.read(), path.name)


def build_manifest(paths: Sequence[Path]) -> Tuple[Dict[str, Any], Dict[str, bytes]]:
    if not paths:
        raise ValueError("at least one input is required")

    sources: List[Dict[str, Any]] = []
    all_containers: Dict[str, bytes] = {}
    seen_paths: Set[Path] = set()
    for input_path in paths:
        resolved = input_path.resolve()
        if resolved in seen_paths:
            continue
        seen_paths.add(resolved)
        if not input_path.is_file():
            raise FileNotFoundError(str(input_path))
        source, containers = scan_file(input_path)
        sources.append(source)
        for digest, container in containers.items():
            if digest in all_containers and all_containers[digest] != container:
                raise RuntimeError("SHA-256 collision across input files")
            all_containers[digest] = container

    sources.sort(key=lambda item: (item["name"], item["sha256"]))
    return {
        "schema": MANIFEST_SCHEMA,
        "schema_version": SCHEMA_VERSION,
        "sources": sources,
    }, all_containers


def extract_containers(containers: Mapping[str, bytes], output_dir: Path) -> int:
    output_dir.mkdir(parents=True, exist_ok=True)
    written = 0
    for digest in sorted(containers):
        destination = output_dir / "{}.dxbc".format(digest)
        data = containers[digest]
        if destination.exists():
            if not destination.is_file() or destination.read_bytes() != data:
                raise FileExistsError("refusing to replace {}".format(destination))
            continue
        with destination.open("xb") as output:
            output.write(data)
        written += 1
    return written


def parse_spirv(blob: bytes) -> Dict[str, Any]:
    if len(blob) < SPIRV_HEADER.size:
        raise ValidationError("SPIR-V artifact is shorter than its header")
    if len(blob) % 4:
        raise ValidationError("SPIR-V artifact is not four-byte aligned")
    magic, version, generator, bound, schema = SPIRV_HEADER.unpack_from(blob)
    if magic != SPIRV_MAGIC:
        raise ValidationError("SPIR-V artifact has invalid magic")
    return {
        "version": "{}.{}".format((version >> 16) & 0xFF, (version >> 8) & 0xFF),
        "version_raw": "0x{:08x}".format(version),
        "generator_raw": "0x{:08x}".format(generator),
        "id_bound": bound,
        "schema": schema,
    }


def _validate_vkd3d_hash(value: str) -> str:
    normalized = value.strip().lower()
    if not re.fullmatch(r"[0-9a-f]{16}", normalized):
        raise ValueError("invalid vkd3d shader hash: {!r}".format(value))
    return normalized


def build_capture_manifest(
    dump_dir: Path,
    selected_hashes: Optional[Set[str]] = None,
    require_pairs: bool = False,
) -> Dict[str, Any]:
    """Inventory hash-paired DXIL/DXBC and SPIR-V artifacts from vkd3d-proton."""

    if not dump_dir.is_dir():
        raise NotADirectoryError(str(dump_dir))
    selected = None
    if selected_hashes is not None:
        selected = {_validate_vkd3d_hash(value) for value in selected_hashes}

    groups: Dict[str, List[Dict[str, Any]]] = {}
    for path in sorted(dump_dir.iterdir(), key=lambda item: item.name):
        if not path.is_file():
            continue
        match = VKD3D_DUMP_NAME.fullmatch(path.name)
        if not match:
            continue
        shader_hash = match.group("hash").lower()
        if selected is not None and shader_hash not in selected:
            continue

        tag = match.group("tag")
        artifact_format = tag.rsplit(".", 1)[-1].lower()
        if artifact_format not in {"dxbc", "dxil", "spv"}:
            continue
        blob = path.read_bytes()
        artifact: Dict[str, Any] = {
            "name": path.name,
            "tag": tag,
            "format": artifact_format,
            "size": len(blob),
            "sha256": sha256_bytes(blob),
        }
        if artifact_format == "spv":
            artifact["spirv"] = parse_spirv(blob)
        else:
            metadata, container = parse_dxbc_container(blob)
            if len(container) != len(blob):
                raise ValidationError(
                    "{} has trailing data after its DXBC container".format(path.name)
                )
            artifact["dxbc"] = metadata
        groups.setdefault(shader_hash, []).append(artifact)

    if selected is not None:
        missing = sorted(selected - set(groups))
        if missing:
            raise ValidationError(
                "selected hashes have no dump artifacts: {}".format(", ".join(missing))
            )
    if not groups:
        raise ValidationError("no vkd3d-proton shader artifacts found")

    shaders: List[Dict[str, Any]] = []
    for shader_hash in sorted(groups):
        artifacts = sorted(groups[shader_hash], key=lambda item: item["name"])
        formats = {artifact["format"] for artifact in artifacts}
        has_source = bool(formats & {"dxbc", "dxil"})
        has_spirv = "spv" in formats
        paired = has_source and has_spirv
        if require_pairs and not paired:
            raise ValidationError(
                "shader {} does not have both DXIL/DXBC and SPIR-V".format(shader_hash)
            )
        shaders.append(
            {
                "vkd3d_shader_hash": shader_hash,
                "has_source_bytecode": has_source,
                "has_spirv": has_spirv,
                "paired": paired,
                "artifacts": artifacts,
            }
        )

    return {
        "schema": CAPTURE_SCHEMA,
        "schema_version": SCHEMA_VERSION,
        "shader_count": len(shaders),
        "paired_shader_count": sum(1 for shader in shaders if shader["paired"]),
        "shaders": shaders,
    }


def json_text(value: Mapping[str, Any]) -> str:
    return json.dumps(value, indent=2, sort_keys=True, ensure_ascii=True) + "\n"


def write_json(value: Mapping[str, Any], destination: str) -> None:
    rendered = json_text(value)
    if destination == "-":
        sys.stdout.write(rendered)
        return

    path = Path(destination)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary: Optional[Path] = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            dir=str(path.parent),
            prefix=path.name + ".",
            delete=False,
        ) as output:
            temporary = Path(output.name)
            output.write(rendered)
        os.replace(str(temporary), str(path))
        temporary = None
    finally:
        if temporary is not None:
            try:
                temporary.unlink()
            except FileNotFoundError:
                pass


def _hashes_from_args(values: Sequence[str], hash_file: Optional[Path]) -> Optional[Set[str]]:
    hashes: Set[str] = {_validate_vkd3d_hash(value) for value in values}
    if hash_file is not None:
        for line in hash_file.read_text(encoding="utf-8").splitlines():
            line = line.split("#", 1)[0].strip()
            if line:
                hashes.add(_validate_vkd3d_hash(line))
    return hashes or None


def command_scan(args: argparse.Namespace) -> int:
    input_paths = [Path(value) for value in args.inputs]
    if args.manifest != "-":
        manifest_path = Path(args.manifest).resolve()
        if manifest_path in {path.resolve() for path in input_paths}:
            raise ValueError("manifest path must not be one of the input binaries")
    manifest, containers = build_manifest(input_paths)
    write_json(manifest, args.manifest)
    if args.extract_dir:
        count = extract_containers(containers, Path(args.extract_dir))
        print(
            "wrote {} new unique containers to {} (generated binaries must not be committed)".format(
                count, args.extract_dir
            ),
            file=sys.stderr,
        )
    if args.fail_on_invalid and any(
        source["invalid_marker_count"] for source in manifest["sources"]
    ):
        return 2
    return 0


def command_verify(args: argparse.Namespace) -> int:
    expected = json.loads(Path(args.manifest).read_text(encoding="utf-8"))
    actual, _ = build_manifest([Path(value) for value in args.inputs])
    if expected != actual:
        print("manifest does not match the supplied binaries", file=sys.stderr)
        return 1
    print("verified {} source binaries".format(len(actual["sources"])))
    return 0


def command_capture_manifest(args: argparse.Namespace) -> int:
    hashes = _hashes_from_args(args.hashes, Path(args.hash_file) if args.hash_file else None)
    manifest = build_capture_manifest(Path(args.dump_dir), hashes, args.require_pairs)
    write_json(manifest, args.manifest)
    return 0


def build_argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Validate embedded DXBC/DXIL and inventory vkd3d-proton shader captures."
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    scan = subparsers.add_parser(
        "scan", help="scan binaries and emit a deterministic JSON manifest"
    )
    scan.add_argument("inputs", nargs="+", help="DLL or other binary files to scan")
    scan.add_argument("--manifest", default="-", help="manifest path, or - for standard output")
    scan.add_argument(
        "--extract-dir",
        help="explicitly write unique containers as SHA-256-named .dxbc files",
    )
    scan.add_argument(
        "--fail-on-invalid",
        action="store_true",
        help="return status 2 when a DXBC marker is not a valid container",
    )
    scan.set_defaults(function=command_scan)

    verify = subparsers.add_parser(
        "verify", help="re-scan binaries and compare an existing manifest"
    )
    verify.add_argument("manifest", help="manifest created by the scan command")
    verify.add_argument("inputs", nargs="+", help="the original DLL or binary files")
    verify.set_defaults(function=command_verify)

    capture = subparsers.add_parser(
        "capture-manifest", help="pair and validate vkd3d-proton .dxil/.dxbc and .spv dumps"
    )
    capture.add_argument("dump_dir", help="directory named by VKD3D_SHADER_DUMP_PATH")
    capture.add_argument("--manifest", default="-", help="manifest path, or - for standard output")
    capture.add_argument(
        "--hash", dest="hashes", action="append", default=[], help="include one 16-digit vkd3d hash"
    )
    capture.add_argument("--hash-file", help="newline-separated vkd3d hashes; # starts a comment")
    capture.add_argument(
        "--require-pairs",
        action="store_true",
        help="require both source DXIL/DXBC and translated SPIR-V for every hash",
    )
    capture.set_defaults(function=command_capture_manifest)
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = build_argument_parser()
    args = parser.parse_args(argv)
    try:
        return int(args.function(args))
    except (FileNotFoundError, NotADirectoryError, ValidationError, ValueError) as error:
        parser.error(str(error))
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
