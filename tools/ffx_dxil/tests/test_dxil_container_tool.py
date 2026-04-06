from __future__ import annotations

import importlib.util
import json
import os
from pathlib import Path
import struct
import tempfile
import unittest


TOOL_PATH = Path(__file__).resolve().parents[1] / "dxil_container_tool.py"
SPEC = importlib.util.spec_from_file_location("dxil_container_tool", TOOL_PATH)
assert SPEC is not None and SPEC.loader is not None
tool = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(tool)


KNOWN_SDK23 = {
    "amd_fidelityfx_upscaler_dx12.dll": {
        "sha256": "d0dcccc74a43c44ba435b7a369b456e0970d8a4464e4bd683119b374f2c9fb46",
        "markers": 1028,
        "unique": 885,
    },
    "amd_fidelityfx_denoiser_dx12.dll": {
        "sha256": "48f1e5888ba6a0a3d59a98b9751e37c392b0f7b8c223d0082d5c1f40642879d3",
        "markers": 630,
        "unique": 630,
    },
    "amd_fidelityfx_framegeneration_dx12.dll": {
        "sha256": "02297beedd285e822d3a64f314cf00faf378dcec0edc47ff0c4dd71b3a8c2f18",
        "markers": 487,
        "unique": 486,
    },
}


def make_dxil_payload() -> bytes:
    bitcode = b"BC\xc0\xde" + bytes(range(32))
    payload_size = tool.DXIL_PROGRAM_HEADER.size + len(bitcode)
    assert payload_size % 4 == 0
    return tool.DXIL_PROGRAM_HEADER.pack(
        0x00050066,
        payload_size // 4,
        struct.unpack("<I", b"DXIL")[0],
        0x00000106,
        16,
        len(bitcode),
    ) + bitcode


def make_container(parts=None) -> bytes:
    if parts is None:
        parts = [(b"SFI0", b"\x00" * 8), (b"DXIL", make_dxil_payload())]
    header_size = tool.DXBC_HEADER.size + 4 * len(parts)
    offsets = []
    chunks = bytearray()
    for fourcc, payload in parts:
        while (header_size + len(chunks)) % 4:
            chunks.append(0)
        offsets.append(header_size + len(chunks))
        chunks.extend(tool.DXBC_PART_HEADER.pack(fourcc, len(payload)))
        chunks.extend(payload)
    while (header_size + len(chunks)) % 4:
        chunks.append(0)
    size = header_size + len(chunks)
    header = tool.DXBC_HEADER.pack(b"DXBC", bytes(range(16)), 1, size, len(parts))
    return header + struct.pack("<{}I".format(len(offsets)), *offsets) + chunks


class ContainerTests(unittest.TestCase):
    def test_parses_dxil_metadata(self):
        container = make_container()
        metadata, parsed = tool.parse_dxbc_container(container)
        self.assertEqual(parsed, container)
        self.assertTrue(metadata["has_dxil"])
        self.assertEqual(metadata["dxil_programs"][0]["shader_kind"], "compute")
        self.assertEqual(metadata["dxil_programs"][0]["shader_model"], "6.6")
        self.assertEqual(metadata["dxil_programs"][0]["dxil_version"], "1.6")

    def test_rejects_truncated_container(self):
        with self.assertRaisesRegex(tool.ValidationError, "source file"):
            tool.parse_dxbc_container(make_container()[:-4])

    def test_rejects_overlapping_parts(self):
        container = bytearray(make_container())
        first_offset = struct.unpack_from("<I", container, tool.DXBC_HEADER.size)[0]
        struct.pack_into("<I", container, tool.DXBC_HEADER.size + 4, first_offset + 4)
        with self.assertRaisesRegex(tool.ValidationError, "overlap|FourCC"):
            tool.parse_dxbc_container(bytes(container))

    def test_scan_deduplicates_and_records_invalid_marker(self):
        container = make_container()
        blob = b"prefix-DXBC-not-a-container" + container + b"gap" + container
        source, extracted = tool.scan_bytes(blob, "fixture.dll")
        self.assertEqual(source["dxbc_marker_count"], 3)
        self.assertEqual(source["valid_container_occurrence_count"], 2)
        self.assertEqual(source["unique_container_count"], 1)
        self.assertEqual(source["invalid_marker_count"], 1)
        self.assertEqual(source["part_fourcc_counts"], {"DXIL": 1, "SFI0": 1})
        self.assertEqual(source["shader_model_counts"], {"6.6": 1})
        self.assertEqual(len(extracted), 1)

    def test_manifest_is_independent_of_input_order(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            first = root / "z.dll"
            second = root / "a.dll"
            first.write_bytes(make_container())
            second.write_bytes(b"PE-prefix" + make_container([(b"TEST", b"abcd")]))
            forward, _ = tool.build_manifest([first, second])
            reverse, _ = tool.build_manifest([second, first])
            self.assertEqual(tool.json_text(forward), tool.json_text(reverse))
            self.assertEqual([item["name"] for item in forward["sources"]], ["a.dll", "z.dll"])
            self.assertNotIn(str(root), tool.json_text(forward))

    def test_extract_uses_content_hash_and_does_not_replace(self):
        container = make_container()
        digest = tool.sha256_bytes(container)
        with tempfile.TemporaryDirectory() as temporary:
            output = Path(temporary)
            self.assertEqual(tool.extract_containers({digest: container}, output), 1)
            self.assertEqual(tool.extract_containers({digest: container}, output), 0)
            (output / (digest + ".dxbc")).write_bytes(b"wrong")
            with self.assertRaises(FileExistsError):
                tool.extract_containers({digest: container}, output)


class CaptureManifestTests(unittest.TestCase):
    def test_pairs_dxil_and_spirv(self):
        container = make_container()
        spirv = tool.SPIRV_HEADER.pack(tool.SPIRV_MAGIC, 0x00010600, 0, 23, 0)
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            (root / "0123456789abcdef.dxil").write_bytes(container)
            (root / "0123456789abcdef.spv").write_bytes(spirv)
            manifest = tool.build_capture_manifest(root, require_pairs=True)
            self.assertEqual(manifest["shader_count"], 1)
            self.assertEqual(manifest["paired_shader_count"], 1)
            self.assertEqual(manifest["shaders"][0]["artifacts"][1]["spirv"]["version"], "1.6")

    def test_selected_hash_must_exist(self):
        with tempfile.TemporaryDirectory() as temporary:
            with self.assertRaisesRegex(tool.ValidationError, "no dump artifacts"):
                tool.build_capture_manifest(Path(temporary), {"0123456789abcdef"})


def discover_local_sdk23_dlls():
    roots = []
    configured = os.environ.get("FFX_SDK_23_ROOT")
    if configured:
        configured_path = Path(configured)
        roots.extend([configured_path, configured_path / "Kits/FidelityFX/signedbin"])
    temporary_root = Path(os.environ.get("TMPDIR", "/tmp"))
    roots.extend(temporary_root.glob("fsr-sdk-2.3.*/Kits/FidelityFX/signedbin"))
    roots.extend(
        temporary_root.glob("ffx230-*/source/FidelityFX-SDK-2.3.*/Kits/FidelityFX/signedbin")
    )

    names = (
        "amd_fidelityfx_upscaler_dx12.dll",
        "amd_fidelityfx_denoiser_dx12.dll",
        "amd_fidelityfx_framegeneration_dx12.dll",
    )
    discovered = {}
    for root in roots:
        for name in names:
            candidate = root / name
            if candidate.is_file():
                discovered.setdefault(name, candidate)
    return [discovered[name] for name in names if name in discovered]


class LocalSdk23IntegrationTests(unittest.TestCase):
    def test_installed_effect_dlls_when_present(self):
        dlls = discover_local_sdk23_dlls()
        if not dlls:
            self.skipTest("set FFX_SDK_23_ROOT or unpack SDK 2.3 to enable this integration test")

        for dll in dlls:
            with self.subTest(dll=dll.name):
                source, _ = tool.scan_file(dll)
                self.assertGreater(source["valid_container_occurrence_count"], 0)
                self.assertGreater(source["unique_dxil_container_count"], 0)
                self.assertEqual(source["invalid_marker_count"], 0)
                known = KNOWN_SDK23[dll.name]
                if source["sha256"] == known["sha256"]:
                    self.assertEqual(source["dxbc_marker_count"], known["markers"])
                    self.assertEqual(
                        source["valid_container_occurrence_count"], known["markers"]
                    )
                    self.assertEqual(source["unique_container_count"], known["unique"])
                    self.assertEqual(source["unique_dxil_container_count"], known["unique"])
                # Serializing twice is a guard against accidental timestamps or path leakage.
                self.assertEqual(
                    json.dumps(source, sort_keys=True),
                    json.dumps(tool.scan_file(dll)[0], sort_keys=True),
                )


if __name__ == "__main__":
    unittest.main()
