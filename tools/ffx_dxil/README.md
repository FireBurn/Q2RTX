# FidelityFX DXBC/DXIL audit tools

`dxil_container_tool.py` provides reproducible, dependency-free inspection of
shader containers embedded in FidelityFX provider DLLs. It also inventories
the matching source-bytecode and SPIR-V files emitted by vkd3d-proton.

The normal `scan` operation writes **metadata only**. Its JSON contains no
shader bytecode, absolute input paths, timestamps, or other host-specific data,
so two scans of identical inputs are byte-for-byte comparable.

## Scan SDK provider DLLs

Use DLLs obtained through their normal licensed distribution. For SDK 2.3:

```sh
python3 tools/ffx_dxil/dxil_container_tool.py scan \
  /path/to/FidelityFX-SDK/Kits/FidelityFX/signedbin/amd_fidelityfx_upscaler_dx12.dll \
  /path/to/FidelityFX-SDK/Kits/FidelityFX/signedbin/amd_fidelityfx_denoiser_dx12.dll \
  /path/to/FidelityFX-SDK/Kits/FidelityFX/signedbin/amd_fidelityfx_framegeneration_dx12.dll \
  --fail-on-invalid \
  --manifest build/ffx-dxil/sdk-2.3.json
```

The manifest records:

- the name, byte size, and SHA-256 of each source DLL;
- every `DXBC` marker and the reason any marker failed validation;
- unique containers, duplicate file offsets, header digest, and SHA-256;
- part FourCCs, sizes, offsets, and payload SHA-256 values;
- DXIL shader kind/model, DXIL version, and LLVM bitcode metadata;
- aggregate container bytes, FourCCs, shader models, and DXIL versions.

Recheck a manifest against the original binaries with:

```sh
python3 tools/ffx_dxil/dxil_container_tool.py verify \
  build/ffx-dxil/sdk-2.3.json \
  /path/to/amd_fidelityfx_upscaler_dx12.dll \
  /path/to/amd_fidelityfx_denoiser_dx12.dll \
  /path/to/amd_fidelityfx_framegeneration_dx12.dll
```

### Optional local extraction

Extraction is deliberately opt-in:

```sh
python3 tools/ffx_dxil/dxil_container_tool.py scan /path/to/provider.dll \
  --manifest build/ffx-dxil/provider.json \
  --extract-dir build/ffx-dxil/containers
```

Containers are named by SHA-256 and existing, differing files are never
overwritten. Keep extracted binaries in an ignored build directory. **Do not
commit or redistribute extracted shaders or model data.** They retain the
licensing and distribution restrictions of their source DLL. This tool neither
decrypts nor reconstructs HLSL source; it carves already embedded DXBC
containers and validates their layout.

## Inventory a vkd3d-proton capture

After following [CAPTURE.md](CAPTURE.md), select the shader hashes belonging to
the effect and create a pairing manifest:

```sh
python3 tools/ffx_dxil/dxil_container_tool.py capture-manifest \
  /absolute/path/to/empty-vkd3d-dump \
  --hash-file build/ffx-dxil/fsr-hashes.txt \
  --require-pairs \
  --manifest build/ffx-dxil/fsr-capture.json
```

`--hash` may be repeated instead of using a file. A capture manifest validates
each DXIL/DXBC container and SPIR-V header, pairs files by vkd3d's 16-digit
shader hash, and records artifact hashes. It intentionally does not copy or
embed the artifacts.

## Tests

Run the standard-library test suite with:

```sh
python3 -m unittest discover -s tools/ffx_dxil/tests -v
```

Synthetic tests always run. If an unpacked SDK 2.3 is found below `/tmp`, the
three effect DLLs are scanned as integration tests. Set `FFX_SDK_23_ROOT` to an
SDK root or its `signedbin` directory to test another location.

The SDK 2.3.0 signed binaries used while developing this tool validate as
follows. Counts distinguish all embedded occurrences from unique containers:

| Provider | DXBC occurrences | Unique DXIL containers |
| --- | ---: | ---: |
| Upscaler | 1,028 | 885 |
| Denoiser | 630 | 630 |
| Frame generation | 487 | 486 |

All 2,145 markers in those three DLLs passed structural validation. The local
integration test pins these counts when the input SHA-256 matches the known
SDK 2.3.0 files and otherwise applies version-independent validation.
