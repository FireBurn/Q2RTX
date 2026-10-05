# DLSS reference inputs

Acquired on 2026-09-15 at the user's request. Files live under the ignored
`build/vendor-sdks/` directory; none are linked into or packaged with Q2RTX.

## DLSS 4.x / 4.5

The official [NVIDIA DLSS SDK v310.9.1](https://github.com/NVIDIA/DLSS/releases/tag/v310.9.1),
commit `374959484e79a640feaba44c93ac8cfb0a03f5b5`, is staged at
`build/vendor-sdks/dlss-310.9.1/`. It includes the public headers, documentation,
license, and x86-64 Windows/Linux release runtimes for super resolution,
ray reconstruction and frame generation. SDK runtime numbering (310.9.1)
is distinct from the DLSS marketing generation.

Each of the 37 downloaded files was checked against the pinned repository's
Git blob hash and length, then assigned a SHA-256 in the local acquisition
manifest. The three Windows DLLs report file version 310.9.1.0.

## DLSS 5 neural rendering

The official SDK tree above contains no neural-rendering runtime. The
separate `nvngx_dlssnr.dll` was obtained from the community-hosted
[310.8.0 archive](https://github.com/RankFTW/rhi-repo/releases/tag/dlssnr-310.8.0)
and staged at `build/vendor-sdks/dlss5-310.8.0-community/nvngx_dlssnr.dll`.
This is a mirrored binary, not an official public SDK download. The original
archive is retained alongside it. No patched RTX40/SF variant was selected.

- Archive SHA-256: `388c0a7912e15ec911b9c9e11a692142b11fe387ddf2b637d8c358138fffb3ac`
- DLL SHA-256: `e16bcf15e16e13f527491cdf7845b2fe6521a738d8f7c9c721866a8496e1fc8e`
- DLL length: 165,840,496 bytes; PE file version: 310.8.0.0.

The archive matches the release API's SHA-256 and length. `osslsigncode 2.9`
successfully verified NVIDIA Corporation's Authenticode signatures on this
DLL and all three official Windows DLLs, including certificate/CRL checks.
The signatures establish signed-file integrity, not portability or execution
correctness. No DLSS DLL has been executed during this acquisition.

## Reproducibility and next step

`build/vendor-sdks/dlss-acquisition.json` records the pinned official commit,
download URLs, archive member, file lengths, SHA-256s, versions and signature
log paths. `build/vendor-sdks/acquire_dlss.py` recreates the downloads without
overwriting differing files. License text is retained with the official SDK.

Static scanning of the four Windows DLLs found zero valid DXBC/DXIL containers;
the metadata manifest is `build/vendor-sdks/dlss-containers.json`. This scanner
does not recover compressed/runtime-generated code or NVIDIA machine code.
These inputs remove the missing-runtime blocker. Subsequent kernel and weight
discovery is recorded below; reference execution, native translation and
cross-vendor validation remain unimplemented. Downloading these files is not
a working DLSS backend.

## CUDA extraction result

Further inspection found CUDA fatbinary envelopes rather than DXIL.
`../cuda_fatbin_tool.py` now inventories bounded version-1 envelopes and can
extract them into an ignored directory. It checks envelope bounds, not CUDA
payload validity; NVIDIA `cuobjdump` provides the second check.

```sh
python3 tools/ffx_dxil/cuda_fatbin_tool.py \
  build/vendor-sdks/dlss5-310.8.0-community/nvngx_dlssnr.dll \
  --manifest build/vendor-sdks/nr-fatbins.json \
  --extract-dir build/vendor-sdks/nr-fatbins
```

All 313 envelope occurrences across the four Windows DLLs were extracted.
`cuobjdump 13.0.39 -ptx` returned success for every unique extracted bundle:

| Runtime | Bundles | PTX modules | Kernel entry occurrences | PTX targets |
| --- | ---: | ---: | ---: | --- |
| Super resolution | 169 | 169 | 182 | sm_80, sm_89 |
| Ray reconstruction | 59 | 59 | 72 | sm_89 |
| Frame generation | 70 | 101 | 102 | sm_89, sm_120 |
| Neural rendering | 15 | 15 | 231 | sm_120 |

These counts include architecture variants and utility kernels, not just
neural inference kernels. Decoding PTX is not validation of execution.
The SR, RR and NR modules include FP8 `mma.sync` matrix instructions;
SR, FG and NR also include FP16 matrix instructions. Porting must preserve
their lane layout, precision and synchronization, or explicitly validate
any replacement arithmetic. A GPU-vendor-independent implementation has
not yet been produced.

Local artifacts:

- `build/vendor-sdks/cuda-envelopes.json` and `cuda-envelopes/`:
  DLL offsets, hashes and extracted bundles.
- `build/vendor-sdks/cuda-ptx-inventory.json` and `cuda-ptx/`:
  decoded PTX, targets, kernel names and matrix instruction inventory.
- `build/vendor-sdks/inspect_cuda.py`: reproduces the cuobjdump inventory.

The cuobjdump executable came from NVIDIA's
[CUDA 13.0.0 redistributable manifest](https://developer.download.nvidia.com/compute/cuda/redist/redistrib_13.0.0.json).
Its archive SHA-256 was checked against
`2385804b0a628b98b3ed96a19c803143d5c2e2fcc5139f633450862cf7755d65`.

## Neural-rendering weight resource

The NR DLL contains a `WEIGHTS_HT` PE resource of 147,695,410 bytes.
`../dlss_nr_weights.py` indexes its observed length-prefixed serialization,
validating all boundaries and consuming the resource exactly. It recovered
153 named blobs, totalling 147,689,898 payload bytes. Names identify layers
such as `block0.layer0.layer`; tensor types, shapes, packing and graph
connectivity remain unverified. These are serialized blobs, not an exported
ONNX model or a demonstrated runnable network.

```sh
python3 tools/ffx_dxil/dlss_nr_weights.py \
  build/vendor-sdks/dlss5-310.8.0-community/nvngx_dlssnr.dll \
  --manifest build/vendor-sdks/dlss5-weights.json \
  --extract-dir build/vendor-sdks/dlss5-weights
```

This command uses `pefile` for PE resource lookup; metadata includes input,
resource and per-blob hashes. Extraction is optional and refuses differing
existing files. Keep all payloads local under their source terms. Next steps
are decoding the tensor layout and graph, then translating and validating
the actual kernel sequence against a reference.
