# Q2RTX FidelityFX development guide

This repository is being extended with reusable native-Vulkan implementations
of FidelityFX temporal upscaling, frame generation, and denoising.  Work must
remain useful to Q2RTX while keeping provider-neutral pieces separable for
other Vulkan applications.

## Required working practice

- Read `HANDOVER.md` and `TODO.md` before changing FidelityFX or temporal code.
- Update `HANDOVER.md` after every meaningful implementation or validation
  milestone and before ending a session.  It must describe reality, including
  incomplete or broken paths.
- Update task state in `TODO.md` alongside the code change that advances it.
- Never describe a feature as working until it has passed a live visual test
  with Vulkan validation enabled.  A successful compile is not feature proof.
- Preserve unrelated user changes.  This worktree may be intentionally dirty.
- Use `apply_patch` for hand-written text/source changes.  Generated shaders
  and mechanical formatter output may be produced by their documented tools.
- Do not commit captured proprietary shader/model payloads, DLLs, RenderDoc
  captures, or extracted bytecode.  `tools/ffx_dxil/` intentionally ignores
  those outputs.
- Public SDK 2.3 FSR3.1.5 HLSL must be generated with
  `extern/ffx-vulkan/tools/generate_fsr3_3_1_5_spirv.sh`.  It packs each pass's
  non-sampler descriptor bindings, emits the checked embedded C bundle, and
  rejects collisions; do not restore broad 1000/2000/3000 category shifts,
  which faulted RADV during pipeline creation.

## Ground truth architecture

The provider-neutral input boundary is
`src/refresh/vkpt/temporal_contract.h`.  Q2RTX populates it in
`src/refresh/vkpt/temporal.c` immediately after `vkpt_interleave`, and validates
the ABI, requested slots, producer state, extents, motion/camera metadata, and
depth conventions immediately before provider dispatch.

Temporal upscaling order is:

1. ray tracing and denoising/compositing;
2. checkerboard interleave to dense render-resolution inputs;
3. temporal upscaler (it replaces Q2RTX TAA/TAAU);
4. display-resolution bloom, tone mapping, and display effects;
5. UI composition;
6. present.

Do not feed FSR already-TAA-accumulated or tone-mapped color.  Do not put UI
into scene color used for frame interpolation.

Swapchain images must not be transitioned or otherwise used before
acquisition.  `swap_chain_image_initialized` tracks first use so the initial
`UNDEFINED -> PRESENT_SRC_KHR` transition is recorded in the acquired frame
submission that waits on `image_available`.

Q2RTX signal conventions currently exposed by the contract:

- `FLAT_COLOR`: dense render-resolution RGBA16F linear HDR, stored at 128x
  semantic radiance (`value_scale = 1/128`); alpha is checkerboard metadata.
- `FLAT_MOTION.xy`: dense normalized-UV current-to-previous motion
  (`previousUV - currentUV`).  Multiply by render dimensions for pixel units.
  It is jitter-free; camera jitter is supplied separately to the provider.
- `camera.jitter_render_pixels`: the offset Q2RTX adds to its ray sample.
  FSR's inverse input lookup uses the opposite sign; the adapter must negate it.
- `TEMPORAL_VIEW_Z`: dense R32F, positive-forward primary-surface view Z;
  sky is `PRIMARY_RAY_T_MAX`.  It is currently valid only on one GPU.
- `TEMPORAL_DEVICE_DEPTH`: dense R32F conventional finite device depth derived
  from the same positive-forward projection as view Z; near is 0, far/sky is
  1.  It is also currently valid only on one GPU.
- `TEMPORAL_REACTIVE_MASK` and `TEMPORAL_COMPOSITION_MASK`: dense R8 UNORM
  FSR3 history-control inputs authored by primary-ray material classification.
  Transparent/water/glass/warped/screen paths receive both masks; view-model
  pixels receive the reactive mask. Sky and ordinary opaque geometry are zero.
- `TEMPORAL_NORMALS`, `TEMPORAL_ALBEDO`, and `TEMPORAL_ROUGHNESS`: dense
  provider-neutral primary-material inputs exported at checkerboard interleave.
  Normals are geometric linear XYZ in `[-1, 1]`; they are deliberately not yet
  advertised as a complete Ray Regeneration signal set.
- Camera metadata includes a positive vertical FOV derived from `abs(P[5])`
  (Q2RTX flips Vulkan Y) and `view_space_to_meters = 0.0254` for the engine's
  one-inch-per-world-unit convention.
- A dedicated UI texture and the full Ray Regeneration signal set are not yet
  available.
- Non-rectilinear projections must fall back to the existing renderer.

## FSR4 facts that must not regress

The pre-existing FSR4 code was a hand-written prototype, not AMD's provider.
The screenshot rectangle exactly matched the internal render extent because
the provider used wrong resource domains and guessed dispatch sizes.  Forced
history reset hid the rectangle by disabling temporal accumulation.

The INT8/DOT4 v07 model requires all of the following:

- exact per-pass dispatch dimensions from `ffx_fsr4_schedule.*`;
- exact activation scratch sizes: 20,880,256 bytes through 1080p output,
  83,232,256 through 4K, and 332,352,256 above 4K;
- the selected 89,216-byte `initializers.bin` at the model initializer binding;
- the selected 1,024-byte pre-pass `cbPass_Weights` at binding 34;
- a separate sampler binding (currently binding 35); a sampled image and a
  sampler cannot both occupy binding 0 in Vulkan;
- matching shader permutations for motion/depth/exposure/colorspace;
- the jitter-free motion-vector permutation
  (`FFX_MLSR_JITTERED_MOTION_VECTORS=0`);
- normalized provider motion scale (`API scale / motion texture dimensions`);
- the provider-recommended jitter phase count (32 samples at Performance 2x),
  not Q2RTX's fixed 128-sample TAA cycle;
- conversion from Q2RTX ray-sample jitter to FSR API jitter by negating XY;
- display-sized history/reprojection/recurrent resources and deterministic
  one-time initialization;
- an explicit sampled exposure identity of 1.0, never a generic zero clear;
- the optional SPD graph, when `flt_fsr4_auto_exposure=1`: a 2×1
  current/previous R32F exposure image, R32_UINT atomic counter, R32F mip-5,
  the reflected 32-byte cbuffer, and 64×64 workgroup setup. Disabling it must
  restore the 1.0 identity rather than reuse an old exposure value;
- staging resources kept alive until their recorded uploads finish.

The backend must fail closed: mandatory pipeline creation, descriptor
allocation/writes, resource registration, scheduling, and execution errors are
propagated.  Pipeline handles encode `pass + 1` so the pre pass is never a null
handle.  Generated shaders use the reflected 104-byte UBO at binding 43 and no
push-constant range.  Partial allocations must unwind cleanly and destruction
must remain idempotent.

Never restore the per-frame history/reprojection clear workaround.  It erases
the temporal signal instead of fixing the provider.

FSR 4.1.1, Ray Regeneration, and ML Frame Generation are binary-only in AMD
SDK 2.3 and officially DX12-only.  Shader extraction is demonstrably possible,
but shader blobs alone do not supply pass selection, resources, constants,
weights, barriers, scheduling, or presentation.  Treat those ports as measured
reverse-engineering research and keep an honest fallback.

tools/ffx_dxil/reference_harness/fsr_provider_probe.cpp is the controlled AMD
SDK 2.3 DX12/Wine entry point for that research. On this RX 6800M it enumerated
only analytical upscaler 3.1.5/2.3.4 providers, and a 4.1.1 API context selected
3.1.5 for a successful dispatch. That is a measured current RDNA2 fallback
result, not evidence that official neural FSR4 is available.

FSR3 analytical upscaling/frame interpolation has public source and is the
highest-confidence native-Vulkan path for RDNA2.  The old AMD Vulkan
frame-interpolation swapchain wrapper is Windows-specific and assumes distinct
queues unavailable on this machine; build a portable explicit presenter.

## Reusable modules

- `extern/ffx-vulkan/`: standalone Vulkan C ABI, validation, capability probe,
  pinned MIT-licensed AMD FSR3 1.1.4 host runtime, native Vulkan compute
  backend, and generated shader permutations.  The opaque public C upscaler
  and analytical frame-generation APIs own their aligned backend/scratch and
  shared resources and record work into an application command buffer.  The
  latter now submits reset plus temporal Optical-Flow/Frame-Interpolation
  frames on the RX 6800M with zero validation messages. Q2RTX now selects the
  upscaler through `flt_upscaler 1` and exposes experimental
  `flt_frame_generation 1`: a `minImageCount + 2`-image (five on the
  RX 6800M/RADV test surface), two-acquire/two-present
  generated-then-real path with UI replay and safe real-frame fallback. Its
  blocking second acquire reserves the real image instead of spuriously
  falling back at a zero-timeout probe. Generated and real submissions now use
  separate descriptors, one UI upload, and queue order rather than a CPU replay
  wait. The reusable backend retains eight dynamic-view generations because FI
  records both Prepare and Dispatch per rendered frame; this fixed the prior
  in-flight image-view VUID on the two-frame Q2RTX host. Rolling
  rendered/generated presentation cadence is reported only after successful
  WSI pairs; explicit WSI pacing and VRR/VSync policy are still required before
  performance claims. The application must ensure GPU completion before destroying either
  context.
  It also contains the exact public SDK v2.3.0 FSR3.1.5 source closure under
  `upstream/ffx-2.3.0`. Its object-only `fsr3-host-3.1.5-scaffold` target and
  always-on graph test are host-port gates, not a runnable implementation. The
  graph test proves the reset and temporal-RCAS scheduler resource/job contract
  before the dedicated SDK-2.3 Vulkan `FfxInterface` resource/job bridge is
  introduced; only then can it replace or augment the proven 1.1.4 path. Its
  first pipeline callback layer now selects the embedded module by scheduler
  name and returns its reflected resource slots. It now owns SDK-created
  buffers/images and mip views and records its ordered initialization copies,
  but does not yet import application images or record barriers/compute jobs.
  The fixed Q2-compatible 3.1.5 SPIR-V set is now
  generated under `generated/ffx-2.3.0/vk/fsr3upscaler-q2-v2` (ten pass
  wrappers plus AccumulateSharpen) with a pinned DXC,
  SHA-256, and Vulkan 1.2 validation. It deliberately uses distinct binding
  ranges for SRV/UAV/samplers/CBVs; do not mix it with DX12 table bindings. A
  small public reflection helper reads those bindings from the actual SPIR-V;
  the SDK-2.3 bridge must use it rather than hand-copy DX12 root signatures.
  Its catalogue accepts only the generated base and `ACCUMULATE_SHARPEN`
  permutations, failing closed for unsupported wave/FP16/jitter/depth profiles.
- `tools/ffx_dxil/`: deterministic DXBC/DXIL scanner, verifier, extractor, and
  vkd3d DXIL-to-SPIR-V capture-manifest tooling.
- `src/refresh/vkpt/fsr4/ffx_fsr4_schedule.*`: renderer-independent exact
  INT8/DOT4 schedule and model/scratch helpers.
- `src/refresh/vkpt/fsr4/ffx_fsr4_assets.*`: dependency-free v07 asset
  contract for coherent static/DRS model, tensor tier, SPIR-V, initializer,
  pass-0 weights, RCAS, and SPD names. `extern/ffx-vulkan` exports it as
  `ffx-vulkan::fsr4-v07-assets`, and Q2RTX uses that same selector.

## Build and validation

From the repository root:

```sh
cmake -S . -B build -DBUILD_TESTING=ON
cmake --build build -j"$(nproc)"
ctest --test-dir build --output-on-failure
```

Standalone reusable module:

```sh
cmake -S extern/ffx-vulkan -B build/ffx-vulkan
cmake --build build/ffx-vulkan -j"$(nproc)"
ctest --test-dir build/ffx-vulkan --output-on-failure
build/ffx-vulkan/ffx_vk_capability_probe
```

DXIL tooling:

```sh
python -m unittest discover -s tools/ffx_dxil/tests -v
python tools/ffx_dxil/dxil_container_tool.py --help
```

For shader changes, run `spirv-val` on every generated SPIR-V module and run
the ABI/reflection validator documented beside the FSR4 compile script.  Live
tests must capture the full Vulkan validation message, not a clipped overlay.

The known-good FSR4 shader compiler is
`/home/fireburn/DirectXShaderCompiler/build-release/bin/dxc`.  Generate and
validate every fixed preset plus DRS with:

```sh
DXC_BIN=/home/fireburn/DirectXShaderCompiler/build-release/bin/dxc \
FSR4_VALIDATE=1 \
src/refresh/vkpt/fsr4/compile_shaders_fsr4.sh \
  /home/fireburn/FidelityFX-SDK_WithFSR4 all baseq2/fsr4_shaders
src/refresh/vkpt/fsr4/compile_shaders_fsr4.sh \
  /home/fireburn/FidelityFX-SDK_WithFSR4 drs baseq2/fsr4_shaders
for preset in native quality balanced performance ultraperf drs; do
  python3 src/refresh/vkpt/fsr4/validate_fsr4_spirv.py \
    baseq2/fsr4_shaders --preset "$preset"
done
```

The validator must check image-write component counts as well as bindings and
formats.  DX12 permits a `float3` write to an RGBA resource; Vulkan explicit
`Rgba16f` storage images require four components.

The primary test GPU is an RX 6800M (NAVI22/RDNA2) using RADV.  It exposes
FP16, INT8, signed integer dot product, timeline semaphore, synchronization2,
and the required compute-derivative feature.  Capability presence does not
imply official AMD support or image correctness.
