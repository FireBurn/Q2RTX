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
- `ffx-vulkan::effects` is both a vendored and installed C++ integration
  target. Its installed export must expose only `include/` and versioned
  targets—not `upstream/`, generated-source paths, or Q2RTX paths—and changes
  to that export require the clean `examples/installed-full-stack` consumer
  contract.
- Use the read-only `fsr_diagnostics` console command for a live provider
  audit. It must report the resolved provider/reason, temporal input contract,
  the most recent retained temporal reset frame/reason bits,
  current FSR4 model/permutation/memory, current/effective bounded DRS scale
  when the DRS model is selected, all FSR3/FI effect memory, and state without
  mutating a context or recording GPU work.
- Frame-generation FPS telemetry must use completed logical-render cadence,
  never CPU duration between generated/real `vkQueuePresentKHR` submissions.
  A rate-gated fallback learns FI cost and requires that headroom on recovery;
  do not replace it with a simple threshold +2 loop, which oscillates when
  disabling FI itself raises the frame rate.
- A nonzero FG safety floor may make one guarded attempt, but must never
  visibly calibrate itself through repeated generated/real mode toggles. Keep
  the measured FI-cost estimate and conservative recovery headroom; zero is
  the explicit unrestricted diagnostic setting.
- Camera-cut classification is shared through
  `ffx-vulkan::temporal-lifecycle`, not duplicated per provider. It resets
  history for a teleport over 256 world units, a turn over 90 degrees, a lens
  jump over 0.35 radians, or invalid camera data; the boundary tests must stay
  strict about ordinary motion and exact threshold values.
- A successful FI/OF reset dispatch seeds its history but cannot synthesize a
  valid intermediate image. The paired presenter must use the real scene for
  that one slot, including resets caused internally when the rate gate resumes;
  use `ffxVkFrameGenerationShouldPresentGenerated` and retain its policy test.
- The FSR3.1.6 FI output is a public presentation image and may be either
  RGBA8 or RGBA16F. Its `rw_output` declaration must therefore use the Vulkan
  formatless (`unknown`) storage-image form, while internal compact FI images
  remain exactly typed. Do not enable DXC's global unknown-image-format switch:
  it also makes image-texel-pointer resources invalid Vulkan SPIR-V.
- A source-v07 FSR4 model cvar change must rebuild before availability is
  resolved. Do not leave graph recreation solely in the dispatch path: the
  resolver otherwise rejects the mismatched old graph forever as a pending
  model switch. `fsr4_recreate_context` owns the required device-idle hitch.
- A binary semaphore passed to WSI can remain pending after its render fence.
  On either transition between ordinary and paired generated→real presentation,
  use `ffxVkFrameGenerationTransitionNeedsQuiescence` and wait for the present
  queue once before reusing its presentation semaphores; never add that stall to
  either steady presentation path.
- A map/menu transition may skip a logical trace while advancing a frame slot.
  Before a slot's trace semaphore is signalled again, the transfer handoff must
  consume both the normal previous trace signal and any stale current-slot
  trace signal. Do not assume the preceding-slot wait alone proves current
  trace-semaphore reuse is safe.

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

Console screenshot readback happens after the frame's normal present.  It must
acquire its own local WSI image, wait on that acquire semaphore, copy it, and
present it again with a separate completion semaphore; never reuse
`current_swap_chain_image_index` after presentation or alter that renderer
state for a screenshot.

HDR captures use `screenshothdr`, not a PNG/JPEG command.  Its host-readable
image transitions must use HOST -> ALL_COMMANDS before the copy and
ALL_COMMANDS -> HOST afterwards: a generic ALL_COMMANDS -> ALL_COMMANDS
barrier is invalid when either access mask is `HOST_READ`.

Generated/real WSI acquisition goes through
`ffxVkFrameGenerationAcquirePair`. If its second acquire cannot form a pair,
the first image remains acquired and must be rendered/presented as the normal
one-image fallback; do not acquire a third image or reuse the un-signalled
second semaphore.

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
- `flt_temporal_debug_view`: a presentation-only semantic inspector for every
  dense temporal input. It must use the input's `valid_extent`, not its
  allocation extent; it does not mutate provider state. It suspends analytical
  frame generation and removes its FIFO swapchain request while nonzero. The
  swapchain-creation and per-frame policy decisions must use that identical
  condition; disagreement causes a resize/context-recreation loop.
  Values 10-12 additionally inspect current reconstructed FSR output and the
  FSR4-v07 provider's borrowed history/reprojected surfaces. Values 13-15
  inspect the provider-neutral Ray-Regeneration-compatible material inputs:
  octahedral-normal/linear-roughness/category, sqrt diffuse albedo, and sqrt
  specular albedo respectively. Values 16-19 inspect Q2RTX's raw direct and
  indirect diffuse/specular radiance partitions. Values 20-21 visualize their
  first-lobe hit distances from alpha (black means untraced); view 22 shows
  the primary direct-sun blocker distance (black untraced, white FP16-max
  exposed). Contract v10 pairs that image with the exact resolved GPU sun
  emission, surface-to-light direction, and angular radius only after the
  primary-ray readback ring has fence-retired the matching physical-sky
  update. It also exports actual camera-position delta and maps the complete
  Q2RTX input set into `ffx-vulkan::rayregeneration-contract` for a live
  provider-neutral preflight. Q2RTX stores its direct-shadow direction from
  surface to sun; the bridge negates it for the provider's light-to-target
  convention. It remains provider-neutral groundwork, not
  proof that a neural RR provider is active. Those two FSR4
  private surfaces must be
  obtained only through `ffxFsr4GetDebugResource`; do not expose their handles
  as general renderer resources or record writes to them.
- `TEMPORAL_REACTIVE_MASK` and `TEMPORAL_COMPOSITION_MASK`: dense R8 UNORM
  FSR3 history-control inputs authored by primary-ray material classification.
  Transparent/water/glass/warped/screen paths receive both masks; view-model
  pixels receive the reactive mask. Sky and ordinary opaque geometry are zero.
- `TEMPORAL_NORMALS`, `TEMPORAL_ALBEDO`, and `TEMPORAL_ROUGHNESS`: dense
  provider-neutral primary-material inputs exported at checkerboard interleave.
  Normals are geometric linear XYZ in `[-1, 1]`.
- `TEMPORAL_DENOISER_NORMAL_ROUGHNESS_MATERIAL`,
  `TEMPORAL_DENOISER_DIFFUSE_ALBEDO`, and
  `TEMPORAL_DENOISER_SPECULAR_ALBEDO`: compact Ray-Regeneration-compatible
  material inputs exported at the same point. The first is oct-normal,
  linear roughness, and category 0; the latter two are sqrt encoded. They are
  a provider-neutral input foundation, not evidence that AMD's neural RR
  binary provider is available or active.
- `TEMPORAL_RR_DIRECT_DIFFUSE`, `TEMPORAL_RR_INDIRECT_DIFFUSE`,
  `TEMPORAL_RR_DIRECT_SPECULAR`, and `TEMPORAL_RR_INDIRECT_SPECULAR`: dense
  unfiltered Q2RTX lighting partitions. Direct diffuse is the direct-lighting
  high-frequency channel; indirect diffuse is Q2RTX's current low-frequency
  SH coefficient; direct specular is preserved before indirect accumulation;
  indirect specular is the nonnegative remaining combined SPEC energy. These
  use the FSR RR-compatible alpha contract: direct alpha is non-negative and
  otherwise undefined, while indirect alpha is the physically traced first
  lobe segment distance (`10000` for Q2RTX's finite sky miss, negative when
  that lobe was not traced). Later bounces remain associated with that first
  lobe. They are useful provider-neutral radiance groundwork, but do not
  substitute for dominant-visibility inputs.
- `ffx-vulkan::rayregeneration-contract` validates a reusable Vulkan host's
  RR-style image, alpha, camera, jitter, and motion metadata before a provider
  is attached. Contract v4 represents all seven independently selectable
  signal inputs: four radiance partitions, dominant-light visibility, ambient
  occlusion, and specular occlusion. The latter two are R8_UNORM [0,1] optional
  additions; one primary radiance or dominant-light signal remains required.
  Q2RTX currently exports the first five, not separate AO/specular occlusion.
  It also requires a three-component RR motion scale and previous-minus-current
  camera delta. Q2RTX contract v11 exports dense `TEMPORAL_RR_MOTION`: XY is
  `PreviousUV-CurrentUV` and Z is previous-minus-current signed linear view-Z.
  Never feed `FLAT_MOTION` to an RR provider: its Z is radial/reflection-
  denoiser metadata, not an RR linear depth delta.
  It also validates full-resolution per-signal output bindings and optional
  checkerboard flags/origins (checkerboard input half-width, output full-width)
  before a provider records work.
  The validator does not inspect GPU pixels, record commands, or supply a
  neural provider. Do not use a superficial Q2RTX call to it as evidence that
  the renderer meets official RR input requirements; the dominant-light bridge
  remains required.
- Camera metadata includes a positive vertical FOV derived from `abs(P[5])`
  (Q2RTX flips Vulkan Y) and `view_space_to_meters = 0.0254` for the engine's
  one-inch-per-world-unit convention.
- Contract version 4 records the last successfully presented camera and emits
  `VKPT_TEMPORAL_RESET_CAMERA_CUT` for conservative teleport, >90-degree
  transform, or substantial lens discontinuities. Do not turn ordinary
  motion-vector reprojection into a reset.
- A Wayland/SDL focus transition must suspend paired analytical presentation
  and request `VKPT_TEMPORAL_RESET_FOCUS_CHANGED` on both edges. The first
  recovered pair uses the real scene to seed fresh FI/OF history; never
  interpolate across a compositor-deferred presentation gap.
- A dedicated alpha UI texture is available for analytical frame generation;
  the full Ray Regeneration signal set is not.
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

The generated SPIR-V graphs, manifests, and six model-specific initializer/
pre-pass-weight `.bin` pairs are versioned together.  Their source MIT notice
and exact upstream revision are retained in
`baseq2/fsr4_shaders/LICENSE-FSR4-v07.txt`; keep every pair synchronized with
its manifest. `fsr4_v07_assets` verifies every tracked pair against its
manifest's size and SHA-256. `CONFIG_VKPT_INSTALL_FSR4_V07_ASSETS` may package
this complete set, but it remains the older source-v07 model rather than FSR
4.1.1. When Linux packaging enables that option, CMake must require the notice,
all six initializer/pre-weight pairs, and their manifests at configure time;
never make a runnable FSR4 selection depend on an optional, unchecked loose
directory.

The backend must fail closed: mandatory pipeline creation, descriptor
allocation/writes, resource registration, scheduling, and execution errors are
propagated.  Pipeline handles encode `pass + 1` so the pre pass is never a null
handle.  Generated shaders use the reflected 104-byte UBO at binding 43 and no
push-constant range.  Partial allocations must unwind cleanly and destruction
must remain idempotent.

Never restore the per-frame history/reprojection clear workaround.  It erases
the temporal signal instead of fixing the provider.

FSR 4.1.1, Ray Regeneration, ML Frame Generation, and Radiance Caching are
binary-only in AMD SDK 2.3 and officially DX12-only.  Shader extraction is demonstrably possible,
but shader blobs alone do not supply pass selection, resources, constants,
weights, barriers, scheduling, or presentation.  Treat those ports as measured
reverse-engineering research and keep an honest fallback.
AMD's official SDK 2.3 prebuilt archive splits signed effect DLLs into sample
release directories rather than `Kits/FidelityFX/signedbin`; the metadata-only
`tools/ffx_dxil` integration test accepts either layout. Keep every downloaded
DLL, manifest, capture, and any opt-in extraction under an ignored build or
temporary directory—never add those payloads to Q2RTX or `ffx-vulkan`.

The temporal diagnostics menu must retain the four read-only official-provider
rows: FSR4.1.1, Ray Regeneration, ML Frame Generation, and Radiance Caching.
Their labels distinguish this status from the runnable source-v07 FSR4 and
analytical FSR3 FI/OF choices;
keep the values short enough to fit the 52-character static column rather than
silently ellipsizing the DX12/native-Vulkan or RX 9000 boundary.

`CONFIG_VKPT_FSR3` is optional.  If it is disabled, requests for either FSR3
upscaler or analytical frame-generation debug output must fail closed with an
explicit not-compiled reason; they must never fall through to source-v07 FSR4.

tools/ffx_dxil/reference_harness/fsr_provider_probe.cpp is the controlled AMD
SDK 2.3 DX12/Wine entry point for that research. On this RX 6800M it enumerated
only analytical upscaler 3.1.5/2.3.4 providers, and a 4.1.1 API context selected
3.1.5 for a successful dispatch. That is a measured current RDNA2 fallback
result, not evidence that official neural FSR4 is available. The probe also
creates a Frame Generation 4 context and, with the full SDK's
`ffx_denoiser.h`, a Ray Regeneration 1.2 context independently before querying
the selected provider. The deliberately reduced vendored source closure omits
that header, and its build must report the omission rather than manufacture an
internal provider type. Every run ends with stable
`FFX_PROVIDER_PROBE_RESULT` records; future hardware/driver revalidation must
use their actual selected names/IDs and return codes, not only a successful 4.x
API-context creation.

The full current SDK probe also covers Radiance Caching 0.9.0. On this RDNA2
adapter it enumerated provider 0.9.0 but context creation returned error 6
after vkd3d reported missing WMMA support. This is a measured
unavailable-provider result, not evidence that Radiance Caching can run here.

FSR3 analytical upscaling/frame interpolation has public source and is the
highest-confidence native-Vulkan path for RDNA2.  The old AMD Vulkan
frame-interpolation swapchain wrapper is Windows-specific and assumes distinct
queues unavailable on this machine; build a portable explicit presenter.

The historical FSR4 source-v07 SDK has a different public ABI from SDK 2.3.
Use the legacy source-v07 provider probe rather than the 4.1.1 probe for it.
On the selected RDNA2 adapter, its untouched signed loader enumerated only
3.1.5 and 2.3.4. With the community PROTON_FSR4_RDNA3_UPGRADE=1 and
FSR4_UPGRADE=1 switches it enumerated and created 4.0.2; adding the cached
amdxcffx64 compatibility DLL alone did not change that outcome. Its one
controlled dispatch returned success from FFX but command-list closure failed
because vkd3d could not expose WMMA. Thus this is evidence of a historical
DX12 provider-selection workaround, not a verified RDNA2 execution route or a
replacement for Q2RTX's native Vulkan source-v07 implementation. Never call
it official FSR4.1.1.

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
  in-flight image-view VUID on the two-frame Q2RTX host. Q2RTX reports logical
  rendered cadence and a nominal 2x generated rate only while a WSI pair was
  active; explicit WSI pacing and VRR/VSync policy are still required before
  performance claims. The application must ensure GPU completion before destroying either
  context.
  `VKPT_IMG_FSR_RCAS_OUTPUT` is source-v07 FSR4 recurrent storage, never an
  analytical-FI output target. FI/OF must use the dedicated
  `VKPT_IMG_FSR_FRAMEGEN_OUTPUT` image for its generated presentation; the
  root `framegen_target_isolation` test enforces this ownership boundary.
  The reusable SDK-3.1.6 FI wrapper enforces the general rule too: a dispatch
  output must not alias its prepared colour, depth, motion, or optional
  distortion input. Keep that fail-closed check in place for non-Q2 hosts.
  It also contains the exact public SDK v2.3.0 FSR3.1.5 source closure under
  `upstream/ffx-2.3.0`. Its object-only `fsr3-host-3.1.5-scaffold` target and
  graph test remain host-port gates, but the dedicated SDK-2.3 Vulkan
  `FfxInterface` bridge is now runnable through the opaque 3.1.5 public ABI
  and experimental Q2RTX `flt_upscaler 3` path. It selects embedded modules
  by scheduler name, imports application images, owns SDK resources/mip views,
  records initialization copies, barriers, descriptors, and compute jobs, and
  restores caller image states after unregistering them. The complete 3.1.5
  source closure is compiled with a private `ffxVk315...` prefix because its
  public C symbols otherwise collide with the simultaneously linked 1.1.4
  runtime. The portable ABI intentionally remains versioned and opaque.

  `ffx-vulkan::radiancecache-contract` is intentionally only a public
  host-buffer contract: it validates the five application-owned buffer roles,
  their Vulkan access states, finite hyperparameters, and the two 32-bit atomic
  counters. It must not be presented as Radiance Caching inference, training,
  sample generation, or an official provider. Q2RTX has no producer for those
  buffers and must keep the current official-provider diagnostic unavailable
  on the RX 6800M.
  The fixed Q2-compatible 3.1.5 SPIR-V set is now
  generated under `generated/ffx-2.3.0/vk/fsr3upscaler-q2-v2` (ten pass
  wrappers plus AccumulateSharpen) with a pinned DXC,
  SHA-256, and Vulkan 1.2 validation. It uses the documented LDS-only SPD
  permutation (`FFX_SPD_NO_WAVE_OPERATIONS=1`) so every intermediate
  reduction has an explicit workgroup barrier; GPU-AV found races in the
  generic wave-based variant on the RDNA2 Vulkan target. It deliberately uses
  distinct binding
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

`extern/ffx-vulkan` is the canonical home for the reusable FSR4 v07 Vulkan
provider and public headers. Q2RTX links `ffx-vulkan::fsr4-v07-vulkan`; do not
reintroduce a renderer-private copy. The provider uses versioned
`ffxFsr4V07…` entry points and an explicit backend-interface setter so it can
coexist with AMD's unversioned SDK symbols. Its dependency-complete FSR4
subset installs as a CMake package; the full FSR3 source closure is consumed
with `add_subdirectory` as documented in its README.

For a Vulkan application that vendors this directory, prefer the documented
`ffx-vulkan::effects` convenience target or start from
`examples/full-stack`. It links the compatible versioned FSR3 1.1.4, 3.1.5,
and 3.1.6 targets, FSR4 v07, and the WSI policy without pretending that FSR4
v07 is FSR 4.1.1. The host CMake project must enable both C and C++.

`ffx-vulkan` normally ships no source-v07 model data. A distributor that has a
matching licensed bundle can enable
`FFX_VK_PORTABLE_INSTALL_FSR4_V07_ASSETS` with
`FFX_VK_PORTABLE_FSR4_V07_ASSET_DIR`; CMake validates and installs only the
notice, five shared assets, and six model-prefixed SPIR-V/initializer/
pre-weight/manifest sets. The installed package exposes its opt-in location as
`FFX_VK_FSR4_V07_ASSET_DIR`; never copy arbitrary neighbouring blobs.
The installed-package consumer CTest must assert the complete 288-file opt-in
payload, including RCAS and SPD, whenever that installation option is enabled.

FSR4 v07 callers must bracket every provider dispatch with
`ffxFsr4VkBeginFrame(interface, frame_id)` and retire it with
`ffxFsr4VkRetireFrame(interface, completed_frame_id)` only after their GPU
fence signals. This is the portable ownership boundary for descriptor sets,
staged uploads, and host-visible constant-buffer partitions; do not restore a
renderer-specific automatic pool rotation.

Before every FSR4 dispatch, register each external image with
`ffxFsr4VkSetExternalImageState`. `FfxApiResource` only transports a view, so
the separate state record supplies the real image plus current and restored
Vulkan layout/stage/access. The backend transitions imports to `GENERAL` for
compute and restores them at unregister. Inputs must declare
`FFX_API_RESOURCE_STATE_COMPUTE_READ`; output must declare
`FFX_API_RESOURCE_STATE_UNORDERED_ACCESS`. Do not weaken this back into an
implicit Q2RTX-only GENERAL-layout assumption.

The FSR4 provider validates every supplied SPIR-V module's descriptor-set,
binding, and descriptor-type ABI before it creates pipelines. Keep that
fail-closed check when updating generated assets; the standalone layout test
must cover every model preset and tensor tier. Pipeline descriptor layouts are
then reflected per module (not a generic 44-binding superset), with immutable
samplers and a required write for every declared non-sampler binding.

The installed package also exports
`ffx-vulkan::framegeneration-presenter-policy`. It contains only reusable WSI
policy (FIFO selection, image-count/pair validation, and image/GPU semaphore
indexing, reset-slot generated-image suppression, and mode-transition
quiescence); applications must retain acquire, submit, present, fence, and
platform-window ownership. Q2RTX must use this policy rather than recreating
those invariants locally.

Frame generation consumes the HUDless `VKPT_IMG_TAA_OUTPUT` scene. FSR3.1.4,
FSR3.1.5, and the source-v07 FSR4 provider all publish their reconstructed
display image there, so each may feed the public analytical FI/OF schedulers;
this must never be labelled AMD ML Frame Generation. When it is active, UI is rendered once to a lazy per-frame-slot RGBA16F texture with
premultiplied RGB/source alpha and published as
`VKPT_TEMPORAL_UI_SEPARATE_TEXTURE`; final presentation composites
`ui.rgb + scene.rgb * (1-ui.a)`. Do not replace this with a second direct UI
draw unless falling back after alpha-target creation/recording failed.

`ffx-vulkan::fsr3-vk-framegeneration-3.1.6` owns fixed maximum-render and
display-sized shared resources. A host must recreate its context after a
resize: `RecordPrepare` rejects render extents above the maximum and any scene
colour extent other than the fixed display extent, while `RecordDispatch`
requires that exact display extent. An all-zero interpolation rectangle means
the entire display; every other rectangle must be non-empty and contained in
it. Keep these fail-closed guards and their API-smoke coverage: otherwise a
stale resize can record clipped generated-frame work and look like strobing.
All host-provided FI/OF motion, camera, timing, and luminance floats must also
be finite; timing/near/far/view scale/FOV must be positive and luminance has a
non-negative ordered range. Do not pass NaN/Inf through to the SDK: normal C++
comparisons do not reject them and a bad constant can become a temporal flash.

DXIL tooling:

```sh
python -m unittest discover -s tools/ffx_dxil/tests -v
python tools/ffx_dxil/dxil_container_tool.py --help
```

For shader changes, run `spirv-val` on every generated SPIR-V module and run
the ABI/reflection validator documented beside the FSR4 compile script.  Live
tests must capture the full Vulkan validation message, not a clipped overlay.

The known-good FSR4 shader compiler is
`/home/fireburn/DirectXShaderCompiler/build-release/bin/dxc`, built from the
separate converter checkout at commit `3b80347af` (`Fix Objective-C rewriter
build with current LLVM`).  That compatibility change is committed in the
converter repository rather than being an untracked local workaround. Generate
and validate every fixed preset plus DRS with:

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

## Generated Q2RTX shader-table ABI

`global_textures.h` is a host/SPIR-V ABI, not merely a shared include.  The
sampled `TEX_*` bindings begin immediately after the global `IMG_*` storage
image table, so adding or reordering an image changes every sampled framebuffer
binding.  `shader_global_texture_abi` reflects the generated `shader_vkpt/*.spv`
modules and rejects an old table.  Keep `compileShaders.cmake`'s output-clean
stamp and `setup/package_shaders.cmake`'s fresh-archive behavior: incremental
`7z a` archives and loose user shader caches can otherwise retain modules built
against a previous table.  Bump the launcher shader-layout marker whenever this
table changes; it preserves the old loose cache as a recoverable backup.

## Packaged and portable launches

`Q2RTX_DATA_DIR`, used by the installed launcher for portable/staged data, is
also honored by the client before the `/usr/share/quake2rtx` default. Do not
remove this: otherwise a staged executable can silently load stale system
shader archives. Keep the RTX Video page as navigation only; its FSR controls
belong on `temporal_settings`, which is protected by `video_menu_layout` so
low-height modes remain readable.
