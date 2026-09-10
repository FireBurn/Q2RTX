# Capturing selected FidelityFX pipelines through Proton

A raw shader dump is not a runnable FidelityFX implementation. The provider's
pass selection, root parameters, constants, resources, model uploads,
synchronization, and temporal lifetime are equally important. This procedure
captures both sides: translated shader pairs and the D3D12 command stream that
defines how they are used.

## 1. Establish a controlled reference

Use a legally installed provider DLL with a small D3D12 reference application
or SDK sample under a current Proton/vkd3d-proton. Record all of the following
before capture:

- provider DLL SHA-256 and file/product version;
- Proton, vkd3d-proton, Mesa/RADV, kernel, and GPU/firmware versions;
- GPU PCI ID and advertised wave/matrix capabilities;
- render and display resolution, quality mode, HDR state, and effect flags.

Run a minimal matrix of scenes: static camera, camera translation, disocclusion,
animated geometry, emissives, particles/transparency, and a resolution or
camera cut that forces history reset. A single pretty frame cannot establish a
correct temporal implementation.

## 2. Dump vkd3d source bytecode and translated SPIR-V

`VKD3D_SHADER_DUMP_PATH` emits files named `$hash.{dxbc,dxil,spv}`. Use a new,
empty directory for every run because vkd3d-proton does not overwrite existing
dumps. Disabling its cache ensures the relevant pipelines are compiled during
the run.

For a direct Proton invocation:

```sh
capture_dir="$(mktemp -d /tmp/q2rtx-ffx-capture.XXXXXX)"
VKD3D_SHADER_DUMP_PATH="$capture_dir" \
VKD3D_SHADER_CACHE_PATH=0 \
VKD3D_TIMESTAMP_PROFILE="$capture_dir/dispatch-profile.csv" \
proton run /path/to/reference.exe
```

For a Steam launch option, create an empty directory first and use its absolute
path:

```text
VKD3D_SHADER_DUMP_PATH=/absolute/path/to/empty-dump VKD3D_SHADER_CACHE_PATH=0 VKD3D_TIMESTAMP_PROFILE=/absolute/path/to/dispatch-profile.csv %command%
```

The timestamp profile requires vkd3d-proton built with
`-Denable_profiling=true`; omit that variable when using a build without it.
When available, it helps narrow the capture to compute PSOs and reports shader
and root-signature hashes. It does not record sufficient bindings or constant
data by itself.

Create a RenderDoc capture of the same controlled frame. A vkd3d-proton build
configured with RenderDoc support can also use `VKD3D_AUTO_CAPTURE_SHADER` with
a selected hash and `VKD3D_AUTO_CAPTURE_COUNTS` to target the submission. The
upstream
[`docs/single_dispatch_capture.md`](https://github.com/HansKristian-Work/vkd3d-proton/blob/master/docs/single_dispatch_capture.md)
describes the exporter plugin and standalone `d3d12-replayer` flow. A
sufficiently recent build embeds
`NonSemantic.dxil-spirv.signature` records in SPIR-V; retain them because they
map push constants and descriptor tables back to the D3D12 root signature.

After identifying effect hashes in the timing profile and RenderDoc event list,
put one lowercase 16-digit hash on each line of a local text file and run the
`capture-manifest` command documented in [README.md](README.md). Keep the raw
dump outside source control.

## 3. Record orchestration, not only shaders

For a trace-enabled vkd3d reference runtime, `dispatch_trace.py TRACE OUTPUT`
indexes direct compute calls with retained shader, root-signature, constant
buffer addresses and descriptor-table addresses. OUTPUT must not exist.
It rejects unresolved shaders and indirect/bundle execution rather than
silently making an incomplete direct-call sequence look complete. Its JSON
is explicitly not replay-ready: it does not recover descriptor contents,
resource lifetimes, synchronization, or constant sizes/bytes. Keep the index
beside the private capture, not in distributable shader assets.

For every dispatch, record these fields in execution order:

- vkd3d shader hash, root-signature hash, and `Dispatch(x, y, z)` dimensions;
- reflected/local workgroup size and active permutation/configuration flags;
- every root constant and constant-buffer byte, including padding;
- descriptor register, space, array index, view type, and sampler;
- resource format, extent, mips, layers, usage, and logical role;
- resource state/layout before and after the pass, barriers, queue, and semaphore
  or fence dependencies;
- scratch aliasing, persistent history ownership, initialization/clear values,
  and destruction/reset points;
- model/weight upload contents, alignment, destination offset, and the first
  dispatch that consumes them;
- indirect-dispatch argument buffers and the pass that produces them.

Also document the external signal contracts that cannot be inferred from DXIL:

- input/output color domain, exposure and pre-exposure;
- depth convention, linearization, sign, units, and reversed-Z state;
- motion-vector direction, units, jitter inclusion, and render/display scaling;
- jitter sequence and phase count;
- reactive, transparency/composition, HUD-less, and frame-generation masks;
- frame IDs, camera cuts, resize/reset behavior, and valid-history rules.

Frame generation additionally needs the exact `Configure -> Prepare -> Dispatch`
ordering, real/generated presentation IDs, optical-flow cadence, UI composition
mode, swapchain pacing, and async-compute ownership transitions.

## 4. Prove an extracted graph incrementally

Replay or port one dispatch at a time and capture its intermediate output.
Compare against the reference using floating-point error images and NaN/Inf
counts before enabling the next pass. Validate at least native and non-native
render sizes, motion, disocclusion, camera cuts, and repeated history resets.

Do not treat matching final output on a static frame as proof: temporal feedback
can hide wrong motion scaling, uninitialized history, missing barriers, or an
incorrect permutation for several frames.

## Provenance boundary

Commit manifests, hashes, capture notes, and independently written integration
code only. Do not commit or publish provider DLLs, extracted DXIL, translated
SPIR-V, weights, RenderDoc captures containing proprietary buffers, or
standalone replay packages unless their license explicitly permits it.
