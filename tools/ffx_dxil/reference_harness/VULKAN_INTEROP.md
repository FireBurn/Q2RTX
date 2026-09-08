# Official provider interoperability investigation

Status: 2026-09-08. The SDK 4.1.1 provider executes the deterministic
reset/history/reseed probe through Wine/vkd3d on local RDNA2. Native Q2RTX
integration is not implemented. RDNA4 testing is available from the user;
the remote operating system has not yet been specified.

## Inspected implementations

OptiScaler v0.9.4 (`7534ad00`) implements Vulkan/DX12 sharing in
`OptiScaler/upscalers/IFeature_VkwDx12.cpp`. It creates D3D12 shared resource
handles, imports them with `VK_EXTERNAL_MEMORY_HANDLE_TYPE_D3D12_RESOURCE_BIT`,
and imports shared D3D12 fences with
`VK_EXTERNAL_SEMAPHORE_HANDLE_TYPE_D3D12_FENCE_BIT`.
This is a Windows Vulkan interface, including when executed through Wine.
It is not an external-FD interface usable directly by native Linux Q2RTX.

Source: https://github.com/optiscaler/OptiScaler/blob/7534ad00bf9e590eedb99e8dd9fd8c89dae3654f/OptiScaler/upscalers/IFeature_VkwDx12.cpp

Proton's `wineopenxr/vkd3d-proton-interop.h` declares Vulkan handle/resource
queries and queue locking. `ID3D12DXVKInteropDevice2` adds interop command
queues/allocators and begin/end Vulkan command-buffer access. The device
extension also declares `CreateResourceFromBorrowedHandle`. These are
promising same-process, same-device interfaces; they do not make a VkImage
handle valid on a different VkDevice or in another process.

Source: https://github.com/ValveSoftware/Proton/blob/proton_11.0/wineopenxr/vkd3d-proton-interop.h

## Next executable milestone

The optional PE loader now invokes the Unix helper successfully on both
shared texture and fence handles: `FFX_UNIX_FD` reports status=0, valid=1,
unload=0 for each. All three GPU round trips and four checker frames pass in
`/tmp/q2rtx-int8-runtime.NAfvoo/interop-unix-fd-linked.log`. The loader refuses
runtime versions other than Wine 11.17. Enable only for this experiment with
`FFX_PROBE_UNIX_FD_LIB='\??\Z:\tmp\q2rtx-probe-unix-fd.so'` (an NT path).
The helper must link explicitly against the installed
`/usr/lib/wine-staging-11.17/wine/x86_64-unix/ntdll.so`; the earlier unlinked
build failed to load. Append that input to the compile command below.
Descriptors are currently validated and immediately closed, not transferred
or Vulkan-imported. Next implement Unix socket SCM_RIGHTS transport with
explicit memory/semaphore metadata and test imports on a native device.

Shared-resource testing now passes: the probe creates the texture with
`D3D12_HEAP_FLAG_SHARED`, exports and reopens it using DX12 shared-handle
methods, then performs the Vulkan clear/readback on the reopened resource.
Its completion fence is also exported/reopened with `D3D12_FENCE_FLAG_SHARED`.
The 64/64 pixel check and all four provider checks pass in
`interop-shared.log`. This does not yet test two devices or processes.

Wine source audit at commit `c860583954f29d7523d48e480aaf445d4217ccde`:
`win32u/vulkan.c` maps Windows memory handles to host opaque FDs (or DMA-BUF),
and semaphore handles to opaque FDs. Exported allocations use
`vkGetMemoryFdKHR` internally and register the FD as a D3DKMT resource.
`win32u/d3dkmt.c` retrieves local-object FDs using
`wine_server_handle_to_fd`. These are internal Unix-side paths, not a public
Windows Vulkan FD API. The installed Wine build must be checked separately;
the pinned upstream implementation is architectural evidence only. Next
investigate a versioned Unix helper boundary capable of duplicating and
transferring those FDs with explicit resource metadata and ownership.

Sources:
- https://github.com/ValveSoftware/wine/blob/c860583954f29d7523d48e480aaf445d4217ccde/dlls/win32u/vulkan.c
- https://github.com/ValveSoftware/wine/blob/c860583954f29d7523d48e480aaf445d4217ccde/dlls/win32u/d3dkmt.c

Installed-runtime follow-up: `/etc/eselect/wine/bin/wine` resolves to
`/usr/bin/wine-staging-11.17`. Dynamic-symbol inspection of its
`x86_64-unix/ntdll.so` confirms `wine_server_call`, `wine_server_send_fd`,
`wine_server_fd_to_handle`, and `wine_server_handle_to_fd` are exported.
Its `win32u.so` does **not** export `d3dkmt_object_get_fd` or
`d3dkmt_open_resource`; directly resolving those private helpers will not work.

The matching upstream wine-11.17 sources distinguish a shared wrapper handle
from the underlying FD-bearing object. `d3dkmt_object_open` sends a server
request with the shared handle and object type; the reply supplies an object
handle, which the local FD helper then translates. A prototype must use
matching Wine server protocol definitions inside a Unix library, close the
temporary object handle and duplicated FD, and transfer the FD over a Unix
socket. It must reject a mismatched Wine version/protocol rather than guess
request numbers or inspect private in-memory object layouts. Installed SDK
headers include `wine/unixlib.h` but not server protocol headers, so a
matching Wine source/development input is required for this experiment.
This is an experimental route, not a stable distribution dependency yet.

Matching source:
- https://github.com/wine-mirror/wine/blob/wine-11.17/dlls/win32u/d3dkmt.c
- https://github.com/wine-mirror/wine/blob/wine-11.17/server/d3dkmt.c

`probe_unix_fd.c` now implements the Unix half of this experiment: open a
resource/synchronization object from its shared handle, obtain its FD, check
it with `fstat`, and close both temporary handles. It does not send the FD or
import it into another Vulkan device yet. Its protocol-961 build guard rejects
other headers. The PE caller must verify the runtime Wine version before
invoking it; `probe_unix_loader.h` now provides that optional caller for the
standalone probe, not the game. Compilation passed with:

```sh
cc -std=c11 -fPIC -shared -Wall -Wextra -Werror -D__WINESRC__ \
  -I/tmp/q2rtx-wine-headers.IYXf7k -I/usr/include/wine \
  -I/usr/include/wine/windows \
  tools/ffx_dxil/reference_harness/probe_unix_fd.c \
  /usr/lib/wine-staging-11.17/wine/x86_64-unix/ntdll.so \
  -o /tmp/q2rtx-probe-unix-fd.so
```

The temporary include directory contains unmodified `wine/server.h` and
`wine/server_protocol.h` downloaded from upstream tag wine-11.17. Only compile
coverage existed initially; live shared texture/fence invocation now passes
as recorded above. This still does not prove native-device importability.

The enabled-extension query on the actual Wine-facing provider device reports
`enabled_memory_fd=0 enabled_semaphore_fd=0 enabled_memory_win32=1`.
The same invocation passes both 64-word/pixel interop round trips and all four
220/220 checker checks (`interop-external.log`). This is a device-enabled
extension list, not a physical-device capability list: it says nothing about
the underlying Unix driver's FD support. Direct FD export through this
Windows Vulkan device is not available. Investigate Wine's Win32 external
memory/fence translation before choosing a native Linux transport; never
reinterpret the resulting Windows handles as Unix descriptors.

The independent 8x8 texture test now passes as well: a D3D12-owned RGBA8
image is cleared and copied using Vulkan after explicit UNDEFINED ->
TRANSFER_DST -> TRANSFER_SRC transitions. DX12 submission and readback
verify all 64 pixels. The resource is private to this test and discarded,
so it does not yet test restoring a host image for a subsequent FSR dispatch.
Next apply interop to actual provider inputs/outputs and establish native
Linux external-memory transport.

A real buffer round trip now passes locally: the probe creates a vkd3d interop
allocator, obtains its Vulkan command buffer, records `vkCmdFillBuffer` on a
D3D12 readback resource's Vulkan buffer, ends interop and submits through
D3D12. After fence/device verification all 64 CPU-read words match the known
pattern. The subsequent four official-provider pixel checks also pass.
This establishes same-process buffer command interoperability, not image
sharing or transport into the native Linux process. Next apply the pattern
to a texture with explicit layout transitions.

`probe_interop.h` now exercises the base interface's device, queue, image
and layout queries on each dispatch's actual output. The local four-frame
forced-INT8 run returns valid/non-null handles, queue family 0 and image
layout 1 (GENERAL), while all pixel checks pass. It does not yet submit a
Vulkan command through those handles. Ordinary Windows without the interface
reports unavailable and can still run the DX12 pixel probe.

The reference probe now queries both `ID3D12DXVKInteropDevice` and
`ID3D12DXVKInteropDevice2` using their published IIDs. On the local
vkd3d-proton `634d341a5a312a3`, both return S_OK and non-null interfaces.
The same invocation still passes all four provider pixel checks. This removes
interface availability as the next uncertainty; method-level resource and
command-buffer interoperability remains untested.

Extend the Windows reference probe to query the vkd3d interop interface and
record a Vulkan copy on the provider device, with a known pattern checked
after a DX12 dispatch. Verify image layouts and queue synchronization before
involving Q2RTX. On Windows, instead evaluate shared D3D12 resources/fences
against an application-owned Vulkan device matching the same adapter.

For native Linux, a Wine helper must expose shareable allocation and
synchronization handles through an explicit transport, or an in-process Wine
bridge must preserve the relevant calling conventions and Vulkan dispatch
ownership. Raw Vulkan handles and Win32 HANDLE values cannot be sent to the
native process and treated as imported Linux file descriptors. Prototype the
image/fence round trip first; do not add a selectable game provider before it
works. A CPU-copy helper may aid diagnosis but is not the intended final
real-time GPU integration.

The reusable boundary should own provider contexts and transfer resources,
while the host retains its device, queues and presentation. Require adapter
identity, exact format/extent support, explicit completion and resize teardown.
Use the existing native FSR implementation on any failed capability check.
The official DLLs remain caller-supplied; an eligibility override stays an
explicit experimental option and is unnecessary for the RDNA4 control.
