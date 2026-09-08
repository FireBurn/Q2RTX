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
