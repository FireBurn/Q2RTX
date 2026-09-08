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
