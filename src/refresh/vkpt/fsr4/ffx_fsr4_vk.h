// ffx_fsr4_vk.h
// Vulkan backend for FidelityFX FSR4 (INT8 / dot4add path)
//
// Drop-in replacement for the DX12 backend. Implements the FfxInterface
// callback table so the existing ffx_provider_fsr4 dispatch logic can run
// on Vulkan without modification.
//
// Requirements:
//   Vulkan 1.3  (or 1.2 + VK_KHR_shader_integer_dot_product)
//   VK_KHR_shader_float16_int8          (for int8 storage in SPIR-V)
//   VK_KHR_16bit_storage                (for float16 I/O)
//   VK_KHR_shader_integer_dot_product   (core in 1.3, promoted from extension)
//
// Build FSR4 INT8 shaders to SPIR-V with:
//   dxc -spirv -T cs_6_4 -enable-16bit-types -HV 2021
//       -DWMMA_ENABLED=0 -DFSR4_ENABLE_DOT4=1
//       -fspv-target-env=vulkan1.3
//       -fspv-extension=SPV_KHR_integer_dot_product
//       -I <sdk>/upscalers/fsr4/dx12
//       -I <sdk>/api/internal/dx12
//       -E <entry> <shader>.hlsl -Fo <shader>.spv

#pragma once

#include <vulkan/vulkan.h>
#include <stdint.h>
#include <stddef.h>

#include "ffx_types_q2rtx.h"

#ifdef __cplusplus
extern "C" {
#endif

// ---------------------------------------------------------------------------
// Opaque context handed back to the caller. Store the VkDevice and the
// allocator so the backend can create/destroy its own resources.
// ---------------------------------------------------------------------------
typedef struct FfxFsr4VkContext FfxFsr4VkContext;

// ---------------------------------------------------------------------------
// One SPIR-V blob. Caller owns the memory; must remain valid until
// ffxFsr4VkDestroyContext().
// ---------------------------------------------------------------------------
typedef struct FfxFsr4VkShaderBlob {
    const uint32_t* spirv;      ///< Pointer to SPIR-V words
    size_t          sizeBytes;  ///< Size in bytes (must be multiple of 4)
    const char*     entryPoint; ///< e.g. "main" or the DXC-exported name
} FfxFsr4VkShaderBlob;

// FSR4 has: pass0 (pre), pass1-12 (model), pass13 (post), rcas, spd_auto_exposure
#define FFX_FSR4_VK_PASS_COUNT 16

typedef struct FfxFsr4VkCreateInfo {
    VkDevice                    device;
    VkPhysicalDevice            physicalDevice;

    // Optional custom allocator; pass NULL for default VkAllocationCallbacks
    const VkAllocationCallbacks* allocator;

    // All SPIR-V blobs for each shader pass.
    // Index mapping (same as DX12 pass IDs):
    //   [0]    = pre-pass  (fsr4_model_v07_i8_<preset>/pre.hlsl)
    //   [1-12] = model passes (passes_<res>.hlsl, MLSR_PASS_N define)
    //   [13]   = post-pass (post.hlsl)
    //   [14]   = RCAS sharpening (rcas.hlsl)
    //   [15]   = SPD auto-exposure (spd_auto_exposure.hlsl)
    FfxFsr4VkShaderBlob shaders[FFX_FSR4_VK_PASS_COUNT];

    // Scratch memory for the backend's internal bookkeeping.
    // Allocate at least ffxFsr4VkGetScratchMemorySize() bytes.
    void*  scratchBuffer;
    size_t scratchBufferSize;
} FfxFsr4VkCreateInfo;

// ---------------------------------------------------------------------------
// Query minimum scratch memory required.
// ---------------------------------------------------------------------------
size_t ffxFsr4VkGetScratchMemorySize(void);

// ---------------------------------------------------------------------------
// Create a Vulkan backend context and fill in the FfxInterface table.
// outInterface->device will be set to the opaque FfxFsr4VkContext*.
// ---------------------------------------------------------------------------
VkResult ffxFsr4VkCreateContext(
    const FfxFsr4VkCreateInfo* createInfo,
    FfxInterface*              outInterface   ///< Populated on success
);

// ---------------------------------------------------------------------------
// Destroy and free all Vulkan objects created by ffxFsr4VkCreateContext().
// Call AFTER destroying the FfxEffectContext that used this backend.
// ---------------------------------------------------------------------------
void ffxFsr4VkDestroyContext(FfxFsr4VkContext* ctx);

#ifdef __cplusplus
}
#endif
