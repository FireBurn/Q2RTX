// ffx_fsr4_vk.c  —  Vulkan backend for FidelityFX FSR4 (INT8 / dot4add path)
//
// Implements all 19 FfxInterface function pointers so the existing
// ffx_provider_fsr4 dispatch logic runs on Vulkan unchanged.
//
// Two descriptor set layouts are used:
//
//   LAYOUT_MODEL  (passes 1-12): 4 storage buffers
//     Compiled with: -fvk-t-shift 0 0 -fvk-u-shift 2 0
//     bind 0  STORAGE_BUFFER  input SRV    (t0 ByteAddressBuffer)
//     bind 1  STORAGE_BUFFER  weights SRV  (t1 ByteAddressBuffer)
//     bind 2  STORAGE_BUFFER  output UAV   (u0 RWByteAddressBuffer)
//     bind 3  STORAGE_BUFFER  scratch UAV  (u1 RWByteAddressBuffer)
//
//   LAYOUT_FULL (passes 0, 13, 14, 15): textures + images + buffers + cbuffers
//     Compiled with: -fvk-t-shift 0 0 -fvk-s-shift 0 0 -fvk-u-shift 21 0 -fvk-b-shift 43 0
//     bind  0..20  COMBINED_IMAGE_SAMPLER  t0..t20 (Texture2D SRVs + samplers)
//     bind 21..33  STORAGE_IMAGE           u0..u12 (RWTexture2D UAVs)
//     bind 32      STORAGE_BUFFER          u11=ScratchBuffer (pre/post/spd)
//                  STORAGE_IMAGE           u11=rw_rcas_output (rcas only)
//     bind 34      UNIFORM_BUFFER          cbPass_Weights
//     bind 43      UNIFORM_BUFFER          b0 (MLSR_Optimized_Constants / cbRCAS / etc.)

#include "ffx_fsr4_vk.h"
#include <string.h>
#include <stdio.h>
#include <stdlib.h>

// ── tuneable limits ──────────────────────────────────────────────────────────
#define MAX_RESOURCES       192
#define MAX_PIPELINES        32
#define MAX_PENDING_JOBS     96
#define MAX_DESC_SETS        64   /* per pool — need 14 per frame, 3 pools */
#define NUM_POOL_FRAMES       3   /* triple-buffered descriptor pools */
#define MAX_STAGING          32
#define CBUF_RING_BYTES    (512 * 1024)

// Slot offsets inside LAYOUT_FULL (matching -fvk-*-shift flags)
#define SLOT_SRV_TEX_BASE    0    // t0..t20 → COMBINED_IMAGE_SAMPLER
#define SLOT_UAV_BASE       21    // u0..u12 → STORAGE_IMAGE (or STORAGE_BUFFER for u11)
#define SLOT_SCRATCH        32    // u11+21  (STORAGE_BUFFER in pre/post, STORAGE_IMAGE in rcas)
#define SLOT_CBUF_WEIGHTS   34    // cbPass_Weights → UNIFORM_BUFFER
#define SLOT_CBUF_MAIN      43    // b0+43   → UNIFORM_BUFFER
#define FULL_LAYOUT_COUNT   44    // bindings 0..43

// ── internal types ───────────────────────────────────────────────────────────

typedef enum { RES_NONE=0, RES_BUFFER=1, RES_IMAGE=2 } ResKind;

typedef struct {
    ResKind        kind;
    VkBuffer       buf;
    VkDeviceMemory mem;
    VkDeviceSize   size;
    VkImage        img;
    VkImageView    view;
    VkFormat       fmt;
    uint32_t       w, h;
    int            external;
    int            needsInit;  /* 1 = image needs UNDEFINED→GENERAL transition */
} VkRes;

typedef enum { PIPE_MODEL=0, PIPE_FULL=1 } PipeKind;

typedef struct {
    VkPipeline            pipeline;
    VkPipelineLayout      layout;
    VkDescriptorSetLayout dsLayout;
    PipeKind              kind;
    uint32_t              cbufBytes;
} VkPipe;

typedef struct { FfxGpuJobDescription d; } Job;

typedef struct {
    VkBuffer       buf;
    VkDeviceMemory mem;
    uint32_t       dstIdx;
    VkDeviceSize   size;
} Staging;

struct FfxFsr4VkContext {
    VkDevice                     dev;
    VkPhysicalDevice             phys;
    const VkAllocationCallbacks* alloc;

    VkDescriptorPool  pool[NUM_POOL_FRAMES];
    uint32_t          poolIdx;        /* cycles 0,1,2 each frame */
    VkSampler         linearSampler;

    VkDescriptorSetLayout modelLayout;
    VkDescriptorSetLayout fullLayout;
    VkDescriptorSetLayout fullLayoutRcas;

    VkBuffer       cbRing;
    VkDeviceMemory cbMem;
    uint8_t*       cbMap;
    uint32_t       cbOff;

    VkRes    res[MAX_RESOURCES];
    uint32_t resCount;

    VkPipe   pipe[MAX_PIPELINES];

    Job      jobs[MAX_PENDING_JOBS];
    uint32_t jobCount;

    Staging  staging[MAX_STAGING];
    uint32_t stagingCount;

    FfxFsr4VkShaderBlob blobs[FFX_FSR4_VK_PASS_COUNT];
};

// ── helpers ──────────────────────────────────────────────────────────────────

static uint32_t findMemType(VkPhysicalDevice phys, uint32_t bits, VkMemoryPropertyFlags f)
{
    VkPhysicalDeviceMemoryProperties p;
    vkGetPhysicalDeviceMemoryProperties(phys, &p);
    for (uint32_t i = 0; i < p.memoryTypeCount; i++)
        if ((bits & (1u<<i)) && (p.memoryTypes[i].propertyFlags & f) == f)
            return i;
    return UINT32_MAX;
}

static VkResult mkBuf(FfxFsr4VkContext* c, VkDeviceSize sz,
                      VkBufferUsageFlags usage, VkMemoryPropertyFlags mf,
                      VkBuffer* ob, VkDeviceMemory* om)
{
    VkBufferCreateInfo bi;
    memset(&bi, 0, sizeof(bi));
    bi.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO;
    bi.size=sz; bi.usage=usage; bi.sharingMode=VK_SHARING_MODE_EXCLUSIVE;
    VkResult r = vkCreateBuffer(c->dev, &bi, c->alloc, ob);
    if (r) return r;
    VkMemoryRequirements mr;
    vkGetBufferMemoryRequirements(c->dev, *ob, &mr);
    uint32_t mt = findMemType(c->phys, mr.memoryTypeBits, mf);
    if (mt == UINT32_MAX) { vkDestroyBuffer(c->dev, *ob, c->alloc); return VK_ERROR_OUT_OF_DEVICE_MEMORY; }
    VkMemoryAllocateInfo ai;
    memset(&ai, 0, sizeof(ai));
    ai.sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO;
    ai.allocationSize=mr.size; ai.memoryTypeIndex=mt;
    r = vkAllocateMemory(c->dev, &ai, c->alloc, om);
    if (r) { vkDestroyBuffer(c->dev, *ob, c->alloc); return r; }
    return vkBindBufferMemory(c->dev, *ob, *om, 0);
}

static VkResult mkImg(FfxFsr4VkContext* c, uint32_t w, uint32_t h,
                      VkFormat fmt, VkImageUsageFlags usage,
                      VkImage* oi, VkDeviceMemory* om, VkImageView* ov)
{
    VkImageCreateInfo ii;
    memset(&ii, 0, sizeof(ii));
    ii.sType = VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO;
    ii.imageType=VK_IMAGE_TYPE_2D; ii.format=fmt;
    ii.extent=(VkExtent3D){w, h>0?h:1, 1};
    ii.mipLevels=1; ii.arrayLayers=1;
    ii.samples=VK_SAMPLE_COUNT_1_BIT;
    ii.tiling=VK_IMAGE_TILING_OPTIMAL; ii.usage=usage;
    ii.sharingMode=VK_SHARING_MODE_EXCLUSIVE;
    ii.initialLayout=VK_IMAGE_LAYOUT_UNDEFINED;
    VkResult r = vkCreateImage(c->dev, &ii, c->alloc, oi);
    if (r) return r;
    VkMemoryRequirements mr;
    vkGetImageMemoryRequirements(c->dev, *oi, &mr);
    uint32_t mt = findMemType(c->phys, mr.memoryTypeBits, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT);
    if (mt == UINT32_MAX) { vkDestroyImage(c->dev, *oi, c->alloc); return VK_ERROR_OUT_OF_DEVICE_MEMORY; }
    VkMemoryAllocateInfo ai;
    memset(&ai, 0, sizeof(ai));
    ai.sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO;
    ai.allocationSize=mr.size; ai.memoryTypeIndex=mt;
    r = vkAllocateMemory(c->dev, &ai, c->alloc, om);
    if (r) { vkDestroyImage(c->dev, *oi, c->alloc); return r; }
    r = vkBindImageMemory(c->dev, *oi, *om, 0);
    if (r) { vkFreeMemory(c->dev,*om,c->alloc); vkDestroyImage(c->dev,*oi,c->alloc); return r; }
    VkImageViewCreateInfo vi;
    memset(&vi, 0, sizeof(vi));
    vi.sType = VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO;
    vi.image=*oi; vi.viewType=VK_IMAGE_VIEW_TYPE_2D; vi.format=fmt;
    vi.subresourceRange=(VkImageSubresourceRange){VK_IMAGE_ASPECT_COLOR_BIT,0,1,0,1};
    r = vkCreateImageView(c->dev, &vi, c->alloc, ov);
    if (r) { vkFreeMemory(c->dev,*om,c->alloc); vkDestroyImage(c->dev,*oi,c->alloc); }
    return r;
}

static VkFormat toVkFmt(FfxSurfaceFormat f)
{
    switch(f) {
    case FFX_SURFACE_FORMAT_R32G32B32A32_FLOAT:  return VK_FORMAT_R32G32B32A32_SFLOAT;
    case FFX_SURFACE_FORMAT_R16G16B16A16_FLOAT:  return VK_FORMAT_R16G16B16A16_SFLOAT;
    case FFX_SURFACE_FORMAT_R16G16_FLOAT:        return VK_FORMAT_R16G16_SFLOAT;
    case FFX_SURFACE_FORMAT_R32_FLOAT:           return VK_FORMAT_R32_SFLOAT;
    case FFX_SURFACE_FORMAT_R16_FLOAT:           return VK_FORMAT_R16_SFLOAT;
    case FFX_SURFACE_FORMAT_R8G8B8A8_UNORM:      return VK_FORMAT_R8G8B8A8_UNORM;
    case FFX_SURFACE_FORMAT_R11G11B10_FLOAT:     return VK_FORMAT_B10G11R11_UFLOAT_PACK32;
    case FFX_SURFACE_FORMAT_R16G16_UINT:         return VK_FORMAT_R16G16_UINT;
    case FFX_SURFACE_FORMAT_R8_UNORM:            return VK_FORMAT_R8_UNORM;
    default:                                      return VK_FORMAT_R16G16B16A16_SFLOAT;
    }
}

// ── descriptor set layout builders ───────────────────────────────────────────

static VkResult buildModelLayout(FfxFsr4VkContext* c)
{
    VkDescriptorSetLayoutBinding b[4];
    memset(b, 0, sizeof(b));
    for (int i=0; i<4; i++) {
        b[i].binding=i;
        b[i].descriptorType=VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
        b[i].descriptorCount=1;
        b[i].stageFlags=VK_SHADER_STAGE_COMPUTE_BIT;
    }
    VkDescriptorSetLayoutCreateInfo ci;
    memset(&ci, 0, sizeof(ci));
    ci.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO;
    ci.bindingCount=4; ci.pBindings=b;
    return vkCreateDescriptorSetLayout(c->dev, &ci, c->alloc, &c->modelLayout);
}

static VkResult buildFullLayoutVariant(FfxFsr4VkContext* c,
                                        VkDescriptorType scratch_type,
                                        VkDescriptorSetLayout* outLayout)
{
    const uint32_t N = FULL_LAYOUT_COUNT;
    VkDescriptorSetLayoutBinding* b =
        (VkDescriptorSetLayoutBinding*)calloc(N, sizeof(*b));

    // Default all bindings to COMBINED_IMAGE_SAMPLER with immutable sampler.
    // Unused bindings are harmless — shaders won't access them.
    for (uint32_t i=0; i<N; i++) {
        b[i].binding=i;
        b[i].descriptorType=VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
        b[i].descriptorCount=1;
        b[i].stageFlags=VK_SHADER_STAGE_COMPUTE_BIT;
        b[i].pImmutableSamplers=&c->linearSampler;
    }

    // UAV bindings 21..33 (u0..u12): STORAGE_IMAGE by default
    for (uint32_t u=0; u<=12; u++) {
        uint32_t s = SLOT_UAV_BASE + u;
        b[s].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_IMAGE;
        b[s].pImmutableSamplers = NULL;
    }

    // u11 (binding 32): per-variant type — STORAGE_BUFFER for pre/post/spd,
    // STORAGE_IMAGE for rcas
    b[SLOT_SCRATCH].descriptorType = scratch_type;
    b[SLOT_SCRATCH].pImmutableSamplers = NULL;

    // cbPass_Weights (binding 34): UNIFORM_BUFFER
    b[SLOT_CBUF_WEIGHTS].descriptorType = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER;
    b[SLOT_CBUF_WEIGHTS].pImmutableSamplers = NULL;

    // Main constants (binding 43): UNIFORM_BUFFER
    b[SLOT_CBUF_MAIN].descriptorType = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER;
    b[SLOT_CBUF_MAIN].pImmutableSamplers = NULL;

    VkDescriptorSetLayoutCreateInfo ci;
    memset(&ci, 0, sizeof(ci));
    ci.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO;
    ci.bindingCount=N; ci.pBindings=b;
    VkResult r = vkCreateDescriptorSetLayout(c->dev, &ci, c->alloc, outLayout);
    free(b);
    return r;
}

static VkResult buildFullLayout(FfxFsr4VkContext* c)
{
    // pre/post/spd: ScratchBuffer at binding 32 is STORAGE_BUFFER
    VkResult r = buildFullLayoutVariant(c, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
                                         &c->fullLayout);
    if (r != VK_SUCCESS) return r;

    // rcas: rw_rcas_output at binding 32 is STORAGE_IMAGE
    r = buildFullLayoutVariant(c, VK_DESCRIPTOR_TYPE_STORAGE_IMAGE,
                                &c->fullLayoutRcas);
    return r;
}

// ── pipeline creation ─────────────────────────────────────────────────────────

static VkResult mkPipeline(FfxFsr4VkContext* c, uint32_t passIdx, VkPipe* out)
{
    const FfxFsr4VkShaderBlob* blob = &c->blobs[passIdx];

    PipeKind kind = (passIdx==0||passIdx==13||passIdx==14||passIdx==15)
                    ? PIPE_FULL : PIPE_MODEL;
    out->kind    = kind;
    out->dsLayout = (kind==PIPE_MODEL) ? c->modelLayout
                  : (passIdx==14)     ? c->fullLayoutRcas
                  :                     c->fullLayout;

    static const uint32_t cbSz[FFX_FSR4_VK_PASS_COUNT] = {
        88,  // pass 0:  pre
        0,0,0,0,0,0,0,0,0,0,0,0,  // passes 1-12
        88,  // pass 13: post
        16,  // pass 14: RCAS
        32,  // pass 15: SPD
    };
    out->cbufBytes = cbSz[passIdx];

    VkPushConstantRange pcr = {VK_SHADER_STAGE_COMPUTE_BIT, 0, 0};
    if (out->cbufBytes > 0 && out->cbufBytes <= 256) pcr.size = out->cbufBytes;

    VkPipelineLayoutCreateInfo plci;
    memset(&plci, 0, sizeof(plci));
    plci.sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO;
    plci.setLayoutCount=1; plci.pSetLayouts=&out->dsLayout;
    plci.pushConstantRangeCount = pcr.size ? 1 : 0;
    plci.pPushConstantRanges    = pcr.size ? &pcr : NULL;

    VkResult r = vkCreatePipelineLayout(c->dev, &plci, c->alloc, &out->layout);
    if (r) return r;

    VkShaderModuleCreateInfo smci;
    memset(&smci, 0, sizeof(smci));
    smci.sType = VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO;
    smci.codeSize=blob->sizeBytes; smci.pCode=blob->spirv;
    VkShaderModule sm;
    r = vkCreateShaderModule(c->dev, &smci, c->alloc, &sm);
    if (r) { vkDestroyPipelineLayout(c->dev,out->layout,c->alloc); return r; }

    VkComputePipelineCreateInfo cpci;
    memset(&cpci, 0, sizeof(cpci));
    cpci.sType = VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO;
    cpci.stage.sType=VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
    cpci.stage.stage=VK_SHADER_STAGE_COMPUTE_BIT;
    cpci.stage.module=sm;
    cpci.stage.pName=blob->entryPoint ? blob->entryPoint : "main";
    cpci.layout=out->layout;

    r = vkCreateComputePipelines(c->dev, VK_NULL_HANDLE, 1, &cpci, c->alloc, &out->pipeline);
    vkDestroyShaderModule(c->dev, sm, c->alloc);
    if (r) vkDestroyPipelineLayout(c->dev, out->layout, c->alloc);
    return r;
}

// ── FfxInterface callbacks ────────────────────────────────────────────────────

static FfxVersionNumber cbGetSDKVersion(FfxInterface* I) {
    (void)I;
    return FFX_SDK_MAKE_VERSION(FFX_SDK_VERSION_MAJOR,FFX_SDK_VERSION_MINOR,FFX_SDK_VERSION_PATCH);
}

static FfxErrorCode cbGetEffectGpuMemUsage(FfxInterface* I, FfxUInt32 id, FfxApiEffectMemoryUsage* o) {
    (void)I;(void)id; if(o) memset(o,0,sizeof(*o)); return FFX_OK;
}

static FfxErrorCode cbCreateBackendCtx(FfxInterface* I, FfxEffect eff,
    FfxEffectBindlessConfig* bc, FfxUInt32* outId)
{
    (void)eff;(void)bc;
    FfxFsr4VkContext* c = (FfxFsr4VkContext*)I->device;
    if (outId) *outId = 0;
    for (int i=0; i<FFX_FSR4_VK_PASS_COUNT; i++) {
        if (!c->blobs[i].spirv) continue;
        VkResult r = mkPipeline(c, i, &c->pipe[i]);
        if (r != VK_SUCCESS) return FFX_ERROR_BACKEND_API_ERROR;
    }
    return FFX_OK;
}

static FfxErrorCode cbGetDeviceCaps(FfxInterface* I, FfxDeviceCapabilities* o) {
    (void)I;
    if (!o) return FFX_ERROR_INVALID_ARGUMENT;
    memset(o,0,sizeof(*o));
    o->minimumSupportedShaderModel=FFX_SHADER_MODEL_6_4;
    o->fp16Supported=1; o->int8Supported=1;
    return FFX_OK;
}

static FfxErrorCode cbDestroyBackendCtx(FfxInterface* I, FfxUInt32 id) {
    (void)id;
    FfxFsr4VkContext* c = (FfxFsr4VkContext*)I->device;

    // Wait for all GPU work to finish before destroying anything.
    // This prevents "vkDestroyPipeline on in-use" errors when context
    // is recreated while a command buffer is still in flight.
    vkDeviceWaitIdle(c->dev);

    // Destroy pipelines
    for (int i=0; i<MAX_PIPELINES; i++) {
        if (c->pipe[i].pipeline) {
            vkDestroyPipeline(c->dev, c->pipe[i].pipeline, c->alloc);
            vkDestroyPipelineLayout(c->dev, c->pipe[i].layout, c->alloc);
            c->pipe[i].pipeline=VK_NULL_HANDLE;
        }
    }

    // Free all non-external resources (internal textures, buffers) so they
    // don't leak across context recreations.
    for (uint32_t i=0; i<c->resCount; i++) {
        VkRes* r = &c->res[i];
        if (r->kind != RES_NONE && !r->external) {
            if(r->buf)  vkDestroyBuffer(c->dev, r->buf, c->alloc);
            if(r->view) vkDestroyImageView(c->dev, r->view, c->alloc);
            if(r->img)  vkDestroyImage(c->dev, r->img, c->alloc);
            if(r->mem)  vkFreeMemory(c->dev, r->mem, c->alloc);
        }
        memset(r, 0, sizeof(*r));
    }
    c->resCount = 0;

    // Reset cbuffer ring offset
    c->cbOff = 0;

    // Reset all descriptor pools
    for (int p = 0; p < NUM_POOL_FRAMES; p++)
        vkResetDescriptorPool(c->dev, c->pool[p], 0);
    c->poolIdx = 0;

    return FFX_OK;
}

static FfxErrorCode cbCreateRes(FfxInterface* I,
    const FfxCreateResourceDescription* desc, FfxUInt32 id, FfxResourceInternal* out)
{
    (void)id;
    FfxFsr4VkContext* c = (FfxFsr4VkContext*)I->device;
    if (c->resCount >= MAX_RESOURCES) return FFX_ERROR_OUT_OF_RANGE;

    uint32_t idx = c->resCount++;
    VkRes* r = &c->res[idx];
    memset(r, 0, sizeof(*r));

    const int isTexture = (desc->type==FFX_RESOURCE_TYPE_TEXTURE2D ||
                           desc->type==FFX_RESOURCE_TYPE_TEXTURE1D ||
                           desc->type==FFX_RESOURCE_TYPE_TEXTURE3D);
    if (isTexture) {
        VkFormat fmt = toVkFmt(desc->format);
        VkImageUsageFlags usage =
            VK_IMAGE_USAGE_STORAGE_BIT | VK_IMAGE_USAGE_SAMPLED_BIT |
            VK_IMAGE_USAGE_TRANSFER_DST_BIT | VK_IMAGE_USAGE_TRANSFER_SRC_BIT;
        VkResult vr = mkImg(c, desc->width, desc->height>0?desc->height:1,
                            fmt, usage, &r->img, &r->mem, &r->view);
        if (vr) { c->resCount--; return FFX_ERROR_BACKEND_API_ERROR; }
        r->kind=RES_IMAGE; r->fmt=fmt; r->w=desc->width; r->h=desc->height>0?desc->height:1;
        r->needsInit=1;  /* needs UNDEFINED→GENERAL transition before first use */
    } else {
        VkDeviceSize sz = desc->initDataSize>0
            ? (VkDeviceSize)desc->initDataSize
            : (VkDeviceSize)((desc->width?desc->width:1)*4);
        VkResult vr = mkBuf(c, sz,
            VK_BUFFER_USAGE_STORAGE_BUFFER_BIT|VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT|
            VK_BUFFER_USAGE_TRANSFER_DST_BIT|VK_BUFFER_USAGE_TRANSFER_SRC_BIT,
            VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT, &r->buf, &r->mem);
        if (vr) { c->resCount--; return FFX_ERROR_BACKEND_API_ERROR; }
        r->kind=RES_BUFFER; r->size=sz;

        // Upload initial data via a staging buffer kept alive until ExecuteGpuJobs
        if (desc->initData && desc->initDataSize>0) {
            if (c->stagingCount >= MAX_STAGING) return FFX_ERROR_OUT_OF_RANGE;
            Staging* st = &c->staging[c->stagingCount++];
            vr = mkBuf(c, desc->initDataSize,
                VK_BUFFER_USAGE_TRANSFER_SRC_BIT,
                VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT|VK_MEMORY_PROPERTY_HOST_COHERENT_BIT,
                &st->buf, &st->mem);
            if (vr) { c->resCount--; return FFX_ERROR_BACKEND_API_ERROR; }
            void* p;
            vkMapMemory(c->dev, st->mem, 0, desc->initDataSize, 0, &p);
            memcpy(p, desc->initData, desc->initDataSize);
            vkUnmapMemory(c->dev, st->mem);
            st->dstIdx=idx; st->size=(VkDeviceSize)desc->initDataSize;
        }
    }
    out->internalIndex=idx;
    return FFX_OK;
}

static FfxErrorCode cbRegisterRes(FfxInterface* I,
    const FfxApiResource* in, FfxUInt32 id, FfxResourceInternal* out)
{
    (void)id;
    FfxFsr4VkContext* c = (FfxFsr4VkContext*)I->device;
    if (!in||!in->resource) { out->internalIndex=UINT32_MAX; return FFX_OK; }
    if (c->resCount >= MAX_RESOURCES) return FFX_ERROR_OUT_OF_RANGE;

    uint32_t idx = c->resCount++;
    VkRes* r = &c->res[idx];
    memset(r, 0, sizeof(*r));
    r->external = 1;

    // Caller convention:
    //   Texture: resource = VkImageView,  description.type = TEXTURE2D
    //   Buffer:  resource = VkBuffer,     description.type = BUFFER
    if (in->description.type==FFX_RESOURCE_TYPE_TEXTURE2D ||
        in->description.type==FFX_RESOURCE_TYPE_TEXTURE1D) {
        r->kind=RES_IMAGE;
        r->view=(VkImageView)in->resource;
        r->w=in->description.width;
        r->h=in->description.height>0?in->description.height:1;
        r->fmt=toVkFmt(in->description.format);
    } else {
        r->kind=RES_BUFFER;
        r->buf=(VkBuffer)in->resource;
        r->size=in->description.width;
    }
    out->internalIndex=idx;
    return FFX_OK;
}

static FfxApiResource cbGetRes(FfxInterface* I, FfxResourceInternal ri) {
    FfxFsr4VkContext* c = (FfxFsr4VkContext*)I->device;
    FfxApiResource o={0};
    if (ri.internalIndex<c->resCount)
        o.resource = (c->res[ri.internalIndex].kind==RES_IMAGE)
            ? (void*)c->res[ri.internalIndex].view
            : (void*)c->res[ri.internalIndex].buf;
    return o;
}

static FfxErrorCode cbUnregisterRes(FfxInterface* I, FfxCommandList cmd, FfxUInt32 id) {
    (void)cmd;(void)id;
    FfxFsr4VkContext* c = (FfxFsr4VkContext*)I->device;
    uint32_t n=0;
    for (uint32_t i=0; i<c->resCount; i++) {
        if (c->res[i].external) memset(&c->res[i],0,sizeof(c->res[i]));
        else { if(i!=n) c->res[n]=c->res[i]; n++; }
    }
    c->resCount=n;
    return FFX_OK;
}

static FfxErrorCode cbRegStaticRes(FfxInterface* I,
    const FfxStaticResourceDescription* d, FfxUInt32 id)
{ (void)I;(void)d;(void)id; return FFX_OK; }

static FfxApiResourceDescription cbGetResDesc(FfxInterface* I, FfxResourceInternal ri) {
    FfxFsr4VkContext* c = (FfxFsr4VkContext*)I->device;
    FfxApiResourceDescription o={0};
    if (ri.internalIndex<c->resCount) {
        VkRes* r=&c->res[ri.internalIndex];
        o.width  = (r->kind==RES_IMAGE)?r->w:(uint32_t)r->size;
        o.height = (r->kind==RES_IMAGE)?r->h:1;
        o.depth=1; o.mipCount=1;
        o.type=(r->kind==RES_IMAGE)?FFX_RESOURCE_TYPE_TEXTURE2D:FFX_RESOURCE_TYPE_BUFFER;
    }
    return o;
}

static FfxErrorCode cbDestroyRes(FfxInterface* I, FfxResourceInternal ri, FfxUInt32 id) {
    (void)id;
    FfxFsr4VkContext* c = (FfxFsr4VkContext*)I->device;
    if (ri.internalIndex>=c->resCount) return FFX_OK;
    VkRes* r=&c->res[ri.internalIndex];
    if (!r->external) {
        if(r->buf)  vkDestroyBuffer(c->dev,r->buf,c->alloc);
        if(r->view) vkDestroyImageView(c->dev,r->view,c->alloc);
        if(r->img)  vkDestroyImage(c->dev,r->img,c->alloc);
        if(r->mem)  vkFreeMemory(c->dev,r->mem,c->alloc);
    }
    memset(r,0,sizeof(*r));
    return FFX_OK;
}

static FfxErrorCode cbMapRes(FfxInterface* I, FfxResourceInternal ri, void** p) {
    FfxFsr4VkContext* c=(FfxFsr4VkContext*)I->device;
    if (ri.internalIndex>=c->resCount) return FFX_ERROR_INVALID_ARGUMENT;
    return vkMapMemory(c->dev,c->res[ri.internalIndex].mem,0,VK_WHOLE_SIZE,0,p)==VK_SUCCESS
        ? FFX_OK : FFX_ERROR_BACKEND_API_ERROR;
}

static FfxErrorCode cbUnmapRes(FfxInterface* I, FfxResourceInternal ri) {
    FfxFsr4VkContext* c=(FfxFsr4VkContext*)I->device;
    if (ri.internalIndex>=c->resCount) return FFX_ERROR_INVALID_ARGUMENT;
    vkUnmapMemory(c->dev, c->res[ri.internalIndex].mem);
    return FFX_OK;
}

static FfxErrorCode cbStageCbuf(FfxInterface* I, void* data, FfxUInt32 sz, FfxConstantBuffer* out) {
    FfxFsr4VkContext* c = (FfxFsr4VkContext*)I->device;
    uint32_t off = (c->cbOff + 255u) & ~255u;
    if (off + sz > CBUF_RING_BYTES) off = 0;
    memcpy(c->cbMap + off, data, sz);
    c->cbOff = off + sz;
    // Encode ring offset as the resource pointer — cbufBytes known at dispatch
    out->resource.resource = (void*)(uintptr_t)off;
    return FFX_OK;
}

static FfxErrorCode cbCreatePipeline(FfxInterface* I, FfxShaderBlob* blob,
    const FfxPipelineDescription* pd, FfxUInt32 id, FfxPipelineState* out)
{
    (void)id; (void)blob;
    (void)I; // pipeline already built in cbCreateBackendCtx

    uint32_t passIdx = 1; // default: inner model pass
    if (pd) { /* name is array, always valid when pd != NULL */
        const wchar_t* n = pd->name;
        const wchar_t* e = n;
        while (*e) e++;
        while (e>n && e[-1]>=L'0' && e[-1]<=L'9') e--;
        if (*e >= L'0') passIdx = (uint32_t)wcstoul(e, NULL, 10);
        if (wcsstr(n, L"PRE"))  passIdx=0;
        else if (wcsstr(n, L"POST")) passIdx=13;
        else if (wcsstr(n, L"RCAS")) passIdx=14;
        else if (wcsstr(n, L"SPD"))  passIdx=15;
    }
    if (passIdx >= FFX_FSR4_VK_PASS_COUNT) passIdx=1;

    out->pipeline = (FfxPipeline)(uintptr_t)passIdx;

    if (blob) {
        out->srvTextureCount = blob->srvTextureCount;
        out->uavTextureCount = blob->uavTextureCount;
        out->srvBufferCount  = blob->srvBufferCount;
        out->uavBufferCount  = blob->uavBufferCount;
        out->constCount      = blob->cbvCount;
    } else {
        out->srvBufferCount=2; out->uavBufferCount=2;
    }
    return FFX_OK;
}

static FfxErrorCode cbDestroyPipeline(FfxInterface* I, FfxPipelineState* ps, FfxUInt32 id) {
    (void)I;(void)id; if(ps) ps->pipeline=NULL; return FFX_OK;
}

static FfxErrorCode cbScheduleJob(FfxInterface* I, const FfxGpuJobDescription* j) {
    FfxFsr4VkContext* c = (FfxFsr4VkContext*)I->device;
    if (c->jobCount >= MAX_PENDING_JOBS) return FFX_ERROR_OUT_OF_RANGE;
    c->jobs[c->jobCount++].d = *j;
    return FFX_OK;
}

// ── image layout barrier ──────────────────────────────────────────────────────

static void imgBarrier(VkCommandBuffer cmd, VkImage img,
    VkImageLayout from, VkImageLayout to,
    VkAccessFlags sa, VkAccessFlags da,
    VkPipelineStageFlags ss, VkPipelineStageFlags ds)
{
    VkImageMemoryBarrier b;
    memset(&b, 0, sizeof(b));
    b.sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER;
    b.srcAccessMask=sa; b.dstAccessMask=da;
    b.oldLayout=from; b.newLayout=to;
    b.srcQueueFamilyIndex=b.dstQueueFamilyIndex=VK_QUEUE_FAMILY_IGNORED;
    b.image=img;
    b.subresourceRange=(VkImageSubresourceRange){VK_IMAGE_ASPECT_COLOR_BIT,0,1,0,1};
    vkCmdPipelineBarrier(cmd,ss,ds,0,0,NULL,0,NULL,1,&b);
}

// ── descriptor writing helpers ────────────────────────────────────────────────

static void writeModelDs(FfxFsr4VkContext* c, VkDescriptorSet ds,
    const FfxComputeJobDescription* cj)
{
    VkWriteDescriptorSet wr[4];
    VkDescriptorBufferInfo bi[4];
    memset(wr,0,sizeof(wr)); memset(bi,0,sizeof(bi));
    uint32_t wc=0;

    // binding 0: SRV buffer[0] (input, t0→0)
    if (cj->pipeline.srvBufferCount>0) {
        uint32_t idx=cj->srvBuffers[0].resource.internalIndex;
        if (idx<c->resCount && c->res[idx].kind==RES_BUFFER) {
            bi[wc]=(VkDescriptorBufferInfo){c->res[idx].buf,0,VK_WHOLE_SIZE};
            wr[wc]=(VkWriteDescriptorSet){VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET,
                NULL,ds,0,0,1,VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,NULL,&bi[wc],NULL}; wc++;
        }
    }
    // binding 1: SRV buffer[1] (weights, t1→1)
    if (cj->pipeline.srvBufferCount>1) {
        uint32_t idx=cj->srvBuffers[1].resource.internalIndex;
        if (idx<c->resCount && c->res[idx].kind==RES_BUFFER) {
            bi[wc]=(VkDescriptorBufferInfo){c->res[idx].buf,0,VK_WHOLE_SIZE};
            wr[wc]=(VkWriteDescriptorSet){VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET,
                NULL,ds,1,0,1,VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,NULL,&bi[wc],NULL}; wc++;
        }
    }
    // binding 2: UAV buffer[0] (output, u0→2)
    if (cj->pipeline.uavBufferCount>0) {
        uint32_t idx=cj->uavBuffers[0].resource.internalIndex;
        if (idx<c->resCount && c->res[idx].kind==RES_BUFFER) {
            bi[wc]=(VkDescriptorBufferInfo){c->res[idx].buf,0,VK_WHOLE_SIZE};
            wr[wc]=(VkWriteDescriptorSet){VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET,
                NULL,ds,2,0,1,VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,NULL,&bi[wc],NULL}; wc++;
        }
    }
    // binding 3: UAV buffer[1] (scratch, u1→3)
    if (cj->pipeline.uavBufferCount>1) {
        uint32_t idx=cj->uavBuffers[1].resource.internalIndex;
        if (idx<c->resCount && c->res[idx].kind==RES_BUFFER) {
            bi[wc]=(VkDescriptorBufferInfo){c->res[idx].buf,0,VK_WHOLE_SIZE};
            wr[wc]=(VkWriteDescriptorSet){VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET,
                NULL,ds,3,0,1,VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,NULL,&bi[wc],NULL}; wc++;
        }
    }
    if (wc) vkUpdateDescriptorSets(c->dev,wc,wr,0,NULL);
}

static void writeFullDs(FfxFsr4VkContext* c, VkDescriptorSet ds,
    const FfxComputeJobDescription* cj)
{
    // Upper bound on writes: 44 bindings + spare
    enum { MAX_WR=50 };
    VkWriteDescriptorSet wr[MAX_WR];
    VkDescriptorImageInfo  imgs[21+13];   // SRV textures + UAV textures
    VkDescriptorBufferInfo bufs[8];       // UAV bufs + scratch + cbuffers
    memset(wr,0,sizeof(wr)); memset(imgs,0,sizeof(imgs)); memset(bufs,0,sizeof(bufs));
    uint32_t wc=0, ic=0, bc=0;

    // ── SRV textures t0..t20 → bindings 0..20 (COMBINED_IMAGE_SAMPLER) ──
    // Use VK_IMAGE_LAYOUT_GENERAL for all images.  Internal images (history,
    // recurrent, reprojected, exposure) are also used as UAV STORAGE_IMAGE in
    // other passes, so they must stay in GENERAL.  External images (TAA output)
    // are transitioned to GENERAL by fsr.c before dispatch.
    for (uint32_t i=0; i<cj->pipeline.srvTextureCount && i<21; i++) {
        uint32_t idx=cj->srvTextures[i].resource.internalIndex;
        if (idx<c->resCount && c->res[idx].kind==RES_IMAGE) {
            imgs[ic]=(VkDescriptorImageInfo){c->linearSampler, c->res[idx].view,
                VK_IMAGE_LAYOUT_GENERAL};
            wr[wc]=(VkWriteDescriptorSet){VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET,
                NULL,ds,SLOT_SRV_TEX_BASE+i,0,1,
                VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER,&imgs[ic],NULL,NULL};
            ic++; wc++;
        }
    }

    // ── UAV textures u0..u12 → bindings 21..33 (STORAGE_IMAGE) ──
    for (uint32_t i=0; i<cj->pipeline.uavTextureCount && i<13; i++) {
        uint32_t idx=cj->uavTextures[i].resource.internalIndex;
        if (idx<c->resCount && c->res[idx].kind==RES_IMAGE) {
            imgs[ic]=(VkDescriptorImageInfo){VK_NULL_HANDLE, c->res[idx].view,
                VK_IMAGE_LAYOUT_GENERAL};
            wr[wc]=(VkWriteDescriptorSet){VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET,
                NULL,ds,SLOT_UAV_BASE+i,0,1,
                VK_DESCRIPTOR_TYPE_STORAGE_IMAGE,&imgs[ic],NULL,NULL};
            ic++; wc++;
        }
    }

    // ── UAV buffers → STORAGE_BUFFER at u-register + SLOT_UAV_BASE ──
    // Use uavBufferBindings[i].bindingIndex (the u-register number) to compute
    // the Vulkan binding.  If bindingIndex is 0 and i>0, treat as unset and skip.
    for (uint32_t i=0; i<cj->pipeline.uavBufferCount; i++) {
        uint32_t idx=cj->uavBuffers[i].resource.internalIndex;
        if (idx>=c->resCount || c->res[idx].kind!=RES_BUFFER) continue;
        uint32_t u_reg = cj->pipeline.uavBufferBindings[i].bindingIndex;
        if (u_reg == 0 && i > 0) continue;  // bindingIndex not set — skip
        uint32_t binding = SLOT_UAV_BASE + u_reg;
        bufs[bc]=(VkDescriptorBufferInfo){c->res[idx].buf,0,VK_WHOLE_SIZE};
        wr[wc]=(VkWriteDescriptorSet){VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET,
            NULL,ds,binding,0,1,
            VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,NULL,&bufs[bc],NULL};
        bc++; wc++;
    }

    // ── SRV buffers → STORAGE_BUFFER at t-register (binding = t_reg, shift=0) ──
    for (uint32_t i=0; i<cj->pipeline.srvBufferCount; i++) {
        uint32_t idx=cj->srvBuffers[i].resource.internalIndex;
        if (idx>=c->resCount || c->res[idx].kind!=RES_BUFFER) continue;
        uint32_t binding = cj->pipeline.srvBufferBindings[i].bindingIndex;
        if (binding == 0 && i > 0) continue;  // bindingIndex not set — skip
        bufs[bc]=(VkDescriptorBufferInfo){c->res[idx].buf,0,VK_WHOLE_SIZE};
        wr[wc]=(VkWriteDescriptorSet){VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET,
            NULL,ds,binding,0,1,
            VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,NULL,&bufs[bc],NULL};
        bc++; wc++;
    }

    // ── Main constants (b0) → binding 43 (UNIFORM_BUFFER) ──
    if (cj->pipeline.constCount>0) {
        uint32_t off=(uint32_t)(uintptr_t)cj->cbs[0].resource.resource;
        bufs[bc]=(VkDescriptorBufferInfo){c->cbRing, off, 256};
        wr[wc]=(VkWriteDescriptorSet){VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET,
            NULL,ds,SLOT_CBUF_MAIN,0,1,
            VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER,NULL,&bufs[bc],NULL};
        bc++; wc++;
    }

    // ── Pass weights (cbPass_Weights) → binding 34 (UNIFORM_BUFFER) ──
    if (cj->pipeline.constCount>1) {
        uint32_t off=(uint32_t)(uintptr_t)cj->cbs[1].resource.resource;
        bufs[bc]=(VkDescriptorBufferInfo){c->cbRing, off, 256};
        wr[wc]=(VkWriteDescriptorSet){VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET,
            NULL,ds,SLOT_CBUF_WEIGHTS,0,1,
            VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER,NULL,&bufs[bc],NULL};
        bc++; wc++;
    }

    if (wc) vkUpdateDescriptorSets(c->dev, wc, wr, 0, NULL);
}

// ── execute all queued jobs ───────────────────────────────────────────────────

static FfxErrorCode cbExecuteJobs(FfxInterface* I, FfxCommandList cl, FfxUInt32 id) {
    (void)id;
    FfxFsr4VkContext* c = (FfxFsr4VkContext*)I->device;
    VkCommandBuffer cmd = (VkCommandBuffer)cl;

    // Triple-buffered pool cycling: reset the pool from 2 frames ago
    // (guaranteed idle by Q2RTX's double-buffered fence wait) and use
    // the current frame's pool for new allocations.
    uint32_t cur  = c->poolIdx % NUM_POOL_FRAMES;
    uint32_t old  = (c->poolIdx + 1) % NUM_POOL_FRAMES;  // 2 frames ago
    vkResetDescriptorPool(c->dev, c->pool[old], 0);
    VkDescriptorPool activePool = c->pool[cur];
    c->poolIdx++;

    // Flush staging uploads first (recorded before any compute work)
    for (uint32_t i=0; i<c->stagingCount; i++) {
        Staging* st=&c->staging[i];
        VkBufferCopy bc={0,0,st->size};
        vkCmdCopyBuffer(cmd, st->buf, c->res[st->dstIdx].buf, 1, &bc);
    }
    if (c->stagingCount>0) {
        VkMemoryBarrier mb;
    memset(&mb, 0, sizeof(mb));
    mb.sType = VK_STRUCTURE_TYPE_MEMORY_BARRIER;
        mb.srcAccessMask=VK_ACCESS_TRANSFER_WRITE_BIT;
        mb.dstAccessMask=VK_ACCESS_SHADER_READ_BIT;
        vkCmdPipelineBarrier(cmd,
            VK_PIPELINE_STAGE_TRANSFER_BIT,
            VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
            0,1,&mb,0,NULL,0,NULL);
        for (uint32_t i=0; i<c->stagingCount; i++) {
            vkDestroyBuffer(c->dev, c->staging[i].buf, c->alloc);
            vkFreeMemory(c->dev, c->staging[i].mem, c->alloc);
        }
        c->stagingCount=0;
    }

    // Transition any internal images that haven't been initialised yet
    // from UNDEFINED to GENERAL (needed on first frame).
    {
        int init_count = 0;
        for (uint32_t i = 0; i < c->resCount; i++) {
            if (c->res[i].kind == RES_IMAGE && c->res[i].needsInit && c->res[i].img) {
                imgBarrier(cmd, c->res[i].img,
                    VK_IMAGE_LAYOUT_UNDEFINED, VK_IMAGE_LAYOUT_GENERAL,
                    0, VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_SHADER_WRITE_BIT,
                    VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT);
                c->res[i].needsInit = 0;
                init_count++;
            }
        }
        (void)init_count;
    }

    for (uint32_t ji=0; ji<c->jobCount; ji++) {
        FfxGpuJobDescription* job=&c->jobs[ji].d;

        if (job->jobType==FFX_GPU_JOB_CLEAR_FLOAT) {
            uint32_t idx=job->clearJobDescriptor.target.internalIndex;
            if (idx<c->resCount) {
                VkRes* r=&c->res[idx];
                if (r->kind==RES_BUFFER)
                    vkCmdFillBuffer(cmd,r->buf,0,VK_WHOLE_SIZE,0);
                else if (r->kind==RES_IMAGE && r->img) {
                    imgBarrier(cmd,r->img,
                        VK_IMAGE_LAYOUT_UNDEFINED,VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,
                        0,VK_ACCESS_TRANSFER_WRITE_BIT,
                        VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT,VK_PIPELINE_STAGE_TRANSFER_BIT);
                    VkClearColorValue cv={{0,0,0,0}};
                    VkImageSubresourceRange sr={VK_IMAGE_ASPECT_COLOR_BIT,0,1,0,1};
                    vkCmdClearColorImage(cmd,r->img,VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,&cv,1,&sr);
                    imgBarrier(cmd,r->img,
                        VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,VK_IMAGE_LAYOUT_GENERAL,
                        VK_ACCESS_TRANSFER_WRITE_BIT,VK_ACCESS_SHADER_READ_BIT|VK_ACCESS_SHADER_WRITE_BIT,
                        VK_PIPELINE_STAGE_TRANSFER_BIT,VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT);
                }
            }
        }
        else if (job->jobType==FFX_GPU_JOB_COPY) {
            uint32_t si=job->copyJobDescriptor.src.internalIndex;
            uint32_t di=job->copyJobDescriptor.dst.internalIndex;
            if (si<c->resCount && di<c->resCount &&
                c->res[si].kind==RES_BUFFER && c->res[di].kind==RES_BUFFER) {
                VkBufferCopy bc={
                    job->copyJobDescriptor.srcOffset,
                    job->copyJobDescriptor.dstOffset,
                    job->copyJobDescriptor.size};
                vkCmdCopyBuffer(cmd,c->res[si].buf,c->res[di].buf,1,&bc);
            }
        }
        else if (job->jobType==FFX_GPU_JOB_COMPUTE) {
            FfxComputeJobDescription* cj=&job->computeJobDescriptor;
            uint32_t passIdx=(uint32_t)(uintptr_t)cj->pipeline.pipeline;
            if (passIdx>=FFX_FSR4_VK_PASS_COUNT) continue;
            VkPipe* bp=&c->pipe[passIdx];
            if (!bp->pipeline) continue;

            // Allocate descriptor set from this frame's pool
            VkDescriptorSetAllocateInfo dsai;
    memset(&dsai, 0, sizeof(dsai));
    dsai.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO;
            dsai.descriptorPool=activePool;
            dsai.descriptorSetCount=1;
            dsai.pSetLayouts=&bp->dsLayout;
            VkDescriptorSet ds;
            if (vkAllocateDescriptorSets(c->dev,&dsai,&ds)!=VK_SUCCESS) continue;

            if (bp->kind==PIPE_FULL) writeFullDs(c,ds,cj);
            else                     writeModelDs(c,ds,cj);

            vkCmdBindPipeline(cmd,VK_PIPELINE_BIND_POINT_COMPUTE,bp->pipeline);
            vkCmdBindDescriptorSets(cmd,VK_PIPELINE_BIND_POINT_COMPUTE,
                bp->layout,0,1,&ds,0,NULL);

            // Push constants for cbuffer (≤256 bytes guaranteed by Vulkan spec)
            if (bp->cbufBytes>0 && bp->cbufBytes<=256 && cj->pipeline.constCount>0) {
                uint32_t off=(uint32_t)(uintptr_t)cj->cbs[0].resource.resource;
                vkCmdPushConstants(cmd,bp->layout,
                    VK_SHADER_STAGE_COMPUTE_BIT,0,bp->cbufBytes,c->cbMap+off);
            }

            vkCmdDispatch(cmd,cj->dimensions[0],cj->dimensions[1],cj->dimensions[2]);

            // Memory barrier between compute passes — ensures writes from this
            // dispatch are visible to the next dispatch (pre→model→post chain).
            {
                VkMemoryBarrier mb;
                memset(&mb, 0, sizeof(mb));
                mb.sType = VK_STRUCTURE_TYPE_MEMORY_BARRIER;
                mb.srcAccessMask = VK_ACCESS_SHADER_WRITE_BIT;
                mb.dstAccessMask = VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_SHADER_WRITE_BIT;
                vkCmdPipelineBarrier(cmd,
                    VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
                    VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
                    0, 1, &mb, 0, NULL, 0, NULL);
            }

            // Inter-pass memory barrier
            VkMemoryBarrier mb;
    memset(&mb, 0, sizeof(mb));
    mb.sType = VK_STRUCTURE_TYPE_MEMORY_BARRIER;
            mb.srcAccessMask=VK_ACCESS_SHADER_WRITE_BIT;
            mb.dstAccessMask=VK_ACCESS_SHADER_READ_BIT|VK_ACCESS_SHADER_WRITE_BIT;
            vkCmdPipelineBarrier(cmd,
                VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
                VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
                0,1,&mb,0,NULL,0,NULL);
        }
    }

    c->jobCount=0;
    return FFX_OK;
}

static FfxErrorCode cbSwapChain(FfxFrameGenerationConfig const* cfg){ (void)cfg; return FFX_OK; }
static void cbRegCbAlloc(FfxInterface* I, FfxConstantBufferAllocator a){ (void)I;(void)a; }

// ── public API ────────────────────────────────────────────────────────────────

size_t ffxFsr4VkGetScratchMemorySize(void) { return sizeof(FfxFsr4VkContext); }

VkResult ffxFsr4VkCreateContext(const FfxFsr4VkCreateInfo* ci, FfxInterface* out)
{
    if (!ci||!out||!ci->scratchBuffer) return VK_ERROR_INITIALIZATION_FAILED;
    if (ci->scratchBufferSize < sizeof(FfxFsr4VkContext)) return VK_ERROR_INITIALIZATION_FAILED;

    FfxFsr4VkContext* c = (FfxFsr4VkContext*)ci->scratchBuffer;
    memset(c, 0, sizeof(*c));
    c->dev=ci->device; c->phys=ci->physicalDevice; c->alloc=ci->allocator;
    /* Deep-copy the SPIR-V blobs so this context owns the data.
     * The caller may free its copy immediately after this call returns. */
    memcpy(c->blobs, ci->shaders, sizeof(c->blobs));
    for (int i = 0; i < FFX_FSR4_VK_PASS_COUNT; i++) {
        if (!c->blobs[i].spirv || !c->blobs[i].sizeBytes) continue;
        uint32_t* copy = (uint32_t*)malloc(c->blobs[i].sizeBytes);
        if (!copy) { /* unwind already-copied blobs and fail */
            for (int j = 0; j < i; j++) { free((void*)c->blobs[j].spirv); c->blobs[j].spirv = NULL; }
            return VK_ERROR_OUT_OF_HOST_MEMORY;
        }
        memcpy(copy, c->blobs[i].spirv, c->blobs[i].sizeBytes);
        c->blobs[i].spirv = copy;
    }

    VkResult r;

    // Sampler (shared, baked into LAYOUT_FULL immutable samplers)
    VkSamplerCreateInfo sci;
    memset(&sci, 0, sizeof(sci));
    sci.sType = VK_STRUCTURE_TYPE_SAMPLER_CREATE_INFO;
    sci.magFilter=VK_FILTER_LINEAR; sci.minFilter=VK_FILTER_LINEAR;
    sci.mipmapMode=VK_SAMPLER_MIPMAP_MODE_LINEAR;
    sci.addressModeU=sci.addressModeV=sci.addressModeW=VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE;
    sci.maxLod=VK_LOD_CLAMP_NONE;
    r=vkCreateSampler(c->dev,&sci,c->alloc,&c->linearSampler);
    if (r) return r;

    // Descriptor set layouts
    r=buildModelLayout(c); if (r) return r;
    r=buildFullLayout(c);  if (r) return r;

    // cbuffer ring buffer (host-visible, persistently mapped)
    r=mkBuf(c, CBUF_RING_BYTES,
        VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT|VK_BUFFER_USAGE_STORAGE_BUFFER_BIT,
        VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT|VK_MEMORY_PROPERTY_HOST_COHERENT_BIT,
        &c->cbRing, &c->cbMem);
    if (r) return r;
    vkMapMemory(c->dev, c->cbMem, 0, CBUF_RING_BYTES, 0, (void**)&c->cbMap);

    // Triple-buffered descriptor pools — one per in-flight frame.
    // At the start of each frame, reset the pool from 2 frames ago (guaranteed
    // complete) and allocate from the current frame's pool.
    VkDescriptorPoolSize ps[] = {
        {VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,         MAX_DESC_SETS * 16},
        {VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER,         MAX_DESC_SETS},
        {VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, MAX_DESC_SETS * 21},
        {VK_DESCRIPTOR_TYPE_STORAGE_IMAGE,          MAX_DESC_SETS * 13},
    };
    VkDescriptorPoolCreateInfo dpci;
    memset(&dpci, 0, sizeof(dpci));
    dpci.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO;
    dpci.flags=VK_DESCRIPTOR_POOL_CREATE_FREE_DESCRIPTOR_SET_BIT;
    dpci.maxSets=MAX_DESC_SETS;
    dpci.poolSizeCount=4; dpci.pPoolSizes=ps;
    for (int p = 0; p < NUM_POOL_FRAMES; p++) {
        r=vkCreateDescriptorPool(c->dev, &dpci, c->alloc, &c->pool[p]);
        if (r) return r;
    }
    c->poolIdx = 0;

    // Populate FfxInterface table
    out->fpGetSDKVersion                     = cbGetSDKVersion;
    out->fpGetEffectGpuMemoryUsage           = cbGetEffectGpuMemUsage;
    out->fpCreateBackendContext              = cbCreateBackendCtx;
    out->fpGetDeviceCapabilities             = cbGetDeviceCaps;
    out->fpDestroyBackendContext             = cbDestroyBackendCtx;
    out->fpCreateResource                    = cbCreateRes;
    out->fpRegisterResource                  = cbRegisterRes;
    out->fpGetResource                       = cbGetRes;
    out->fpUnregisterResources               = cbUnregisterRes;
    out->fpRegisterStaticResource            = cbRegStaticRes;
    out->fpGetResourceDescription            = cbGetResDesc;
    out->fpDestroyResource                   = cbDestroyRes;
    out->fpMapResource                       = cbMapRes;
    out->fpUnmapResource                     = cbUnmapRes;
    out->fpStageConstantBufferDataFunc       = cbStageCbuf;
    out->fpCreatePipeline                    = cbCreatePipeline;
    out->fpDestroyPipeline                   = cbDestroyPipeline;
    out->fpScheduleGpuJob                    = cbScheduleJob;
    out->fpExecuteGpuJobs                    = cbExecuteJobs;
    out->fpSwapChainConfigureFrameGeneration = cbSwapChain;
    out->fpRegisterConstantBufferAllocator   = cbRegCbAlloc;

    out->scratchBuffer=ci->scratchBuffer;
    out->scratchBufferSize=ci->scratchBufferSize;
    out->device=(FfxDevice)c;
    return VK_SUCCESS;
}

void ffxFsr4VkDestroyContext(FfxFsr4VkContext* c)
{
    if (!c) return;
    /* Free the deep-copied SPIR-V blobs allocated in ffxFsr4VkCreateContext. */
    for (int i = 0; i < FFX_FSR4_VK_PASS_COUNT; i++) {
        free((void*)c->blobs[i].spirv);
        c->blobs[i].spirv = NULL;
    }
    for (uint32_t i=0; i<c->resCount; i++) {
        VkRes* r=&c->res[i];
        if (!r->external) {
            if(r->buf)  vkDestroyBuffer(c->dev,r->buf,c->alloc);
            if(r->view) vkDestroyImageView(c->dev,r->view,c->alloc);
            if(r->img)  vkDestroyImage(c->dev,r->img,c->alloc);
            if(r->mem)  vkFreeMemory(c->dev,r->mem,c->alloc);
        }
    }
    for (uint32_t i=0; i<c->stagingCount; i++) {
        vkDestroyBuffer(c->dev,c->staging[i].buf,c->alloc);
        vkFreeMemory(c->dev,c->staging[i].mem,c->alloc);
    }
    if (c->cbMap)  vkUnmapMemory(c->dev,c->cbMem);
    if (c->cbRing) vkDestroyBuffer(c->dev,c->cbRing,c->alloc);
    if (c->cbMem)  vkFreeMemory(c->dev,c->cbMem,c->alloc);
    for (int p = 0; p < NUM_POOL_FRAMES; p++)
        if (c->pool[p]) vkDestroyDescriptorPool(c->dev,c->pool[p],c->alloc);
    if (c->modelLayout)  vkDestroyDescriptorSetLayout(c->dev,c->modelLayout,c->alloc);
    if (c->fullLayout)   vkDestroyDescriptorSetLayout(c->dev,c->fullLayout,c->alloc);
    if (c->fullLayoutRcas) vkDestroyDescriptorSetLayout(c->dev,c->fullLayoutRcas,c->alloc);
    if (c->linearSampler) vkDestroySampler(c->dev,c->linearSampler,c->alloc);
    // Pipelines are destroyed by cbDestroyBackendCtx (called by the provider layer)
}

// ---------------------------------------------------------------------------
// Helpers for narrow provider-side direct Vulkan access.
//
// Note: the FfxInterface indirection is intentional — the provider holds
// an FfxInterface* and an FfxResourceInternal index, not a VkImage handle.
// These helpers let the provider translate (interface, ri) → VkImage so it
// can issue its own vkCmd calls (e.g. vkCmdClearColorImage on internal
// history textures) without the interface needing a generic "clear" job.
// ---------------------------------------------------------------------------
VkImage ffxFsr4VkGetImage(FfxInterface* iface, FfxResourceInternal ri)
{
    if (!iface || !iface->device) return VK_NULL_HANDLE;
    FfxFsr4VkContext* c = (FfxFsr4VkContext*)iface->device;
    if (ri.internalIndex >= c->resCount) return VK_NULL_HANDLE;
    VkRes* r = &c->res[ri.internalIndex];
    return (r->kind == RES_IMAGE) ? r->img : VK_NULL_HANDLE;
}

uint32_t ffxFsr4VkGetWidth(FfxInterface* iface, FfxResourceInternal ri)
{
    if (!iface || !iface->device) return 0;
    FfxFsr4VkContext* c = (FfxFsr4VkContext*)iface->device;
    if (ri.internalIndex >= c->resCount) return 0;
    VkRes* r = &c->res[ri.internalIndex];
    return (r->kind == RES_IMAGE) ? r->w : 0;
}

uint32_t ffxFsr4VkGetHeight(FfxInterface* iface, FfxResourceInternal ri)
{
    if (!iface || !iface->device) return 0;
    FfxFsr4VkContext* c = (FfxFsr4VkContext*)iface->device;
    if (ri.internalIndex >= c->resCount) return 0;
    VkRes* r = &c->res[ri.internalIndex];
    return (r->kind == RES_IMAGE) ? r->h : 0;
}

int ffxFsr4VkNeedsInitialTransition(FfxInterface* iface, FfxResourceInternal ri)
{
    if (!iface || !iface->device) return 0;
    FfxFsr4VkContext* c = (FfxFsr4VkContext*)iface->device;
    if (ri.internalIndex >= c->resCount) return 0;
    VkRes* r = &c->res[ri.internalIndex];
    return (r->kind == RES_IMAGE) ? r->needsInit : 0;
}

void ffxFsr4VkMarkImageInitialized(FfxInterface* iface, FfxResourceInternal ri)
{
    if (!iface || !iface->device) return;
    FfxFsr4VkContext* c = (FfxFsr4VkContext*)iface->device;
    if (ri.internalIndex >= c->resCount) return;
    VkRes* r = &c->res[ri.internalIndex];
    if (r->kind == RES_IMAGE) r->needsInit = 0;
}
