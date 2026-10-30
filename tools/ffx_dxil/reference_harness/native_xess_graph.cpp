// SPDX-License-Identifier: MIT
// Persistent fixed-profile graph experiment. Inputs/outputs cross the host;
// all fourteen passes and temporal history remain on one Vulkan device.
#include "native_replay_common.h"
#include "ffx_vk_xess_fixed_context.hpp"
#include "ffx_vk_xess_recovered.h"
#include <nlohmann/json.hpp>
#include <filesystem>
#include <map>
#include <memory>
#include <limits>
using Json=nlohmann::json;
namespace fs=std::filesystem;

static void barrier(VkCommandBuffer cmd,VkPipelineStageFlags src,VkPipelineStageFlags dst,VkAccessFlags from,VkAccessFlags to) {
    VkMemoryBarrier b{VK_STRUCTURE_TYPE_MEMORY_BARRIER};b.srcAccessMask=from;b.dstAccessMask=to;
    vkCmdPipelineBarrier(cmd,src,dst,0,1,&b,0,nullptr,0,nullptr);
}
static std::vector<uint32_t> shaderWords(const std::string& path) {
    auto bytes=read(path.c_str());if(bytes.size()%4) throw std::runtime_error("Invalid shader size");
    std::vector<uint32_t> result(bytes.size()/4);std::memcpy(result.data(),bytes.data(),bytes.size());return result;
}
static ffxvk::XeSSFixedModel modelFromBundle(const Json& bundle) {
    const auto& conv=bundle.at("convolutions");const auto& resolve=bundle.at("resolve");
    ffxvk::XeSSFixedModel result;
    result.prepareShader=shaderWords(bundle.at("prepare"));result.resolveShader=shaderWords(resolve.at("shader"));
    if(conv.at("passes").size()!=12 || conv.at("images").size()!=13)
        throw std::runtime_error("Unsupported captured graph");
    const auto& passes=conv.at("passes");
    std::array<std::string,13> ids;ids[0]=conv.at("input").get<std::string>();
    for(unsigned i=0;i<12;++i) ids[i+1]=passes.at(i).at("images").at(2).get<std::string>();
    for(unsigned i=0;i<13;++i) {
        if(conv.at("images").at(ids[i]).get<std::array<uint32_t,3>>()!=ffxvk::XeSSFixedContext::tensorShapes[i])
            throw std::runtime_error("Unsupported tensor shape");
        for(unsigned j=0;j<i;++j) if(ids[i]==ids[j]) throw std::runtime_error("Aliased tensor graph");
    }
    for(unsigned i=0;i<12;++i) {
        const auto& pass=passes.at(i);auto& target=result.convolutions[i];
        if(pass.at("images").at(0)!=ids[i] || pass.at("images").at(1)!=ids[ffxvk::XeSSFixedContext::skipInputs[i]])
            throw std::runtime_error("Unsupported skip connection");
        target.shader=shaderWords(pass.at("shader"));
        const auto constants=pass.at("push").get<std::array<uint32_t,6>>();
        const auto shape=ffxvk::XeSSFixedContext::tensorShapes[i<5?i:i+1];
        if(constants[0]!=shape[0] || constants[1]!=shape[1] || constants[4]!=0 || constants[5]!=3 ||
            pass.at("groups").get<std::array<uint32_t,3>>()!=ffxvk::XeSSFixedContext::dispatchGroups[i])
            throw std::runtime_error("Unsupported convolution constants");
        for(unsigned j=0;j<3;++j) {
            const auto& buffer=conv.at("buffers").at(pass.at("buffers").at(j).get<std::string>());
            auto bytes=read(buffer.at("file").get<std::string>().c_str());
            if(bytes.size()!=buffer.at("size").get<size_t>()) throw std::runtime_error("Truncated model data");
            target.buffers[j].assign(bytes.begin(),bytes.end());
        }
    }
    for(unsigned j=0;j<3;++j) {
        auto bytes=read(resolve.at("buffers").at(j).at("file").get<std::string>().c_str());
        if(bytes.size()!=resolve.at("buffers").at(j).at("view_bytes").get<size_t>()) throw std::runtime_error("Truncated resolve data");
        result.resolveBuffers[j].assign(bytes.begin(),bytes.end());
    }
    return result;
}
static FfxVkXeSSRecoveredModel modelSpans(const ffxvk::XeSSFixedModel& model) {
    FfxVkXeSSRecoveredModel result{};result.structSize=sizeof(result);
    auto words=[](const std::vector<uint32_t>& data){return FfxVkXeSSRecoveredBlob{data.data(),data.size()*4};};
    auto bytes=[](const std::vector<uint8_t>& data){return FfxVkXeSSRecoveredBlob{data.data(),data.size()};};
    result.prepareShader=words(model.prepareShader);result.resolveShader=words(model.resolveShader);
    for(unsigned i=0;i<12;++i) {
        const auto& source=model.convolutions[i];auto& target=result.convolutions[i];
        target.shader=words(source.shader);target.weights=bytes(source.buffers[0]);
        target.scales=bytes(source.buffers[1]);target.biases=bytes(source.buffers[2]);
    }
    result.resolveWeights=bytes(model.resolveBuffers[0]);result.resolveScales=bytes(model.resolveBuffers[1]);
    result.resolveBiases=bytes(model.resolveBuffers[2]);return result;
}

int main(int argc,char** argv) try {
    if(argc!=5 && argc!=9) throw std::runtime_error("Usage: native_xess_graph MODEL.xessvk_OR_resolved-bundle.json frames.json INPUT_DIR NEW_OUTPUT_DIR [inputWidth inputHeight outputWidth outputHeight]");
    const bool engineInputs=std::getenv("XESS_REPLAY_ENGINE_INPUTS")!=nullptr;
    FfxVkXeSSReplayExtent extent{753,424,1280,720};
    if(argc==9) {
        uint32_t* dimensions[]={&extent.inputWidth,&extent.inputHeight,&extent.outputWidth,&extent.outputHeight};
        for(unsigned i=0;i<4;++i) {
            size_t consumed=0;auto value=std::stoul(argv[5+i],&consumed);
            if(consumed!=std::strlen(argv[5+i]) || value>UINT32_MAX) throw std::runtime_error("Invalid dimensions");
            *dimensions[i]=static_cast<uint32_t>(value);
        }
    }
    FfxVkXeSSReplayExtent enabled{};
    if(!ffxVkXeSSRecoveredGetExtent({extent.outputWidth,extent.outputHeight},&enabled) ||
        extent.inputWidth!=enabled.inputWidth || extent.inputHeight!=enabled.inputHeight)
        throw std::runtime_error("Unverified output profile");
    auto modelFile=read(argv[1]);
    const bool packed=modelFile.size()>=8 && std::memcmp(modelFile.data(),"XESSVK1\0",8)==0;
    FfxVkXeSSRecoveredModel packedModel{};
    Json bundle,frames;
    if(packed) {
        char reason[256]{};
        if(ffxVkXeSSRecoveredModelFromBlob(modelFile.data(),modelFile.size(),&packedModel,reason,sizeof(reason))!=VK_SUCCESS)
            throw std::runtime_error(reason);
    } else std::ifstream(argv[1])>>bundle;
    std::ifstream(argv[2])>>frames;
    if((!packed && (bundle.at("schema")!="xess.native-bundle" || bundle.at("version")!=1)) || frames.size()!=4)
        throw std::runtime_error("Unsupported graph profile");
    if(!frames[0].at("reset").get<bool>()) throw std::runtime_error("Initial frame must reset");
    if(!fs::create_directory(argv[4])) throw std::runtime_error("Output directory must be new");
    if(!packed && bundle.at("convolutions").at("passes").size()!=12) throw std::runtime_error("Expected twelve convolutions");
    VkApplicationInfo app{VK_STRUCTURE_TYPE_APPLICATION_INFO};app.apiVersion=VK_API_VERSION_1_3;
    VkInstanceCreateInfo ii{VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO};ii.pApplicationInfo=&app;VkInstance instance;check(vkCreateInstance(&ii,nullptr,&instance));
    uint32_t count=0;check(vkEnumeratePhysicalDevices(instance,&count,nullptr));std::vector<VkPhysicalDevice> devices(count);
    check(vkEnumeratePhysicalDevices(instance,&count,devices.data()));uint32_t family=UINT32_MAX;
    for(auto candidate:devices) {
        VkPhysicalDeviceVulkan13Features f13{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_3_FEATURES};
        VkPhysicalDeviceVulkan12Features f12{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_2_FEATURES};f12.pNext=&f13;
        VkPhysicalDeviceFeatures2 f{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_FEATURES_2};f.pNext=&f12;vkGetPhysicalDeviceFeatures2(candidate,&f);
        VkPhysicalDeviceProperties p;vkGetPhysicalDeviceProperties(candidate,&p);
        if(p.apiVersion<VK_API_VERSION_1_3 || !f12.runtimeDescriptorArray || !f12.bufferDeviceAddress || !f12.shaderInt8 ||
           !f12.uniformBufferStandardLayout || !f13.shaderIntegerDotProduct || !f.features.shaderStorageImageWriteWithoutFormat ||
           !f.features.shaderSampledImageArrayDynamicIndexing || !f.features.shaderStorageImageArrayDynamicIndexing ||
           !f.features.shaderStorageBufferArrayDynamicIndexing || !f.features.shaderUniformBufferArrayDynamicIndexing ||
           p.limits.maxPerStageDescriptorStorageImages<32 || p.limits.maxPerStageDescriptorSampledImages<24 ||
           p.limits.maxPerStageDescriptorUniformBuffers<24 || p.limits.maxPerStageDescriptorStorageBuffers<6) continue;
        uint32_t n=0;vkGetPhysicalDeviceQueueFamilyProperties(candidate,&n,nullptr);std::vector<VkQueueFamilyProperties> q(n);
        vkGetPhysicalDeviceQueueFamilyProperties(candidate,&n,q.data());
        for(uint32_t i=0;i<n;++i) if(q[i].queueFlags&VK_QUEUE_COMPUTE_BIT) {physical=candidate;family=i;break;}
        if(physical) {std::printf("PERSISTENT_DEVICE %s\n",p.deviceName);break;}
    }
    if(!physical) throw std::runtime_error("No device supporting recovered graph");
    VkPhysicalDeviceVulkan13Features f13{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_3_FEATURES};f13.shaderIntegerDotProduct=VK_TRUE;
    VkPhysicalDeviceVulkan12Features f12{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_2_FEATURES};f12.pNext=&f13;
    f12.runtimeDescriptorArray=f12.bufferDeviceAddress=f12.shaderInt8=f12.uniformBufferStandardLayout=VK_TRUE;
    VkPhysicalDeviceFeatures2 f{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_FEATURES_2};f.pNext=&f12;
    f.features.shaderStorageImageWriteWithoutFormat=f.features.shaderSampledImageArrayDynamicIndexing=VK_TRUE;
    f.features.shaderStorageImageArrayDynamicIndexing=f.features.shaderStorageBufferArrayDynamicIndexing=f.features.shaderUniformBufferArrayDynamicIndexing=VK_TRUE;
    float priority=1;VkDeviceQueueCreateInfo qi{VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO};qi.queueFamilyIndex=family;qi.queueCount=1;qi.pQueuePriorities=&priority;
    VkDeviceCreateInfo di{VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO};di.pNext=&f;di.queueCreateInfoCount=1;di.pQueueCreateInfos=&qi;
    check(vkCreateDevice(physical,&di,nullptr,&device));VkQueue queue;vkGetDeviceQueue(device,family,0,&queue);

    ffxvk::XeSSResources resources(physical,device);
    std::unique_ptr<FfxVkXeSSRecoveredContext,decltype(&ffxVkXeSSRecoveredDestroy)> context(nullptr,ffxVkXeSSRecoveredDestroy);
    char diagnostic[256]{};
    auto apiCheck=[&](VkResult result){if(result!=VK_SUCCESS) throw std::runtime_error(std::string("XeSS C API: ")+diagnostic);};
    // The experimental harness owns this device. On failure, retire any work
    // before the shared resource owner's destructor runs.
    struct RetireOnFailure {
        bool active=true;
        ~RetireOnFailure() {if(active) vkDeviceWaitIdle(device);}
    } retire;
    auto makeBuffer=[&](size_t n,VkBufferUsageFlags usage){
        auto b=resources.createBuffer(n,usage);return Buffer{b.handle,b.memory,b.mapped};};
    auto makeInputs=[&](){return std::array<ffxvk::XeSSImage,4>{
        resources.createImage(engineInputs?extent.outputWidth:extent.inputWidth,engineInputs?extent.outputHeight:extent.inputHeight,VK_FORMAT_R16G16B16A16_SFLOAT),
        resources.createImage(engineInputs?extent.outputWidth:extent.inputWidth,engineInputs?extent.outputHeight:extent.inputHeight,engineInputs?VK_FORMAT_R16G16B16A16_SFLOAT:VK_FORMAT_R16G16_SFLOAT),
        resources.createImage(engineInputs?extent.outputWidth:extent.inputWidth,engineInputs?extent.outputHeight:extent.inputHeight,VK_FORMAT_R32_SFLOAT),
        resources.createImage(extent.outputWidth,extent.outputHeight,VK_FORMAT_R16G16B16A16_SFLOAT)};};
    const std::array<std::array<ffxvk::XeSSImage,4>,2> imageSets={makeInputs(),makeInputs()};
    auto model=packed?ffxvk::XeSSFixedModel{}:modelFromBundle(bundle);
    auto spans=packed?packedModel:modelSpans(model);
    // C callers may load byte-oriented, unaligned file storage. The bridge
    // copies shader words into aligned owned memory before Vulkan sees them.
    std::vector<uint8_t> unalignedPrepare(spans.prepareShader.sizeBytes+1);
    std::memcpy(unalignedPrepare.data()+1,spans.prepareShader.data,spans.prepareShader.sizeBytes);
    spans.prepareShader.data=unalignedPrepare.data()+1;
    FfxVkXeSSRecoveredCreateInfo create{};create.structSize=sizeof(create);create.abiVersion=FFX_VK_XESS_RECOVERED_ABI_VERSION;
    create.physicalDevice=physical;create.device=device;create.extent=extent;create.frameSlots=2;create.model=&spans;
    auto truncatedModel=spans;--truncatedModel.resolveBiases.sizeBytes;
    auto invalidCreate=create;invalidCreate.model=&truncatedModel;
    FfxVkXeSSRecoveredContext* rejectedContext=nullptr;
    if(ffxVkXeSSRecoveredCreate(&invalidCreate,&rejectedContext,diagnostic,sizeof(diagnostic))==VK_SUCCESS ||
        rejectedContext || !diagnostic[0]) throw std::runtime_error("Truncated C model accepted or no diagnostic");
    FfxVkXeSSRecoveredContext* created=nullptr;
    apiCheck(ffxVkXeSSRecoveredCreate(&create,&created,diagnostic,sizeof(diagnostic)));context.reset(created);
    // The context copied model data into its resources; the caller can drop it.
    model={};
    std::vector<char>().swap(modelFile);
    std::vector<uint8_t>().swap(unalignedPrepare);
    if(packed) std::puts("PACKED_MODEL entries=53 source_storage_released=1");
    const VkFormat formats[]={VK_FORMAT_R16G16B16A16_SFLOAT,engineInputs?VK_FORMAT_R16G16B16A16_SFLOAT:VK_FORMAT_R16G16_SFLOAT,VK_FORMAT_R32_SFLOAT,VK_FORMAT_R16G16B16A16_SFLOAT};
    auto bindingsFor=[&](const std::array<ffxvk::XeSSImage,4>& inputs){
        FfxVkXeSSRecoveredImages bindings{};bindings.structSize=sizeof(bindings);
        FfxVkXeSSRecoveredImage* external[]={&bindings.color,&bindings.motion,&bindings.depth,&bindings.output};
        for(unsigned i=0;i<4;++i) *external[i]={inputs[i].handle,inputs[i].view,
            {i==3?extent.outputWidth:extent.inputWidth,i==3?extent.outputHeight:extent.inputHeight},formats[i]};
        return bindings;
    };
    auto initializeInputs=[&](VkCommandBuffer cmd,const std::array<ffxvk::XeSSImage,4>& inputs){
        for(size_t index=0;index<inputs.size();++index) {
            const auto& input=inputs[index];
            VkImageMemoryBarrier b{VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER};b.image=input.handle;
            b.oldLayout=VK_IMAGE_LAYOUT_UNDEFINED;b.newLayout=VK_IMAGE_LAYOUT_GENERAL;
            b.srcQueueFamilyIndex=b.dstQueueFamilyIndex=VK_QUEUE_FAMILY_IGNORED;
            b.subresourceRange={VK_IMAGE_ASPECT_COLOR_BIT,0,1,0,1};b.dstAccessMask=VK_ACCESS_TRANSFER_WRITE_BIT;
            vkCmdPipelineBarrier(cmd,VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT,VK_PIPELINE_STAGE_TRANSFER_BIT,0,0,nullptr,0,nullptr,1,&b);
            VkClearColorValue clear{};
            if(engineInputs && index<3) {
                clear.float32[0]=index==0?3200.f:index==1?.75f:.01f;
                clear.float32[1]=index==0?1600.f:-.5f;
                clear.float32[2]=7;clear.float32[3]=-13;
            }
            vkCmdClearColorImage(cmd,b.image,VK_IMAGE_LAYOUT_GENERAL,&clear,1,&b.subresourceRange);
        }
    };
    const FfxVkXeSSRecoveredConversion conversion={sizeof(FfxVkXeSSRecoveredConversion),1.f/128.f,
        float(extent.inputWidth),float(extent.inputHeight),128.f};
    auto bindInputs=[&](uint32_t slot,const FfxVkXeSSRecoveredImages& images) {
        return engineInputs?ffxVkXeSSRecoveredBindConverted(context.get(),slot,&images,&conversion,diagnostic,sizeof(diagnostic)):
            ffxVkXeSSRecoveredBind(context.get(),slot,&images,diagnostic,sizeof(diagnostic));
    };
    const size_t inputPixels=size_t(extent.inputWidth)*extent.inputHeight;
    const size_t outputBytes=size_t(extent.outputWidth)*extent.outputHeight*8;
    auto makeUploads=[&](){return std::array<Buffer,3>{makeBuffer(inputPixels*8,VK_BUFFER_USAGE_TRANSFER_SRC_BIT),
        makeBuffer(inputPixels*(engineInputs?8:4),VK_BUFFER_USAGE_TRANSFER_SRC_BIT),makeBuffer(inputPixels*4,VK_BUFFER_USAGE_TRANSFER_SRC_BIT)};};
    const std::array<std::array<Buffer,3>,2> uploadSets={makeUploads(),makeUploads()};
    const std::array<Buffer,2> outputs={makeBuffer(outputBytes,VK_BUFFER_USAGE_TRANSFER_DST_BIT),makeBuffer(outputBytes,VK_BUFFER_USAGE_TRANSFER_DST_BIT)};
    VkCommandPoolCreateInfo cpi{VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO};cpi.queueFamilyIndex=family;cpi.flags=VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT;
    VkCommandPool commands;check(vkCreateCommandPool(device,&cpi,nullptr,&commands));
    VkCommandBufferAllocateInfo cai{VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO};cai.commandPool=commands;cai.level=VK_COMMAND_BUFFER_LEVEL_PRIMARY;cai.commandBufferCount=2;
    std::array<VkCommandBuffer,2> buffers;check(vkAllocateCommandBuffers(device,&cai,buffers.data()));
    VkCommandBuffer cmd=buffers[0];
    // Exercise a valid recording discarded before submission. Recovery must
    // reinitialize internal layouts/history; no GPU work has run yet.
    VkCommandBufferBeginInfo discarded{VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO};
    check(vkBeginCommandBuffer(cmd,&discarded));
    FfxVkXeSSReplayFrame discardedFrame{0,0,0,0,1};
    if(ffxVkXeSSRecoveredRecord(context.get(),cmd,&discardedFrame,0,diagnostic,sizeof(diagnostic))==VK_SUCCESS)
        throw std::runtime_error("Unbound context accepted");
    const auto initialBindings=bindingsFor(imageSets[0]);
    apiCheck(bindInputs(0,initialBindings));
    initializeInputs(cmd,imageSets[0]);
    apiCheck(ffxVkXeSSRecoveredRecord(context.get(),cmd,&discardedFrame,0,diagnostic,sizeof(diagnostic)));
    check(vkEndCommandBuffer(cmd));apiCheck(ffxVkXeSSRecoveredDiscardHistory(context.get()));
    VkFenceCreateInfo fi{VK_STRUCTURE_TYPE_FENCE_CREATE_INFO};VkFence fence;check(vkCreateFence(device,&fi,nullptr,&fence));
    float previousX=0,previousY=0;
    for(unsigned frame=0;frame<4;++frame) {
        const unsigned slot=frame%2;cmd=buffers[slot];
        const auto& inputs=imageSets[slot];const auto& uploads=uploadSets[slot];
        const auto bindings=bindingsFor(inputs);
        apiCheck(bindInputs(slot,bindings));
        auto invalidBindings=bindings;invalidBindings.color.extent.width=extent.inputWidth-1;
        const bool rejected=bindInputs(slot,invalidBindings)!=VK_SUCCESS;
        if(!rejected) throw std::runtime_error("Unsupported image extent accepted");
        if(engineInputs) {
            auto invalidConversion=conversion;invalidConversion.outputScale=0;
            if(ffxVkXeSSRecoveredBindConverted(context.get(),slot,&bindings,&invalidConversion,diagnostic,sizeof(diagnostic))==VK_SUCCESS)
                throw std::runtime_error("Invalid conversion accepted");
        }
        auto jitter=frames.at(frame).at("jitter").get<std::array<float,2>>();
        FfxVkXeSSReplayFrame info{jitter[0],jitter[1],previousX,previousY,frames.at(frame).at("reset").get<bool>()?1u:0u};
        const char* names[]={"color.rgba16f",engineInputs?"motion.rgba16f":"motion.rg16f","depth.r32f"};
        for(unsigned i=0;i<3;++i) {
            auto data=read((fs::path(argv[3])/("frame-"+std::to_string(frame))/names[i]).c_str());
            if(data.size()!=inputPixels*(i==0 || (i==1 && engineInputs)?8:4)) throw std::runtime_error("Input size mismatch");
            std::memcpy(uploads[i].mapped,data.data(),data.size());
        }
        check(vkResetCommandBuffer(cmd,0));VkCommandBufferBeginInfo bi{VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO};check(vkBeginCommandBuffer(cmd,&bi));
        if(frame<2) initializeInputs(cmd,inputs);
        barrier(cmd,VK_PIPELINE_STAGE_ALL_COMMANDS_BIT,VK_PIPELINE_STAGE_TRANSFER_BIT|VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
            VK_ACCESS_MEMORY_READ_BIT|VK_ACCESS_MEMORY_WRITE_BIT,VK_ACCESS_TRANSFER_WRITE_BIT|VK_ACCESS_SHADER_READ_BIT|VK_ACCESS_SHADER_WRITE_BIT);
        VkBufferImageCopy copy{};copy.imageSubresource={VK_IMAGE_ASPECT_COLOR_BIT,0,0,1};copy.imageExtent={extent.inputWidth,extent.inputHeight,1};
        for(unsigned i=0;i<3;++i) vkCmdCopyBufferToImage(cmd,uploads[i].handle,inputs[i].handle,VK_IMAGE_LAYOUT_GENERAL,1,&copy);
        barrier(cmd,VK_PIPELINE_STAGE_TRANSFER_BIT,VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,VK_ACCESS_TRANSFER_WRITE_BIT,VK_ACCESS_SHADER_READ_BIT|VK_ACCESS_SHADER_WRITE_BIT);
        auto invalidFrame=info;invalidFrame.jitterX=std::numeric_limits<float>::quiet_NaN();
        if(ffxVkXeSSRecoveredRecord(context.get(),cmd,&invalidFrame,slot,diagnostic,sizeof(diagnostic))==VK_SUCCESS)
            throw std::runtime_error("Nonfinite jitter accepted");
        if(ffxVkXeSSRecoveredRecord(context.get(),cmd,&info,ffxVkXeSSRecoveredFrameSlotCount(context.get()),diagnostic,sizeof(diagnostic))==VK_SUCCESS)
            throw std::runtime_error("Invalid frame slot accepted");
        if(!frame) {
            invalidFrame=info;invalidFrame.reset=0;
            if(ffxVkXeSSRecoveredRecord(context.get(),cmd,&invalidFrame,slot,diagnostic,sizeof(diagnostic))==VK_SUCCESS)
                throw std::runtime_error("Unseeded history accepted");
        }
        apiCheck(ffxVkXeSSRecoveredRecord(context.get(),cmd,&info,slot,diagnostic,sizeof(diagnostic)));
        barrier(cmd,VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,VK_PIPELINE_STAGE_TRANSFER_BIT,VK_ACCESS_SHADER_WRITE_BIT,VK_ACCESS_TRANSFER_READ_BIT);
        copy.imageExtent={extent.outputWidth,extent.outputHeight,1};vkCmdCopyImageToBuffer(cmd,inputs[3].handle,VK_IMAGE_LAYOUT_GENERAL,outputs[slot].handle,1,&copy);
        barrier(cmd,VK_PIPELINE_STAGE_TRANSFER_BIT,VK_PIPELINE_STAGE_HOST_BIT,VK_ACCESS_TRANSFER_WRITE_BIT,VK_ACCESS_HOST_READ_BIT);
        check(vkEndCommandBuffer(cmd));
        if(slot==1) {
            // Record both frames before submitting either. Rebinding slot 1
            // must not invalidate the executable command buffer for slot 0.
            VkSubmitInfo submit{VK_STRUCTURE_TYPE_SUBMIT_INFO};submit.commandBufferCount=2;submit.pCommandBuffers=buffers.data();
            check(vkQueueSubmit(queue,1,&submit,fence));
            check(vkWaitForFences(device,1,&fence,VK_TRUE,30000000000ull));check(vkResetFences(device,1,&fence));
            for(unsigned index=0;index<2;++index) {
                std::ofstream file(fs::path(argv[4])/("frame-"+std::to_string(frame-1+index)+".rgba16f"),std::ios::binary);
                file.write(static_cast<const char*>(outputs[index].mapped),outputBytes);file.close();if(!file) throw std::runtime_error("Output write failed");
            }
        }
        previousX=jitter[0];previousY=jitter[1];std::printf("PERSISTENT_FRAME frame=%u reset=%u\n",frame,info.reset);std::fflush(stdout);
    }
    vkDestroyFence(device,fence,nullptr);vkDestroyCommandPool(device,commands,nullptr);
    std::puts("SHARED_CONTEXT_CONTRACTS alternating_bindings=pass invalid_extent=pass invalid_jitter=pass initial_reset=pass discarded_recording=pass unbound_rejection=pass frame_slots=2 batch_size=2 invalid_slot=pass c_api=pass truncated_model=pass unaligned_shader=pass");
    if(engineInputs) std::puts("ENGINE_INPUT_CONVERSION color_scale=0.0078125 output_scale=128 motion=uv_to_pixels metadata=discarded padded_inputs=poisoned");
    context.reset();resources.clear();
    retire.active=false;
    vkDestroyDevice(device,nullptr);vkDestroyInstance(instance,nullptr);return 0;
} catch(const std::exception& e) {std::fprintf(stderr,"%s\n",e.what());return 1;}
