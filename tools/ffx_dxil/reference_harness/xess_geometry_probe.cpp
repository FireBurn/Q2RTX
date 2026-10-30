// SPDX-License-Identifier: MIT
// Emit generated geometry/constants for independent comparison with traces.
#include "ffx_vk_xess_fixed_context.hpp"
#include <cstdio>
#include <cstdlib>

static void array(const uint32_t* values,size_t count) {
    std::printf("[");
    for(size_t i=0;i<count;++i) std::printf("%s%u",i?",":"",values[i]);
    std::printf("]");
}
int main(int argc,char** argv) try {
    if(argc!=5) throw std::invalid_argument("Usage: xess_geometry_probe inputWidth inputHeight outputWidth outputHeight");
    FfxVkXeSSReplayExtent extent{};
    uint32_t* dimensions[]={&extent.inputWidth,&extent.inputHeight,&extent.outputWidth,&extent.outputHeight};
    for(unsigned i=0;i<4;++i) {
        char* end=nullptr;auto value=std::strtoul(argv[i+1],&end,10);
        if(end==argv[i+1] || *end || value>UINT32_MAX) throw std::invalid_argument("Invalid extent");
        *dimensions[i]=static_cast<uint32_t>(value);
    }
    ffxvk::XeSSFixedGeometry geometry(extent);
    std::printf("{\"tensors\":[");
    for(unsigned i=0;i<13;++i) {if(i)std::printf(",");array(geometry.tensors[i].data(),3);}
    std::printf("],\"convolution_groups\":[");
    for(unsigned i=0;i<12;++i) {if(i)std::printf(",");array(geometry.groups[i].data(),3);}
    std::printf("],\"prepare_groups\":");array(geometry.prepare.data(),3);
    std::printf(",\"resolve_groups\":");array(geometry.resolve.data(),3);
    std::printf(",\"frames\":[");
    constexpr float jitter[4][2]={{.25f,-.25f},{-.25f,.25f},{.125f,.375f},{.25f,-.25f}};
    float previousX=0,previousY=0;
    for(unsigned frame=0;frame<4;++frame) {
        FfxVkXeSSReplayFrame info{jitter[frame][0],jitter[frame][1],previousX,previousY,frame==0||frame==3?1u:0u};
        uint32_t prepare[30],resolve[23];
        if(!ffxVkXeSSReplayConstantsForExtent(&extent,&info,prepare,resolve)) throw std::runtime_error("Invalid frame");
        std::printf("%s{\"prepare\":",frame?",":"");array(prepare,30);
        std::printf(",\"resolve\":");array(resolve,23);std::printf("}");
        previousX=info.jitterX;previousY=info.jitterY;
    }
    std::printf("]}\n");return 0;
} catch(const std::exception& error) {std::fprintf(stderr,"%s\n",error.what());return 1;}
