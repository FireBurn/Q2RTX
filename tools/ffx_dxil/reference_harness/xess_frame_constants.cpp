// SPDX-License-Identifier: MIT
#include "ffx_vk_xess_replay_constants.h"
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <stdexcept>
#include <cstdio>

int main(int argc, char** argv) try {
    if (argc != 8) throw std::runtime_error("Usage: xess_frame_constants jitterX jitterY previousX previousY reset prepare.bin resolve.bin");
    float values[4];
    for (int i=0;i<4;++i) {
        char* end=nullptr; values[i]=std::strtof(argv[i+1],&end);
        if (end==argv[i+1] || *end) throw std::runtime_error("Invalid jitter");
    }
    if (std::strcmp(argv[5],"0") && std::strcmp(argv[5],"1")) throw std::runtime_error("Invalid reset");
    FfxVkXeSSReplayFrame frame{values[0],values[1],values[2],values[3],uint32_t(argv[5][0]-'0')};
    uint32_t prepare[30],resolve[23];
    if (!ffxVkXeSSReplayConstants(&frame,prepare,resolve)) throw std::runtime_error("Unsupported frame parameters");
    for (int i=0;i<2;++i) {
        std::ofstream file(argv[6+i],std::ios::binary);
        file.write(reinterpret_cast<const char*>(i?resolve:prepare),i?sizeof(resolve):sizeof(prepare));
        file.close(); if (!file) throw std::runtime_error("Cannot write constants");
    }
    return 0;
} catch (const std::exception& e) { std::fprintf(stderr,"%s\n",e.what()); return 1; }
