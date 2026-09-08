// SPDX-License-Identifier: MIT
// Local diagnostic protocol only: no Vulkan allocation metadata yet.
#ifndef PROBE_FD_TRANSPORT_H
#define PROBE_FD_TRANSPORT_H
#include <stdint.h>
struct probe_fd_packet {
    uint32_t magic;
    uint32_t version;
    uint32_t kind; // 0: texture allocation, 1: fence semaphore
    uint32_t reserved;
};
#define PROBE_FD_MAGIC UINT32_C(0x46535246)
#endif
