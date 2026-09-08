// SPDX-License-Identifier: MIT
// Native Linux receiver for the isolated Wine GPU-descriptor experiment.
#define _GNU_SOURCE
#include <stdio.h>
#include <string.h>
#include <unistd.h>
#include <poll.h>
#include <sys/socket.h>
#include <sys/un.h>
#include <sys/stat.h>
#include <sys/time.h>
#include "probe_fd_transport.h"
#include "probe_native_semaphore.h"

static int receive_one(int server, unsigned expected)
{
    struct pollfd ready = {.fd = server, .events = POLLIN};
    struct ucred peer;
    socklen_t peer_size = sizeof(peer);
    struct timeval timeout = {.tv_sec = 10};
    struct probe_fd_packet packet = {0};
    struct iovec iov = {&packet, sizeof(packet)};
    union { struct cmsghdr align; char bytes[CMSG_SPACE(sizeof(int) * 8)]; } control = {0};
    struct msghdr message = {0};
    unsigned count = 0;
    int client, valid = 1, received = -1;
    if (poll(&ready, 1, 45000) != 1) return 0;
    client = accept4(server, NULL, NULL, SOCK_CLOEXEC);
    if (client < 0) return 0;
    if (setsockopt(client, SOL_SOCKET, SO_RCVTIMEO, &timeout, sizeof(timeout)) ||
        getsockopt(client, SOL_SOCKET, SO_PEERCRED, &peer, &peer_size) || peer.uid != geteuid()) {
        close(client);
        return 0;
    }
    message.msg_iov = &iov;
    message.msg_iovlen = 1;
    message.msg_control = control.bytes;
    message.msg_controllen = sizeof(control.bytes);
    ssize_t size = recvmsg(client, &message, MSG_CMSG_CLOEXEC);
    if (size != sizeof(packet) || (message.msg_flags & (MSG_TRUNC | MSG_CTRUNC))) valid = 0;
    for (struct cmsghdr *cmsg = size >= 0 ? CMSG_FIRSTHDR(&message) : NULL;
         cmsg; cmsg = CMSG_NXTHDR(&message, cmsg)) {
        if (cmsg->cmsg_len < CMSG_LEN(0)) { valid = 0; break; }
        if (cmsg->cmsg_level != SOL_SOCKET || cmsg->cmsg_type != SCM_RIGHTS) { valid = 0; continue; }
        size_t bytes = cmsg->cmsg_len - CMSG_LEN(0);
        for (size_t i = 0; i + sizeof(int) <= bytes; i += sizeof(int)) {
            int fd;
            struct stat info;
            memcpy(&fd, (char *)CMSG_DATA(cmsg) + i, sizeof(fd));
            if (fstat(fd, &info)) valid = 0;
            if (received < 0) received = fd;
            else close(fd);
            ++count;
        }
    }
    valid &= count == 1 && packet.magic == PROBE_FD_MAGIC && packet.version == 2 &&
        packet.kind == expected && packet.reserved == 0;
    printf("FFX_NATIVE_FD kind=%u descriptors=%u valid=%d\n", packet.kind, count, valid);
    if (valid && packet.kind == 1) {
        valid = probe_native_semaphore(received, packet.device_uuid);
        received = -1;
    }
    if (received >= 0) close(received);
    close(client);
    return valid;
}

int main(int argc, char **argv)
{
    struct sockaddr_un address = {.sun_family = AF_UNIX};
    if (argc != 2 || strlen(argv[1]) >= sizeof(address.sun_path)) return 2;
    strcpy(address.sun_path, argv[1]);
    umask(0077);
    int server = socket(AF_UNIX, SOCK_SEQPACKET | SOCK_CLOEXEC, 0);
    if (server < 0) return 1;
    // Never remove a pre-existing socket or other caller-owned path.
    if (bind(server, (struct sockaddr *)&address, sizeof(address))) { close(server); return 1; }
    int valid = !listen(server, 2) && receive_one(server, 0) && receive_one(server, 1);
    close(server);
    unlink(argv[1]);
    return valid ? 0 : 1;
}
