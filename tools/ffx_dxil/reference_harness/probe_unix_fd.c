// SPDX-License-Identifier: MIT
// Experimental Wine 11.17 Unix library. Not a portable provider interface.
// Build against matching Wine source server headers, never guessed opcodes.
#define _GNU_SOURCE
#define WINE_UNIX_LIB
#include <stdint.h>
#include <string.h>
#include <unistd.h>
#include <sys/stat.h>
#include <sys/socket.h>
#include <sys/un.h>
#include <sys/time.h>
#include "probe_fd_transport.h"
#include <ntstatus.h>
#include <wine/unixlib.h>
#include <wine/server.h>

#if SERVER_PROTOCOL_VERSION != 961
#error This experiment requires the Wine 11.17 server protocol (961).
#endif

// Fixed-width PE/Unix boundary. The caller must first verify Wine 11.17.
struct probe_fd_args {
    uint32_t abi_version;
    uint32_t shared_handle;
    uint32_t is_semaphore;
    uint32_t fd_valid;
};

static int transfer_fd(int fd, uint32_t kind)
{
    const char *path = getenv("FFX_PROBE_FD_SOCKET");
    struct sockaddr_un address = {.sun_family = AF_UNIX};
    struct ucred peer;
    socklen_t peer_size = sizeof(peer);
    struct timeval timeout = {.tv_sec = 10};
    struct probe_fd_packet packet = {PROBE_FD_MAGIC, 1, kind, 0};
    struct iovec iov = {&packet, sizeof(packet)};
    union { struct cmsghdr align; char bytes[CMSG_SPACE(sizeof(int))]; } control = {0};
    struct msghdr message = {0};
    struct cmsghdr *cmsg;
    int sock, result = -1;
    if (!path || !*path) return 0;
    if (strlen(path) >= sizeof(address.sun_path)) return -1;
    strcpy(address.sun_path, path);
    sock = socket(AF_UNIX, SOCK_SEQPACKET | SOCK_CLOEXEC, 0);
    if (sock < 0) return -1;
    if (setsockopt(sock, SOL_SOCKET, SO_SNDTIMEO, &timeout, sizeof(timeout)) ||
        connect(sock, (struct sockaddr *)&address, sizeof(address)) ||
        getsockopt(sock, SOL_SOCKET, SO_PEERCRED, &peer, &peer_size) || peer.uid != geteuid())
        goto done;
    message.msg_iov = &iov;
    message.msg_iovlen = 1;
    message.msg_control = control.bytes;
    message.msg_controllen = sizeof(control.bytes);
    cmsg = CMSG_FIRSTHDR(&message);
    cmsg->cmsg_level = SOL_SOCKET;
    cmsg->cmsg_type = SCM_RIGHTS;
    cmsg->cmsg_len = CMSG_LEN(sizeof(fd));
    memcpy(CMSG_DATA(cmsg), &fd, sizeof(fd));
    if (sendmsg(sock, &message, MSG_NOSIGNAL) == sizeof(packet)) result = 0;
done:
    close(sock);
    return result;
}

static NTSTATUS inspect_shared_fd(void *opaque)
{
    struct probe_fd_args *args = opaque;
    HANDLE object = NULL;
    NTSTATUS status;
    int fd = -1;
    struct stat info;

    if (!args || args->abi_version != 1 || args->is_semaphore > 1)
        return STATUS_INVALID_PARAMETER;
    args->fd_valid = 0;
    if (!args->shared_handle) return STATUS_INVALID_HANDLE;

    SERVER_START_REQ(d3dkmt_object_open)
    {
        req->type = args->is_semaphore ? D3DKMT_SYNC : D3DKMT_RESOURCE;
        req->handle = args->shared_handle;
        status = wine_server_call(req);
        if (!status) object = wine_server_ptr_handle(reply->handle);
    }
    SERVER_END_REQ;
    if (status) return status;

    status = wine_server_handle_to_fd(object, GENERIC_ALL, &fd, NULL);
    if (!status) {
        if (fstat(fd, &info) || transfer_fd(fd, args->is_semaphore)) status = STATUS_UNSUCCESSFUL;
        else args->fd_valid = 1;
        close(fd);
    }
    NtClose(object);
    return status;
}

const unixlib_entry_t __wine_unix_call_funcs[] = {inspect_shared_fd};
