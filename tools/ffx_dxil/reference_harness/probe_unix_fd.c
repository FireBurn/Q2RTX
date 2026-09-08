// SPDX-License-Identifier: MIT
// Experimental Wine 11.17 Unix library. Not a portable provider interface.
// Build against matching Wine source server headers, never guessed opcodes.
#define WINE_UNIX_LIB
#include <stdint.h>
#include <string.h>
#include <unistd.h>
#include <sys/stat.h>
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
        if (fstat(fd, &info)) status = STATUS_UNSUCCESSFUL;
        else args->fd_valid = 1;
        close(fd);
    }
    NtClose(object);
    return status;
}

const unixlib_entry_t __wine_unix_call_funcs[] = {inspect_shared_fd};
