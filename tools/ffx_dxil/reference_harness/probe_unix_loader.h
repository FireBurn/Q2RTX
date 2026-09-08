// SPDX-License-Identifier: MIT
// Optional Wine 11.17 private-ABI diagnostic, never used by native providers.
template<typename T> static T probe_ntdll_proc(HMODULE module, const char* name)
{
    FARPROC address = GetProcAddress(module, name);
    T result = nullptr;
    static_assert(sizeof(result) == sizeof(address), "Win64 pointer ABI");
    std::memcpy(&result, &address, sizeof(result));
    return result;
}

static bool inspect_unix_shared_handle(HANDLE shared, bool semaphore)
{
    wchar_t path[32768];
    DWORD length = GetEnvironmentVariableW(L"FFX_PROBE_UNIX_FD_LIB", path, 32768);
    if (!length) return true; // Explicit opt-in only.
    if (length >= 32768 || reinterpret_cast<uintptr_t>(shared) > UINT32_MAX) return false;
    HMODULE ntdll = GetModuleHandleW(L"ntdll.dll");
    using Version = const char* (__cdecl*)();
    using Query = LONG (WINAPI*)(HANDLE, const void*, int, void*, SIZE_T, SIZE_T*);
    using Dispatch = LONG (WINAPI*)(UINT64, unsigned, void*);
    auto version = probe_ntdll_proc<Version>(ntdll, "wine_get_version");
    auto query = probe_ntdll_proc<Query>(ntdll, "NtQueryVirtualMemory");
    auto dispatcher = probe_ntdll_proc<Dispatch*>(ntdll, "__wine_unix_call_dispatcher");
    if (!version || std::strcmp(version(), "11.17") || !query || !dispatcher || !*dispatcher) {
        std::fprintf(stderr, "Unix FD experiment requires Wine 11.17.\n");
        return false;
    }
    struct UnicodeName { USHORT length, capacity; wchar_t* buffer; };
    UnicodeName name{static_cast<USHORT>(length * 2), static_cast<USHORT>(length * 2), path};
    UINT64 library[2]{};
    // Wine 11.17 winternl.h MemoryWineLoadUnixLibByName / UnloadUnixLib.
    LONG status = query(GetCurrentProcess(), &name, 1002, library, sizeof(library), nullptr);
    if (status) {
        std::fprintf(stderr, "Unix FD library load failed: 0x%08lx\n", static_cast<unsigned long>(status));
        return false;
    }
    struct Args { UINT32 abi, handle, semaphore, valid; };
    Args args{1, static_cast<UINT32>(reinterpret_cast<uintptr_t>(shared)), semaphore ? 1u : 0u, 0};
    status = (*dispatcher)(library[1], 0, &args);
    LONG unload = query(GetCurrentProcess(), &library[0], 1004, nullptr, 0, nullptr);
    std::printf("FFX_UNIX_FD kind=%s status=0x%08lx valid=%u unload=0x%08lx\n",
        semaphore ? "fence" : "texture", static_cast<unsigned long>(status), args.valid,
        static_cast<unsigned long>(unload));
    return !status && args.valid == 1 && !unload;
}
