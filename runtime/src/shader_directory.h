// Shared shader-library location for the Vulkan and DirectStorage runtimes.
#pragma once
#include <windows.h>
#include <bit>
#include <filesystem>
#include <fstream>
#include <iterator>
#include <memory>
#include <string>

namespace leonida::shader_paths {
// The folder make_game recorded in shader_path.txt at `config`, or empty without one. Throws when the path cannot be
// converted: MSVC's STL reports a Windows path that is not UTF-8 as std::system_error, libstdc++ as filesystem_error.
inline std::filesystem::path shader_directory_from_file(const std::filesystem::path& config) {
    std::ifstream file(config, std::ios::binary);
    if (!file) return {};
    const std::string text{std::istreambuf_iterator<char>(file), std::istreambuf_iterator<char>()};
    if (text.empty() || text.find('\0') != std::string::npos) return {};

    // make_game records an absolute Unix path on Linux. Resolve it inside the running Wine prefix.
    if (text.front() == '/') {
        using WineDosPath = wchar_t* (CDECL*)(const char*);
        const auto to_dos = std::bit_cast<WineDosPath>(
            ::GetProcAddress(::GetModuleHandleW(L"kernel32.dll"), "wine_get_dos_file_name"));
        if (!to_dos) return {};
        const auto release = [](wchar_t* path) { ::HeapFree(::GetProcessHeap(), 0, path); };
        const std::unique_ptr<wchar_t, decltype(release)> path(to_dos(text.c_str()), release);
        if (path) return std::filesystem::path(path.get());
        return {};
    }
    return std::filesystem::path(std::u8string(text.begin(), text.end()));
}

inline std::filesystem::path configured(std::string* error = nullptr, const std::filesystem::path& config = {}) {
    try {
        constexpr wchar_t env_name[] = L"LEONIDA_SHADERS";
        if (const DWORD size = ::GetEnvironmentVariableW(env_name, nullptr, 0)) {
            std::wstring value(size, L'\0');
            const DWORD length = ::GetEnvironmentVariableW(env_name, value.data(), size);
            if (length && length < size) {
                value.resize(length);
                return std::filesystem::path(value);
            }
        }
        if (!config.empty()) return shader_directory_from_file(config);
        std::wstring exe(32768, L'\0');
        const DWORD length = ::GetModuleFileNameW(nullptr, exe.data(), static_cast<DWORD>(exe.size()));
        if (!length || length >= exe.size()) return {};
        exe.resize(length);
        return shader_directory_from_file(std::filesystem::path(exe).parent_path() / L"shader_path.txt");
    } catch (const std::exception& e) {
        if (error) *error = e.what();
        return {};
    }
}
} // namespace leonida::shader_paths
