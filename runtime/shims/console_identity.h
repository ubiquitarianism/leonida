// Which console the title is told it runs on: one setting, two answers.
//
// The title learns its console model two ways. Some code calls XSystemGetDeviceType; GTA VI's own code instead
// reads HKLM\Software\Microsoft\Windows NT\CurrentVersion\OEM, value ConsoleMode (a DWORD), and takes a missing
// value for a PC (docs/runtime_imports.md, "Console identity in the registry"). Both answers come from
// LEONIDA_DEVICE_TYPE through this header so they can never disagree.
//
// The ConsoleMode numbers (0x60 Series S, 0x64 Series X) are what VIctory's xsystem.rs maps (commit 2849477,
// GPL-3.0-or-later); 0x68 for the Scarlett devkit is the value xbox-emulator's xpal and gdk/hook.rs use (commit
// 0284e83, GPL-3.0).
#pragma once
#include <windows.h>

#include <cstdint>
#include <cstdlib>

namespace rt::shims::console_identity {

// XSystemDeviceType, in the order of the public documentation.
enum class DeviceType : uint32_t {
    Unknown = 0, Pc = 1, XboxOne = 2, XboxOneS = 3, XboxOneX = 4, XboxOneXDevkit = 5,
    XboxScarlettLockhart = 6, XboxScarlettAnaconda = 7, XboxScarlettDevkit = 8,
};

// The device type the title is told: LEONIDA_DEVICE_TYPE=<0..8>, read at every call so a test or a launcher change is
// seen at once. Text that is not a number counts as 0; a number outside 0..8 or an unset variable gives Series X, as
// XSystemGetDeviceType always did (the package is the Xbox Series build).
inline uint32_t configured_device_type() {
    char s[16];
    const DWORD n = GetEnvironmentVariableA("LEONIDA_DEVICE_TYPE", s, sizeof(s));
    if (n > 0 && n < sizeof(s)) {
        const int v = std::atoi(s);
        if (v >= 0 && v <= 8) return static_cast<uint32_t>(v);
    }
    return static_cast<uint32_t>(DeviceType::XboxScarlettAnaconda);
}

// The registry's ConsoleMode for a device type. False for the types that have none (Unknown, Pc and the Xbox One
// family): the title takes a missing value for a PC, so the registry shim leaves those to the host registry.
inline bool console_mode_for(uint32_t device_type, DWORD& mode) {
    switch (static_cast<DeviceType>(device_type)) {
    case DeviceType::XboxScarlettLockhart: mode = 0x60; return true;
    case DeviceType::XboxScarlettAnaconda: mode = 0x64; return true;
    case DeviceType::XboxScarlettDevkit: mode = 0x68; return true;
    default: return false;
    }
}

}  // namespace rt::shims::console_identity
