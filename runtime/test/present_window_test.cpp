// The submitting thread does no message dispatch. Each hidden presentation window must still
// process messages, close on its owner thread and allow the other window to keep running.
#include "../shims/present_window.h"

#include <cstdio>
#include <memory>

namespace {
bool send(HWND window, UINT message) {
    DWORD_PTR result = 0;
    return SendMessageTimeoutW(window, message, 0, 0, SMTO_ABORTIFHUNG | SMTO_BLOCK, 2000, &result) != 0;
}
}

int main() {
    HWND control = CreateWindowExW(0, L"STATIC", L"", WS_POPUP, 0, 0, 1, 1, nullptr, nullptr,
                                   GetModuleHandleW(nullptr), nullptr);
    if (!control) {
        std::fprintf(stderr, "no Win32 desktop available (error %lu)\n", GetLastError());
        return 77;
    }
    DestroyWindow(control);

    rt::d3d12_xs::PresentWindow first;
    auto second = std::make_unique<rt::d3d12_xs::PresentWindow>();
    const HWND a = first.create(), b = second->create();
    if (!a || !b || a == b || GetWindowThreadProcessId(a, nullptr) == GetCurrentThreadId() ||
        !send(a, WM_NULL) || !send(b, WM_NULL)) return 1;
    if (!send(a, WM_CLOSE) || IsWindow(a) || !send(b, WM_NULL)) return 1;

    std::thread release([window = std::move(second)]() mutable { window.reset(); });
    release.join();
    if (IsWindow(b)) return 1;
    return 0;
}
