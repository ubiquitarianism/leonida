#pragma once

#include <windows.h>
#include <future>
#include <utility>
#include <thread>

// The thread that owns an HWND must continue dispatching window messages while another
// thread creates pipelines or waits for the GPU. A stop event also wakes an idle window.
namespace rt::d3d12_xs {
class PresentWindow {
public:
    PresentWindow() = default;
    PresentWindow(const PresentWindow&) = delete;
    PresentWindow& operator=(const PresentWindow&) = delete;
    ~PresentWindow() {
        if (stop_) SetEvent(stop_);
        if (owner_.joinable()) owner_.join();
        if (stop_) CloseHandle(stop_);
    }

    HWND create() {
        if (stop_) return nullptr;
        stop_ = CreateEventW(nullptr, TRUE, FALSE, nullptr);
        if (!stop_) return nullptr;
        try {
            std::promise<std::pair<HWND, DWORD>> ready;
            auto result = ready.get_future();
            owner_ = std::thread([stop = stop_, ready = std::move(ready)]() mutable {
                WNDCLASSW cls{};
                cls.lpfnWndProc = &procedure;
                cls.hInstance = GetModuleHandleW(nullptr);
                cls.lpszClassName = L"LeonidaPresentX";
                HWND window = nullptr;
                if (RegisterClassW(&cls) || GetLastError() == ERROR_CLASS_ALREADY_EXISTS)
                    window = CreateWindowExW(0, cls.lpszClassName, L"Leonida", WS_OVERLAPPEDWINDOW,
                                             CW_USEDEFAULT, CW_USEDEFAULT, CW_USEDEFAULT, CW_USEDEFAULT,
                                             nullptr, nullptr, cls.hInstance, nullptr);
                ready.set_value({window, window ? ERROR_SUCCESS : GetLastError()});
                if (!window) return;
                while (MsgWaitForMultipleObjectsEx(1, &stop, INFINITE, QS_ALLINPUT, MWMO_INPUTAVAILABLE) ==
                       WAIT_OBJECT_0 + 1) {
                    MSG msg;
                    while (WaitForSingleObject(stop, 0) == WAIT_TIMEOUT && PeekMessageW(&msg, nullptr, 0, 0, PM_REMOVE)) {
                        if (msg.message == WM_QUIT) return;
                        TranslateMessage(&msg);
                        DispatchMessageW(&msg);
                    }
                }
                if (IsWindow(window)) DestroyWindow(window);
            });
            const auto [window, error] = result.get();
            if (!window) SetLastError(error);
            return window;
        } catch (const std::exception&) {
            SetLastError(ERROR_NOT_ENOUGH_MEMORY);
            return nullptr;
        }
    }

private:
    static LRESULT CALLBACK procedure(HWND window, UINT message, WPARAM wparam, LPARAM lparam) {
        if (message == WM_DESTROY) {
            PostQuitMessage(0);
            return 0;
        }
        return DefWindowProcW(window, message, wparam, lparam);
    }
    HANDLE stop_ = nullptr;
    std::thread owner_;
};
}  // namespace rt::d3d12_xs
