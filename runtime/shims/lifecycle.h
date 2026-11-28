// App lifecycle notifications (RegisterAppStateChangeNotification / UnregisterAppStateChangeNotification) and
// oversized top-level console window fitting + suspend-on-close hook (CreateWindowExA/W).
#pragma once

#include <windows.h>

#include <cstddef>
#include <utility>

namespace rt::lifecycle {

// Scales down (width, height) proportionally to fit inside `area`, leaving fitting sizes and CW_USEDEFAULT / <= 0
// unchanged.
std::pair<int, int> fit_window_to_area(const RECT& area, int width, int height);

// For tests: number of currently registered app-state change callbacks, and synchronous suspend invocation
// (quiesced = TRUE) without terminating the process.
std::size_t registered_notification_count();
void invoke_suspend_callbacks_for_test();

}  // namespace rt::lifecycle
