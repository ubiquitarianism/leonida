#pragma once

// What the runtime logs when a present fails (shims/d3d12_vulkan_device.cpp, present_result), and how often.
// No Vulkan and no Windows headers, so the test runs everywhere; the device file checks the numbers below against
// the real VkResult values with static_asserts.
//
// Before this, every failed present printed "[rt] d3d12: present failed (VkResult <number>)". A presenter that
// fails every frame then writes one line per frame for as long as the title runs: a retail run on an AMD RX 6750 XT
// under Windows, where every present returned VK_ERROR_OUT_OF_DATE_KHR, left 180,031 identical lines in 15 minutes
// (most of a log of about 1 GB together with the D3D12 trace), and the line gave only the number.
//
// Now the result is named, and a run of the same result in a row is reported at its 1st, 10th, 100th, ... failure
// only. When the run ends (a success, or another result) a closing line gives its length if it was longer than what
// the last report said. A different result always starts a new run, so no kind of failure is ever hidden.

#include <cstdint>
#include <string>
#include <vector>

namespace leonida {

// The VkResult values a present can come back with, by name; nullptr for any other number.
inline const char* present_result_name(std::int32_t result) {
    switch (result) {
    case 0: return "VK_SUCCESS";
    case 1: return "VK_NOT_READY";
    case 2: return "VK_TIMEOUT";
    case 1000001003: return "VK_SUBOPTIMAL_KHR";
    case -1: return "VK_ERROR_OUT_OF_HOST_MEMORY";
    case -2: return "VK_ERROR_OUT_OF_DEVICE_MEMORY";
    case -3: return "VK_ERROR_INITIALIZATION_FAILED";
    case -4: return "VK_ERROR_DEVICE_LOST";
    case -11: return "VK_ERROR_FORMAT_NOT_SUPPORTED";
    case -13: return "VK_ERROR_UNKNOWN";
    case -1000000000: return "VK_ERROR_SURFACE_LOST_KHR";
    case -1000000001: return "VK_ERROR_NATIVE_WINDOW_IN_USE_KHR";
    case -1000001004: return "VK_ERROR_OUT_OF_DATE_KHR";
    case -1000255000: return "VK_ERROR_FULL_SCREEN_EXCLUSIVE_MODE_LOST_EXT";
    default: return nullptr;
    }
}

// "VK_ERROR_OUT_OF_DATE_KHR, VkResult -1000001004", or "VkResult -7" for a number without a name here.
inline std::string describe_present_result(std::int32_t result) {
    const char* name = present_result_name(result);
    const std::string number = "VkResult " + std::to_string(result);
    return name ? std::string(name) + ", " + number : number;
}

// Decides which failed presents get a log line. Not thread-safe: the caller serialises note().
class PresentFailureLog {
public:
    // The lines to log for this present's result, without the trailing newline; usually none or one.
    std::vector<std::string> note(std::int32_t result) {
        std::vector<std::string> lines;
        if (run_ != 0 && result != last_) {
            if (run_ > reported_) lines.push_back(closing_line());
            run_ = 0;
            reported_ = 0;
        }
        if (result == 0) return lines;  // VK_SUCCESS
        last_ = result;
        ++run_;
        if (is_power_of_ten(run_)) {
            std::string line = "[rt] d3d12: present failed (" + describe_present_result(result) + ")";
            if (run_ > 1) line += ", " + std::to_string(run_) + " times in a row";
            lines.push_back(std::move(line));
            reported_ = run_;
        }
        return lines;
    }

    // The failures in the current run of one result; 0 after a success.
    std::uint64_t run() const { return run_; }

private:
    static bool is_power_of_ten(std::uint64_t n) {
        while (n >= 10 && n % 10 == 0) n /= 10;
        return n == 1;
    }

    std::string closing_line() const {
        return "[rt] d3d12: present failed (" + describe_present_result(last_) + ") " + std::to_string(run_) +
               " times in a row in all";
    }

    std::int32_t last_ = 0;
    std::uint64_t run_ = 0;       // length of the current run of last_
    std::uint64_t reported_ = 0;  // the run length the last printed line gave
};

}  // namespace leonida
