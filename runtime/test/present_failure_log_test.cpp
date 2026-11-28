// PresentFailureLog (src/present_failure_log.h): the names of the present results, which failures in a run get a
// line (the 1st, 10th, 100th, ...), the closing line when a run ends, and that a different result is never hidden.
// No GPU, no Vulkan, no game data.
#include "../src/present_failure_log.h"

#include <cstdio>

using namespace leonida;

#define EXPECT(condition) do { \
    if (!(condition)) { std::fprintf(stderr, "line %d: %s\n", __LINE__, #condition); return 1; } \
} while (0)

namespace {

constexpr std::int32_t out_of_date = -1000001004;
constexpr std::int32_t host_memory = -1;

// Feeds `count` presents with one result and returns every line they produced.
std::vector<std::string> feed(PresentFailureLog& log, std::int32_t result, int count) {
    std::vector<std::string> all;
    for (int i = 0; i < count; ++i) {
        for (std::string& line : log.note(result)) all.push_back(std::move(line));
    }
    return all;
}

int names() {
    EXPECT(std::string(present_result_name(out_of_date)) == "VK_ERROR_OUT_OF_DATE_KHR");
    EXPECT(std::string(present_result_name(-4)) == "VK_ERROR_DEVICE_LOST");
    EXPECT(std::string(present_result_name(1000001003)) == "VK_SUBOPTIMAL_KHR");
    EXPECT(present_result_name(-7) == nullptr);
    EXPECT(describe_present_result(out_of_date) == "VK_ERROR_OUT_OF_DATE_KHR, VkResult -1000001004");
    EXPECT(describe_present_result(-7) == "VkResult -7");  // a number without a name here is still printed
    return 0;
}

int successes_are_silent() {
    PresentFailureLog log;
    EXPECT(feed(log, 0, 1000).empty());
    EXPECT(log.run() == 0);
    return 0;
}

int single_failures_each_get_a_line() {
    // The shape of d3d12_present_test's injected failures: a failure, a success, the same failure again, another one.
    PresentFailureLog log;
    auto lines = log.note(out_of_date);
    EXPECT(lines.size() == 1);
    EXPECT(lines[0] == "[rt] d3d12: present failed (VK_ERROR_OUT_OF_DATE_KHR, VkResult -1000001004)");
    EXPECT(log.note(0).empty());  // a run of one was already reported in full: no closing line
    lines = log.note(out_of_date);
    EXPECT(lines.size() == 1);
    EXPECT(lines[0] == "[rt] d3d12: present failed (VK_ERROR_OUT_OF_DATE_KHR, VkResult -1000001004)");
    EXPECT(log.note(0).empty());
    lines = log.note(host_memory);
    EXPECT(lines.size() == 1);
    EXPECT(lines[0] == "[rt] d3d12: present failed (VK_ERROR_OUT_OF_HOST_MEMORY, VkResult -1)");
    return 0;
}

int a_long_run_is_reported_at_powers_of_ten() {
    // The measured case: 180,031 presents in a row with the same result.
    PresentFailureLog log;
    const auto lines = feed(log, out_of_date, 180031);
    EXPECT(lines.size() == 6);  // the 1st, 10th, 100th, 1,000th, 10,000th and 100,000th
    EXPECT(lines[0] == "[rt] d3d12: present failed (VK_ERROR_OUT_OF_DATE_KHR, VkResult -1000001004)");
    EXPECT(lines[1] == "[rt] d3d12: present failed (VK_ERROR_OUT_OF_DATE_KHR, VkResult -1000001004), 10 times in a row");
    EXPECT(lines[5] ==
           "[rt] d3d12: present failed (VK_ERROR_OUT_OF_DATE_KHR, VkResult -1000001004), 100000 times in a row");
    EXPECT(log.run() == 180031);

    // The run ends with a success: one closing line with the whole length, then nothing more.
    const auto closing = log.note(0);
    EXPECT(closing.size() == 1);
    EXPECT(closing[0] ==
           "[rt] d3d12: present failed (VK_ERROR_OUT_OF_DATE_KHR, VkResult -1000001004) 180031 times in a row in all");
    EXPECT(log.run() == 0);
    EXPECT(log.note(0).empty());
    return 0;
}

int a_run_that_ends_on_a_reported_count_has_no_closing_line() {
    PresentFailureLog log;
    EXPECT(feed(log, out_of_date, 10).size() == 2);  // the 1st and the 10th
    EXPECT(log.note(0).empty());                     // "10 times in a row" was already said
    return 0;
}

int another_result_closes_the_run_and_starts_its_own() {
    PresentFailureLog log;
    EXPECT(feed(log, out_of_date, 25).size() == 2);  // the 1st and the 10th
    const auto lines = log.note(host_memory);        // a different failure is never folded into the run before it
    EXPECT(lines.size() == 2);
    EXPECT(lines[0] ==
           "[rt] d3d12: present failed (VK_ERROR_OUT_OF_DATE_KHR, VkResult -1000001004) 25 times in a row in all");
    EXPECT(lines[1] == "[rt] d3d12: present failed (VK_ERROR_OUT_OF_HOST_MEMORY, VkResult -1)");
    EXPECT(log.run() == 1);

    // Alternating results: every single one is logged, none is counted with the other.
    PresentFailureLog alternating;
    int printed = 0;
    for (int i = 0; i < 50; ++i) printed += int(alternating.note(i % 2 ? host_memory : out_of_date).size());
    EXPECT(printed == 50);
    return 0;
}

}  // namespace

int main() {
    if (int r = names()) return r;
    if (int r = successes_are_silent()) return r;
    if (int r = single_failures_each_get_a_line()) return r;
    if (int r = a_long_run_is_reported_at_powers_of_ten()) return r;
    if (int r = a_run_that_ends_on_a_reported_count_has_no_closing_line()) return r;
    if (int r = another_result_closes_the_run_and_starts_its_own()) return r;
    std::printf("present_failure_log: ok\n");
    return 0;
}
