// The compile helper's file protocol (shims/d3d12_compile_helper.h): request lines, their order, and the markers that
// keep two processes from compiling one entry at once. No Vulkan, no helper process.
#include "../shims/d3d12_compile_helper.h"

#include <chrono>
#include <cstdio>
#include <filesystem>
#include <fstream>
#include <string>
#include <thread>

namespace fs = std::filesystem;
namespace helper = rt::d3d12::compile_helper;

#define CHECK(condition) do {                                        \
    if (!(condition)) {                                              \
        std::fprintf(stderr, "line %d: %s\n", __LINE__, #condition); \
        return 1;                                                    \
    }                                                                \
} while (0)

static int test_lines() {
    const auto line = helper::parse_request(helper::format_request("fspx_compute/lib_cs_00001_ab", 4096));
    CHECK(line.name == "fspx_compute/lib_cs_00001_ab" && line.priority == 4096);
    const auto bare = helper::parse_request("name only");
    CHECK(bare.name == "name only" && bare.priority == 0);  // an older writer's request is urgent
    return 0;
}

static int test_paths() {
    const fs::path file = fs::path("J") / "cs-0123456789abcdef.bin";
    CHECK(helper::request_path(file) == fs::path("J") / "requests" / "cs-0123456789abcdef.req");
    CHECK(helper::failed_path(file) == fs::path("J") / "requests" / "cs-0123456789abcdef.failed");
    CHECK(helper::building_path(file) == fs::path("J") / "requests" / "cs-0123456789abcdef.building");
    return 0;
}

static int test_order(const fs::path& dir) {
    const auto requests = dir / "requests";
    fs::create_directories(requests);
    const auto now = fs::file_time_type::clock::now();
    const auto put = [&](const char* stem, const std::string& line, std::chrono::seconds age) {
        const auto path = requests / (std::string(stem) + ".req");
        std::ofstream(path, std::ios::binary) << line << "\n";
        fs::last_write_time(path, now - age);
    };
    put("cs-0000000000000001", helper::format_request("warm-old", 500000), std::chrono::seconds(30));
    put("cs-0000000000000002", helper::format_request("dispatch-new", 0), std::chrono::seconds(1));
    put("cs-0000000000000003", helper::format_request("dispatch-old", 0), std::chrono::seconds(20));
    put("cs-0000000000000004", helper::format_request("warm-small", 1000), std::chrono::seconds(5));
    std::ofstream(requests / "cs-0000000000000005.failed") << "not a request\n";
    const auto list = helper::list_requests(requests);
    CHECK(list.size() == 4);
    // Urgent (dispatch) first, oldest first among equals, then warm-ups by size.
    CHECK(list[0].name == "dispatch-old" && list[1].name == "dispatch-new");
    CHECK(list[2].name == "warm-small" && list[3].name == "warm-old");
    CHECK(list[0].stem == "cs-0000000000000003");
    return 0;
}

static int test_markers(const fs::path& dir) {
    const auto file = dir / "cs-00000000000000aa.bin";
    CHECK(helper::mark_building(file));
    CHECK(!helper::mark_building(file));  // this process has it already
    helper::unmark_building(file);
    CHECK(!fs::exists(helper::building_path(file)));  // the held handle's close removes the lock file
    CHECK(helper::mark_building(file));
    helper::unmark_building(file);
    // A marker left behind names a process that no longer holds it (a killed helper) and is taken over: it is named
    // after this very process, which is the reused process id that used to make it look busy forever.
    std::ofstream(helper::building_path(file), std::ios::binary | std::ios::trunc)
        << std::to_string(GetCurrentProcessId()) << "\n";
    CHECK(fs::exists(helper::building_path(file)));
    CHECK(helper::mark_building(file));
    helper::unmark_building(file);
    CHECK(!fs::exists(helper::building_path(file)));
    // A lock file another opener holds is a live build: no claim, and nothing removes it.
    const HANDLE held = CreateFileW(helper::building_path(file).c_str(), GENERIC_WRITE, 0, nullptr, CREATE_NEW,
                                    FILE_ATTRIBUTE_NORMAL, nullptr);
    CHECK(held != INVALID_HANDLE_VALUE);
    CHECK(!helper::mark_building(file));
    CHECK(fs::exists(helper::building_path(file)));
    CloseHandle(held);
    fs::remove(helper::building_path(file));
    CHECK(helper::mark_building(file));
    helper::unmark_building(file);
    return 0;
}

// What state() answers for a request the helper has taken: no .req and no .failed means it is not coming (the helper
// removes .req before it writes .failed and before the build that writes the journal file), so it is failed rather
// than pending for as long as the helper lives.
static int test_state(const fs::path& dir) {
    auto& client = helper::Client::instance();
    client.pretend_helper_running_for_test();  // no helper process is started
    const leonida::cached_pso::Identity identity;
    const auto file = dir / "cs-00000000000000bb.bin";
    std::string why;
    CHECK(client.request(file, "fspx_compute/lib_cs_00001", 4000, identity, &why));
    CHECK(fs::exists(helper::request_path(file)));
    CHECK(client.state(file, identity, &why) == helper::Client::State::pending);
    // A bump of the priority keeps a .failed: it is the answer to this same request.
    helper::write_line(helper::failed_path(file), "no compute entry of that name");
    CHECK(client.request(file, "fspx_compute/lib_cs_00001", 0, identity, &why));
    CHECK(fs::exists(helper::failed_path(file)));
    CHECK(helper::read_line(helper::failed_path(file)) == "no compute entry of that name");
    why.clear();
    CHECK(client.state(file, identity, &why) == helper::Client::State::failed);
    CHECK(why.find("no compute entry of that name") != std::string::npos);
    // A fresh request drops an older failure, so asking again can retry.
    CHECK(client.request(file, "fspx_compute/lib_cs_00001", 0, identity, &why));
    CHECK(!fs::exists(helper::failed_path(file)));
    // .req gone, no .failed, no journal file: failed.
    const auto gone = dir / "cs-00000000000000cc.bin";
    CHECK(client.request(gone, "fspx_compute/lib_cs_00002", 0, identity, &why));
    CHECK(client.state(gone, identity, &why) == helper::Client::State::pending);
    fs::remove(helper::request_path(gone));
    why.clear();
    CHECK(client.state(gone, identity, &why) == helper::Client::State::failed);
    CHECK(!why.empty());
    // .req gone because the helper has started building it (it holds the lock file): pending, also past the pending
    // bound, since giving up would only start the same build in the game. Failed once the lock is let go with no file.
    const auto building = dir / "cs-00000000000000ee.bin";
    SetEnvironmentVariableA("LEONIDA_COMPILE_HELPER_PENDING_MS", "100");
    CHECK(client.request(building, "fspx_compute/lib_cs_00004", 0, identity, &why));
    const HANDLE lock = CreateFileW(helper::building_path(building).c_str(), GENERIC_WRITE, 0, nullptr, CREATE_NEW,
                                    FILE_ATTRIBUTE_NORMAL | FILE_FLAG_DELETE_ON_CLOSE, nullptr);
    CHECK(lock != INVALID_HANDLE_VALUE);
    fs::remove(helper::request_path(building));
    CHECK(helper::building_held(building));
    CHECK(client.state(building, identity, &why) == helper::Client::State::pending);
    std::this_thread::sleep_for(std::chrono::milliseconds(250));
    CHECK(client.state(building, identity, &why) == helper::Client::State::pending);
    CloseHandle(lock);
    CHECK(!helper::building_held(building));
    why.clear();
    CHECK(client.state(building, identity, &why) == helper::Client::State::failed);
    SetEnvironmentVariableA("LEONIDA_COMPILE_HELPER_PENDING_MS", nullptr);
    return 0;
}

// A pending request the helper never answers is failed after LEONIDA_COMPILE_HELPER_PENDING_MS, so a skipped dispatch
// becomes a build here instead of never running.
static int test_pending_bound(const fs::path& dir) {
    auto& client = helper::Client::instance();
    client.pretend_helper_running_for_test();
    const leonida::cached_pso::Identity identity;
    const auto file = dir / "cs-00000000000000dd.bin";
    SetEnvironmentVariableA("LEONIDA_COMPILE_HELPER_PENDING_MS", "150");
    std::string why;
    CHECK(client.request(file, "fspx_compute/lib_cs_00003", 0, identity, &why));
    CHECK(client.state(file, identity, &why) == helper::Client::State::pending);
    std::this_thread::sleep_for(std::chrono::milliseconds(300));
    why.clear();
    CHECK(client.state(file, identity, &why) == helper::Client::State::failed);
    CHECK(why.find("150 ms") != std::string::npos);
    SetEnvironmentVariableA("LEONIDA_COMPILE_HELPER_PENDING_MS", nullptr);
    return 0;
}

int main() {
    const auto dir = fs::temp_directory_path() /
        ("leonida_compile_helper_test_" + std::to_string(std::chrono::steady_clock::now().time_since_epoch().count()));
    std::error_code ec;
    fs::create_directories(dir, ec);
    int failed = test_lines();
    if (!failed) failed = test_paths();
    if (!failed) failed = test_order(dir);
    if (!failed) failed = test_markers(dir);
    if (!failed) failed = test_state(dir);
    if (!failed) failed = test_pending_bound(dir);
    fs::remove_all(dir, ec);
    if (failed) return failed;
    std::printf("compile_helper_test: ok\n");
    return 0;
}
