// No SDL, Windows, window, input sampling or worker. The shim instantiates this exact lifecycle with its SDL backend.
#include "../shims/keyboard_pad_connection.h"

#include <cstdio>
#include <cstdlib>
#include <limits>
#include <string>

#define CHECK(x) do { if (!(x)) { std::fprintf(stderr, "line %d: %s\n", __LINE__, #x); std::exit(1); } } while (0)

namespace {
using Connection = rt::gameinput::detail::KeyboardPadConnection<int, int>;

struct Fake {
    std::uint64_t clock = 0;
    unsigned clock_reads = 0;
    int live = 0, handle = 0, serial = 0;
    bool attach_ok = true, open_ok = true, detach_ok = true;
    std::string events;

    std::uint64_t now() { ++clock_reads; return clock; }
    int attach() {
        CHECK(!live && !handle);  // never leak or create two virtual devices
        events += 'A';
        return attach_ok ? (live = ++serial) : 0;
    }
    int open(int id) {
        CHECK(id && id == live && !handle);
        events += 'O';
        return open_ok ? (handle = live + 10) : 0;
    }
    void close(int js) {
        CHECK(js && js == handle);
        events += 'C';
        handle = 0;
    }
    bool detach(int id) {
        CHECK(id && id == serial && !handle);
        events += 'D';
        if (id != live) return false;  // SDL returns false for an already-removed device
        if (detach_ok) live = 0;
        return detach_ok;
    }
    bool is_virtual(int id) const { return id != 0 && id == live; }
    void failed() { events += '!'; }
    void attached() { CHECK(handle); events += '+'; }
    void removed() { CHECK(!live && !handle); events += '-'; }
};

void throttled_recovery() {
    Connection pad;
    Fake b;
    b.attach_ok = false;
    CHECK(!pad.update(false, b) && b.events.empty() && !b.clock_reads);
    CHECK(!pad.update(true, b) && b.events == "A!");  // first failure, including at clock zero
    for (b.clock = 2'000; b.clock < 1'000'000; b.clock += 2'000) CHECK(!pad.update(true, b));
    b.clock = 999'999;
    CHECK(!pad.update(true, b) && b.events == "A!");
    ++b.clock;
    CHECK(!pad.update(true, b) && b.events == "A!A");  // no extra warning
    b.clock += 8'500'000;
    CHECK(!pad.update(true, b) && b.events == "A!AA");  // one attempt, no catch-up loop
    b.attach_ok = true;
    b.clock += 999'999;
    CHECK(!pad.update(true, b));
    ++b.clock;
    CHECK(pad.update(true, b) && b.events == "A!AAAO+");
    CHECK(pad.id == b.live && pad.js == b.handle);
    const unsigned previous = b.clock_reads;
    for (int n = 0; n < 500; ++n) CHECK(pad.update(true, b));
    CHECK(b.events == "A!AAAO+" && previous == b.clock_reads);  // hot path does not even read the retry clock
    CHECK(!pad.update(false, b) && b.events == "A!AAAO+CD-" && !pad.id && !pad.js);
    CHECK(!pad.update(false, b) && b.events == "A!AAAO+CD-");
    b.attach_ok = false;
    CHECK(!pad.update(true, b) && b.events == "A!AAAO+CD-A!");  // new outage reports once
    CHECK(!pad.update(false, b));
    b.attach_ok = true;
    CHECK(pad.update(true, b) && b.events == "A!AAAO+CD-A!AO+");  // no old cooldown after off
}

void partial_open_and_cleanup() {
    Connection pad;
    Fake b;
    b.open_ok = false;
    CHECK(!pad.update(true, b) && b.events == "AO!D" && !pad.id && !pad.js);
    b.clock += 1'000'000;
    b.detach_ok = false;
    CHECK(!pad.update(true, b) && b.events == "AO!DAOD");
    CHECK(pad.id == b.live && !pad.js);
    b.clock += 1'000'000;
    CHECK(!pad.update(true, b) && b.events == "AO!DAODD");  // no new attach while old ID is owned
    b.open_ok = b.detach_ok = true;
    b.clock += 1'000'000;
    CHECK(pad.update(true, b) && b.events == "AO!DAODDDAO+");
    b.events.clear();
    b.detach_ok = false;
    CHECK(!pad.update(false, b) && b.events == "CD" && !pad.js && pad.id);
    CHECK(!pad.update(false, b) && b.events == "CD");  // no second close/removal or per-poll detach
    b.clock += 1'000'000;
    CHECK(!pad.update(false, b) && b.events == "CDD");
    b.detach_ok = true;
    CHECK(pad.update(true, b) && b.events == "CDDD-AO+");  // finish real removal before reattaching
    CHECK(!pad.update(false, b) && b.events == "CDDD-AO+CD-");
}

void elapsed_clock_rollover() {
    Connection pad;
    Fake b;
    b.clock = std::numeric_limits<std::uint64_t>::max() - 9;
    b.attach_ok = false;
    CHECK(!pad.update(true, b));
    b.clock += 999'999;
    b.attach_ok = true;
    CHECK(!pad.update(true, b) && b.events == "A!");
    ++b.clock;
    CHECK(pad.update(true, b) && b.events == "A!AO+");
}

void already_removed_during_cleanup() {
    Connection pad;
    Fake b;
    b.open_ok = b.detach_ok = false;
    CHECK(!pad.update(true, b) && b.events == "AO!D" && pad.id == b.live && !pad.js);
    b.live = 0;  // SDL no longer owns this virtual ID when cleanup retries
    b.clock += 1'000'000;
    b.open_ok = b.detach_ok = true;
    CHECK(pad.update(true, b) && b.events == "AO!DDAO+" && pad.id == 2);
}
}  // namespace

int main() {
    throttled_recovery();
    partial_open_and_cleanup();
    elapsed_clock_rollover();
    already_removed_during_cleanup();
    std::puts("keyboard pad retry: cadence, recovery, partial ownership, off/re-enable and rollover passed");
}
