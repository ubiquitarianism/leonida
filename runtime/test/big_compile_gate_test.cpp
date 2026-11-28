// LargeModuleGate with a budget that is read (runtime/src/large_module_gate.h): who gets in, in which order and when, and
// the settings. The threads that wait for a place are real; every check that waits for one gives up after 10 s, so a
// missing wake-up fails the test and does not hang it. Fixed budgets (vk_pipeline_check's) are large_module_gate_test.cpp's.
// No Vulkan.
#include "../src/large_module_gate.h"

#include <atomic>
#include <chrono>
#include <condition_variable>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <functional>
#include <memory>
#include <mutex>
#include <thread>
#include <vector>

namespace {

using namespace std::chrono_literals;
using leonida::LargeModuleGate;
using Work = LargeModuleGate::Work;

// Ends the process: the threads that the failed check left waiting would keep the rest from ending.
#define CHECK(condition) do {                                                           \
    if (!(condition)) {                                                                 \
        std::fprintf(stderr, "FAIL line %d: %s\n", __LINE__, #condition);               \
        std::fflush(stderr);                                                            \
        std::_Exit(1);                                                                  \
    }                                                                                   \
} while (0)

bool eventually(const std::function<bool()>& what) {
    const auto end = std::chrono::steady_clock::now() + 10s;
    while (!what()) {
        if (std::chrono::steady_clock::now() > end) return false;
        std::this_thread::sleep_for(1ms);
    }
    return true;
}

// A thread in enter() for one estimate, which keeps its place until leave() is called.
class Entrant {
public:
    Entrant(LargeModuleGate& gate, std::uint64_t estimate, Work work = Work::demand) {
        thread_ = std::thread([&gate, estimate, work, this] {
            LargeModuleGate::Ticket place(gate, estimate, work);
            in_ = true;
            std::unique_lock lock(mutex_);
            go_.wait(lock, [this] { return leave_; });
            lock.unlock();
        });
    }
    ~Entrant() {
        if (thread_.joinable()) {
            leave();
            join();
        }
    }
    Entrant(const Entrant&) = delete;
    Entrant& operator=(const Entrant&) = delete;
    bool in() const { return in_; }
    void leave() {
        std::lock_guard lock(mutex_);
        leave_ = true;
        go_.notify_all();
    }
    void join() { thread_.join(); }

private:
    std::atomic<bool> in_{false};
    std::mutex mutex_;
    std::condition_variable go_;
    bool leave_ = false;
    std::thread thread_;
};

bool waiting(LargeModuleGate& gate, Work work, std::size_t count) {
    return eventually([&] { return gate.waiting(work) == count; });
}
bool in(const Entrant& e) { return eventually([&] { return e.in(); }); }
// The time a wrong admission would need to happen.
bool stays_out(const Entrant& e) {
    std::this_thread::sleep_for(50ms);
    return !e.in();
}

LargeModuleGate fixed(std::uint64_t budget) {
    return LargeModuleGate([budget] { return budget; });
}

void bound_and_one_always() {
    auto gate = fixed(10);
    Entrant big(gate, 25);  // more than the whole budget: it compiles alone
    CHECK(in(big));
    Entrant first(gate, 6), second(gate, 4);
    CHECK(stays_out(first) && stays_out(second));
    big.leave();
    big.join();
    CHECK(in(first) && in(second));  // 10 fits exactly
    Entrant third(gate, 1);
    CHECK(waiting(gate, Work::demand, 1) && stays_out(third));
    first.leave();
    CHECK(in(third));
    CHECK(gate.inflight() == 2);
    second.leave();
    third.leave();
    second.join();
    third.join();
    first.join();
    CHECK(eventually([&] { return gate.inflight() == 0; }));
}

// Whatever the sizes, what is in fits, or is one alone; everything gets in, once.
void many_threads() {
    for (const std::uint64_t budget : {0u, 1u, 10u, 40u}) {
        auto gate = fixed(budget);
        std::atomic<std::uint64_t> used{0};
        std::atomic<unsigned> inside{0}, most{0};
        std::atomic<int> over{0}, done{0};
        std::vector<std::thread> threads;
        for (unsigned t = 0; t < 8; ++t)
            threads.emplace_back([&, t] {
                for (unsigned i = 0; i < 150; ++i) {
                    const std::uint64_t estimate = 1 + (t * 7 + i) % 13;
                    const LargeModuleGate::Ticket place(gate, estimate,
                                                        (t + i) % 3 ? Work::demand : Work::speculative);
                    const std::uint64_t sum = used += estimate;
                    const unsigned n = ++inside;
                    if (sum > budget && n > 1) ++over;
                    unsigned m = most;
                    while (n > m && !most.compare_exchange_weak(m, n)) {}
                    for (int spin = 0; spin < 20; ++spin) std::this_thread::yield();
                    --inside;
                    used -= estimate;
                    ++done;
                }
            });
        for (auto& t : threads) t.join();
        CHECK(over == 0);
        CHECK(done == 8 * 150);
        CHECK(gate.inflight() == 0);
        CHECK(budget < 40 || most > 1);  // and where there is room, several are in
    }
}

// Those that wait get in in the order they came, but for what fits now and also beside the first in line: that one finds
// its room when those that were in before it leave, whatever passed it.
void order() {
    auto gate = fixed(10);
    Entrant first(gate, 8);
    CHECK(in(first));
    Entrant big(gate, 9);
    CHECK(waiting(gate, Work::demand, 1));
    Entrant in_the_way(gate, 2);  // it fits beside the first, and not beside the big one
    CHECK(waiting(gate, Work::demand, 2));
    Entrant little(gate, 1);
    CHECK(in(little));
    Entrant next(gate, 1);  // beside the big one there is room for one of these
    CHECK(waiting(gate, Work::demand, 3));
    CHECK(stays_out(big) && stays_out(in_the_way) && stays_out(next));
    little.leave();
    little.join();
    CHECK(stays_out(big) && stays_out(in_the_way) && stays_out(next));  // nor when the one that passed has left
    first.leave();
    CHECK(in(big));
    CHECK(in(next));  // the first in line is now in_the_way, which it can pass
    CHECK(stays_out(in_the_way));
    big.leave();
    CHECK(in(in_the_way));
    in_the_way.leave();
    next.leave();
}

// What passes the first in line is bounded: a stream of small ones does not keep it out.
void a_stream_of_small_ones_does_not_keep_out() {
    auto gate = fixed(100);
    Entrant first(gate, 50);
    CHECK(in(first));
    Entrant big(gate, 70);
    CHECK(waiting(gate, Work::demand, 1));
    std::vector<std::unique_ptr<Entrant>> stream;
    for (int i = 0; i < 3; ++i) {  // 30 beside the big one's 70, one after another
        stream.push_back(std::make_unique<Entrant>(gate, 10));
        CHECK(in(*stream.back()));
        if (i < 2) {  // one of them stays in
            stream.back()->leave();
            stream.back()->join();
        }
    }
    Entrant more(gate, 10);
    CHECK(waiting(gate, Work::demand, 2));
    CHECK(stays_out(more));  // it fits (70 of 100 with the rest), but the first in line has been passed enough
    first.leave();
    CHECK(in(big));  // 10 of the stream and 70
    CHECK(in(more));
    big.leave();
    more.leave();
    stream.back()->leave();
}

// Of those that can pass, the one that came first does.
void those_that_pass_keep_their_order() {
    for (int round = 0; round < 10; ++round) {
        auto gate = fixed(20);
        Entrant first(gate, 15), other(gate, 5);
        CHECK(in(first) && in(other));
        Entrant big(gate, 12);
        CHECK(waiting(gate, Work::demand, 1));
        Entrant a(gate, 4);
        CHECK(waiting(gate, Work::demand, 2));
        Entrant b(gate, 4);
        CHECK(waiting(gate, Work::demand, 3));
        other.leave();
        CHECK(in(a));
        CHECK(stays_out(b) && stays_out(big));  // 19 of 20
        first.leave();
        CHECK(in(big) && in(b));
        big.leave();
        a.leave();
        b.leave();
    }
}

// One that needs the machine for itself is not passed: what got in beside it would keep it out.
void larger_than_the_budget_is_not_passed() {
    auto gate = fixed(10);
    Entrant first(gate, 5);
    CHECK(in(first));
    Entrant alone(gate, 25);
    CHECK(waiting(gate, Work::demand, 1));
    Entrant little(gate, 1);
    CHECK(waiting(gate, Work::demand, 2));
    CHECK(stays_out(little));
    first.leave();
    CHECK(in(alone));
    CHECK(stays_out(little));
    alone.leave();
    CHECK(in(little));
    little.leave();
}

// The same for the speculative ones, within their quarter.
void speculative_ones_pass_each_other() {
    auto gate = fixed(100);
    Entrant first(gate, 20, Work::speculative);
    CHECK(in(first));
    Entrant big(gate, 18, Work::speculative);
    CHECK(waiting(gate, Work::speculative, 1));
    Entrant little(gate, 5, Work::speculative);
    CHECK(in(little));  // 25 of 25, and 23 with the big one
    Entrant next(gate, 1, Work::speculative);
    CHECK(waiting(gate, Work::speculative, 2));
    little.leave();
    little.join();
    CHECK(in(next));  // 21 of 25, and 24 of what may pass
    Entrant over(gate, 2, Work::speculative);
    CHECK(waiting(gate, Work::speculative, 2));
    CHECK(stays_out(over) && stays_out(big));  // it would be 26
    first.leave();
    CHECK(in(big) && in(over));
    big.leave();
    next.leave();
    over.leave();
}

// The first in the queue leaves it, and the next one, which fits as well, gets in at once and not when something leaves.
void one_admission_wakes_the_next() {
    auto gate = fixed(10);
    Entrant first(gate, 10);
    CHECK(in(first));
    Entrant a(gate, 4), b(gate, 4), c(gate, 2);
    CHECK(waiting(gate, Work::demand, 3));
    first.leave();
    CHECK(in(a) && in(b) && in(c));
    CHECK(gate.inflight() == 3);
    a.leave();
    b.leave();
    c.leave();
}

void leaving_wakes_all_that_fit() {
    auto gate = fixed(10);
    Entrant a(gate, 5), b(gate, 5);
    CHECK(in(a) && in(b));
    Entrant c(gate, 3), d(gate, 2);
    CHECK(waiting(gate, Work::demand, 2) && stays_out(c));
    a.leave();
    CHECK(in(c) && in(d));
    b.leave();
    c.leave();
    d.leave();
}

// What a submission waits for goes first, whenever it came.
void demand_before_speculative() {
    auto gate = fixed(10);
    Entrant first(gate, 10);
    CHECK(in(first));
    Entrant speculative(gate, 1, Work::speculative);
    CHECK(waiting(gate, Work::speculative, 1));
    Entrant demand(gate, 9);
    CHECK(waiting(gate, Work::demand, 1));
    first.leave();
    CHECK(in(demand));  // the speculative one came first and would have fitted
    CHECK(stays_out(speculative));
    demand.leave();
    CHECK(in(speculative));
    speculative.leave();
}

// Waiting demand keeps out speculative work that fits, also the one that is first in line.
void waiting_demand_keeps_speculative_out() {
    auto gate = fixed(100);
    Entrant first(gate, 40, Work::demand);
    CHECK(in(first));
    Entrant blocked(gate, 100);  // waits for the first
    CHECK(waiting(gate, Work::demand, 1));
    Entrant speculative(gate, 5, Work::speculative);  // a quarter holds it
    CHECK(waiting(gate, Work::speculative, 1));
    CHECK(stays_out(speculative));
    first.leave();
    CHECK(in(blocked));
    CHECK(stays_out(speculative));
    blocked.leave();
    CHECK(in(speculative));
    speculative.leave();
}

// While a submission has more to build, the room between two of its modules is not a speculative one's.
void a_demand_keeps_speculative_out() {
    auto gate = fixed(100);
    auto one = std::make_unique<LargeModuleGate::Demand>(gate), other = std::make_unique<LargeModuleGate::Demand>(gate);
    Entrant speculative(gate, 10, Work::speculative);
    CHECK(waiting(gate, Work::speculative, 1));
    CHECK(stays_out(speculative));  // it fits, and nothing is in
    for (int i = 0; i < 2; ++i) {
        Entrant built(gate, 10);
        CHECK(in(built));
        built.leave();
        built.join();
        CHECK(stays_out(speculative));  // between two of the submission's modules
    }
    one.reset();
    CHECK(stays_out(speculative));  // another submission has some
    other.reset();
    CHECK(in(speculative));  // the end of the last one wakes it
    speculative.leave();
}

// A speculative one gets in within a quarter of the budget, and what is free of it is there for a demand.
void speculative_share() {
    auto gate = fixed(100);
    Entrant demand(gate, 30);
    CHECK(in(demand));
    Entrant over(gate, 26, Work::speculative);  // 56 of 100, but more than 25 with the rest
    CHECK(waiting(gate, Work::speculative, 1));
    CHECK(stays_out(over));
    demand.leave();
    CHECK(in(over));  // alone, whatever its size
    Entrant quick(gate, 70);
    CHECK(in(quick));  // 96 of 100: the speculative one does not count against it
    quick.leave();
    over.leave();
    quick.join();
    over.join();
    CHECK(eventually([&] { return gate.inflight() == 0; }));
    Entrant share(gate, 25, Work::speculative);
    CHECK(in(share));
    Entrant second(gate, 1, Work::speculative);
    CHECK(waiting(gate, Work::speculative, 1));  // one more byte is past the quarter
    CHECK(stays_out(second));
    Entrant next(gate, 75);
    CHECK(in(next));  // the whole rest
    CHECK(stays_out(second));
    next.leave();
    share.leave();
    CHECK(in(second));
    second.leave();
}

// Speculative ones get in in the order they came, and alone when nothing is in.
void speculative_order_and_alone() {
    auto gate = fixed(0);  // unknown memory: one at a time
    Entrant a(gate, 5, Work::speculative);
    CHECK(in(a));
    Entrant b(gate, 5, Work::speculative);
    CHECK(waiting(gate, Work::speculative, 1));
    Entrant c(gate, 5, Work::demand);
    CHECK(waiting(gate, Work::demand, 1));
    CHECK(stays_out(b) && stays_out(c));
    a.leave();
    CHECK(in(c));
    CHECK(stays_out(b));
    c.leave();
    CHECK(in(b));
    b.leave();
}

// The budget is read when a place is asked for and when one is given back, not once.
void budget_follows_the_moment() {
    std::atomic<std::uint64_t> free{100};
    LargeModuleGate gate([&] { return free.load(); });
    CHECK(gate.budget() == 100);
    Entrant a(gate, 30), b(gate, 10);
    CHECK(in(a) && in(b));
    free = 35;
    Entrant c(gate, 20);  // 60 of 100 a moment ago
    CHECK(waiting(gate, Work::demand, 1));
    CHECK(stays_out(c));
    free = 100;
    b.leave();
    CHECK(in(c));  // 50 of 100 now; it would not be within 35
    CHECK(gate.inflight() == 2);
    a.leave();
    c.leave();
    a.join();
    c.join();
    // A budget that shrinks while a module is in does not throw it out, and holds the next.
    Entrant d(gate, 100);
    CHECK(in(d));
    free = 1;
    Entrant e(gate, 1);
    CHECK(waiting(gate, Work::demand, 1));
    CHECK(stays_out(e));
    CHECK(gate.inflight() == 1);
    d.leave();
    CHECK(in(e));
}

// An estimate counts for as long as its module is in, whatever the driver has taken of it already.
void estimates_stay_reserved() {
    auto gate = fixed(100);  // the driver has taken nothing yet: the memory is all there
    Entrant a(gate, 60);
    CHECK(in(a));
    Entrant b(gate, 60);
    CHECK(waiting(gate, Work::demand, 1));
    CHECK(stays_out(b));
    a.leave();
    CHECK(in(b));
    b.leave();
}

// A ticket gives its place back once, however it ends.
void tickets() {
    auto gate = fixed(10);
    {
        LargeModuleGate::Ticket a(gate, 6, Work::speculative);
        LargeModuleGate::Ticket moved = std::move(a);
        CHECK(gate.inflight() == 1);
        LargeModuleGate::Ticket b;
        b = LargeModuleGate::Ticket(gate, 3);
        CHECK(gate.inflight() == 2);
        b.release();
        b.release();
        CHECK(gate.inflight() == 1);
    }
    CHECK(gate.inflight() == 0);
}

// The memory budget's setting: what it says, and nothing else.
void settings() {
    using leonida::large_compile_budget;
    const auto check = [](const char* text, bool valid, bool has, std::uint64_t mib) {
        const auto s = large_compile_budget(text);
        CHECK(s.valid == valid);
        CHECK(s.fixed.has_value() == has);
        if (has) CHECK(*s.fixed == mib << 20);
    };
    check(nullptr, true, false, 0);
    check("", true, false, 0);
    check("0", true, true, 0);
    check("1", true, true, 1);
    check("512", true, true, 512);
    check("0512", true, true, 512);
    check("12a", false, false, 0);
    check(" 5", false, false, 0);
    check("-1", false, false, 0);
    check("1.5", false, false, 0);
    check("99999999999999999999999", true, true, UINT64_MAX >> 20);  // saturates: no limit, not a small number
    CHECK(*large_compile_budget("99999999999999999999999").fixed >= (UINT64_MAX >> 20) << 20);
}

// The share of the memory, and what the machine gives now: nothing unknown on Windows and Linux, and no more than three
// quarters of either limit.
void machine() {
    using leonida::large_compile_share;
    CHECK(large_compile_share(1000, 4000) == 750);
    CHECK(large_compile_share(4000, 1000) == 750);  // the commit limit has less room than the RAM
    CHECK(large_compile_share(0, 4000) == 0);
    const std::uint64_t budget = leonida::large_compile_budget_bytes();
#if defined(_WIN32) || defined(__linux__)
    CHECK(budget > 0);
#endif
    // Read a moment apart on a running machine: the budget may not exceed either limit by more than that moment.
    const std::uint64_t slack = std::uint64_t(256) << 20;
    CHECK(budget <= leonida::available_memory_bytes() / 4 * 3 + slack);
    CHECK(leonida::large_module_estimate(1 << 20) == (std::uint64_t(1) << 20) * leonida::kBytesPerSpirvByte);
#ifdef _WIN32
    MEMORYSTATUSEX status{};
    status.dwLength = sizeof(status);
    CHECK(GlobalMemoryStatusEx(&status) != 0);
    CHECK(budget <= status.ullAvailPageFile / 4 * 3 + slack);
#endif
}

}  // namespace

int main() {
    bound_and_one_always();
    many_threads();
    order();
    a_stream_of_small_ones_does_not_keep_out();
    those_that_pass_keep_their_order();
    larger_than_the_budget_is_not_passed();
    speculative_ones_pass_each_other();
    one_admission_wakes_the_next();
    leaving_wakes_all_that_fit();
    demand_before_speculative();
    waiting_demand_keeps_speculative_out();
    a_demand_keeps_speculative_out();
    speculative_share();
    speculative_order_and_alone();
    budget_follows_the_moment();
    estimates_stay_reserved();
    tickets();
    settings();
    machine();
    std::puts("big_compile_gate: ok");
    return 0;
}
