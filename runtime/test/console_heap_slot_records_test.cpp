// leonida::ConsoleHeapSlotRecords (src/console_heap_slot_records.h): a change of the address space finds exactly the
// console-heap slots whose V#s it touches, without walking the heap. Portable, CPU only, no game data.
//
// A small model stands in for the device: backings (device ranges), alias ranges (title VA -> arena offset) and a
// resolver that follows DeviceState::vsharp_address's order (an alias first, refused when its copy is not backed; then
// the base as an arena offset). fix() is fix_console_slot: it resolves both halves of a slot, writes the "copy" and the
// records. apply() is apply_slot_changes: the slots touched() names are fixed again. After every step the copy is
// compared with a full re-resolution of every slot: a slot left naming a range that went (a GPU page fault in the
// runtime) or a waiting slot left null after its buffer came fails the comparison.
#include "../src/console_heap_slot_records.h"

#include <array>
#include <cstdint>
#include <cstdio>
#include <iterator>
#include <map>
#include <random>
#include <vector>

#define EXPECT(condition) do { \
    if (!(condition)) { std::fprintf(stderr, "line %d: %s\n", __LINE__, #condition); return 1; } \
} while (0)

namespace {

using leonida::ConsoleHeapSlotRecords;
using leonida::SlotChange;

constexpr std::uint64_t kArena = 0x7f0000000000ull;  // the arena's device address
constexpr std::uint64_t kMask = (1ull << 48) - 1;

struct HalfWords {
    std::uint64_t base = 0, span = 0;  // span 0: no V# in this half
};

struct Model {
    using Words = HalfWords;
    struct Alias {
        std::uint64_t end, shader;
    };
    std::vector<std::array<Words, 2>> slots;
    std::vector<std::array<std::uint64_t, 2>> copy;  // the device address each half names, 0: null
    std::map<std::uint64_t, std::uint64_t> backings;  // device start -> end
    std::map<std::uint64_t, Alias> aliases;           // title VA -> range
    ConsoleHeapSlotRecords records;
    std::vector<SlotChange> changes;
    std::uint64_t fixes = 0;

    explicit Model(std::size_t n) : slots(n), copy(n, {0, 0}) { records.reset(n); }

    bool backed(std::uint64_t address, std::uint64_t span) const {
        auto it = backings.upper_bound(address);
        if (it == backings.begin()) return false;
        --it;
        return address >= it->first && address + span <= it->second;
    }
    // As vsharp_address: 0 when nothing backs it, with `refused` an alias copy that is not backed.
    std::uint64_t resolve(std::uint64_t base, std::uint64_t span, std::uint64_t& refused) const {
        refused = 0;
        auto it = aliases.upper_bound(base);
        if (it != aliases.begin() && base < std::prev(it)->second.end) {
            --it;
            const std::uint64_t to = kArena + it->second.shader + (base - it->first);
            if (base + span > it->second.end || !backed(to, span)) {
                refused = to;
                return 0;
            }
            return to;
        }
        if (base < (1ull << 40) && backed(kArena + base, span)) return kArena + base;
        return 0;
    }
    void fix(std::size_t slot) {
        ++fixes;
        for (unsigned half = 0; half < 2; ++half) {
            const Words& w = slots[slot][half];
            if (!w.span) {
                records.set(slot, half, ConsoleHeapSlotRecords::none, 0, 0, 0);
                copy[slot][half] = 0;
                continue;
            }
            std::uint64_t refused = 0;
            const std::uint64_t to = resolve(w.base, w.span, refused);
            records.set(slot, half, to ? ConsoleHeapSlotRecords::resolved : ConsoleHeapSlotRecords::waiting, w.base,
                        w.span, to ? to : refused);
            copy[slot][half] = to;
        }
    }
    void write(std::size_t slot, Words a, Words b = Words{}) {
        slots[slot] = {a, b};
        fix(slot);
    }
    void zero(std::size_t slot) {
        slots[slot] = {};
        copy[slot] = {0, 0};
        records.clear(slot);
    }
    // The changes the device notes (DeviceState::backing, alias_changed).
    void add_backing(std::uint64_t offset, std::uint64_t bytes) {
        backings[kArena + offset] = kArena + offset + bytes;
        changes.push_back({SlotChange::target_added, kArena + offset, kArena + offset + bytes});
        changes.push_back({SlotChange::source_added, offset & kMask, (offset & kMask) + bytes});
    }
    void release_backing(std::uint64_t offset) {
        const auto it = backings.find(kArena + offset);
        changes.push_back({SlotChange::target_gone, it->first, it->second});
        backings.erase(it);
    }
    void add_alias(std::uint64_t title, std::uint64_t bytes, std::uint64_t shader) {
        for (auto it = aliases.begin(); it != aliases.end();) {
            if (it->first < title + bytes && it->second.end > title) {
                remove_change(it->first, it->second);
                it = aliases.erase(it);
            } else {
                ++it;
            }
        }
        aliases[title] = {title + bytes, shader};
        changes.push_back({SlotChange::source_added, title, title + bytes});
    }
    void remove_alias(std::uint64_t title) {
        const auto it = aliases.find(title);
        remove_change(it->first, it->second);
        aliases.erase(it);
    }
    void remove_change(std::uint64_t title, const Alias& a) {
        changes.push_back({SlotChange::source_gone, title, a.end});
        changes.push_back({SlotChange::target_gone, kArena + a.shader, kArena + a.shader + (a.end - title)});
    }
    // apply_slot_changes: the slots looked at (index entries) and fixed again.
    std::uint64_t looked = 0;
    std::vector<std::uint32_t> touched;
    void apply() {
        const std::uint64_t before = records.looked_at();
        records.touched(changes.data(), changes.size(), touched);
        looked = records.looked_at() - before;
        changes.clear();
        for (const std::uint32_t slot : touched) fix(slot);
    }
    // Every slot as a full re-resolution would leave it: the copy the walk this replaced would have produced.
    bool consistent() const {
        for (std::size_t slot = 0; slot < slots.size(); ++slot)
            for (unsigned half = 0; half < 2; ++half) {
                const Words& w = slots[slot][half];
                std::uint64_t refused = 0;
                const std::uint64_t want = w.span ? resolve(w.base, w.span, refused) : 0;
                if (copy[slot][half] != want) {
                    std::fprintf(stderr, "slot %zu half %u names 0x%llx, a full re-resolution gives 0x%llx\n", slot, half,
                                 static_cast<unsigned long long>(copy[slot][half]),
                                 static_cast<unsigned long long>(want));
                    return false;
                }
            }
        return true;
    }
};

// The design's case: 60,010 live slots, 10 of them name buffer B. Releasing B looks at those 10 and fixes only them;
// the walk this replaced looked at all 60,010.
int release_looks_at_its_slots_only() {
    constexpr std::size_t kSlots = 65536;
    Model m(kSlots);
    constexpr std::uint64_t a = 0x100000, b = 0x900000;
    m.add_backing(a, 0x10000);
    m.add_backing(b, 0x10000);
    m.changes.clear();
    for (std::size_t slot = 0; slot < 60000; ++slot) m.write(slot, {a + (slot % 64) * 0x100, 0x100});
    for (std::size_t slot = 60000; slot < 60010; ++slot) m.write(slot, {b + (slot - 60000) * 0x100, 0x100});
    EXPECT(m.records.recorded() == 60010 && m.records.waiting_count() == 0 && m.consistent());
    m.fixes = 0;
    m.release_backing(b);
    m.apply();
    const std::uint64_t looked = m.looked;
    EXPECT(looked <= 10);
    EXPECT(m.touched.size() == 10 && m.touched.front() == 60000 && m.touched.back() == 60009);
    EXPECT(m.fixes == 10 && m.records.waiting_count() == 10 && m.consistent());
    // Nothing changed: nothing looked at.
    m.apply();
    EXPECT(m.looked == 0 && m.touched.empty());
    std::printf("release: 60,010 live slots, 10 name the buffer: %llu looked at, 10 fixed again\n",
                static_cast<unsigned long long>(looked));
    return 0;
}

// An alias range removed, replaced and added again; a slot overwritten or zeroed after it was recorded; a release that
// only partly overlaps a slot's range; two slots naming one backing; a buffer that comes back.
int cases() {
    Model m(4096);
    constexpr std::uint64_t title = 0x200000000000ull, copy_at = 0x400000, other_copy = 0x800000, c = 0x1000000;
    m.add_backing(copy_at, 0x20000);
    m.add_backing(other_copy, 0x20000);
    m.add_backing(c, 0x10000);
    m.add_alias(title, 0x20000, copy_at);
    m.apply();
    for (std::size_t slot = 0; slot < 8; ++slot) m.write(slot, {title + slot * 0x1000, 0x800});
    EXPECT(m.copy[3][0] == kArena + copy_at + 0x3000 && m.consistent());

    // Alias change: the range moves to another copy; every slot through it follows, nothing else is fixed.
    m.write(100, {c, 0x100});
    m.add_alias(title, 0x20000, other_copy);
    m.apply();
    EXPECT(m.touched.size() == 8 && m.copy[3][0] == kArena + other_copy + 0x3000 && m.consistent());
    // Removed: they wait (null in the copy); added again: they are fixed when it comes.
    m.remove_alias(title);
    m.apply();
    EXPECT(m.touched.size() == 8 && m.copy[3][0] == 0 && m.records.waiting_count() == 8 && m.consistent());
    m.add_alias(title, 0x20000, copy_at);
    m.apply();
    EXPECT(m.touched.size() == 8 && m.records.waiting_count() == 0 && m.consistent());

    // A slot overwritten after it was recorded (it names C now), and one zeroed: a release of their old copy does not
    // touch them; the zeroed one keeps no record.
    m.write(5, {c + 0x200, 0x100});
    m.zero(6);
    m.release_backing(copy_at);
    m.apply();
    EXPECT(m.touched.size() == 6);  // slots 0-4 and 7
    for (const std::uint32_t slot : m.touched) EXPECT(slot != 5 && slot != 6 && slot != 100);
    EXPECT(m.copy[5][0] == kArena + c + 0x200 && m.records.state(6, 0) == ConsoleHeapSlotRecords::none);
    EXPECT(m.consistent());

    // Partial overlap: a release that covers only the tail of a slot's range touches it; one that ends right before
    // its range does not. The two slots naming the same bytes are both fixed.
    m.add_backing(0x2000000, 0x1000);
    m.add_backing(0x2001000, 0x3000);
    m.apply();
    m.write(200, {0x2001800, 0x400});
    m.write(201, {0x2001800, 0x400});
    m.write(202, {0x2000000, 0x1000});  // ends where the second backing starts
    EXPECT(m.copy[200][0] == kArena + 0x2001800 && m.copy[202][0] == kArena + 0x2000000);
    m.release_backing(0x2001000);
    m.apply();
    EXPECT(m.touched.size() == 2 && m.touched[0] == 200 && m.touched[1] == 201);
    EXPECT(m.copy[200][0] == 0 && m.copy[201][0] == 0 && m.copy[202][0] != 0 && m.consistent());
    // The buffer comes back: the waiting slots are fixed again by the addition, not by a probe.
    m.add_backing(0x2001000, 0x3000);
    m.apply();
    EXPECT(m.touched.size() == 2 && m.copy[200][0] == kArena + 0x2001800 && m.copy[201][0] == kArena + 0x2001800);
    EXPECT(m.consistent());

    // The counter half on its own, and a refused alias copy: waiting on the copy's address, fixed when it is backed.
    constexpr std::uint64_t title2 = 0x300000000000ull, unbacked = 0x3000000;
    m.add_alias(title2, 0x10000, unbacked);
    m.apply();
    m.write(300, {c, 0x40}, {title2 + 0x40, 4});
    EXPECT(m.copy[300][0] == kArena + c && m.copy[300][1] == 0 && m.records.state(300, 1) == ConsoleHeapSlotRecords::waiting);
    m.add_backing(unbacked, 0x10000);
    m.apply();
    EXPECT(m.copy[300][1] == kArena + unbacked + 0x40 && m.consistent());

    // A large range in a granule before the release: found through its span class.
    m.add_backing(0x10000000, 0x400000);
    m.apply();
    m.write(400, {0x10000000, 0x400000});
    m.release_backing(0x10000000);
    m.apply();
    EXPECT(m.touched.size() == 1 && m.touched[0] == 400 && m.copy[400][0] == 0 && m.consistent());

    // The heap destroyed (its records with it): a new heap of the same size starts with none.
    m.records.reset(4096);
    m.release_backing(c);
    m.apply();
    EXPECT(m.touched.empty() && m.records.recorded() == 0);
    std::printf("cases: alias moved, removed and added again; overwritten and zeroed slots; partial overlap; two slots "
                "on one backing; a buffer back; a refused alias copy backed later; a large range; a destroyed heap\n");
    return 0;
}

// Random writes, zeroes, releases, additions and alias changes; after each round of changes the copy must equal a full
// re-resolution, and the entries looked at must stay near what the changes touch.
int differential() {
    std::mt19937_64 rng(0x5107);
    constexpr std::size_t kSlots = 3000;
    Model m(kSlots);
    constexpr std::uint64_t kGranule = 0x10000;
    std::vector<std::uint64_t> titles;
    for (int i = 0; i < 6; ++i) titles.push_back(0x100000000000ull + std::uint64_t(i) * 0x100000);
    const auto pick = [&](std::uint64_t n) { return std::uint64_t(rng() % n); };
    std::uint64_t looked = 0, touched = 0;
    for (int round = 0; round < 400; ++round) {
        for (int op = 0; op < 20; ++op) {
            const auto slot = std::size_t(pick(kSlots));
            switch (pick(9)) {
            case 0: case 1: case 2: {  // a V# into an arena offset, one through an alias, or a counter half too
                const auto span = 1 + pick(pick(4) ? 0x2000 : 0x80000);
                Model::Words a{pick(2) ? pick(64) * kGranule / 2 + pick(0x800) : titles[pick(6)] + pick(0x80000), span};
                Model::Words b{};
                if (!pick(4)) b = {pick(64) * kGranule / 2, 4};
                m.write(slot, a, b);
                break;
            }
            case 3: m.zero(slot); break;
            case 4: {
                const auto at = pick(64) * kGranule / 2;
                const auto it = m.backings.find(kArena + at);
                if (it == m.backings.end()) {
                    // Not over another backing.
                    const auto bytes = (1 + pick(8)) * kGranule / 4;
                    bool free = true;
                    for (const auto& [lo, hi] : m.backings) free &= !(lo < kArena + at + bytes && hi > kArena + at);
                    if (free) m.add_backing(at, bytes);
                } else {
                    m.release_backing(at);
                }
                break;
            }
            case 5: case 6: {
                const auto title = titles[pick(6)] + pick(4) * 0x20000;
                m.add_alias(title, (1 + pick(4)) * 0x8000, pick(64) * kGranule / 2);
                break;
            }
            case 7:
                if (!m.aliases.empty()) {
                    auto it = m.aliases.begin();
                    std::advance(it, std::ptrdiff_t(pick(m.aliases.size())));
                    m.remove_alias(it->first);
                }
                break;
            default: m.fix(slot); break;
            }
        }
        m.apply();
        looked += m.looked;
        touched += m.touched.size();
        if (!m.consistent()) {
            std::fprintf(stderr, "round %d\n", round);
            return 1;
        }
    }
    EXPECT(m.records.recorded() > 0);
    // The model's address space is a few MiB, so most changes touch most of it: this case checks results, not cost.
    std::printf("differential: 400 rounds, copy equal to a full re-resolution after each; %llu entries looked at, "
                "%llu slots fixed again, %zu halves recorded at the end\n",
                static_cast<unsigned long long>(looked), static_cast<unsigned long long>(touched),
                m.records.recorded());
    return 0;
}

}  // namespace

int main() {
    if (release_looks_at_its_slots_only()) return 1;
    if (cases()) return 1;
    if (differential()) return 1;
    std::printf("console_heap_slot_records_test: ok\n");
    return 0;
}
