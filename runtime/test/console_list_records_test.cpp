// BL-3: the console list's inline records replay as the measured vtable calls (src/console_list_records.h).
// The DLL side is modelled independently from docs/d3d12_command_objects.md §3.5a. Each command sequence runs with
// the cursor at its limit (all vtable calls) and with room (records replayed before each call and at Close); the two
// streams must match. No game files, no GPU, no Windows.
#include "console_list_records.h"

#include <cstdio>
#include <cstring>
#include <random>
#include <string>
#include <variant>

namespace cl = leonida::console_list;

namespace {
int failures = 0;

void check(bool ok, const std::string& what) {
    if (!ok) {
        std::printf("FAIL %s\n", what.c_str());
        ++failures;
    }
}

struct Dispatch { std::uint32_t x, y, z; };
struct DispatchW { std::uint32_t x, y, z, w; };
struct Constant { unsigned root; std::uint32_t value; unsigned offset; };
struct Constants { unsigned root; std::vector<std::uint32_t> values; };
struct Table { unsigned root; std::uint32_t address; };
struct RootView { unsigned root; cl::RootKind kind; std::uint64_t va; };
struct Pipeline { std::uintptr_t object; };
using Command = std::variant<Dispatch, DispatchW, Constant, Constants, Table, RootView, Pipeline>;

// The 28 bytes of a pipeline object (+0x10..+0x2B). This runtime owns the object, so it chooses them: a marker
// (bit 30 set, never a root header) and the address.
constexpr std::uint32_t kPipelineMarker = 0x50534F31;  // 'PSO1', bit 30 set: never a root header
std::array<std::uint8_t, cl::pipeline_record_bytes> pipeline_bytes(std::uintptr_t object) {
    std::array<std::uint8_t, cl::pipeline_record_bytes> b{};
    std::memcpy(b.data(), &kPipelineMarker, 4);
    const std::uint64_t o = object;
    std::memcpy(b.data() + 4, &o, 8);
    return b;
}
std::uintptr_t resolve(std::span<const std::uint8_t, cl::pipeline_record_bytes> b) {
    std::uint32_t marker;
    std::memcpy(&marker, b.data(), 4);
    if (marker != kPipelineMarker) return 0;
    std::uint64_t o;
    std::memcpy(&o, b.data() + 4, 8);
    return static_cast<std::uintptr_t>(o);
}

// The fallback, from the measured shapes (§3.3a, §3.5a), not from the decoder.
cl::Call vtable_call(const Command& c) {
    cl::Call call;
    if (auto* d = std::get_if<Dispatch>(&c)) {
        call.slot = 14; call.args = {d->x, d->y, d->z, 0};
    } else if (auto* d = std::get_if<DispatchW>(&c)) {
        call.slot = 80; call.args = {d->x, d->y, d->z, d->w};
    } else if (auto* k = std::get_if<Constant>(&c)) {
        call.slot = 37; call.args = {k->root, k->value, k->offset, 0};
    } else if (auto* k = std::get_if<Constants>(&c)) {
        call.slot = 39; call.args = {k->root, k->values.size(), 0, 0}; call.constants = k->values;
    } else if (auto* t = std::get_if<Table>(&c)) {
        call.slot = 35; call.args = {t->root, t->address, 0, 0};
    } else if (auto* v = std::get_if<RootView>(&c)) {
        call.slot = v->kind == cl::RootKind::cbv ? 41 : v->kind == cl::RootKind::srv ? 43 : 45;
        call.args = {v->root, v->va, 0, 0};
    } else if (auto* p = std::get_if<Pipeline>(&c)) {
        call.slot = 25; call.args = {p->object, 0, 0, 0}; call.pipeline = p->object;
    }
    return call;
}

struct List {
    std::array<std::uint32_t, cl::max_root_parameters> table{};  // +0x344: filled by the runtime
    std::vector<std::uint8_t> records;
    std::size_t limit = 0;          // bytes of room; 0 = cursor at its limit
    std::vector<cl::Call> stream;   // what reaches the real list
    bool decode_failed = false;

    void flush() {
        const cl::Decoded d = cl::decode(records, resolve);
        if (d.error) {
            std::printf("  decode stopped at +%zu: %s\n", d.error->offset, d.error->what);
            decode_failed = true;
        }
        stream.insert(stream.end(), d.calls.begin(), d.calls.end());
        records.clear();
    }
    void call(const cl::Call& c) {  // pending records go first
        flush();
        stream.push_back(c);
    }
    void put(std::uint32_t w) {
        const auto* p = reinterpret_cast<const std::uint8_t*>(&w);
        records.insert(records.end(), p, p + 4);
    }
};

// Inline when the record fits and the table entry allows it, else the vtable.
void issue(List& list, const Command& c) {
    std::vector<std::uint32_t> words;
    std::vector<std::uint8_t> raw;  // the pipeline record is bytes, not words
    if (auto* d = std::get_if<Dispatch>(&c)) {
        words = {cl::dispatch_header, d->x, d->y, d->z};
    } else if (auto* d = std::get_if<DispatchW>(&c)) {
        words = {cl::dispatch_w_header, d->x, d->y, d->z, d->w};
    } else if (auto* k = std::get_if<Constant>(&c)) {
        words = {list.table[k->root] + 4 * k->offset, k->value};
    } else if (auto* k = std::get_if<Constants>(&c)) {
        words = {list.table[k->root] + (static_cast<std::uint32_t>(k->values.size() - 1) << 16)};
        words.insert(words.end(), k->values.begin(), k->values.end());
    } else if (auto* t = std::get_if<Table>(&c)) {
        words = {list.table[t->root], t->address};
    } else if (auto* v = std::get_if<RootView>(&c)) {
        words = {list.table[v->root], static_cast<std::uint32_t>(v->va), static_cast<std::uint32_t>(v->va >> 32)};
    } else if (auto* p = std::get_if<Pipeline>(&c)) {
        const auto b = pipeline_bytes(p->object);
        raw.assign(b.begin(), b.end());
    }
    const std::size_t size = raw.empty() ? 4 * words.size() : raw.size();
    const bool is_root = !std::holds_alternative<Dispatch>(c) && !std::holds_alternative<DispatchW>(c) &&
                         !std::holds_alternative<Pipeline>(c);
    const bool blocked = is_root && (words[0] & cl::inline_blocked_bits);
    if (blocked || list.records.size() + size > list.limit) {
        list.call(vtable_call(c));
        return;
    }
    if (raw.empty()) {
        for (std::uint32_t w : words) list.put(w);
    } else {
        list.records.insert(list.records.end(), raw.begin(), raw.end());
    }
}

// kinds[root] is the root parameter's kind; the runtime fills the table from it.
std::vector<cl::RootKind> random_signature(std::mt19937& rng) {
    const unsigned n = 1 + rng() % 12;
    std::vector<cl::RootKind> kinds(n);
    for (auto& k : kinds) k = static_cast<cl::RootKind>(1 + rng() % 5);
    return kinds;
}

Command random_command(std::mt19937& rng, const std::vector<cl::RootKind>& sig) {
    const unsigned pick = rng() % 6;
    if (pick == 0) return Dispatch{rng(), 1 + rng() % 64, 1 + rng() % 4};
    if (pick == 1) return DispatchW{rng(), 1, 1, (rng() & 1) ? 0x8020u : 0x20u};
    if (pick == 2) return Pipeline{static_cast<std::uintptr_t>(0x10000 + 0x100 * (rng() % 4096))};
    const unsigned root = rng() % sig.size();
    switch (sig[root]) {
    case cl::RootKind::constants:
        if (rng() & 1) return Constant{root, rng(), static_cast<unsigned>(rng() % 64)};
        else {
            std::vector<std::uint32_t> v(2 + rng() % 63);
            for (auto& x : v) x = rng();
            return Constants{root, v};
        }
    case cl::RootKind::table: return Table{root, rng()};
    default: return RootView{root, sig[root], (std::uint64_t{rng()} << 32) | rng()};
    }
}

void fill_table(List& list, const std::vector<cl::RootKind>& sig) {
    for (unsigned i = 0; i < sig.size(); ++i) list.table[i] = cl::root_header(i, sig[i]);
}

void check_paths_agree() {
    std::mt19937 rng(0xB13);
    unsigned sequences = 0, inline_records = 0;
    for (const std::size_t limit : {std::size_t{0}, std::size_t{8}, std::size_t{20}, std::size_t{64},
                                    std::size_t{300}, std::size_t{4096}}) {
        for (int round = 0; round < 200; ++round) {
            const auto sig = random_signature(rng);
            std::vector<Command> cmds(1 + rng() % 40);
            for (auto& c : cmds) c = random_command(rng, sig);

            List vtable_only;  // cursor at its limit
            fill_table(vtable_only, sig);
            List with_room;
            fill_table(with_room, sig);
            with_room.limit = limit;
            for (const auto& c : cmds) {
                issue(vtable_only, c);
                const std::size_t before = with_room.records.size();
                issue(with_room, c);
                if (with_room.records.size() > before) ++inline_records;
            }
            vtable_only.flush();  // Close
            with_room.flush();
            ++sequences;
            check(!with_room.decode_failed, "limit " + std::to_string(limit) + ": every record decodes");
            check(vtable_only.stream == with_room.stream,
                  "limit " + std::to_string(limit) + " round " + std::to_string(round) +
                      ": the replayed records give the vtable calls in the same order");
        }
    }
    std::printf("  %u sequences, %u commands written inline and replayed\n", sequences, inline_records);
    check(inline_records > 1000, "the inline path was exercised");
}

// One of each measured form.
void check_each_form() {
    List list;
    const std::vector<cl::RootKind> sig = {cl::RootKind::constants, cl::RootKind::table, cl::RootKind::cbv,
                                           cl::RootKind::srv, cl::RootKind::uav};
    fill_table(list, sig);
    list.limit = 4096;
    issue(list, Constants{0, std::vector<std::uint32_t>(19, 7)});  // the builder's 19 at 0x14e3b
    issue(list, Constant{0, 0xABCD, 3});                           // + 0xc at 0x19fca
    issue(list, Table{1, 0x12345678});
    issue(list, RootView{2, cl::RootKind::cbv, 0x1'2345'6789ull});
    issue(list, RootView{3, cl::RootKind::srv, 0xFFFF'0000'0000'0010ull});
    issue(list, RootView{4, cl::RootKind::uav, 8});
    issue(list, Pipeline{0x7000});
    issue(list, Dispatch{5, 1, 1});
    issue(list, DispatchW{9, 1, 1, 0x8020});
    // 4+4n, 8, 8, 12 x3, 28, 16, 20.
    check(list.records.size() == (4 + 4 * 19) + 8 + 8 + 12 * 3 + 28 + 16 + 20, "record sizes match §3.5a");
    const cl::Decoded d = cl::decode(list.records, resolve);
    check(!d.error, "the nine forms decode");
    const std::vector<unsigned> slots = {39, 37, 35, 41, 43, 45, 25, 14, 80};
    check(d.calls.size() == slots.size(), "nine calls");
    for (std::size_t i = 0; i < slots.size() && i < d.calls.size(); ++i)
        check(d.calls[i].slot == slots[i], "call " + std::to_string(i) + " is slot " + std::to_string(slots[i]));
    if (d.calls.size() == slots.size()) {
        check(d.calls[0].args[1] == 19 && d.calls[0].constants.size() == 19, "19 constants");
        check(d.calls[1].args[2] == 3 && d.calls[1].args[1] == 0xABCD, "one constant at offset 3");
        check(d.calls[3].args[1] == 0x1'2345'6789ull, "root CBV keeps all 64 address bits");
        check(d.calls[4].args[1] == 0xFFFF'0000'0000'0010ull, "root SRV keeps the high half");
        check(d.calls[8].args[3] == 0x8020, "slot 80 keeps W");
    }
}

void check_header_encoding() {
    for (unsigned root = 0; root < cl::max_root_parameters; ++root)
        for (unsigned k = 1; k <= 5; ++k) {
            const std::uint32_t h = cl::root_header(root, static_cast<cl::RootKind>(k));
            check((h & cl::inline_blocked_bits) == 0, "a table entry lets the DLL write inline");
            check(h != cl::dispatch_header && h != cl::dispatch_w_header, "a table entry is not a dispatch header");
            // The largest adjustments the DLL adds stay inside their fields.
            check(((h + (63u << 16)) >> 22) == (h >> 22), "a 64-constant count does not reach the root index");
            check(((h + 4 * 63) >> 8) == (h >> 8), "offset 63 does not reach the kind");
        }
}

std::vector<std::uint8_t> bytes_of(std::initializer_list<std::uint32_t> words) {
    std::vector<std::uint8_t> b(4 * words.size());
    std::memcpy(b.data(), std::data(words), b.size());
    return b;
}

void expect_error(const std::vector<std::uint8_t>& b, std::size_t offset, std::size_t calls_before,
                  const char* what) {
    const cl::Decoded d = cl::decode(b, resolve);
    check(d.error.has_value(), std::string(what) + ": refused");
    if (d.error) check(d.error->offset == offset, std::string(what) + ": at its offset");
    check(d.calls.size() == calls_before, std::string(what) + ": the records before it are kept");
}

void check_refusals() {
    const std::uint32_t t = cl::root_header(2, cl::RootKind::table);
    const std::uint32_t k = cl::root_header(0, cl::RootKind::constants);
    const std::uint32_t v = cl::root_header(1, cl::RootKind::srv);
    expect_error(bytes_of({cl::dispatch_header, 1, 1, 1, 0xDEADBEEF}), 16, 1, "an unknown header");
    expect_error(bytes_of({cl::dispatch_header, 1, 1}), 0, 0, "a truncated dispatch");
    expect_error(bytes_of({cl::dispatch_w_header, 1, 1, 1}), 0, 0, "a truncated slot-80 record");
    expect_error(bytes_of({v, 1}), 0, 0, "a truncated root descriptor");
    expect_error(bytes_of({k + (3u << 16), 1, 2}), 0, 0, "fewer constants than the count");
    expect_error(bytes_of({k + (1u << 16) + 4, 1, 2}), 0, 0, "a count and an offset together");
    expect_error(bytes_of({t + 4, 1}), 0, 0, "an offset on a descriptor table");
    expect_error(bytes_of({cl::root_tag | (6u << 8), 1}), 0, 0, "a header naming no kind");
    expect_error(bytes_of({k | 0x800, 1}), 0, 0, "a header with a bit the runtime never sets");
    expect_error(bytes_of({k | 0x40000000, 1}), 0, 0, "a header with bit 30 set");
    // Without a resolver a pipeline record is unknown, not skipped.
    List list;
    list.limit = 64;
    issue(list, Pipeline{0x7000});
    check(cl::decode(list.records).error.has_value(), "a pipeline record without a resolver is refused");
    std::vector<std::uint8_t> odd = bytes_of({cl::dispatch_header, 1, 1, 1});
    odd.push_back(0);
    expect_error(odd, 16, 0, "a byte count that is not whole dwords");
    check(!cl::decode({}).error && cl::decode({}).calls.empty(), "no records, no calls");
}
}  // namespace

int main() {
    check_header_encoding();
    check_each_form();
    check_refusals();
    check_paths_agree();
    if (failures) {
        std::printf("RESULT: FAIL (%d failures)\n", failures);
        return 1;
    }
    std::printf("RESULT: PASS\n");
    return 0;
}
