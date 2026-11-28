// Large compute modules that are work for a driver's compiler, not padding: a register file in a Function array that every
// instruction loads from and stores to (as the translator's modules have it), integer instructions, selections, nested
// conditionals and loops, and loads from memory the compiler knows nothing about. The same instructions run on the CPU, so
// the value that a dispatch stores is known. Made from a seed; no game data.
#pragma once

#include <cstddef>
#include <cstdint>
#include <initializer_list>
#include <random>
#include <utility>
#include <vector>

namespace bigmodule {

using Words = std::vector<std::uint32_t>;

inline void op(Words& out, std::uint16_t code, std::initializer_list<std::uint32_t> operands) {
    out.push_back((std::uint32_t(operands.size() + 1) << 16) | code);
    out.insert(out.end(), operands.begin(), operands.end());
}

enum class Kind : std::uint8_t { add, sub, mul, xor_, and_, or_, shl, shr, select, load, if_, end_if, loop, end_loop };

struct Instruction {
    Kind kind;
    std::uint8_t d = 0, a = 0, b = 0, c = 0;  // registers; c is a shift count or a loop count for those kinds
};

constexpr unsigned kRegisters = 48, kConstants = 64, kDepth = 3;
constexpr unsigned kLoadBase = 256, kLoadWords = 64;  // the words of the buffer the loads read (the CPU takes them as 0)

struct Program {
    std::vector<Instruction> code;
    std::uint32_t constants[kConstants];

    // What the module stores for the dispatch with this root constant.
    std::uint32_t run(std::uint32_t root_constant) const {
        std::uint32_t v[kRegisters];
        for (unsigned i = 0; i < kRegisters; ++i) v[i] = constants[i % kConstants] ^ (root_constant + i);
        walk(0, v);
        std::uint32_t out = 0;
        for (unsigned i = 0; i < kRegisters; ++i) out ^= v[i];
        return out;
    }

private:
    // Runs from `at` to the end marker of its block and returns the index after it.
    std::size_t walk(std::size_t at, std::uint32_t* v) const {
        for (; at < code.size();) {
            const Instruction& i = code[at];
            const std::uint32_t k = constants[i.c % kConstants];
            switch (i.kind) {
            case Kind::add: v[i.d] = v[i.a] + v[i.b] + k; break;
            case Kind::sub: v[i.d] = v[i.a] - v[i.b] - k; break;
            case Kind::mul: v[i.d] = v[i.a] * v[i.b] * k; break;
            case Kind::xor_: v[i.d] = v[i.a] ^ v[i.b] ^ k; break;
            case Kind::and_: v[i.d] = v[i.a] & (v[i.b] | k); break;
            case Kind::or_: v[i.d] = v[i.a] | (v[i.b] & k); break;
            case Kind::shl: v[i.d] = (v[i.a] << (i.c & 31)) ^ v[i.b]; break;
            case Kind::shr: v[i.d] = (v[i.a] >> (i.c & 31)) ^ v[i.b]; break;
            case Kind::select: v[i.d] = v[i.a] < v[i.b] ? v[i.c % kRegisters] : v[i.a]; break;
            case Kind::load: v[i.d] = v[i.b]; break;  // + the loaded word, which is 0
            case Kind::if_: {
                std::size_t after = at + 1;
                if (v[i.a] < v[i.b])
                    after = walk(at + 1, v);
                else
                    after = skip(at + 1);
                at = after;
                continue;
            }
            case Kind::loop: {
                std::size_t after = at + 1;
                for (unsigned n = 0; n < i.c; ++n) after = walk(at + 1, v);
                at = after;
                continue;
            }
            case Kind::end_if:
            case Kind::end_loop: return at + 1;
            }
            ++at;
        }
        return at;
    }
    std::size_t skip(std::size_t at) const {
        for (int depth = 1; at < code.size(); ++at) {
            const Kind k = code[at].kind;
            if (k == Kind::if_ || k == Kind::loop) ++depth;
            if ((k == Kind::end_if || k == Kind::end_loop) && !--depth) return at + 1;
        }
        return at;
    }
};

// A module of about `bytes` bytes of SPIR-V (a few KiB more), and the program it runs. `seed` decides everything. It has the
// translator's push-constant block: s[4]:s[5] is the console VA of a root UAV, s[6] a root constant. A dispatch stores
// Program::run(s[6]) at the UAV's word s[6], and reads its words kLoadBase to kLoadBase + kLoadWords - 1, which have to
// be 0.
inline Words make(std::size_t bytes, std::uint64_t seed, Program* out = nullptr) {
    std::mt19937_64 random(seed);
    Program program;
    for (auto& c : program.constants) c = std::uint32_t(random());
    // Ids.
    enum : std::uint32_t {
        id_void = 1, id_fn, id_main, id_uint, id_ulong, id_bool, id_block, id_sarr, id_c32, id_ptr_push_block,
        id_push, id_ptr_push_uint, id_ptr_push_ulong, id_data, id_ptr_data, id_ptr_phys_uint, id_regs, id_cregs,
        id_ptr_regs, id_ptr_fn_uint, id_vregs, id_entry, id_c64_32, id_c64_2, id_mask, id_base,
        id_index0 = 40,                            // uint constants 0..127
        id_const0 = id_index0 + 128,               // the program's constants
        id_counter0 = id_const0 + kConstants,      // loop counters by depth
        id_first = id_counter0 + kDepth + 1,
    };
    std::uint32_t next = id_first;
    Words w = {0x07230203u, 0x00010500u, 0u, 0u, 0u};
    op(w, 17, {1});     // Shader
    op(w, 17, {11});    // Int64
    op(w, 17, {5347});  // PhysicalStorageBufferAddresses
    op(w, 14, {5348, 1});
    op(w, 15, {5, id_main, 0x6e69616du, 0u, id_push});
    op(w, 16, {id_main, 17, 1, 1, 1});
    op(w, 71, {id_sarr, 6, 4});
    op(w, 71, {id_block, 2});
    op(w, 72, {id_block, 0, 35, 0});
    op(w, 72, {id_block, 1, 35, 128});
    op(w, 72, {id_block, 2, 35, 136});
    op(w, 71, {id_data, 2});
    op(w, 72, {id_data, 0, 35, 0});
    op(w, 19, {id_void});
    op(w, 33, {id_fn, id_void});
    op(w, 21, {id_uint, 32, 0});
    op(w, 21, {id_ulong, 64, 0});
    op(w, 20, {id_bool});
    op(w, 43, {id_uint, id_c32, 32});
    op(w, 28, {id_sarr, id_uint, id_c32});
    op(w, 30, {id_block, id_sarr, id_ulong, id_ulong});
    op(w, 32, {id_ptr_push_block, 9, id_block});
    op(w, 59, {id_ptr_push_block, id_push, 9});
    op(w, 32, {id_ptr_push_uint, 9, id_uint});
    op(w, 32, {id_ptr_push_ulong, 9, id_ulong});
    op(w, 30, {id_data, id_uint});
    op(w, 32, {id_ptr_data, 5349, id_data});
    op(w, 32, {id_ptr_phys_uint, 5349, id_uint});
    op(w, 43, {id_uint, id_cregs, kRegisters});
    op(w, 28, {id_regs, id_uint, id_cregs});
    op(w, 32, {id_ptr_regs, 7, id_regs});
    op(w, 32, {id_ptr_fn_uint, 7, id_uint});
    op(w, 43, {id_ulong, id_c64_32, 32, 0});
    op(w, 43, {id_ulong, id_c64_2, 2, 0});
    op(w, 43, {id_uint, id_mask, kLoadWords - 1});
    op(w, 43, {id_uint, id_base, kLoadBase});
    for (std::uint32_t i = 0; i < 128; ++i) op(w, 43, {id_uint, id_index0 + i, i});
    for (std::uint32_t i = 0; i < kConstants; ++i) op(w, 43, {id_uint, id_const0 + i, program.constants[i]});
    op(w, 54, {id_void, id_main, 0, id_fn});
    op(w, 248, {id_entry});
    op(w, 59, {id_ptr_regs, id_vregs, 7});
    for (std::uint32_t i = 0; i <= kDepth; ++i) op(w, 59, {id_ptr_fn_uint, id_counter0 + i, 7});
    const auto unary = [&](std::uint16_t code, std::uint32_t type, std::uint32_t a) {
        op(w, code, {type, next, a});
        return next++;
    };
    const auto binary = [&](std::uint16_t code, std::uint32_t type, std::uint32_t a, std::uint32_t b) {
        op(w, code, {type, next, a, b});
        return next++;
    };
    const auto push_word = [&](std::uint32_t index) {
        op(w, 65, {id_ptr_push_uint, next, id_push, id_index0, id_index0 + index});
        return unary(61, id_uint, next++);
    };
    const std::uint32_t lo = push_word(4), hi = push_word(5);
    op(w, 65, {id_ptr_push_ulong, next, id_push, id_index0 + 1});
    const std::uint32_t heap = unary(61, id_ulong, next++);
    const std::uint32_t lo64 = unary(113, id_ulong, lo), hi64 = unary(113, id_ulong, hi);
    const std::uint32_t high = binary(196, id_ulong, hi64, id_c64_32);
    const std::uint32_t va = binary(197, id_ulong, high, lo64);
    const std::uint32_t base = binary(128, id_ulong, va, heap);
    const std::uint32_t root = push_word(6);
    const auto slot = [&](std::uint32_t r) {
        op(w, 65, {id_ptr_fn_uint, next, id_vregs, id_index0 + r});
        return next++;
    };
    const auto load_reg = [&](std::uint32_t r) { return unary(61, id_uint, slot(r)); };
    const auto store_reg = [&](std::uint32_t r, std::uint32_t value) {
        const std::uint32_t p = slot(r);
        op(w, 62, {p, value});
    };
    for (std::uint32_t i = 0; i < kRegisters; ++i) {
        const std::uint32_t seeded = binary(128, id_uint, root, id_index0 + i);
        store_reg(i, binary(198, id_uint, id_const0 + i % kConstants, seeded));
    }
    struct Open {
        Kind kind;
        std::uint32_t merge, header, cont, counter;
        unsigned left;
    };
    std::vector<Open> open;
    unsigned loops = 0;
    const auto pick = [&](std::uint32_t n) { return std::uint8_t(random() % n); };
    const auto close = [&] {
        const Open o = open.back();
        open.pop_back();
        if (o.kind == Kind::if_) {
            op(w, 249, {o.merge});
            op(w, 248, {o.merge});
            program.code.push_back({Kind::end_if});
        } else {
            op(w, 249, {o.cont});
            op(w, 248, {o.cont});
            const std::uint32_t n = unary(61, id_uint, o.counter);
            op(w, 62, {o.counter, binary(128, id_uint, n, id_index0 + 1)});
            op(w, 249, {o.header});
            op(w, 248, {o.merge});
            program.code.push_back({Kind::end_loop});
            --loops;
        }
    };
    while (w.size() * 4 + 256 < bytes) {
        const std::uint32_t roll = std::uint32_t(random() % 100);
        if (roll < 4 && open.size() < kDepth) {
            const std::uint8_t a = pick(kRegisters), b = pick(kRegisters);
            const std::uint32_t condition = binary(176, id_bool, load_reg(a), load_reg(b));  // ULessThan
            const std::uint32_t then = next++, merge = next++;
            op(w, 247, {merge, 0});
            op(w, 250, {condition, then, merge});
            op(w, 248, {then});
            open.push_back({Kind::if_, merge, 0, 0, 0, 10 + unsigned(random() % 30)});
            program.code.push_back({Kind::if_, 0, a, b, 0});
        } else if (roll < 5 && open.size() < kDepth && loops < 1) {
            const std::uint8_t count = std::uint8_t(2 + random() % 3);
            const std::uint32_t counter = id_counter0 + loops++;
            op(w, 62, {counter, id_index0});
            const std::uint32_t header = next++, check = next++, body = next++, cont = next++, merge = next++;
            op(w, 249, {header});
            op(w, 248, {header});
            op(w, 246, {merge, cont, 0});
            op(w, 249, {check});
            op(w, 248, {check});
            const std::uint32_t n = unary(61, id_uint, counter);
            const std::uint32_t more = binary(176, id_bool, n, id_index0 + count);
            op(w, 250, {more, body, merge});
            op(w, 248, {body});
            open.push_back({Kind::loop, merge, header, cont, counter, 20 + unsigned(random() % 20)});
            program.code.push_back({Kind::loop, 0, 0, 0, count});
        } else {
            Instruction i{Kind(random() % 10), pick(kRegisters), pick(kRegisters), pick(kRegisters), pick(kConstants)};
            if (i.kind == Kind::load && random() % 2) i.kind = Kind::add;
            const std::uint32_t k = id_const0 + i.c % kConstants;
            std::uint32_t value = 0;
            const std::uint32_t a = load_reg(i.a), b = load_reg(i.b);
            switch (i.kind) {
            case Kind::add: value = binary(128, id_uint, binary(128, id_uint, a, b), k); break;
            case Kind::sub: value = binary(130, id_uint, binary(130, id_uint, a, b), k); break;
            case Kind::mul: value = binary(132, id_uint, binary(132, id_uint, a, b), k); break;
            case Kind::xor_: value = binary(198, id_uint, binary(198, id_uint, a, b), k); break;
            case Kind::and_: value = binary(199, id_uint, a, binary(197, id_uint, b, k)); break;
            case Kind::or_: value = binary(197, id_uint, a, binary(199, id_uint, b, k)); break;
            case Kind::shl: value = binary(198, id_uint, binary(196, id_uint, a, id_index0 + (i.c & 31)), b); break;
            case Kind::shr: value = binary(198, id_uint, binary(194, id_uint, a, id_index0 + (i.c & 31)), b); break;
            case Kind::select: {
                const std::uint32_t c = load_reg(i.c % kRegisters);
                const std::uint32_t less = binary(176, id_bool, a, b);
                op(w, 169, {id_uint, next, less, c, a});
                value = next++;
                break;
            }
            default: {  // load
                const std::uint32_t index = binary(128, id_uint, binary(199, id_uint, a, id_mask), id_base);
                const std::uint32_t offset = binary(196, id_ulong, unary(113, id_ulong, index), id_c64_2);
                const std::uint32_t address = binary(128, id_ulong, base, offset);
                const std::uint32_t data = unary(120, id_ptr_data, address);
                op(w, 65, {id_ptr_phys_uint, next, data, id_index0});
                const std::uint32_t pointer = next++;
                op(w, 61, {id_uint, next, pointer, 2, 4});
                const std::uint32_t loaded = next++;
                value = binary(128, id_uint, b, loaded);
                break;
            }
            }
            store_reg(i.d, value);
            program.code.push_back(i);
        }
        if (!open.empty() && !open.back().left--) close();
    }
    while (!open.empty()) close();
    std::uint32_t result = load_reg(0);
    for (std::uint32_t i = 1; i < kRegisters; ++i) result = binary(198, id_uint, result, load_reg(i));
    const std::uint32_t out_offset = binary(196, id_ulong, unary(113, id_ulong, root), id_c64_2);
    const std::uint32_t out_data = unary(120, id_ptr_data, binary(128, id_ulong, base, out_offset));
    op(w, 65, {id_ptr_phys_uint, next, out_data, id_index0});
    op(w, 62, {next++, result, 2, 4});
    op(w, 253, {});
    op(w, 56, {});
    w[3] = next;
    if (out) *out = std::move(program);
    return w;
}

}  // namespace bigmodule
