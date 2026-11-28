// CPU-only SPIR-V validation and execution test for runtime/src/depth24_fragment.h (D24 and Xbox D16S8 fallback).
// Verifies structural SPIR-V invariants (bounds, section order, pre-function declarations, interface variables,
// execution modes) and interprets the emitted SPIR-V instructions directly across all 65,536 16-bit codes,
// half-step ties, IEEE-754 specials, depth bias/slope/clamp, and 24-bit parity. No Vulkan device or game data.
#include "../src/depth24_fragment.h"

#include <algorithm>
#include <array>
#include <bit>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <limits>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <vector>

#define EXPECT(cond)                                                                                           \
    do {                                                                                                       \
        if (!(cond)) {                                                                                         \
            std::fprintf(stderr, "depth24_fragment_test: line %d failed: %s\n", __LINE__, #cond);              \
            return 1;                                                                                          \
        }                                                                                                      \
    } while (0)

namespace {

using namespace leonida;
using namespace leonida::spirv24;

std::uint32_t f_bits(float f) {
    std::uint32_t u = 0;
    std::memcpy(&u, &f, 4);
    return u;
}

float u_float(std::uint32_t u) {
    float f = 0.0f;
    std::memcpy(&f, &u, 4);
    return f;
}

double round_half_even(double x) {
    const double r = std::floor(x + 0.5);
    return r - x == 0.5 && std::fmod(r, 2.0) != 0.0 ? r - 1.0 : r;
}

std::uint32_t ref_quantize_code(float z, double max_levels, std::uint32_t max_code) {
    if (!(z > 0.0f)) return 0u;
    if (z >= 1.0f) return max_code;
    const float product = z * static_cast<float>(max_levels);
    return static_cast<std::uint32_t>(round_half_even(double(product)));
}

float ref_quantize_float(float z, double max_levels, std::uint32_t max_code) {
    const std::uint32_t d = ref_quantize_code(z, max_levels, max_code);
    return static_cast<float>(double(d) / max_levels);
}

// Synthetic SPIR-V fragment shader that either leaves FragDepth untouched or writes its own FragDepth
// (optionally with an extra early OpReturn branch, DepthGreater conservative mode, or EarlyFragmentTests).
std::vector<std::uint32_t> make_synthetic_fragment(bool writes_depth, bool early_tests, bool conservative_greater,
                                                   bool two_returns) {
    constexpr std::uint32_t op_branch_conditional = 250;
    constexpr std::uint32_t id_main = 1, id_void = 2, id_fn = 3, id_label0 = 4, id_f32 = 5, id_pf_out = 6,
                            id_frag_depth = 7, id_v4f = 8, id_pv4_in = 9, id_frag_coord = 10, id_bool = 11,
                            id_true = 12, id_label1 = 13, id_label2 = 14, id_c = 15, id_z = 16, bound = 17;
    std::vector<std::uint32_t> w = {0x07230203u, 0x00010000u, 0, bound, 0};
    auto put = [&](Inst i) { w.insert(w.end(), i.begin(), i.end()); };
    put(make(op_capability, {1}));
    put(make(op_memory_model, {0, 1}));
    Inst ep = make(op_entry_point, {model_fragment, id_main});
    append_string(ep, "main");
    if (writes_depth) {
        ep.push_back(id_frag_coord);
        ep.push_back(id_frag_depth);
    }
    fix_length(ep);
    put(ep);
    put(make(op_execution_mode, {id_main, mode_origin_upper_left}));
    if (early_tests) put(make(op_execution_mode, {id_main, mode_early_fragment_tests}));
    if (conservative_greater) put(make(op_execution_mode, {id_main, mode_depth_greater}));
    if (writes_depth) {
        put(make(op_decorate, {id_frag_coord, decoration_builtin, builtin_frag_coord}));
        put(make(op_decorate, {id_frag_depth, decoration_builtin, builtin_frag_depth}));
    }
    put(make(op_type_void, {id_void}));
    put(make(op_type_function, {id_fn, id_void}));
    if (writes_depth) {
        put(make(op_type_float, {id_f32, 32}));
        put(make(op_type_pointer, {id_pf_out, storage_output, id_f32}));
        put(make(op_type_vector, {id_v4f, id_f32, 4}));
        put(make(op_type_pointer, {id_pv4_in, storage_input, id_v4f}));
        if (two_returns) {
            put(make(op_type_bool, {id_bool}));
            put(make(41 /* OpConstantTrue */, {id_bool, id_true}));
        }
        put(make(op_variable, {id_pv4_in, id_frag_coord, storage_input}));
        put(make(op_variable, {id_pf_out, id_frag_depth, storage_output}));
    }
    put(make(op_function, {id_void, id_main, 0, id_fn}));
    put(make(op_label, {id_label0}));
    if (writes_depth) {
        put(make(op_load, {id_v4f, id_c, id_frag_coord}));
        put(make(op_composite_extract, {id_f32, id_z, id_c, 2}));
        put(make(op_store, {id_frag_depth, id_z}));
        if (two_returns) {
            put(make(op_branch_conditional, {id_true, id_label1, id_label2}));
            put(make(op_label, {id_label1}));
            put(make(op_return, {}));
            put(make(op_label, {id_label2}));
        }
    }
    put(make(op_return, {}));
    put(make(op_function_end, {}));
    return w;
}

// Structural SPIR-V checker + interpreter for depth24_patch_fragment output.
struct SpvProgram {
    std::uint32_t bound = 0;
    std::uint32_t glsl_id = 0;
    std::uint32_t frag_coord_var = 0;
    std::uint32_t frag_depth_var = 0;
    bool has_depth_replacing = false;
    bool has_conservative_depth = false;
    bool frag_depth_in_interface = false;
    bool frag_coord_in_interface = false;
    std::unordered_map<std::uint32_t, std::uint32_t> const_u32;
    std::unordered_map<std::uint32_t, bool> const_bool;
    std::vector<Inst> body;

    bool parse_and_validate(const std::vector<std::uint32_t>& words, std::string& err) {
        if (words.size() < 5 || words[0] != 0x07230203u) {
            err = "bad magic";
            return false;
        }
        bound = words[3];
        std::vector<Inst> insts;
        for (std::size_t w = 5; w < words.size();) {
            const std::uint32_t count = words[w] >> 16;
            if (count == 0 || w + count > words.size()) {
                err = "truncated instruction";
                return false;
            }
            insts.emplace_back(words.begin() + std::ptrdiff_t(w), words.begin() + std::ptrdiff_t(w + count));
            w += count;
        }

        // Section order check: Capability(0) -> ExtInstImport(1) -> MemoryModel(2) -> EntryPoint(3)
        // -> ExecutionMode(4) -> Decorations(5) -> Globals/Types/Constants/Variables(6) -> Functions(7).
        int section = 0;
        std::unordered_set<std::uint32_t> defined_globals;
        std::unordered_set<std::uint32_t> interface_ids;
        bool in_fn = false;
        std::size_t body_start = 0;

        auto check_id = [&](std::uint32_t id) { return id > 0 && id < bound; };

        for (std::size_t idx = 0; idx < insts.size(); ++idx) {
            const Inst& i = insts[idx];
            const std::uint32_t op = opcode(i);
            if (!in_fn) {
                int s = section;
                if (op == op_capability) s = 0;
                else if (op == op_ext_inst_import) s = 1;
                else if (op == op_memory_model) s = 2;
                else if (op == op_entry_point) s = 3;
                else if (op == op_execution_mode || op == op_execution_mode_id) s = 4;
                else if (op == op_decorate) s = 5;
                else if (is_global_decl(op)) s = 6;
                else if (op == op_function) s = 7;
                if (s < section) {
                    err = "out-of-order SPIR-V section";
                    return false;
                }
                section = s;
            }

            if (op == op_ext_inst_import) {
                if (i.size() < 3 || !check_id(i[1])) {
                    err = "bad ExtInstImport";
                    return false;
                }
                std::string name((i.size() - 2) * 4, '\0');
                std::memcpy(name.data(), &i[2], name.size());
                if (std::strcmp(name.c_str(), "GLSL.std.450") == 0) glsl_id = i[1];
                defined_globals.insert(i[1]);
            } else if (op == op_entry_point) {
                const std::size_t names = 3 + string_words(i, 3);
                for (std::size_t w = names; w < i.size(); ++w) interface_ids.insert(i[w]);
            } else if (op == op_execution_mode && i.size() >= 3) {
                if (i[2] == mode_depth_replacing) has_depth_replacing = true;
                if (i[2] == mode_depth_greater || i[2] == mode_depth_less || i[2] == mode_depth_unchanged)
                    has_conservative_depth = true;
            } else if (op == op_decorate && i.size() == 4 && i[2] == decoration_builtin) {
                if (i[3] == builtin_frag_coord) frag_coord_var = i[1];
                if (i[3] == builtin_frag_depth) frag_depth_var = i[1];
            } else if (!in_fn && is_global_decl(op)) {
                std::uint32_t res = 0;
                if (op == op_constant && i.size() == 4) {
                    res = i[2];
                    const_u32[res] = i[3];
                } else if (op == 41 /* OpConstantTrue */ && i.size() == 3) {
                    res = i[2];
                    const_bool[res] = true;
                } else if (op == op_variable && i.size() >= 4) {
                    res = i[2];
                } else if (op >= op_type_void && op <= 39 && i.size() >= 2) {
                    res = i[1];
                }
                if (res) {
                    if (!check_id(res) || !defined_globals.insert(res).second) {
                        err = "duplicate or out-of-bounds global id";
                        return false;
                    }
                }
            } else if (op == op_function) {
                in_fn = true;
                body_start = idx;
                body.push_back(i);
            } else if (in_fn) {
                body.push_back(i);
                if (op == op_function_end) in_fn = false;
            }
        }
        if (body_start == 0 || body.empty()) {
            err = "missing function body";
            return false;
        }
        frag_depth_in_interface = frag_depth_var && interface_ids.count(frag_depth_var) != 0;
        frag_coord_in_interface = frag_coord_var && interface_ids.count(frag_coord_var) != 0;

        // Verify every SSA operand in the function body is defined (either globally before OpFunction or earlier in
        // the body) and every result id is unique and < bound.
        std::unordered_set<std::uint32_t> defs = defined_globals;
        for (const Inst& i : body) {
            const std::uint32_t op = opcode(i);
            if (op == op_function || op == op_label || op == op_return || op == op_function_end) {
                if (op == op_label && i.size() == 2) defs.insert(i[1]);
                continue;
            }
            if (op == 250 /* OpBranchConditional */) {
                if (i.size() != 4 || !defs.count(i[1])) {
                    err = "undefined branch condition";
                    return false;
                }
                continue;
            }
            if (op == op_store) {
                if (i.size() != 3 || !defs.count(i[1]) || !defs.count(i[2])) {
                    err = "undefined operand in OpStore";
                    return false;
                }
                continue;
            }
            if (i.size() < 3 || !check_id(i[2])) {
                err = "bad result id in function body";
                return false;
            }
            if (!defs.count(i[1])) {
                err = "undefined result type id";
                return false;
            }
            const std::size_t first_arg = op == op_ext_inst ? 5 : 3;
            if (op == op_ext_inst && (i.size() < 5 || i[3] != glsl_id)) {
                err = "bad ExtInst import id";
                return false;
            }
            for (std::size_t a = first_arg; a < i.size(); ++a) {
                if (op == op_composite_extract && a > 3) continue;  // literal component index
                if (!defs.count(i[a])) {
                    err = "undefined operand id in function body (missing from prime_quantise?)";
                    return false;
                }
            }
            if (!defs.insert(i[2]).second) {
                err = "duplicate result id in function body";
                return false;
            }
        }
        return true;
    }

    // Executes the entry function for a given FragCoord.z and screen-space slopes (dz_dx, dz_dy).
    bool eval(float z_in, float dz_dx, float dz_dy, float& z_out, std::string& err) const {
        std::unordered_map<std::uint32_t, std::uint32_t> u32_val = const_u32;
        std::unordered_map<std::uint32_t, bool> bool_val = const_bool;
        std::unordered_map<std::uint32_t, std::array<float, 4>> vec4_val;
        float frag_depth = std::numeric_limits<float>::quiet_NaN();
        bool wrote_depth = false;

        std::unordered_map<std::uint32_t, std::size_t> labels;
        for (std::size_t k = 0; k < body.size(); ++k)
            if (opcode(body[k]) == op_label && body[k].size() == 2) labels[body[k][1]] = k;

        for (std::size_t pc = 0; pc < body.size(); ++pc) {
            const Inst& i = body[pc];
            const std::uint32_t op = opcode(i);
            switch (op) {
            case op_function:
            case op_label:
                break;
            case op_return:
                if (!wrote_depth) {
                    err = "shader returned without writing FragDepth";
                    return false;
                }
                z_out = frag_depth;
                return true;
            case op_function_end:
                err = "fell off end of function";
                return false;
            case 250: {  // OpBranchConditional
                const std::uint32_t target = bool_val.at(i[1]) ? i[2] : i[3];
                pc = labels.at(target);
                break;
            }
            case op_load:
                if (i[3] == frag_coord_var) {
                    vec4_val[i[2]] = {0.5f, 0.5f, z_in, 1.0f};
                } else if (i[3] == frag_depth_var) {
                    u32_val[i[2]] = f_bits(frag_depth);
                } else {
                    err = "unexpected OpLoad variable";
                    return false;
                }
                break;
            case op_store:
                if (i[1] != frag_depth_var) {
                    err = "unexpected OpStore variable";
                    return false;
                }
                frag_depth = u_float(u32_val.at(i[2]));
                wrote_depth = true;
                break;
            case op_composite_extract:
                u32_val[i[2]] = f_bits(vec4_val.at(i[3])[i[4]]);
                break;
            case op_bitcast:
                u32_val[i[2]] = u32_val.at(i[3]);
                break;
            case op_f_add:
                u32_val[i[2]] = f_bits(u_float(u32_val.at(i[3])) + u_float(u32_val.at(i[4])));
                break;
            case op_f_mul:
                u32_val[i[2]] = f_bits(u_float(u32_val.at(i[3])) * u_float(u32_val.at(i[4])));
                break;
            case op_dpdx:
                u32_val[i[2]] = f_bits(dz_dx);
                break;
            case op_dpdy:
                u32_val[i[2]] = f_bits(dz_dy);
                break;
            case op_convert_f_to_u: {
                const float v = u_float(u32_val.at(i[3]));
                if (!(v >= 0.0f) || v > 4294967295.0f) {
                    err = "OpConvertFToU received out-of-range or NaN float";
                    return false;
                }
                u32_val[i[2]] = static_cast<std::uint32_t>(v);
                break;
            }
            case op_convert_u_to_f:
                u32_val[i[2]] = f_bits(static_cast<float>(u32_val.at(i[3])));
                break;
            case op_i_add:
                u32_val[i[2]] = u32_val.at(i[3]) + u32_val.at(i[4]);
                break;
            case op_i_sub:
                u32_val[i[2]] = u32_val.at(i[3]) - u32_val.at(i[4]);
                break;
            case op_shift_left_logical: {
                const std::uint32_t sh = u32_val.at(i[4]);
                if (sh >= 32u) {
                    err = "OpShiftLeftLogical with shift >= 32 (UB in SPIR-V)";
                    return false;
                }
                u32_val[i[2]] = u32_val.at(i[3]) << sh;
                break;
            }
            case 194u: {  // OpShiftRightLogical
                const std::uint32_t sh = u32_val.at(i[4]);
                if (sh >= 32u) {
                    err = "OpShiftRightLogical with shift >= 32 (UB in SPIR-V)";
                    return false;
                }
                u32_val[i[2]] = u32_val.at(i[3]) >> sh;
                break;
            }
            case op_i_equal:
                bool_val[i[2]] = u32_val.at(i[3]) == u32_val.at(i[4]);
                break;
            case op_u_greater_than:
                bool_val[i[2]] = u32_val.at(i[3]) > u32_val.at(i[4]);
                break;
            case op_u_greater_than_equal:
                bool_val[i[2]] = u32_val.at(i[3]) >= u32_val.at(i[4]);
                break;
            case op_select:
                u32_val[i[2]] = bool_val.at(i[3]) ? u32_val.at(i[4]) : u32_val.at(i[5]);
                break;
            case op_ext_inst: {
                const std::uint32_t ext_op = i[4];
                if (ext_op == glsl_round_even) {
                    const float x = u_float(u32_val.at(i[5]));
                    const float r = std::isfinite(x) ? static_cast<float>(round_half_even(double(x))) : x;
                    u32_val[i[2]] = f_bits(r);
                } else if (ext_op == glsl_fabs) {
                    u32_val[i[2]] = f_bits(std::fabs(u_float(u32_val.at(i[5]))));
                } else if (ext_op == glsl_fmin) {
                    // GLSL.std.450 FMin: if one operand is NaN, the other is returned.
                    const float a = u_float(u32_val.at(i[5])), b = u_float(u32_val.at(i[6]));
                    u32_val[i[2]] = f_bits(std::isnan(a) ? b : std::isnan(b) ? a : (a < b ? a : b));
                } else if (ext_op == glsl_fmax) {
                    const float a = u_float(u32_val.at(i[5])), b = u_float(u32_val.at(i[6]));
                    u32_val[i[2]] = f_bits(std::isnan(a) ? b : std::isnan(b) ? a : (a > b ? a : b));
                } else if (ext_op == glsl_find_umsb) {
                    const std::uint32_t v = u32_val.at(i[5]);
                    u32_val[i[2]] = v == 0u ? 0xffffffffu : static_cast<std::uint32_t>(31 - std::countl_zero(v));
                } else {
                    err = "unexpected GLSL.std.450 instruction";
                    return false;
                }
                break;
            }
            default:
                err = "unexpected SPIR-V opcode";
                return false;
            }
        }
        err = "no return reached";
        return false;
    }
};

int test_d16_exhaustive() {
    const Depth24Bias d16_bias{0, 0.0f, 0.0f, 65535.0f};
    const std::vector<std::uint32_t> depth_only = depth24_depth_only_fragment(d16_bias);
    EXPECT(!depth_only.empty());

    std::string why, err;
    const std::vector<std::uint32_t> ps_plain =
        depth24_patch_fragment(make_synthetic_fragment(false, false, true, false), d16_bias, why);
    EXPECT(!ps_plain.empty());
    const std::vector<std::uint32_t> ps_write =
        depth24_patch_fragment(make_synthetic_fragment(true, false, true, true), d16_bias, why);
    EXPECT(!ps_write.empty());

    for (const auto& spv : {depth_only, ps_plain, ps_write}) {
        SpvProgram prog;
        if (!prog.parse_and_validate(spv, err)) {
            std::fprintf(stderr, "D16 validate failed: %s\n", err.c_str());
            return 1;
        }
        EXPECT(prog.has_depth_replacing);
        EXPECT(!prog.has_conservative_depth);
        EXPECT(prog.frag_depth_in_interface);
        EXPECT(prog.frag_coord_in_interface);

        // Exhaustive check of all 65,536 16-bit UNORM codes (round-trip bit-for-bit exact).
        for (std::uint32_t d = 0; d <= 65535u; ++d) {
            const float z = static_cast<float>(double(d) / 65535.0);
            float out = 0.0f;
            if (!prog.eval(z, 0.0f, 0.0f, out, err)) {
                std::fprintf(stderr, "D16 eval failed at d=%u: %s\n", d, err.c_str());
                return 1;
            }
            if (f_bits(out) != f_bits(z)) {
                std::fprintf(stderr, "D16 mismatch at d=%u: got %08x (%.9g), want %08x (%.9g)\n",
                             d, f_bits(out), out, f_bits(z), z);
                return 1;
            }
        }

        // Half-step ties and neighbors across the full 16-bit range.
        for (std::uint32_t d = 0; d < 65535u; d += 17u) {
            const float mid = static_cast<float>((double(d) + 0.5) / 65535.0);
            for (float z : {std::nextafter(mid, 0.0f), mid, std::nextafter(mid, 1.0f)}) {
                float out = 0.0f;
                EXPECT(prog.eval(z, 0.0f, 0.0f, out, err));
                const float want = ref_quantize_float(z, 65535.0, 0xffffu);
                EXPECT(f_bits(out) == f_bits(want));
            }
        }

        // IEEE-754 specials: zeros, subnormals, negatives, infinities, NaNs, >1.0.
        const std::uint32_t specials[] = {
            0x00000000u, 0x80000000u, 0x00000001u, 0x007fffffu, 0x00800000u, 0xbf800000u,
            0xff800000u, 0x7f800000u, 0x7fc00000u, 0x7fa00001u, 0xffc00000u, 0x3f7fffffu,
            0x3f800000u, 0x3f800001u, 0x40000000u, 0x7f7fffffu,
        };
        for (std::uint32_t bits : specials) {
            const float z = u_float(bits);
            float out = 0.0f;
            if (!prog.eval(z, 0.0f, 0.0f, out, err)) {
                std::fprintf(stderr, "D16 special %08x eval failed: %s\n", bits, err.c_str());
                return 1;
            }
            const float want = ref_quantize_float(z, 65535.0, 0xffffu);
            if (f_bits(out) != f_bits(want)) {
                std::fprintf(stderr, "D16 special %08x mismatch: got %08x want %08x\n",
                             bits, f_bits(out), f_bits(want));
                return 1;
            }
        }
    }
    return 0;
}

int test_d16_and_d24_bias_and_clamp() {
    struct Case {
        float levels;
        std::uint32_t max_code;
        int constant;
        float slope;
        float clamp;
        float z;
        float dx;
        float dy;
    };
    const Case cases[] = {
        {65535.0f, 0xffffu, 1, 0.0f, 0.0f, 0.25f, 0.0f, 0.0f},
        {65535.0f, 0xffffu, -3, 2.0f, 0.0f, 0.5f, -0.002f, 0.004f},
        {65535.0f, 0xffffu, 5, 4.0f, 0.01f, 0.5f, 0.01f, 0.0f},     // positive clamp caps bias
        {65535.0f, 0xffffu, -5, -4.0f, -0.01f, 0.5f, 0.01f, 0.0f},  // negative clamp floors bias
        {16777215.0f, 0xffffffu, 1, 0.0f, 0.0f, 0.15625f, 0.0f, 0.0f},
        {16777215.0f, 0xffffffu, 1000, 1.5f, 0.004f, 0.5f, 0.003f, -0.005f},
        {16777215.0f, 0xffffffu, -5, -2.0f, -0.01f, 0.7f, 0.008f, 0.002f},
    };
    for (const Case& c : cases) {
        const Depth24Bias bias{c.constant, c.slope, c.clamp, c.levels};
        const std::vector<std::uint32_t> spv = depth24_depth_only_fragment(bias);
        EXPECT(!spv.empty());
        SpvProgram prog;
        std::string err;
        EXPECT(prog.parse_and_validate(spv, err));
        float out = 0.0f;
        EXPECT(prog.eval(c.z, c.dx, c.dy, out, err));
        float amount = c.slope * std::fmax(std::fabs(c.dx), std::fabs(c.dy)) +
                       depth24_bias_constant(c.constant, c.levels);
        if (c.clamp > 0.0f) amount = std::fmin(amount, c.clamp);
        if (c.clamp < 0.0f) amount = std::fmax(amount, c.clamp);
        const float want = ref_quantize_float(c.z + amount, double(c.levels), c.max_code);
        EXPECT(f_bits(out) == f_bits(want));

        // Shader-written SV_Depth must be quantised WITHOUT rasterizer bias.
        std::string why;
        const std::vector<std::uint32_t> sw =
            depth24_patch_fragment(make_synthetic_fragment(true, false, false, false), bias, why);
        EXPECT(!sw.empty());
        SpvProgram sw_prog;
        EXPECT(sw_prog.parse_and_validate(sw, err));
        float sw_out = 0.0f;
        EXPECT(sw_prog.eval(c.z, c.dx, c.dy, sw_out, err));
        const float sw_want = ref_quantize_float(c.z, double(c.levels), c.max_code);
        EXPECT(f_bits(sw_out) == f_bits(sw_want));
    }
    return 0;
}

int test_d24_boundary_codes() {
    const Depth24Bias d24_bias{};
    const std::vector<std::uint32_t> spv = depth24_depth_only_fragment(d24_bias);
    EXPECT(!spv.empty());
    SpvProgram prog;
    std::string err;
    EXPECT(prog.parse_and_validate(spv, err));

    const std::uint32_t sample_codes[] = {
        0u, 1u, 2u, 3u, 127u, 128u, 255u, 256u, 65535u, 65536u,
        5033000u, 0x7fffffu, 0x800000u, 0xfffffeu, 0xffffffu,
    };
    for (std::uint32_t d : sample_codes) {
        const float z = static_cast<float>(double(d) / 16777215.0);
        float out = 0.0f;
        EXPECT(prog.eval(z, 0.0f, 0.0f, out, err));
        EXPECT(f_bits(out) == f_bits(z));
    }
    return 0;
}

int test_rejections() {
    std::string why;
    const auto base = make_synthetic_fragment(false, false, false, false);
    for (float bad : {0.0f, -1.0f, 255.0f, 65536.0f, std::numeric_limits<float>::quiet_NaN()}) {
        const Depth24Bias bias{0, 0.0f, 0.0f, bad};
        why.clear();
        EXPECT(depth24_patch_fragment(base, bias, why).empty() && !why.empty());
        EXPECT(depth24_depth_only_fragment(bias).empty());
    }
    why.clear();
    EXPECT(depth24_patch_fragment(make_synthetic_fragment(false, true, false, false), {}, why).empty() && !why.empty());
    return 0;
}

}  // namespace

int main() {
    if (test_d16_exhaustive() != 0) return 1;
    if (test_d16_and_d24_bias_and_clamp() != 0) return 1;
    if (test_d24_boundary_codes() != 0) return 1;
    if (test_rejections() != 0) return 1;
    std::printf("depth24_fragment_test: ok (all 65536 D16 codes, ties, specials, bias/clamp, D24 parity)\n");
    return 0;
}
