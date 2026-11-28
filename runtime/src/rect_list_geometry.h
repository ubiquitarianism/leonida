#pragma once

// Console rectangle lists on Vulkan, which has no such primitive.
//
// A rectangle is three corners (v1, v2, v3) of a screen-aligned rectangle and the hardware makes the fourth, for the
// position and for every interpolant, after the vertex stage. A list topology of rectangles (D3D_PRIMITIVE_TOPOLOGY
// 17) and a pipeline whose topology type is 19 are console-only. The fact comes from xvalkyrie (GPL-3.0, commit
// 53af7df77c10, notes in its backlog); no code is taken from it. The corner between the other two (the right angle)
// is not always v1: the title's own full-screen blit (VS_Blit_6e5520d1) gives (-1, 1) (-1, -1) (1, -1), the corner
// at v2, and xcl's notes say the same of a retail title's full-screen pass and list v1, v2 and v3 as possible
// (learned from its docs and tests, GPL-2.0-only, no code taken). Taking v1 there made the parallelogram
// v2 + v3 - v1 = (1, -3), of which only the lower-left half of the screen is on screen.
//
// The fourth vertex depends on what the vertex stage wrote, not on what it read, so the expansion has to sit after
// the vertex stage: a pre-pass over the vertex buffers is wrong for any shader that is not linear in its inputs (and
// the translated vertex shaders fetch by V# themselves, so there is no vertex buffer to expand). The pipeline gets a
// small geometry stage instead, built here from the vertex shader's own SPIR-V: it takes the triangle (v1, v2, v3)
// the list draws, finds the corner between the other two as the one opposite the longest side (from the positions'
// x and y; a degenerate triangle takes v1), makes the fourth corner v4 = the other two's sum minus that corner, and
// emits the rectangle as two triangles with the winding of (v1, v2, v3) that both start at v1, so a flat input
// (taken from a triangle's first vertex) is v1's over the whole rectangle:
//   corner v1: (v1, v2, v4) (v1, v4, v3);  corner v2: (v1, v2, v3) (v1, v3, v4);  corner v3: (v1, v2, v3) (v1, v4, v2). Every Location output of the vertex stage is a vec4 float in the translator's output (rdna2spirv.py),
// and so is the stage's Position; the generated stage reads and writes exactly those.
//
// Not covered, and refused with a reason: a vertex stage that writes ClipDistance, CullDistance, Layer or
// ViewportIndex, or a Location output that is not a vec4 float.
//
// Plain word editing, standard library only.

#include <algorithm>
#include <array>
#include <cstdint>
#include <initializer_list>
#include <map>
#include <set>
#include <string>
#include <vector>

namespace leonida {

// D3D_PRIMITIVE_TOPOLOGY values and the PSO topology type of the console's rectangle and quad lists.
constexpr std::uint32_t topology_rect_list = 17;
constexpr std::uint32_t topology_quad_list = 19;
constexpr std::uint32_t topology_type_rect_list = 19;
constexpr std::array<std::uint32_t, 6> quad_triangle_indices{0, 1, 2, 0, 2, 3};

// The geometry stage for `vertex_words`' outputs; empty, with `why` set, when the vertex stage cannot be expanded.
inline std::vector<std::uint32_t> rect_list_geometry(const std::vector<std::uint32_t>& vertex_words, std::string& why) {
    enum : std::uint32_t {
        op_entry_point = 15, op_type_void = 19, op_type_int = 21, op_type_float = 22,
        op_type_vector = 23, op_type_array = 28, op_type_pointer = 32, op_type_function = 33, op_constant = 43,
        op_type_bool = 20, op_composite_construct = 80, op_composite_extract = 81, op_fmul = 133,
        op_logical_and = 167, op_logical_not = 168, op_select = 169, op_ford_greater_than_equal = 190,
        op_function = 54, op_function_end = 56, op_variable = 59, op_load = 61, op_store = 62, op_access_chain = 65,
        op_decorate = 71, op_fadd = 129, op_fsub = 131, op_emit_vertex = 218, op_end_primitive = 219, op_label = 248, op_return = 253,
        op_capability = 17, op_memory_model = 14, op_execution_mode = 16,
        deco_builtin = 11, deco_location = 30, builtin_position = 0, builtin_point_size = 1,
        storage_input = 1, storage_output = 3
    };
    why.clear();
    if (vertex_words.size() < 6 || vertex_words[0] != 0x07230203u) {
        why = "the vertex shader is not a SPIR-V module";
        return {};
    }
    // What the vertex stage declares: types, Output variables, and the decorations that name them.
    struct Type { std::uint32_t op = 0, component = 0, count = 0, width = 0; };
    std::map<std::uint32_t, Type> types;
    std::map<std::uint32_t, std::uint32_t> variable_type;  // Output variable -> its pointer type
    std::map<std::uint32_t, std::uint32_t> location, builtin;       // variable id -> value
    std::set<std::uint32_t> outputs;
    for (std::size_t i = 5; i < vertex_words.size();) {
        const std::uint32_t op = vertex_words[i] & 0xffffu, n = vertex_words[i] >> 16;
        if (!n || i + n > vertex_words.size()) {
            why = "the vertex shader's SPIR-V is cut short";
            return {};
        }
        const std::uint32_t* w = &vertex_words[i];
        if (op == op_type_float && n >= 3) types[w[1]] = {op, 0, 0, w[2]};
        else if (op == op_type_vector && n >= 4) types[w[1]] = {op, w[2], w[3], 0};
        else if (op == op_type_pointer && n >= 4) types[w[1]] = {op, w[3], w[2], 0};
        else if (op == op_variable && n >= 4 && w[3] == storage_output) outputs.insert(w[2]), variable_type[w[2]] = w[1];
        else if (op == op_decorate && n >= 4 && w[2] == deco_location) location[w[1]] = w[3];
        else if (op == op_decorate && n >= 4 && w[2] == deco_builtin) builtin[w[1]] = w[3];
        i += n;
    }
    std::uint32_t position = 0;
    std::vector<std::uint32_t> params;  // Location values, ascending
    for (std::uint32_t id : outputs) {
        const auto b = builtin.find(id);
        if (b != builtin.end()) {
            if (b->second == builtin_position) position = id;
            else if (b->second != builtin_point_size) {
                why = "the vertex shader writes ClipDistance, CullDistance, Layer or ViewportIndex";
                return {};
            }
            continue;
        }
        const auto l = location.find(id);
        if (l == location.end()) continue;
        const Type& ptr = types[variable_type[id]];
        const Type& vec = types[ptr.component];
        const Type& scalar = types[vec.component];
        // Known limit: every vec4 float output gets the plain a + b - p corner. The translator can export an attribute
        // as the register's raw bits with two packed f16 halves (only the pixel shader knows, through per_vertex in
        // tools/rdna2spirv.py), and nothing on the vertex side tells those apart, so such an output would get a wrong
        // fourth corner. Fixing it needs the pixel shader's words here and a per-channel marker from the translator.
        if (vec.op != op_type_vector || vec.count != 4 || scalar.op != op_type_float || scalar.width != 32) {
            why = "a vertex output is not a vec4 float";
            return {};
        }
        params.push_back(l->second);
    }
    if (!position) {
        why = "the vertex shader writes no Position";
        return {};
    }
    std::sort(params.begin(), params.end());
    params.erase(std::unique(params.begin(), params.end()), params.end());

    // The module: a fixed prelude, then per variable ids, then the body. Ids count up from 1.
    std::uint32_t next = 1;
    const auto id = [&] { return next++; };
    std::vector<std::uint32_t> head, annotations, globals, body;
    const auto emit = [](std::vector<std::uint32_t>& to, std::uint32_t op, std::initializer_list<std::uint32_t> args) {
        to.push_back((std::uint32_t(args.size()) + 1) << 16 | op);
        to.insert(to.end(), args.begin(), args.end());
    };
    const std::uint32_t t_void = id(), t_fn = id(), t_uint = id(), t_float = id(), t_v4 = id(),
                        c3 = id(), t_arr = id(), t_pin_arr = id(), t_pin_v4 = id(),
                        t_pout_v4 = id(), c0 = id(), c1 = id(), c2 = id();
    emit(globals, op_type_void, {t_void});
    emit(globals, op_type_function, {t_fn, t_void});
    emit(globals, op_type_int, {t_uint, 32, 0});
    emit(globals, op_type_float, {t_float, 32});
    emit(globals, op_type_vector, {t_v4, t_float, 4});
    emit(globals, op_constant, {t_uint, c3, 3});
    emit(globals, op_constant, {t_uint, c0, 0});
    emit(globals, op_constant, {t_uint, c1, 1});
    emit(globals, op_constant, {t_uint, c2, 2});
    emit(globals, op_type_array, {t_arr, t_v4, c3});
    emit(globals, op_type_pointer, {t_pin_arr, storage_input, t_arr});
    emit(globals, op_type_pointer, {t_pin_v4, storage_input, t_v4});
    emit(globals, op_type_pointer, {t_pout_v4, storage_output, t_v4});

    struct Var { std::uint32_t in, out; bool is_position; };
    std::vector<Var> vars;
    vars.push_back({id(), id(), true});
    for (std::size_t i = 0; i < params.size(); ++i) vars.push_back({id(), id(), false});
    std::vector<std::uint32_t> io_vars;
    emit(annotations, op_decorate, {vars[0].in, deco_builtin, builtin_position});
    emit(annotations, op_decorate, {vars[0].out, deco_builtin, builtin_position});
    for (std::size_t i = 0; i < params.size(); ++i) {
        emit(annotations, op_decorate, {vars[i + 1].in, deco_location, params[i]});
        emit(annotations, op_decorate, {vars[i + 1].out, deco_location, params[i]});
    }
    for (const Var& v : vars) {
        emit(globals, op_variable, {t_pin_arr, v.in, storage_input});
        emit(globals, op_variable, {t_pout_v4, v.out, storage_output});
        io_vars.push_back(v.in);
        io_vars.push_back(v.out);
    }

    const std::uint32_t main_fn = id(), entry_label = id();
    emit(body, op_function, {t_void, main_fn, 0, t_fn});
    emit(body, op_label, {entry_label});
    // Per variable: the three corners.
    struct Corners { std::uint32_t v[3]; };
    std::vector<Corners> corners;
    const std::uint32_t cs[3] = {c0, c1, c2};
    for (const Var& v : vars) {
        Corners c{};
        for (int k = 0; k < 3; ++k) {
            const std::uint32_t ptr = id();
            c.v[k] = id();
            emit(body, op_access_chain, {t_pin_v4, ptr, v.in, cs[k]});
            emit(body, op_load, {t_v4, c.v[k], ptr});
        }
        corners.push_back(c);
    }
    // The corner between the other two, opposite the longest side: squared lengths from the positions' x and y.
    const std::uint32_t t_bool = id(), t_bv4 = id();
    emit(globals, op_type_bool, {t_bool});
    emit(globals, op_type_vector, {t_bv4, t_bool, 4});
    const auto extract = [&](std::uint32_t vector, std::uint32_t component) {
        const std::uint32_t r = id();
        emit(body, op_composite_extract, {t_float, r, vector, component});
        return r;
    };
    const auto length2 = [&](std::uint32_t a, std::uint32_t b) {
        const std::uint32_t d = id(), xx = id(), yy = id(), sum = id();
        emit(body, op_fsub, {t_v4, d, a, b});
        const std::uint32_t x = extract(d, 0), y = extract(d, 1);
        emit(body, op_fmul, {t_float, xx, x, x});
        emit(body, op_fmul, {t_float, yy, y, y});
        emit(body, op_fadd, {t_float, sum, xx, yy});
        return sum;
    };
    const auto at_least = [&](std::uint32_t a, std::uint32_t b) {
        const std::uint32_t r = id();
        emit(body, op_ford_greater_than_equal, {t_bool, r, a, b});
        return r;
    };
    const auto both = [&](std::uint32_t a, std::uint32_t b) {
        const std::uint32_t r = id();
        emit(body, op_logical_and, {t_bool, r, a, b});
        return r;
    };
    const auto splat = [&](std::uint32_t b) {  // OpSelect on vectors takes a vector condition before SPIR-V 1.4
        const std::uint32_t r = id();
        emit(body, op_composite_construct, {t_bv4, r, b, b, b, b});
        return r;
    };
    const std::uint32_t* p = corners[0].v;  // vars[0] is the Position
    const std::uint32_t l12 = length2(p[0], p[1]), l23 = length2(p[1], p[2]), l31 = length2(p[2], p[0]);
    const std::uint32_t at_v1 = both(at_least(l23, l12), at_least(l23, l31)), not_v1 = id();
    emit(body, op_logical_not, {t_bool, not_v1, at_v1});
    const std::uint32_t at_v2 = both(not_v1, at_least(l31, l12));
    const std::uint32_t c_v1 = splat(at_v1), c_v2 = splat(at_v2);
    const auto pick = [&](std::uint32_t condition, std::uint32_t yes, std::uint32_t no) {
        const std::uint32_t r = id();
        emit(body, op_select, {t_v4, r, condition, yes, no});
        return r;
    };
    // Per variable: the fourth corner, v1 + v2 + v3 - 2 * corner, and the six vertices of the two triangles.
    std::vector<std::array<std::uint32_t, 6>> emitted;
    for (const Corners& c : corners) {
        const std::uint32_t corner = pick(c_v1, c.v[0], pick(c_v2, c.v[1], c.v[2]));
        const std::uint32_t s12 = id(), s123 = id(), less = id(), fourth = id();
        emit(body, op_fadd, {t_v4, s12, c.v[0], c.v[1]});
        emit(body, op_fadd, {t_v4, s123, s12, c.v[2]});
        emit(body, op_fsub, {t_v4, less, s123, corner});
        emit(body, op_fsub, {t_v4, fourth, less, corner});
        emitted.push_back({c.v[0], c.v[1], pick(c_v1, fourth, c.v[2]),
                           c.v[0], pick(c_v2, c.v[2], fourth), pick(c_v1, c.v[2], pick(c_v2, fourth, c.v[1]))});
    }
    for (int triangle = 0; triangle < 2; ++triangle) {
        for (int k = 0; k < 3; ++k) {
            for (std::size_t i = 0; i < vars.size(); ++i) emit(body, op_store, {vars[i].out, emitted[i][triangle * 3 + k]});
            emit(body, op_emit_vertex, {});
        }
        emit(body, op_end_primitive, {});
    }
    emit(body, op_return, {});
    emit(body, op_function_end, {});

    std::vector<std::uint32_t> out{0x07230203u, 0x00010000u, 0, next, 0};
    emit(out, op_capability, {2});  // Geometry
    emit(out, op_memory_model, {0, 1});  // Logical, GLSL450
    std::vector<std::uint32_t> entry{3, main_fn, 0x6e69616du, 0};  // Geometry, %main, "main"
    entry.insert(entry.end(), io_vars.begin(), io_vars.end());
    out.push_back((std::uint32_t(entry.size()) + 1) << 16 | op_entry_point);
    out.insert(out.end(), entry.begin(), entry.end());
    emit(out, op_execution_mode, {main_fn, 22});         // Triangles
    emit(out, op_execution_mode, {main_fn, 0, 1});       // Invocations 1
    emit(out, op_execution_mode, {main_fn, 29});         // OutputTriangleStrip
    emit(out, op_execution_mode, {main_fn, 26, 6});      // OutputVertices 6
    out.insert(out.end(), annotations.begin(), annotations.end());
    out.insert(out.end(), globals.begin(), globals.end());
    out.insert(out.end(), body.begin(), body.end());
    return out;
}

}  // namespace leonida
