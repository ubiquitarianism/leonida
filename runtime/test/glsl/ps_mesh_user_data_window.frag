#version 460
// The pixel shader of the mesh draw in d3d12_graphics_direct_heaps_test: render target 0 gets the heap dwords the
// mesh stage read (its own window at s19/s20, 8 dwords above the vertex and pixel base), render target 1 the pixel
// stage's (s11/s12), so one draw shows both stages of the same mesh pipeline received their heaps.

layout(location = 0) flat in uvec4 mesh_words;
layout(location = 0) out uvec4 mesh_target;
layout(location = 1) out uvec4 ps_target;

layout(push_constant) uniform UserData {
    uint s[32];
    uvec2 heap_base;
    uvec2 code_base;
} pc;

void main() {
    mesh_target = mesh_words;
    ps_target = uvec4(pc.s[11], pc.s[12], 0u, 0u);
}
