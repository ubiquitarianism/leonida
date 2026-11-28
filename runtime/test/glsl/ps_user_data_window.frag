#version 460
// The pixel shader of d3d12_graphics_direct_heaps_test: render target 0 gets the user-data dwords s10..s13 the
// vertex shader read, render target 1 the same dwords as this stage reads them.

layout(location = 0) flat in uvec4 vs_words;
layout(location = 0) out uvec4 vs_target;
layout(location = 1) out uvec4 ps_target;

layout(push_constant) uniform UserData {
    uint s[32];
    uvec2 heap_base;
    uvec2 code_base;
} pc;

void main() {
    vs_target = vs_words;
    ps_target = uvec4(pc.s[10], pc.s[11], pc.s[12], pc.s[13]);
}
