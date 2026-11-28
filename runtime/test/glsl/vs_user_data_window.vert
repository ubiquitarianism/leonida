#version 460
// A vertex shader with no vertex inputs, written to the translator's runtime contract (docs/design.md, "User data"),
// for d3d12_graphics_direct_heaps_test: one triangle covering the viewport that hands the user-data dwords s10..s13
// it reads to the pixel shader. Those are where a directly indexed signature's heaps land after a few dwords of
// explicit root arguments from s8.

layout(location = 0) flat out uvec4 vs_words;

layout(push_constant) uniform UserData {
    uint s[32];
    uvec2 heap_base;
    uvec2 code_base;
} pc;

void main() {
    const vec2 corner = vec2(float(gl_VertexIndex & 2), float((gl_VertexIndex << 1) & 2));
    gl_Position = vec4(corner * 2.0 - 1.0, 0.0, 1.0);
    vs_words = uvec4(pc.s[10], pc.s[11], pc.s[12], pc.s[13]);
}
