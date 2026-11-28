#version 460
// A vertex shader written to the translator's runtime contract (docs/design.md, "Vertex fetch" and "User data"),
// standing in for a translated one in d3d12_draw_test. Inputs sit at location = D3D input register: COLOR is
// register 0 and POSITION register 1 here, the reverse of the input layout's order, so the test shows that the
// layout is matched by semantic. The first root parameter, two root constants, moves the vertex in clip space: s8, s9,
// since graphics root arguments start at s8 in the console layout (docs/d3d12_vulkan.md, D3D-448).

layout(location = 0) in vec4 color;     // COLOR0, input register 0
layout(location = 1) in vec4 position;  // POSITION0, input register 1

layout(location = 0) out vec4 out_color;

layout(push_constant) uniform UserData {
    uint s[32];
    uvec2 heap_base;
    uvec2 code_base;
} pc;

void main() {
    const vec2 offset = vec2(uintBitsToFloat(pc.s[8]), uintBitsToFloat(pc.s[9]));
    gl_Position = vec4(position.xy + offset, 0.0, 1.0);
    out_color = color;
}
