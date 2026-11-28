#version 460
#extension GL_EXT_buffer_reference : require
#extension GL_EXT_buffer_reference_uvec2 : require
// Positive authored fixture. The heap word comes from the directly-indexed signature, not a root table.
// Require the fixture's expected heap and resource addresses before following either reference.
layout(location=0) flat in uvec4 vs_words;
layout(location=0) out uvec4 vs_target;
layout(location=1) out uvec4 ps_target;
layout(push_constant) uniform UserData { uint s[32]; uvec2 heap_base; uvec2 code_base; } pc;
layout(buffer_reference, std430, buffer_reference_align=4) readonly buffer Words { uint w[]; };
uvec2 add64(uvec2 a, uvec2 b) {
    uint carry;
    uint lo=uaddCarry(a.x,b.x,carry);
    return uvec2(lo,a.y+b.y+carry);
}
void main() {
    vs_target=vs_words;
    ps_target=uvec4(0xddddddddu);
    uint heap=pc.s[11];
    if (heap==0u || heap!=pc.s[8]) return;
    Words srv=Words(add64(pc.heap_base,uvec2(heap+32u,0u))); // descriptor slot 1, not slot 0
    uvec2 va=uvec2(srv.w[0],uint(int(srv.w[1]<<16u)>>16));
    if (srv.w[2]<16u || va!=uvec2(pc.s[9],pc.s[10])) return;
    Words input_words=Words(add64(pc.heap_base,va));
    ps_target=uvec4(input_words.w[0],input_words.w[1],input_words.w[2],input_words.w[3]);
}
