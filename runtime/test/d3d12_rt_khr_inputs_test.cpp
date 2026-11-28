// What GetRaytracingAccelerationStructurePrebuildInfo asks vkGetAccelerationStructureBuildSizesKHR for on the KHR path
// (runtime/shims/d3d12_rt_khr_inputs.h): the triangle counts, vertex formats and flags of the build the title then
// records. An indexed geometry with fewer than 3 indices has no triangles; the four 16-bit vertex formats D3D12
// allows are sized as themselves; ALLOW_COMPACTION is asked for. From kevincommits's finding in #1608
// (RT-KHR-BLAS-PRIMCOUNT-AND-PREBUILD). No device, GPU, or game data.
#include "../shims/d3d12_rt_khr_inputs.h"

#include <cstdio>

namespace {

int failures = 0;
void expect(bool ok, const char* what) {
    if (!ok) {
        std::fprintf(stderr, "FAIL: %s\n", what);
        ++failures;
    }
}

D3D12_RAYTRACING_GEOMETRY_DESC triangles(DXGI_FORMAT index_format, UINT index_count, UINT vertex_count,
                                         DXGI_FORMAT vertex_format = DXGI_FORMAT_R32G32B32_FLOAT) {
    D3D12_RAYTRACING_GEOMETRY_DESC d{};
    d.Type = D3D12_RAYTRACING_GEOMETRY_TYPE_TRIANGLES;
    d.Flags = D3D12_RAYTRACING_GEOMETRY_FLAG_OPAQUE;
    d.Triangles.IndexFormat = index_format;
    d.Triangles.IndexCount = index_count;
    d.Triangles.VertexFormat = vertex_format;
    d.Triangles.VertexCount = vertex_count;
    d.Triangles.VertexBuffer.StrideInBytes = 12;
    return d;  // IndexBuffer and VertexBuffer.StartAddress stay 0: a size query reads no memory
}

}  // namespace

int main() {
    using namespace rt::d3d12;

    // Triangle counts: the finding's inputs, and the ones that keep their counts.
    {
        const D3D12_RAYTRACING_GEOMETRY_DESC descs[] = {
            triangles(DXGI_FORMAT_R16_UINT, 0, 3000),
            triangles(DXGI_FORMAT_R32_UINT, 2, 300),
            triangles(DXGI_FORMAT_R32_UINT, 9, 3000),
            triangles(DXGI_FORMAT_UNKNOWN, 0, 9),
        };
        D3D12_BUILD_RAYTRACING_ACCELERATION_STRUCTURE_INPUTS in{};
        in.Type = D3D12_RAYTRACING_ACCELERATION_STRUCTURE_TYPE_BOTTOM_LEVEL;
        in.DescsLayout = D3D12_ELEMENTS_LAYOUT_ARRAY;
        in.NumDescs = 4;
        in.pGeometryDescs = descs;
        const KhrBuildSizeQuery q = khr_build_size_query(in);
        expect(q.type == VK_ACCELERATION_STRUCTURE_TYPE_BOTTOM_LEVEL_KHR, "a BLAS");
        expect(q.geometries.size() == 4 && q.max_primitives.size() == 4, "four geometries");
        if (q.max_primitives.size() == 4) {
            expect(q.max_primitives[0] == 0, "R16_UINT, IndexCount 0, VertexCount 3000: 0 triangles (not 1000)");
            expect(q.max_primitives[1] == 0, "R32_UINT, IndexCount 2, VertexCount 300: 0 triangles (not 100)");
            expect(q.max_primitives[2] == 3, "R32_UINT, IndexCount 9: 3 triangles");
            expect(q.max_primitives[3] == 3, "not indexed, VertexCount 9: 3 triangles");
            expect(q.geometries[0].geometry.triangles.indexType == VK_INDEX_TYPE_UINT16 &&
                       q.geometries[1].geometry.triangles.indexType == VK_INDEX_TYPE_UINT32 &&
                       q.geometries[3].geometry.triangles.indexType == VK_INDEX_TYPE_NONE_KHR,
                   "index types");
            expect(q.geometries[0].flags == VK_GEOMETRY_OPAQUE_BIT_KHR, "opaque geometry");
            expect(q.geometries[0].geometry.triangles.maxVertex == 2999, "maxVertex");
        }
    }

    // Vertex formats: each one D3D12 allows is sized as itself, as the build records it; another one as three floats.
    {
        struct Case {
            DXGI_FORMAT dxgi;
            VkFormat vk;
        };
        const Case cases[] = {
            {DXGI_FORMAT_R32G32B32_FLOAT, VK_FORMAT_R32G32B32_SFLOAT},
            {DXGI_FORMAT_R32G32_FLOAT, VK_FORMAT_R32G32_SFLOAT},
            {DXGI_FORMAT_R16G16B16A16_FLOAT, VK_FORMAT_R16G16B16A16_SFLOAT},
            {DXGI_FORMAT_R16G16_FLOAT, VK_FORMAT_R16G16_SFLOAT},
            {DXGI_FORMAT_R16G16B16A16_SNORM, VK_FORMAT_R16G16B16A16_SNORM},
            {DXGI_FORMAT_R16G16_SNORM, VK_FORMAT_R16G16_SNORM},
        };
        for (const Case& c : cases) {
            expect(khr_vertex_format(c.dxgi) == c.vk, "the build's vertex format");
            const D3D12_RAYTRACING_GEOMETRY_DESC d = triangles(DXGI_FORMAT_UNKNOWN, 0, 3, c.dxgi);
            D3D12_BUILD_RAYTRACING_ACCELERATION_STRUCTURE_INPUTS in{};
            in.Type = D3D12_RAYTRACING_ACCELERATION_STRUCTURE_TYPE_BOTTOM_LEVEL;
            in.DescsLayout = D3D12_ELEMENTS_LAYOUT_ARRAY;
            in.NumDescs = 1;
            in.pGeometryDescs = &d;
            const KhrBuildSizeQuery q = khr_build_size_query(in);
            expect(q.geometries.size() == 1 && q.geometries[0].geometry.triangles.vertexFormat == c.vk,
                   "the size query's vertex format is the build's");
        }
        expect(khr_vertex_format(DXGI_FORMAT_R8G8B8A8_UNORM) == VK_FORMAT_UNDEFINED, "another format: none");
        const D3D12_RAYTRACING_GEOMETRY_DESC d = triangles(DXGI_FORMAT_UNKNOWN, 0, 3, DXGI_FORMAT_R8G8B8A8_UNORM);
        D3D12_BUILD_RAYTRACING_ACCELERATION_STRUCTURE_INPUTS in{};
        in.Type = D3D12_RAYTRACING_ACCELERATION_STRUCTURE_TYPE_BOTTOM_LEVEL;
        in.DescsLayout = D3D12_ELEMENTS_LAYOUT_ARRAY_OF_POINTERS;
        const D3D12_RAYTRACING_GEOMETRY_DESC* pointers[] = {&d};
        in.NumDescs = 1;
        in.ppGeometryDescs = pointers;
        const KhrBuildSizeQuery q = khr_build_size_query(in);
        expect(q.geometries.size() == 1 &&
                   q.geometries[0].geometry.triangles.vertexFormat == VK_FORMAT_R32G32B32_SFLOAT,
               "another format is sized as three floats (array of pointers)");
    }

    // Flags: every one the build forwards, ALLOW_COMPACTION included; PERFORM_UPDATE is a mode.
    {
        const struct {
            D3D12_RAYTRACING_ACCELERATION_STRUCTURE_BUILD_FLAGS d3d12;
            VkBuildAccelerationStructureFlagsKHR vk;
        } cases[] = {
            {D3D12_RAYTRACING_ACCELERATION_STRUCTURE_BUILD_FLAG_ALLOW_UPDATE,
             VK_BUILD_ACCELERATION_STRUCTURE_ALLOW_UPDATE_BIT_KHR},
            {D3D12_RAYTRACING_ACCELERATION_STRUCTURE_BUILD_FLAG_ALLOW_COMPACTION,
             VK_BUILD_ACCELERATION_STRUCTURE_ALLOW_COMPACTION_BIT_KHR},
            {D3D12_RAYTRACING_ACCELERATION_STRUCTURE_BUILD_FLAG_PREFER_FAST_TRACE,
             VK_BUILD_ACCELERATION_STRUCTURE_PREFER_FAST_TRACE_BIT_KHR},
            {D3D12_RAYTRACING_ACCELERATION_STRUCTURE_BUILD_FLAG_PREFER_FAST_BUILD,
             VK_BUILD_ACCELERATION_STRUCTURE_PREFER_FAST_BUILD_BIT_KHR},
            {D3D12_RAYTRACING_ACCELERATION_STRUCTURE_BUILD_FLAG_MINIMIZE_MEMORY,
             VK_BUILD_ACCELERATION_STRUCTURE_LOW_MEMORY_BIT_KHR},
            {D3D12_RAYTRACING_ACCELERATION_STRUCTURE_BUILD_FLAG_PERFORM_UPDATE, 0},
        };
        for (const auto& c : cases) expect(khr_build_flags(c.d3d12) == c.vk, "one build flag");
        const D3D12_RAYTRACING_GEOMETRY_DESC d = triangles(DXGI_FORMAT_UNKNOWN, 0, 3);
        D3D12_BUILD_RAYTRACING_ACCELERATION_STRUCTURE_INPUTS in{};
        in.Type = D3D12_RAYTRACING_ACCELERATION_STRUCTURE_TYPE_BOTTOM_LEVEL;
        in.DescsLayout = D3D12_ELEMENTS_LAYOUT_ARRAY;
        in.Flags = D3D12_RAYTRACING_ACCELERATION_STRUCTURE_BUILD_FLAG_PREFER_FAST_TRACE |
                   D3D12_RAYTRACING_ACCELERATION_STRUCTURE_BUILD_FLAG_ALLOW_COMPACTION;
        in.NumDescs = 1;
        in.pGeometryDescs = &d;
        expect(khr_build_size_query(in).flags == (VK_BUILD_ACCELERATION_STRUCTURE_PREFER_FAST_TRACE_BIT_KHR |
                                                  VK_BUILD_ACCELERATION_STRUCTURE_ALLOW_COMPACTION_BIT_KHR),
               "the size query asks for ALLOW_COMPACTION");
    }

    // A TLAS: its instances.
    {
        D3D12_BUILD_RAYTRACING_ACCELERATION_STRUCTURE_INPUTS in{};
        in.Type = D3D12_RAYTRACING_ACCELERATION_STRUCTURE_TYPE_TOP_LEVEL;
        in.DescsLayout = D3D12_ELEMENTS_LAYOUT_ARRAY_OF_POINTERS;
        in.NumDescs = 7;
        const KhrBuildSizeQuery q = khr_build_size_query(in);
        expect(q.type == VK_ACCELERATION_STRUCTURE_TYPE_TOP_LEVEL_KHR && q.geometries.size() == 1 &&
                   q.geometries[0].geometryType == VK_GEOMETRY_TYPE_INSTANCES_KHR &&
                   q.geometries[0].geometry.instances.arrayOfPointers == VK_TRUE &&
                   q.max_primitives == std::vector<std::uint32_t>{7},
               "a TLAS of 7 instances, as pointers");
    }

    if (failures) {
        std::fprintf(stderr, "%d failure(s)\n", failures);
        return 1;
    }
    std::printf("KHR prebuild: triangle counts, vertex formats and flags as the build records them\n");
    return 0;
}
