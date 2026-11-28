// KHR BuildRaytracingAccelerationStructure emit (runtime/src/vulkan_replay.h): the triangle count of each BLAS
// geometry. An indexed geometry has index_count / 3 triangles, none with fewer than 3 indices, as the XDXR builds
// count them; counting its vertices instead would have the driver read indices from an index buffer the geometry may
// not have (a null address). From kevincommits's finding in #1608 (RT-KHR-BLAS-PRIMCOUNT-AND-PREBUILD).
// No Vulkan loader, GPU, or game data: a fake vkCmdBuildAccelerationStructuresKHR captures the build.
#include "../src/vulkan_replay.h"

#include <cstdint>
#include <cstdio>
#include <vector>

namespace {

struct Captured {
    VkIndexType index_type = VK_INDEX_TYPE_MAX_ENUM;
    VkDeviceAddress index_address = ~VkDeviceAddress(0);
    VkFormat vertex_format = VK_FORMAT_MAX_ENUM;
    std::uint32_t primitives = ~0u;
};

std::vector<Captured> captured;
VkBuildAccelerationStructureFlagsKHR captured_flags = 0;
int builds = 0;

VKAPI_ATTR void VKAPI_CALL rec_build(VkCommandBuffer, std::uint32_t count,
                                     const VkAccelerationStructureBuildGeometryInfoKHR* infos,
                                     const VkAccelerationStructureBuildRangeInfoKHR* const* ranges) {
    ++builds;
    for (std::uint32_t i = 0; i < count; ++i) {
        captured_flags = infos[i].flags;
        for (std::uint32_t g = 0; g < infos[i].geometryCount; ++g) {
            const auto& tr = infos[i].pGeometries[g].geometry.triangles;
            captured.push_back({tr.indexType, tr.indexData.deviceAddress, tr.vertexFormat, ranges[i][g].primitiveCount});
        }
    }
}

VKAPI_ATTR void VKAPI_CALL rec_barrier(VkCommandBuffer, VkPipelineStageFlags, VkPipelineStageFlags, VkDependencyFlags,
                                       std::uint32_t, const VkMemoryBarrier*, std::uint32_t,
                                       const VkBufferMemoryBarrier*, std::uint32_t, const VkImageMemoryBarrier*) {}

int failures = 0;
void expect(bool ok, const char* what) {
    if (!ok) {
        std::fprintf(stderr, "FAIL: %s\n", what);
        ++failures;
    }
}

}  // namespace

int main() {
    using namespace leonida;

    // The shared count itself.
    static_assert(rt_triangle_count(true, 0, 3000) == 0);
    static_assert(rt_triangle_count(true, 2, 300) == 0);
    static_assert(rt_triangle_count(true, 9, 3000) == 3);
    static_assert(rt_triangle_count(false, 0, 9) == 3);
    static_assert(rt_triangle_count(false, 9, 3) == 1);  // an index count without an index type is not read

    ReplayFunctions vk{};
    vk.pipeline_barrier = rec_barrier;
    vk.build_acceleration_structures = rec_build;
    vk.resolve_acceleration_structure = [](std::uint64_t va) -> VkAccelerationStructureKHR {
        return va == 0x10000 ? reinterpret_cast<VkAccelerationStructureKHR>(std::uintptr_t(0xA100)) : VK_NULL_HANDLE;
    };
    vk.heap_base = 0x7000000000;
    const auto resolve_res = [](ResourceId) -> const ResourceBinding* { return nullptr; };

    const auto geometry = [](std::uint32_t index_type, std::uint32_t index_count, std::uint32_t vertex_count,
                             std::uint64_t index_va, std::uint32_t format = VK_FORMAT_R32G32B32_SFLOAT) {
        RtTriangleGeometry t{};
        t.vertex_va = 0x40000;
        t.vertex_count = vertex_count;
        t.vertex_stride = 12;
        t.vertex_format = format;
        t.index_type = index_type;
        t.index_count = index_count;
        t.index_va = index_va;
        return t;
    };

    BuildAccelerationStructure build{};
    build.dst_va = 0x10000;
    build.scratch_va = 0x20000;
    build.build_flags = VK_BUILD_ACCELERATION_STRUCTURE_ALLOW_COMPACTION_BIT_KHR;
    // The finding's two inputs, then an indexed and a non-indexed geometry that keep their counts, then a 16-bit
    // vertex format.
    build.triangles.push_back(geometry(1, 0, 3000, 0));                // UINT16, no indices, no index buffer
    build.triangles.push_back(geometry(2, 2, 300, 0));                 // UINT32, 2 indices, no index buffer
    build.triangles.push_back(geometry(2, 9, 3000, 0x30000));          // UINT32, 3 triangles
    build.triangles.push_back(geometry(0, 0, 9, 0));                   // not indexed, 3 triangles
    build.triangles.push_back(geometry(1, 6, 4, 0x30100, VK_FORMAT_R16G16B16A16_SFLOAT));

    CommandList list;
    expect(list.build_acceleration_structure(build) && list.close(), "record the BLAS build");
    PreparedList prepared;
    expect(prepare(list, resolve_res, vk, prepared) == ReplayResult::success, "the BLAS build prepares");
    emit(prepared, VK_NULL_HANDLE, vk);

    expect(builds == 1 && captured.size() == 5, "one build of five geometries");
    if (captured.size() == 5) {
        expect(captured[0].index_type == VK_INDEX_TYPE_UINT16 && captured[0].index_address == 0,
               "UINT16 with no index buffer keeps its index type and a null index address");
        expect(captured[0].primitives == 0, "UINT16, index_count 0, vertex_count 3000: 0 triangles (not 1000)");
        expect(captured[1].index_type == VK_INDEX_TYPE_UINT32 && captured[1].primitives == 0,
               "UINT32, index_count 2, vertex_count 300: 0 triangles (not 100)");
        expect(captured[2].primitives == 3 && captured[2].index_address == vk.heap_base + 0x30000,
               "UINT32, index_count 9: 3 triangles from its index buffer");
        expect(captured[3].index_type == VK_INDEX_TYPE_NONE_KHR && captured[3].primitives == 3,
               "not indexed, vertex_count 9: 3 triangles");
        expect(captured[4].vertex_format == VK_FORMAT_R16G16B16A16_SFLOAT && captured[4].primitives == 2,
               "a 16-bit vertex format reaches the build");
    }
    expect(captured_flags == VK_BUILD_ACCELERATION_STRUCTURE_ALLOW_COMPACTION_BIT_KHR,
           "the build flags reach the build");

    if (failures) {
        std::fprintf(stderr, "%d failure(s)\n", failures);
        return 1;
    }
    std::printf("KHR BLAS build: an indexed geometry with fewer than 3 indices has no triangles\n");
    return 0;
}
