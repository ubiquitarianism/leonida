// D3D12 acceleration-structure inputs in VK_KHR_acceleration_structure terms, shared by the build that
// BuildRaytracingAccelerationStructure records (d3d12_com.cpp) and the size query of
// GetRaytracingAccelerationStructurePrebuildInfo (d3d12_vulkan_device.cpp), so the size a title is told fits the
// build it then records: the same vertex formats, the same build flags and the same triangle counts.
#pragma once

#include <windows.h>
#include <d3d12.h>
#include <vulkan/vulkan.h>

#include "../src/command_list.h"

#include <cstdint>
#include <vector>

namespace rt::d3d12 {

// The vertex formats D3D12 allows for BLAS builds; any other is VK_FORMAT_UNDEFINED.
inline VkFormat khr_vertex_format(DXGI_FORMAT format) {
    switch (format) {
    case DXGI_FORMAT_R32G32B32_FLOAT: return VK_FORMAT_R32G32B32_SFLOAT;
    case DXGI_FORMAT_R32G32_FLOAT: return VK_FORMAT_R32G32_SFLOAT;
    case DXGI_FORMAT_R16G16B16A16_FLOAT: return VK_FORMAT_R16G16B16A16_SFLOAT;
    case DXGI_FORMAT_R16G16_FLOAT: return VK_FORMAT_R16G16_SFLOAT;
    case DXGI_FORMAT_R16G16B16A16_SNORM: return VK_FORMAT_R16G16B16A16_SNORM;
    case DXGI_FORMAT_R16G16_SNORM: return VK_FORMAT_R16G16_SNORM;
    default: return VK_FORMAT_UNDEFINED;
    }
}

// D3D12_RAYTRACING_ACCELERATION_STRUCTURE_BUILD_FLAGS as build flags. PERFORM_UPDATE is the build's mode, not a flag.
// MINIMIZE_MEMORY is LOW_MEMORY (XDXR: BOX16 nodes in a BLAS).
inline VkBuildAccelerationStructureFlagsKHR khr_build_flags(D3D12_RAYTRACING_ACCELERATION_STRUCTURE_BUILD_FLAGS flags) {
    VkBuildAccelerationStructureFlagsKHR out = 0;
    if (flags & D3D12_RAYTRACING_ACCELERATION_STRUCTURE_BUILD_FLAG_ALLOW_UPDATE)
        out |= VK_BUILD_ACCELERATION_STRUCTURE_ALLOW_UPDATE_BIT_KHR;
    if (flags & D3D12_RAYTRACING_ACCELERATION_STRUCTURE_BUILD_FLAG_ALLOW_COMPACTION)
        out |= VK_BUILD_ACCELERATION_STRUCTURE_ALLOW_COMPACTION_BIT_KHR;
    if (flags & D3D12_RAYTRACING_ACCELERATION_STRUCTURE_BUILD_FLAG_PREFER_FAST_TRACE)
        out |= VK_BUILD_ACCELERATION_STRUCTURE_PREFER_FAST_TRACE_BIT_KHR;
    if (flags & D3D12_RAYTRACING_ACCELERATION_STRUCTURE_BUILD_FLAG_PREFER_FAST_BUILD)
        out |= VK_BUILD_ACCELERATION_STRUCTURE_PREFER_FAST_BUILD_BIT_KHR;
    if (flags & D3D12_RAYTRACING_ACCELERATION_STRUCTURE_BUILD_FLAG_MINIMIZE_MEMORY)
        out |= VK_BUILD_ACCELERATION_STRUCTURE_LOW_MEMORY_BIT_KHR;
    return out;
}

inline VkIndexType khr_index_type(DXGI_FORMAT format) {
    if (format == DXGI_FORMAT_R16_UINT) return VK_INDEX_TYPE_UINT16;
    if (format == DXGI_FORMAT_R32_UINT) return VK_INDEX_TYPE_UINT32;
    return VK_INDEX_TYPE_NONE_KHR;
}

// What vkGetAccelerationStructureBuildSizesKHR is asked for one D3D12 input: one geometry per triangle desc (or the
// instances of a TLAS), the most primitives of each, and the flags. Addresses stay null: a size query reads no memory.
struct KhrBuildSizeQuery {
    VkAccelerationStructureTypeKHR type = VK_ACCELERATION_STRUCTURE_TYPE_BOTTOM_LEVEL_KHR;
    VkBuildAccelerationStructureFlagsKHR flags = 0;
    std::vector<VkAccelerationStructureGeometryKHR> geometries;
    std::vector<std::uint32_t> max_primitives;
};

inline KhrBuildSizeQuery khr_build_size_query(const D3D12_BUILD_RAYTRACING_ACCELERATION_STRUCTURE_INPUTS& in) {
    KhrBuildSizeQuery q;
    q.flags = khr_build_flags(in.Flags);
    const bool pointers = in.DescsLayout == D3D12_ELEMENTS_LAYOUT_ARRAY_OF_POINTERS;
    if (in.Type == D3D12_RAYTRACING_ACCELERATION_STRUCTURE_TYPE_TOP_LEVEL) {
        q.type = VK_ACCELERATION_STRUCTURE_TYPE_TOP_LEVEL_KHR;
        VkAccelerationStructureGeometryKHR g{VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_GEOMETRY_KHR};
        g.geometryType = VK_GEOMETRY_TYPE_INSTANCES_KHR;
        g.geometry.instances.sType = VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_GEOMETRY_INSTANCES_DATA_KHR;
        g.geometry.instances.arrayOfPointers = pointers ? VK_TRUE : VK_FALSE;
        q.geometries.push_back(g);
        q.max_primitives.push_back(in.NumDescs);
        return q;
    }
    for (UINT i = 0; i < in.NumDescs; ++i) {
        const D3D12_RAYTRACING_GEOMETRY_DESC* gd =
            pointers ? (in.ppGeometryDescs ? in.ppGeometryDescs[i] : nullptr)
                     : (in.pGeometryDescs ? &in.pGeometryDescs[i] : nullptr);
        if (!gd || gd->Type != D3D12_RAYTRACING_GEOMETRY_TYPE_TRIANGLES) continue;
        const auto& t = gd->Triangles;
        VkAccelerationStructureGeometryKHR g{VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_GEOMETRY_KHR};
        g.geometryType = VK_GEOMETRY_TYPE_TRIANGLES_KHR;
        g.flags = (gd->Flags & D3D12_RAYTRACING_GEOMETRY_FLAG_OPAQUE) ? VK_GEOMETRY_OPAQUE_BIT_KHR : 0u;
        auto& tr = g.geometry.triangles;
        tr.sType = VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_GEOMETRY_TRIANGLES_DATA_KHR;
        // A format the build refuses still gets a size, as the three-float one (the size query must name a format).
        tr.vertexFormat = khr_vertex_format(t.VertexFormat);
        if (tr.vertexFormat == VK_FORMAT_UNDEFINED) tr.vertexFormat = VK_FORMAT_R32G32B32_SFLOAT;
        tr.vertexStride = t.VertexBuffer.StrideInBytes;
        tr.maxVertex = t.VertexCount ? (t.VertexCount - 1u) : 0u;
        tr.indexType = khr_index_type(t.IndexFormat);
        q.geometries.push_back(g);
        q.max_primitives.push_back(
            leonida::rt_triangle_count(tr.indexType != VK_INDEX_TYPE_NONE_KHR, t.IndexCount, t.VertexCount));
    }
    return q;
}

}  // namespace rt::d3d12
