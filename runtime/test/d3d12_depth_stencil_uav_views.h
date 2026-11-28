#pragma once

#include "d3d12_test_shaders.h"
#include "../src/gfx10_descriptors.h"

#include <cstdio>
#include <cstring>

// Exercise the descriptors as seen by a console shader, including the array size sentinel and invalid ranges.
inline bool check_depth_uav_array_views(ID3D12Device* device, ID3D12Resource* resource, bool depth16) {
    namespace g = leonida::gfx10;
    const UINT layers = resource->GetDesc().DepthOrArraySize;
    const D3D12_DESCRIPTOR_HEAP_DESC heap_desc{D3D12_DESCRIPTOR_HEAP_TYPE_CBV_SRV_UAV, 1,
                                               D3D12_DESCRIPTOR_HEAP_FLAG_NONE, 0};
    d3d12test::Ref<ID3D12DescriptorHeap> heap;
    if (layers < 2 || FAILED(device->CreateDescriptorHeap(&heap_desc, IID_PPV_ARGS(&heap.p)))) return false;
    const auto slot = heap->GetCPUDescriptorHandleForHeapStart();
    const auto read = [&] {
        g::ImageWords words{};
        std::memcpy(words.data(), reinterpret_cast<const void*>(slot.ptr), sizeof(words));
        return words;
    };
    const auto uav = [&](const D3D12_UNORDERED_ACCESS_VIEW_DESC* view) {
        std::memset(reinterpret_cast<void*>(slot.ptr), 0x55, sizeof(g::ImageWords));
        device->CreateUnorderedAccessView(resource, nullptr, view, slot);
        return read();
    };
    const g::ImageWords empty{};
    D3D12_UNORDERED_ACCESS_VIEW_DESC view{};
    view.ViewDimension = D3D12_UAV_DIMENSION_TEXTURE2DARRAY;
    view.Format = depth16 ? DXGI_FORMAT_R16_UINT : DXGI_FORMAT_R32_FLOAT;
    view.Texture2DArray = {0, 0, 1, 0};
    g::Image first{}, tail{};
    if (!g::decode(uav(&view), first) || first.type != g::ImageType::image_2d_array ||
        first.first_layer != 0 || first.layer_count != 1) return false;
    view.Texture2DArray = {0, 1, UINT_MAX, 0};
    const auto remaining = uav(&view);
    view.Texture2DArray.ArraySize = layers - 1;
    if (remaining != uav(&view) || !g::decode(remaining, tail) || tail.address != first.address ||
        tail.first_layer != 1 || tail.layer_count != layers - 1) return false;

    // A depth SRV still names the original image, with the same base layer and count.
    D3D12_SHADER_RESOURCE_VIEW_DESC srv{};
    srv.Format = depth16 ? DXGI_FORMAT_R16_UNORM : DXGI_FORMAT_R32_FLOAT_X8X24_TYPELESS;
    srv.Shader4ComponentMapping = D3D12_DEFAULT_SHADER_4_COMPONENT_MAPPING;
    srv.ViewDimension = D3D12_SRV_DIMENSION_TEXTURE2DARRAY;
    srv.Texture2DArray = {0, 1, 1, layers - 1, 0, 0};
    device->CreateShaderResourceView(resource, &srv, slot);
    g::Image attachment{};
    if (!g::decode(read(), attachment) || attachment.type != g::ImageType::image_2d_array ||
        attachment.address == tail.address || attachment.first_layer != 1 || attachment.layer_count != layers - 1)
        return false;
    if (!depth16) {
        srv.Format = DXGI_FORMAT_X32_TYPELESS_G8X24_UINT;
        srv.Texture2DArray.PlaneSlice = 1;
        device->CreateShaderResourceView(resource, &srv, slot);
        g::Image stencil{};
        if (!g::decode(read(), stencil) || stencil.address != attachment.address ||
            stencil.format != g::stencil_format || stencil.first_layer != 1 || stencil.layer_count != layers - 1)
            return false;
    }
    for (const D3D12_TEX2D_ARRAY_UAV invalid : {
             D3D12_TEX2D_ARRAY_UAV{1, 0, 1, 0}, {0, 0, 0, 0}, {0, layers, UINT_MAX, 0},
             {0, 1, layers, 0}, {0, 0, 1, depth16 ? 1u : 2u}}) {
        view.Texture2DArray = invalid;
        if (uav(&view) != empty) {
            std::fprintf(stderr, "accepted invalid UAV mip %u layers %u+%u plane %u\n",
                         invalid.MipSlice, invalid.FirstArraySlice, invalid.ArraySize, invalid.PlaneSlice);
            return false;
        }
    }
    view.Texture2DArray = {0, 0, 1, 0};
    view.Format = DXGI_FORMAT_R8_UINT;
    if (uav(&view) != empty) return false;
    view.ViewDimension = D3D12_UAV_DIMENSION_TEXTURE2D;
    view.Format = depth16 ? DXGI_FORMAT_R16_UINT : DXGI_FORMAT_R32_FLOAT;
    view.Texture2D = {};
    return uav(&view) == empty && uav(nullptr) == empty;
}
