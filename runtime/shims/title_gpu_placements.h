#pragma once

#include "xg_xs_layout.h"
#include "../src/gfx10_descriptors.h"
#include <d3d12.h>
#include <map>

namespace rt::d3d12 {

// GPU resources can have overlapping lifetimes at one placement VA. Resource references are
// held by the D3D12 caller. The device serializes this index with descriptor registration.
template<class Resource>
class TitleGpuPlacements {
public:
    void add(std::uint64_t address, Resource* resource) { resources_.emplace(address, resource); }

    void remove(std::uint64_t address, const void* owner) {
        auto [it, end] = resources_.equal_range(address);
        for (; it != end; ++it) {
            if (it->second == owner) {
                resources_.erase(it);
                return;
            }
        }
    }

    template<class Matches>
    Resource* find(std::uint64_t va, std::uint64_t alias, Matches matches) const {
        for (const auto address : {va, alias}) {
            if (!address) continue;
            auto [first, end] = resources_.equal_range(address);
            while (first != end) {
                Resource* r = (--end)->second;
                if (matches(*r)) return r;
            }
        }
        return nullptr;
    }

    static bool matches(const rt::xg_xs::ResourceDesc& d, const Resource& resource) {
        const auto& r = resource.desc();
        return UINT(r.Dimension) == d.dimension && r.Width == d.width && r.Height == d.height &&
               r.DepthOrArraySize == d.depth_or_array && (!d.mips || r.MipLevels == d.mips) &&
               r.SampleDesc.Count == d.samples &&
               (UINT(r.Format) == d.format || resource.lists_castable_format(DXGI_FORMAT(d.format)));
    }

    static bool matches(const leonida::gfx10::Image& image, const D3D12_RESOURCE_DESC& d) {
        namespace g = leonida::gfx10;
        if (d.Width != image.width || d.Height != image.height) return false;
        if (g::multisampled(image.type) != (d.SampleDesc.Count > 1)) return false;
        if (d.SampleDesc.Count > 1 && d.SampleDesc.Count != image.samples) return false;
        if (image.first_mip + image.mip_count > d.MipLevels) return false;
        if (image.type == g::ImageType::image_3d)
            return d.Dimension == D3D12_RESOURCE_DIMENSION_TEXTURE3D && image.depth == d.DepthOrArraySize;
        const auto dimension = image.type == g::ImageType::image_1d
                                   ? D3D12_RESOURCE_DIMENSION_TEXTURE1D : D3D12_RESOURCE_DIMENSION_TEXTURE2D;
        return d.Dimension == dimension && image.first_layer + image.layer_count <= d.DepthOrArraySize;
    }

private:
    std::multimap<std::uint64_t, Resource*> resources_;
};

}  // namespace rt::d3d12
