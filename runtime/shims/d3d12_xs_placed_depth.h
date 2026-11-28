// GPU backing for slot-133 depth/UAV textures, keyed by the Xbox placement range on one device.
#pragma once

#include "d3d12_com_object.h"
#include "d3d12_xs_device.h"

#include <algorithm>
#include <bit>
#include <map>
#include <memory>
#include <mutex>
#include <optional>

namespace rt::d3d12_xs {

class PlacedDepthTexture final
    : public d3d12::Object<PlacedDepthTexture, PlacedDepthInterface, ID3D12Object, ID3D12DeviceChild,
                           ID3D12Pageable, ID3D12Resource> {
public:
    static constexpr GUID kTag{0x64472a85, 0x967d, 0x46bc, {0x96, 0xaf, 0x68, 0x1f, 0x50, 0x35, 0x64, 0x71}};

    PlacedDepthTexture(std::shared_ptr<ID3D12Resource> owner, UINT64 address)
        : owner_(std::move(owner)), address_(address) {
        backing = owner_.get();
        placed_depth_vtable.store(*reinterpret_cast<const void* const*>(this), std::memory_order_release);
    }

    HRESULT STDMETHODCALLTYPE QueryInterface(REFIID iid, void** out) override {
        if (iid == kVulkanResourceTag) return backing->QueryInterface(iid, out);
        return Object::QueryInterface(iid, out);
    }
    HRESULT STDMETHODCALLTYPE GetDevice(REFIID iid, void** out) override { return backing->GetDevice(iid, out); }
    D3D12_RESOURCE_DESC STDMETHODCALLTYPE GetDesc() override { return backing->GetDesc(); }
    D3D12_GPU_VIRTUAL_ADDRESS STDMETHODCALLTYPE GetGPUVirtualAddress() override { return address_; }
    HRESULT STDMETHODCALLTYPE GetHeapProperties(D3D12_HEAP_PROPERTIES* props, D3D12_HEAP_FLAGS* flags) override {
        return backing->GetHeapProperties(props, flags);
    }
    HRESULT STDMETHODCALLTYPE Map(UINT index, const D3D12_RANGE* read, void** data) override {
        return backing->Map(index, read, data);
    }
    void STDMETHODCALLTYPE Unmap(UINT index, const D3D12_RANGE* written) override { backing->Unmap(index, written); }
    HRESULT STDMETHODCALLTYPE WriteToSubresource(UINT index, const D3D12_BOX* box, const void* data,
                                                UINT row_pitch, UINT depth_pitch) override {
        return backing->WriteToSubresource(index, box, data, row_pitch, depth_pitch);
    }
    HRESULT STDMETHODCALLTYPE ReadFromSubresource(void* data, UINT row_pitch, UINT depth_pitch,
                                                  UINT index, const D3D12_BOX* box) override {
        return backing->ReadFromSubresource(data, row_pitch, depth_pitch, index, box);
    }

private:
    std::shared_ptr<ID3D12Resource> owner_;
    UINT64 address_;
};

class PlacedDepthTextures {
public:
    // The initial barrier layout and optimized clear are checked for each caller before looking up an alias.
    // Reusing a live compatible placement retains both the attachment image and its writable plane images.
    template<class Create>
    HRESULT create(UINT64 address, const D3D12_RESOURCE_DESC& desc, D3D12_RESOURCE_ALLOCATION_INFO allocation,
                   const D3D12_CLEAR_VALUE* clear, bool opaque_layout, REFIID iid, void** out, Create create_backing) {
        if (!address || !allocation.Alignment || address % allocation.Alignment || !allocation.SizeInBytes ||
            allocation.SizeInBytes > UINT64_MAX - address)
            return E_INVALIDARG;
        const UINT64 end = address + allocation.SizeInBytes;
        // Placed RT/DS textures can inherit contents only when their optimized clear values also match.
        // Compare the defined fields. D3D12_CLEAR_VALUE has union padding, and D16 has no stencil plane.
        std::optional<OptimizedClear> optimized;
        if (clear) optimized = OptimizedClear{clear->Format, std::bit_cast<UINT>(clear->DepthStencil.Depth),
            desc.Format == DXGI_FORMAT_R16_TYPELESS ? UINT8(0) : clear->DepthStencil.Stencil};
        std::shared_ptr<ID3D12Resource> owner;
        try {
            {
                std::lock_guard lock(mutex_);
                // Weak ownership avoids keeping GPU images or the device alive just because an address was used.
                std::erase_if(ranges_, [](const auto& entry) { return entry.second.backing.expired(); });
                // At the same offset two differently shaped opaque depth targets may alternate after
                // an aliasing barrier and reinitialization. The API promises data inheritance only for
                // identical descriptions and optimized clears. Keep that matching backing alive even
                // when another shape is also registered at this address.
                // A range's end is the PC allocation's size, which can run past the title's own footprint (on
                // RADV a large D16 target is sized above the block the title placed it in, and the title's next
                // block starts inside it). A placement that starts inside another range therefore shares nothing
                // with it: between two opaque layouts it gets a backing of its own, since D3D12 leaves an alias's
                // contents undefined until it is reinitialized. Only a non-opaque layout is refused there.
                const auto limit = ranges_.lower_bound(end);
                for (auto it = ranges_.begin(); it != limit; ++it) {
                    const auto& range = it->second;
                    if (range.end <= address) continue;
                    if (it->first != address) {
                        if (!opaque_layout || !range.opaque_layout) return E_NOTIMPL;
                        continue;
                    }
                    const bool same_shape = compatible(desc, range.desc);
                    if (range.end == end && same_shape && range.optimized == optimized) {
                        if (auto live = range.backing.lock()) owner = std::move(live);
                    } else if (!opaque_layout || !range.opaque_layout || same_shape) {
                        return E_NOTIMPL;
                    }
                }
                if (!owner) {
                    ID3D12Resource* resource = nullptr;
                    const HRESULT hr = create_backing(&resource);
                    if (FAILED(hr)) return hr;
                    owner = {resource, [](ID3D12Resource* r) { r->Release(); }};
                    ranges_.emplace(address, Range{end, desc, optimized, owner, opaque_layout});
                }
            }
            auto* alias = new PlacedDepthTexture(std::move(owner), address);
            return d3d12::hand_out(alias, iid, out);
        } catch (const std::bad_alloc&) {
            return E_OUTOFMEMORY;
        }
    }

private:
    // Dimension, mip/sample counts and layout are fixed by the slot handler. The effective alignment and byte
    // interval have already been validated. All remaining usage and image geometry must match for data inheritance.
    static bool compatible(const D3D12_RESOURCE_DESC& a, const D3D12_RESOURCE_DESC& b) {
        return a.Format == b.Format && a.Width == b.Width && a.Height == b.Height &&
               a.DepthOrArraySize == b.DepthOrArraySize && a.SampleDesc.Quality == b.SampleDesc.Quality &&
               a.Flags == b.Flags;
    }
    struct OptimizedClear {
        DXGI_FORMAT format;
        UINT depth_bits;
        UINT8 stencil;
        bool operator==(const OptimizedClear&) const = default;
    };
    struct Range {
        UINT64 end;
        D3D12_RESOURCE_DESC desc;
        std::optional<OptimizedClear> optimized;
        std::weak_ptr<ID3D12Resource> backing;
        bool opaque_layout;
    };
    std::mutex mutex_;
    std::multimap<UINT64, Range> ranges_;
};

} // namespace rt::d3d12_xs
