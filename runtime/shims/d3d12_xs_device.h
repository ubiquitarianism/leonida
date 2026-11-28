// Xbox-ABI proxy around a PC ID3D12Device.
//
// Measured Xbox device deviations used here (vtable indices, x64):
//   slot 26   GetCustomHeapProperties, the PC slot, but the retail executable calls it with the buffer for the
//             returned struct in RCX and the device in RDX (node mask in R8, heap type in R9). A PC C++ method has
//             the device in RCX and the buffer in RDX. The thunk takes the console form when RDX holds a device
//             proxy, and forwards the PC form otherwise. Slot 43 returns a struct too; no call of it by the
//             executable has been seen yet, so it is forwarded in the PC form.
//   slot 41   CreateCommandSignature at its PC position; a console argument of type 105 (a 12-byte dispatch, read
//             from the retail title's call sites) is created as D3D12_INDIRECT_ARGUMENT_TYPE_DISPATCH.
//   slot 61   CreateSamplerX(desc, dest) -> PC CreateSampler (slot 22); the description's PC
//             D3D12_SAMPLER_DESC prefix is passed. The PC slot 61 is CreateMetaCommand. Measured on the retail title's
//             boot. Only for a caller in a guest-mapped image (set_console_caller, below); a host caller reaches the
//             PC CreateMetaCommand.
//   slot 57   the native graphics pipeline state (desc, iid, out). The factory of set_device_native_graphics_factory
//             (below) is asked first with the whole 0x108-byte description (leonida::native_pso::GraphicsDesc); when
//             it answers E_NOTIMPL (none set, no catalogue, or a plan it cannot build) the call goes on as
//             DeserializeGraphicsPipelineStateX(desc, iid, out): converts the Xbox 0x68-byte native graphics PSO
//             description (+0x00 root signature, +0x08/+0x10 state blob, +0x18/+0x20 pipeline input-layout blob,
//             +0x28/+0x30 and +0x48/+0x50 bare RDNA2 programs, +0x38/+0x40 and +0x58/+0x60 kind-2 stage blobs) to
//             D3D12_GRAPHICS_PIPELINE_STATE_DESC and forwards to PC CreateGraphicsPipelineState (slot 10). When the
//             second argument's IID is not IID_ID3D12PipelineState, forwards to PC CreateProtectedResourceSession.
//   slot 62   CreateComputePipelineState(desc, w2, w3, iid, out): two words (0 from XboxBvhBuilder, meaning unknown)
//             before the IID. The PC slot 62 is CreateStateObject.
//   slot 67   CreateComputePipelineState from native code (desc, iid, out): converts the Xbox native-code desc
//             (+0x00 root signature, +0x28 code, +0x30 size) to D3D12_COMPUTE_PIPELINE_STATE_DESC and forwards
//             to PC CreateComputePipelineState (slot 11). The PC slot 67 is CreateProtectedResourceSession1.
//   slot 70   (resource words, view words, destination handle), no return value: a shader-resource view of memory the
//             title placed itself, with no resource (retail boot chain stop 14). The PC slot 70 is CreatePlacedResource1.
//             Layout per VIctory: 14 resource dwords shaped as D3D12_RESOURCE_DESC, 16 view dwords whose first 10 are
//             the PC view description and 10-11 the memory's address. A buffer view gets its V#; a texture the PC
//             null view (no texture exists for the title's memory yet).
//             A shader-resource view that gets nothing this way asks the providers below before the null view:
//             the raw texture view provider for a 2D texture shape, and for the measured raw buffer shape (PR 7066,
//             eugen1763: a desktop D3D12_RESOURCE_DESC of a buffer, a desktop D3D12_SHADER_RESOURCE_VIEW_DESC with
//             R32_TYPELESS, BUFFER and RAW followed at once by a D3D12_GPU_VIRTUAL_ADDRESS) the device's resource
//             provider, which names the buffer a slot-133 call registered for that address.
//   slot 71   the same for an unordered-access view. The PC slot 71 is CreateSamplerFeedbackUnorderedAccessView.
//   slot 74   (pointer, u32, pointer): not measured, returns S_OK. The PC slot 74 is ShaderCacheControl(kinds, control),
//             which reads the first word as flags and returns E_INVALIDARG. Nothing is written, the PC device is not called.
//   slot 76   resource creation -> PC CreateCommittedResource (PC slot 27), with the Xbox layout word ignored.
//   slot 127  CreateCommittedResource3 (heap, flags, desc1, layout, clear, session, castable count/list, riid, out) ->
//             PC ID3D12Device10::CreateCommittedResource3 (PC slot 76), with the Xbox layout word replaced and the
//             Xbox flag bits dropped (RT-5 stop 15, docs/retail_boot_chain.md). E_NOTIMPL when the PC device lacks it.
//   slot 26   GetCustomHeapProperties, called by the retail title with the C ABI: return buffer in RCX, the device
//             in RDX (the PC form is `this` in RCX, return buffer in RDX). Slot 25 (GetResourceAllocationInfo, also a
//             struct wider than 8 bytes) takes the same form by the same rule. Both accept either form.
//   slot 133  CreatePlacedResourceX(va, desc, state, clear, riid, out) -> a buffer on the title's pages at va, made by
//             the handler of set_placed_at_va_handler (below); a depth-stencil texture with unordered access goes to
//             the depth provider (set_depth_stencil_uav_provider) when one is set. Where neither can make it, PC
//             CreateCommittedResource
//             on a DEFAULT heap, with the Xbox flag bits dropped and an all-ones state as COMMON: not backed by the
//             memory at va.
//   slot 48   CreateCommandQueueX(desc, riid, out) -> the same queue proxy as slot 8. The 32-byte Xbox desc starts with
//             a D3D12_COMMAND_QUEUE_DESC; only that prefix is passed on, with the console-only flag bits cleared
//             (DISABLE_GPU_TIMEOUT is kept). The four extra words are logged once.
//   slot 73   CreateCommandListX(desc, allocator, initialState, riid, out) -> the slot 12 console list proxy over PC
//             CreateCommandList(0, type, allocator, initialState). The type is the first word of the Xbox desc; there
//             is no node-mask argument in the console call (public HLE; the PC six-argument form differs). An riid the
//             proxy does not answer (the console's own list IID) is answered with the proxy.
//   slots 71/72  Write32BitValueBottomOfPipeX and Write64BitValueBottomOfPipeX: the console's value writes, with
//             slots 73/74 (Wait32BitValueX, Wait64BitValueX) the waits that go with them. A list the list values
//             track (d3d12_xs_list_values.h, LEONIDA_LIST_VALUES, on by default) records all four, lands each write
//             once its part has run and holds its queue at a wait. Only for a list they do not track
//             (LEONIDA_LIST_VALUES=0, bundles) does LEONIDA_GPU_LABELS=1 record slots 71/72 as GPU completion labels
//             (runtime/src/vulkan_gpu_labels.h): one writer per list. Either writer lands a value at an arena offset
//             in the arena, and one on a page of xmem's that the title's view keeps PAGE_NOACCESS through xmem's own
//             mapping (runtime/src/value_write_sink.h).
//   slots 78/79/80/81  SetFrameIntervalX, ScheduleFrameEventX, WaitFrameEventX, GetFrameStatisticsX. A presentation
//             attached with set_presentation_handler (below) answers them; without one a per-device token state
//             machine does, before the PC vtable is indexed: WaitFrameEventX returns at once with a fresh non-zero
//             token and GetFrameStatisticsX reports no statistics (S_OK and a zero count, the answer the retail boot
//             was measured with; E_POINTER for a null count or for a count above zero with no array). Slots 80/81 are
//             outside a PC ID3D12Device11 table, so they are checked before the beyond-table refusal.
//   slot 86   CreatePipelineState(desc, riid, out) -> PC ID3D12Device2::CreatePipelineState (PC slot 47).
//   slot 101  CreateStateObject(desc, riid, out) -> PC ID3D12Device5::CreateStateObject (PC slot 62, displaced on
//             Xbox by compute-PSO slot 62).
//   slot 108  GetResourceAllocationInfo2(ret, this, visible_mask, count, descs, info1) -> PC
//             ID3D12Device8::GetResourceAllocationInfo2 (PC slot 68), accepting the console C-ABI struct return
//             (return buffer in RCX, device in RDX) as well as the PC C++ order, with each 56-byte
//             D3D12_RESOURCE_DESC expanded to a zero-initialized D3D12_RESOURCE_DESC1 and Xbox layout words and
//             flag bits normalized on the copy.
// The slots 48/73/78/79/80 are the public HLE Freminet/ScarlettX's observation, not Microsoft documentation; slot 81
// is the observed call shape of the retail title. They can be moved with LEONIDA_XS_QUEUE_X_SLOT,
// LEONIDA_XS_LIST_X_SLOT, LEONIDA_XS_FRAME_INTERVAL_SLOT, LEONIDA_XS_SCHEDULE_EVENT_SLOT, LEONIDA_XS_WAIT_EVENT_SLOT
// and LEONIDA_XS_FRAME_STATISTICS_SLOT (3..127; anything else is logged once and ignored).
// The measured slots 9/10 (GetCPU/GPUDescriptorHandleForHeapStart) are ID3D12DescriptorHeap methods, not device
// methods: the PC device's 9/10 are CreateCommandAllocator/CreateGraphicsPipelineState and are forwarded unchanged.
// On the PC heap those getters use the hidden return pointer in RDX; the Xbox ABI returns the 8-byte handle in
// RAX. Device slot 14 therefore wraps the created heap in a heap proxy that performs that conversion. The heap
// proxy holds a reference to the device proxy, and its ID3D12DeviceChild::GetDevice returns that same proxy.
// The proxy's heap slots 9/10 expose only the Xbox shape: a PC C++ caller using the PC header's return convention
// would not receive the value, while GetDesc and the other heap slots forward the PC convention unchanged. A
// direct one-argument call against a real PC heap is invalid, so the device test exercises the conversion on the
// wrapper and compares it with a hidden-pointer call against the real heap.
//
// Every other PC device slot is forwarded by a generated thunk that swaps `this` to the PC device. The proxy's
// vtable therefore follows the PC device order, with slots 62 and 76 and the public HLE's 48/73/78/79/80 deliberately
// shadowed. The full Xbox vtable order is not known from the spec; slots beyond the PC device's own vtable return
// E_NOTIMPL and log once.
//
// Command queues and fences (device slots 8, 75, 36, and 32 for a shared fence) come back wrapped in queue and fence
// proxies of the same kind: the public chain (19 queue slots, 11 or 12 fence slots in SDK 10.0.26100) is forwarded
// in PC order, and a slot past it, up to the end of the proxy's table (128 queue, 64 fence slots), is logged with its
// first four argument words and answered S_OK on a queue or refused with E_NOTIMPL on a fence or allocator. Fence
// slot 12 is served instead: the retail executable reads its return as the address of the fence's 64-bit completed
// value, so it returns a per-fence mirror that a wait on the real fence keeps current (d3d12_xs_device.cpp, "Fence
// slot 12"). Queue slot 35 is the console ExecuteCommandLists(count, lists) variant and is served like PC slot 10, list
// proxies unwrapped. A slot past the table faults on the vtable read inside a
// reserved, never-committed region, and the fault is logged with the slot. That is where the console's queue methods
// (PresentX, SuspendX, ResumeX) will show up; the sibling-layer measurement puts PresentX at queue slot 30 or 32
// and SignalX at 23 or 25 by runtime layout (docs/d3d12_xs_console_abi.md §4, unverified here). The proxy detects
// the layout from a fence reaching slot 16/17, forwards SignalX as PC Signal, routes PresentX (slot 30, 32 in the
// newer layout, or LEONIDA_XS_PRESENT_X_SLOT) to an attached presentation handler or else to the device's own
// presentation core under LEONIDA_VULKAN_DEVICE, and answers the remaining console-only queue methods S_OK (frame
// pacing flows through them) without forwarding, and that
// the console keeps the public slots is an assumption (docs/d3d12_command_objects.md §5.2). Heap, queue and fence proxies are
// unwrapped wherever a device or queue method hands them to the real runtime: queue Signal/Wait (14/15), device
// CreateSharedHandle (31), MakeResident/Evict (34/35), SetEventOnMultipleFenceCompletion (45), SetResidencyPriority
// (46) and EnqueueMakeResident (50). Command lists come back from device slot 12 behind a console-shaped list
// proxy: console vtable order (PC order through slot 26, four console-only extras at 27-30, the PC chain shifted
// +4 through console 59, measured tail slots at 60/61/146), the write cursor pinned at the record limits so a
// title's inline writers always take the vtable path, unmodelled slots recorded and refused. Queue
// ExecuteCommandLists unwraps the list array; the list's own Reset, ExecuteBundle and SetDescriptorHeaps unwrap
// allocator, list and heap proxies. QueryInterface answers only the
// modelled interfaces; one the real object has but the proxy does not model is logged once and refused.
#pragma once

#include <d3d12.h>

#include <cstddef>
#include <cstdint>
#include <memory>
#include <string>
#include <vector>

namespace leonida {
class FrameTokenSource;
}

namespace leonida::native_pso { struct GraphicsDesc; }

namespace rt::xmem { struct PageBatch; }

namespace rt::d3d12_xs {

// Wraps `real` and returns the requested device interface through `out`. Takes ownership of one reference to
// `real` on success; on failure the caller keeps its reference. Returns E_NOINTERFACE if `riid` is not a D3D12
// device interface the PC device implements.
// `console_lists` shapes the command lists CreateCommandList returns: true (the title path) gives the
// console-shaped object the Xbox ABI calls; false hands the real list back for callers compiled against
// the PC headers (the GPU test harnesses), whose vtable indices past slot 26 do not exist in console order.
// D3D12XboxCreateDevice opts in; the default is PC-shaped so direct callers keep the old behaviour.
HRESULT create_device_proxy(ID3D12Device* real, REFIID riid, void** out, bool console_lists = false);

// True when `device` points at a proxy created by create_device_proxy. Used by the tests.
bool is_device_proxy(const void* device);

// The host path of D3D12XboxCreateDevice (BL-2, docs/d3d12_xs_surface.md): the PC D3D12CreateDevice (`create`)
// asked for kHostDeviceFloor, the lowest feature level D3D12 creates a device at. It is a minimum, so asking for it
// refuses no adapter the title can run on and still gives the device its highest level, which FEATURE_LEVELS then
// reports. A null `create`, or S_OK with no device, is E_NOTIMPL / E_FAIL; `*out` is null on failure.
constexpr D3D_FEATURE_LEVEL kHostDeviceFloor = D3D_FEATURE_LEVEL_11_0;
using HostCreateDeviceFn = HRESULT(WINAPI*)(IUnknown* adapter, D3D_FEATURE_LEVEL minimum, REFIID riid, void** out);
HRESULT create_host_device(HostCreateDeviceFn create, IUnknown* adapter, ID3D12Device** out);

// The highest of 11_0, 11_1, 12_0, 12_1 and 12_2 `device` reports through D3D12_FEATURE_FEATURE_LEVELS, or 0 when
// the query fails.
D3D_FEATURE_LEVEL max_feature_level(ID3D12Device* device);

// IDXGIDevice1 on the device. A GDKX title asks its device for IDXGIDevice1, then GetAdapter and EnumOutputs(0),
// to find the output it gives SetFrameIntervalX (docs/d3d12_vulkan.md section 4.5). The proxy answers IDXGIObject,
// IDXGIDevice and IDXGIDevice1 itself, as one COM object with the device (same reference count, QueryInterface
// both ways). GetAdapter and GetParent return the adapter of the device behind the proxy: by default the host DXGI
// adapter with that device's LUID. A provider replaces that: `real` is the device behind the proxy, and it returns
// the adapter as `riid`. rt::d3d12::set_device_outer sets one for the Vulkan-backed device (the layer's own adapter).
//
// A title may ask D3D12XboxCreateDevice (so create_device_proxy) for IDXGIDevice1 directly. What comes back then is
// the proxy's DXGI pointer, not the proxy: is_device_proxy is false for it. Code that takes "the proxy" from a
// creation call goes through device_proxy_primary first.
using DxgiAdapterProvider = HRESULT (*)(ID3D12Device* real, REFIID riid, void** out);
// `device` is the proxy or its DXGI pointer. False, with a line on stderr, when it is neither: nothing is set then.
bool set_device_adapter_provider(void* device, DxgiAdapterProvider provider);
// The Vulkan backend installs its own graphics PSO factory on its outer Xbox device.
class PsoStreamRecord;
using NativeStreamFactory = HRESULT (*)(ID3D12Device*, const D3D12_PIPELINE_STATE_STREAM_DESC&, REFIID, void**,
                                        std::unique_ptr<PsoStreamRecord>);
void set_native_stream_factory(void* device, NativeStreamFactory factory);
using NativeGraphicsFactory = HRESULT (*)(ID3D12Device*, const leonida::native_pso::GraphicsDesc&, REFIID, void**);
void set_device_native_graphics_factory(void* device, NativeGraphicsFactory factory);
// Finds the buffer resource that holds a console GPU virtual address, for device slot 70. `real` is the device behind
// the proxy; `bytes` is how many bytes from `va` the caller wants, so a resource that holds all of them is preferred
// over a shorter alias. Returns the resource with a reference added (the proxy releases it), or null when no resource holds
// `va`; `start` and `size` are then unused, otherwise the resource's first VA and size in bytes.
using ResourceByVaProvider = ID3D12Resource* (*)(ID3D12Device* real, std::uint64_t va, std::uint64_t bytes,
                                                std::uint64_t* start, std::uint64_t* size);
// `device` is the proxy or its DXGI pointer. False, with a line on stderr, when it is neither. Without a provider slot
// 70 does not look for a buffer by its address.
bool set_device_resource_provider(void* device, ResourceByVaProvider provider);
using TextureLeaseProvider = void (*)(IUnknown* resource, void* pages, std::size_t size, UINT layout);
void set_texture_lease_provider(void* device, TextureLeaseProvider provider);
using RawTextureViewProvider = bool (*)(ID3D12Device*, const D3D12_RESOURCE_DESC&, const D3D12_SHADER_RESOURCE_VIEW_DESC&,
                                        std::uint64_t, D3D12_CPU_DESCRIPTOR_HANDLE);
void set_raw_texture_view_provider(void* device, RawTextureViewProvider provider);
using ConsoleConstantView = bool (*)(ID3D12Device*, const D3D12_CONSTANT_BUFFER_VIEW_DESC*, D3D12_CPU_DESCRIPTOR_HANDLE);
void set_constant_view_provider(void* device, ConsoleConstantView provider);
// Installed by the Vulkan backend before handing the device to callers. Reports whether the exact
// CPU span belongs to the heap's live mapped allocation. PC descriptor handles may be opaque.
// (From the patch descriptor-write-through of Sigtryggur-Mani-Gudmundsson/graphics-runtime-patches at d9a0bc1b,
// GPL-3.0.)
using DescriptorMappingProvider = bool (*)(ID3D12DescriptorHeap* heap, SIZE_T start, SIZE_T bytes);
void set_descriptor_mapping_provider(void* device, DescriptorMappingProvider provider);
// Called before publishing an Xbox resource when the backend can implement its depth/stencil UAV views.
using DepthStencilUavProvider = HRESULT (*)(ID3D12Resource* resource, D3D12_HEAP_FLAGS storage_heap_flags);
inline bool depth_uav_description(const D3D12_RESOURCE_DESC& desc) {
    constexpr auto flags = D3D12_RESOURCE_FLAG_ALLOW_DEPTH_STENCIL | D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS;
    return (desc.Flags & flags) == flags && desc.Dimension == D3D12_RESOURCE_DIMENSION_TEXTURE2D &&
           (desc.Format == DXGI_FORMAT_R32G8X24_TYPELESS || desc.Format == DXGI_FORMAT_R16_TYPELESS ||
            desc.Format == static_cast<DXGI_FORMAT>(118)) &&
           desc.DepthOrArraySize && desc.MipLevels == 1 && desc.SampleDesc.Count == 1;
}
using ValidateDepthPlacement = bool (*)(const D3D12_RESOURCE_DESC& native, std::uint32_t layout,
                                        const D3D12_CLEAR_VALUE* clear);
bool set_depth_stencil_uav_provider(void* device, DepthStencilUavProvider provider,
                                    ValidateDepthPlacement validate_placement = nullptr);

// The Vulkan implementation uses this private interface to look up an image for view creation.
inline constexpr GUID kVulkanResourceTag{0x9b3e5f12, 0x7c48, 0x4d0e, {0xa6, 0xb1, 0x2f, 0x8d, 0x4c, 0x7e, 0x9a, 0x35}};

// Identically placed Xbox depth textures have distinct COM identities but refer to one GPU image and layout.
// Only these wrappers carry a backing pointer. Every other resource retains its ordinary pointer-valued IR ID.
struct PlacedDepthInterface : ID3D12Resource {
    ID3D12Resource* backing = nullptr;
};
inline std::atomic<const void*> placed_depth_vtable{nullptr};
inline const ID3D12Resource* texture_backing(const ID3D12Resource* resource) {
    const void* vtable = placed_depth_vtable.load(std::memory_order_acquire);
    if (resource && vtable && *reinterpret_cast<const void* const*>(resource) == vtable)
        return static_cast<const PlacedDepthInterface*>(resource)->backing;
    return resource;
}
// The device proxy an IDXGIDevice pointer handed out by one belongs to, or null.
void* device_proxy_from_dxgi_device(void* dxgi_device);
// The device proxy itself, given the proxy or its DXGI pointer; null for anything else.
void* device_proxy_primary(void* device);

// True when `heap` is a descriptor-heap proxy created by device slot 14.
bool is_heap_proxy(const void* heap);
ID3D12DescriptorHeap* heap_proxy_real_heap(const void* heap);

// Console D3D12 descriptor heaps allocate their backing through XMem (allocator ID 0x80 = eXALLOCAllocatorId_D3D in
// bits 16..23, object type 4 in the low 14 bits). A title that intercepts this allocation via __XMemAllocationHooks__
// returns its own virtual address and expects GetCPUDescriptorHandleForHeapStart() to return that address.
inline constexpr std::uint64_t kXmemDescriptorHeapAttributes = (0x80ULL << 16) | 4ULL;
using DescriptorHeapBackingAlloc = void* (*)(std::size_t byte_size, std::uint64_t attributes);
using DescriptorHeapBackingFree = void (*)(void* ptr, std::uint64_t attributes);
// Installs (or clears when both are null) the backing allocator consulted by CreateDescriptorHeap (slot 14).
void set_descriptor_heap_allocator(DescriptorHeapBackingAlloc alloc, DescriptorHeapBackingFree free);
// Installs the default bridge (used by D3D12XboxCreateDevice) only when no explicit allocator has been set.
void set_default_descriptor_heap_allocator(DescriptorHeapBackingAlloc alloc, DescriptorHeapBackingFree free);
// Translates a CPU descriptor handle from a title-backed descriptor heap to the underlying PC/Vulkan heap handle.
// Returns `handle` unchanged when it does not fall inside any active custom-backed heap.
D3D12_CPU_DESCRIPTOR_HANDLE translate_cpu_descriptor_handle(const void* device, D3D12_CPU_DESCRIPTOR_HANDLE handle);

// Number of PC vtable slots the proxy forwards (detected from the interfaces the PC device supports).
unsigned device_proxy_slot_count(const void* device);

// The PC device behind a proxy. Null when `device` is not a proxy.
ID3D12Device* device_proxy_real_device(const void* device);

// The real object behind any of the child proxies (heap, queue, fence, allocator, and a kind added later, such as a
// console command-list proxy, which joins proxy_real_object), or null when `object` is none of them. For code outside
// this file that is handed a title's object, like the XboxBvhBuilder shim's command lists.
IUnknown* real_object_behind_proxy(const void* object);
// Device slot 133 (the console's CreatePlacedResourceX) makes a buffer on the title's pages at `va`. The proxy does
// not create it itself: the device behind it does, through this handler, which the Vulkan layer registers
// (rt::d3d12::create_buffer_at_host_address). With none registered, a PC device behind the proxy, or a handler
// that fails (a texture, a range the import refuses), the slot creates the committed stand-in described at the top.
// A depth-stencil texture with unordered access goes to the proxy's depth provider first when one is set.
// `clear` is the title's clear value for a render target or depth-stencil texture, else null.
// `layout` is the title's third argument, a D3D12_BARRIER_LAYOUT (0xFFFFFFFF = UNDEFINED), measured on a retail
// boot: 0x6 SHADER_RESOURCE, 0x3 UNORDERED_ACCESS, as PC CreatePlacedResource2 takes it.
using PlacedAtVaHandler = HRESULT (*)(ID3D12Device* real, void* va, const D3D12_RESOURCE_DESC& desc,
                                      std::uint32_t layout, REFIID riid, void** out, const D3D12_CLEAR_VALUE* clear);
void set_placed_at_va_handler(PlacedAtVaHandler handler);

// Called by slot 133 for a buffer, before the handler, with the buffer's range [va, va + width). gta6rt registers
// rt::xmem::commit_physical_range: xmem commits physical pages on first touch, so a range the title has mapped but
// not written yet is still reserved, and the handler's import refuses reserved pages (it would answer with its
// fallback buffer, which the title's writes never reach). A false return is logged once and the handler is still
// called. With none registered nothing is done.
using PlacedAtVaPagesHook = bool (*)(const void* va, std::size_t bytes);
void set_placed_at_va_pages_hook(PlacedAtVaPagesHook hook);
// Test/backend seam for queue slot 31. The proxy serializes the mapping after earlier queue work and before later
// submissions. A null hook leaves the console-only call as a successful no-op when xmem is not linked.
using CopyPageMappingsHook = bool (*)(std::uint32_t, const rt::xmem::PageBatch*, std::uintptr_t,
                                      std::uint32_t, std::uint32_t);
void set_copy_page_mappings_hook(CopyPageMappingsHook hook);
using PinnedTexturePages = std::shared_ptr<std::uint8_t> (*)(const void*, std::size_t);
void set_texture_page_provider(PinnedTexturePages hook);
std::shared_ptr<std::uint8_t> pin_texture_pages(const void* address, std::size_t size);

// Device slot 14's console-extended CreateDescriptorHeap: the description is the PC 16-byte shape followed by a
// console VA and a size (issue #313). The proxy does not create the heap itself: the device behind it registers this
// handler, which the Vulkan layer implements (rt::d3d12::create_console_descriptor_heap). With none registered, or a
// 16-byte PC description (address 0), the slot uses the PC CreateDescriptorHeap.
using ConsoleDescriptorHeapHandler = HRESULT (*)(ID3D12Device* real, const D3D12_DESCRIPTOR_HEAP_DESC* desc,
                                                D3D12_GPU_VIRTUAL_ADDRESS console_address, std::uint64_t console_size,
                                                REFIID riid, void** out);
void set_console_descriptor_heap_handler(ConsoleDescriptorHeapHandler handler);

// Presentation wiring: connects device frame-event slots (78/79/80/81: SetFrameIntervalX, ScheduleFrameEventX,
// WaitFrameEventX, GetFrameStatisticsX; 81 is LEONIDA_XS_FRAME_STATISTICS_SLOT) and queue presentation slots (PresentX
// at 30, at 32 once the newer queue layout is detected, or at LEONIDA_XS_PRESENT_X_SLOT; SuspendX and ResumeX are
// routed only when LEONIDA_XS_SUSPEND_X_SLOT /
// LEONIDA_XS_RESUME_X_SLOT are set) to an external presenter such as rt::d3d12::XboxPresentation without coupling
// d3d12_xs_device to Vulkan headers. Passing nullptr detaches the handler and restores the default mock/fallback
// behavior.
struct PresentationHandler {
    void* context = nullptr;
    HRESULT (*set_frame_interval)(void* context, IUnknown* output, UINT interval_us, UINT latency, UINT flags) = nullptr;
    HRESULT (*schedule_frame_event)(void* context, UINT event, UINT offset_us, UINT flags) = nullptr;
    HRESULT (*wait_frame_event)(void* context, UINT event, DWORD timeout_ms, UINT flags, std::uint64_t* token) = nullptr;
    HRESULT (*get_frame_statistics)(void* context, std::uint64_t token, UINT type_mask, UINT* count, void* out) = nullptr;
    HRESULT (*present_x)(void* context, UINT plane_count, const void* planes, const void* params) = nullptr;
    HRESULT (*suspend_x)(void* context, UINT flags) = nullptr;
    HRESULT (*resume_x)(void* context) = nullptr;
};
bool set_presentation_handler(void* device, const PresentationHandler* handler);

// The device proxy's frame-token source (PresentX paces with the token source the device hands out), or null for
// an object that is not a device proxy.
std::shared_ptr<leonida::FrameTokenSource> device_frame_tokens(const void* device);

// True for slots the proxy handles itself rather than forwarding verbatim: IUnknown (0-2), 8 and 75 (queue
// proxies), 9 (allocator proxies), 14 (descriptor heap proxy), 17-24 (CPU descriptor handle translation), 26 (the
// console's struct-return form), 31/34/35/45/46/50 (proxies unwrapped),
// 32 and 36 (fence proxies), 48/73 and 78/79/80/81 (the Xbox slots above, at their configured indices), 13 (the
// console's format queries), 55, 57, 62, 67, 70,
// 74, 76, 86, 101, 108 and 127. Slot 10 is intentionally false: the measured 9/10 handle conversion is on the heap
// proxy, and device slot 9 is CreateCommandAllocator, which the proxy handles for the allocator proxy.
bool device_slot_overridden(unsigned slot);

// Whether a return address is in an image the guest loader mapped, i.e. the call comes from the title. Device slot 61
// is CreateSamplerX for such a caller and the PC ID3D12Device5::CreateMetaCommand for any other; with no check
// installed every caller is a host one. D3D12XboxCreateDevice installs rt::shims::mapped_image_base.
using ConsoleCaller = bool (*)(std::uint64_t address);
void set_console_caller(ConsoleCaller classify);
// True on the calling thread while the proxy forwards a CheckFeatureSupport (device slot 13) that a console caller
// made, so the device behind it can answer the console's formats.
bool console_feature_query() noexcept;

// Queue and fence proxies created by device slots 8/75 and 36. The real_* accessors return the wrapped object
// (no reference added), or null when the pointer is not such a proxy.
bool is_queue_proxy(const void* queue);
bool is_fence_proxy(const void* fence);
ID3D12CommandQueue* queue_proxy_real_queue(const void* queue);
ID3D12Fence* fence_proxy_real_fence(const void* fence);
// Public slots forwarded (19 for a queue; 11 or 12 for a fence), and the size of the proxy's vtable.
unsigned queue_proxy_slot_count(const void* queue);
unsigned fence_proxy_slot_count(const void* fence);
unsigned queue_proxy_table_size();
unsigned fence_proxy_table_size();
// True for queue slots handled by the proxy instead of forwarded verbatim past IUnknown: 0-2 (IUnknown),
// 7 (GetDevice), 14/15 (Signal/Wait unwrap fences), and 35 (Xbox ExecuteCommandLists variant -> PC slot 10).
bool queue_slot_overridden(unsigned slot);
// Live references on a queue, fence or allocator proxy (0 for anything else). Test helper.
unsigned long command_object_proxy_ref_count(const void* object);
// Per-slot call counters and first-call flags, shared by all queue proxies and all fence proxies.
std::uint64_t queue_slot_hits(unsigned slot);
bool queue_slot_first_call_seen(unsigned slot);
std::uint64_t fence_slot_hits(unsigned slot);
bool fence_slot_first_call_seen(unsigned slot);
// Vtable reads past a queue or fence proxy table that faulted in its guard region (counted by the fault logger).
std::uint64_t queue_table_overruns();
std::uint64_t fence_table_overruns();
// True when the queue, fence and allocator tables sit in front of their guard regions (false only if a reservation
// or the fault logger failed).
bool command_object_tables_guarded();

// Command allocator proxies (device slot 9): the public chain (9 slots: the base chain and Reset) is forwarded,
// slot 9 (console SetAllocatorCallbackX(callback, context, ...)) records the callback and context on the proxy and
// returns S_OK without invoking the callback on the PC/Vulkan path, slots 10..63 are logged and refused with
// E_NOTIMPL, and a read past the table faults in a guard region, as for queues and fences. Slot 9 wraps only an
// allocator that answers kLeonidaCommandAllocatorTag, Leonida's
// own Vulkan-layer allocator: its command lists resolve the proxy (CreateCommandList, Reset). A PC runtime
// allocator goes back unchanged, because the PC runtime's lists would take a proxy for their own allocator; a
// command-list proxy that unwraps at Reset is what lets the host backend wrap allocators too.
// {7d0e6b51-34a2-4f0c-9a61-0c5b2e8f4a10}: the Vulkan layer's CommandAllocator answers it with its implementation.
inline constexpr GUID kLeonidaCommandAllocatorTag{
    0x7d0e6b51, 0x34a2, 0x4f0c, {0x9a, 0x61, 0x0c, 0x5b, 0x2e, 0x8f, 0x4a, 0x10}};
bool is_allocator_proxy(const void* allocator);
// Command-list proxies (device slot 12): the console-shaped object. is_list_proxy / list_proxy_real_list give the
// same access as the queue, fence and allocator helpers; list_slot_hits counts calls per console slot.
bool is_list_proxy(const void* list);
ID3D12CommandList* list_proxy_real_list(const void* list);
std::uint64_t list_slot_hits(unsigned slot);
// The diagnostic name of a command-list slot on the list tail, where the newer console layout sits two slots later
// than the older one from CopyTextureRegionX on (docs/d3d12_xs_console_abi.md 5.3: newer->older pairs 85->83,
// 86->84, 87->85, 90->88, 91->89, 92->90, 95->93, 96->94). `console_layout` is the device-wide layout the queue
// proxy detects (kConsoleLayoutOlder or kConsoleLayoutNewer). Null when the layout is not known yet or the slot is
// not a measured tail slot of that layout, so that a slot number is never given the other layout's name. Naming
// only: no slot is forwarded because of it.
inline constexpr unsigned char kConsoleLayoutOlder = 1;
inline constexpr unsigned char kConsoleLayoutNewer = 2;
const char* list_tail_slot_name(unsigned slot, unsigned char console_layout);
// The wrapped allocator (no reference added), or null when `allocator` is not an allocator proxy.
ID3D12CommandAllocator* allocator_proxy_real_allocator(const void* allocator);
unsigned allocator_proxy_slot_count(const void* allocator);
unsigned allocator_proxy_table_size();
// Callback and context recorded by console allocator slot 9 (SetAllocatorCallbackX), or 0 if unset / not a proxy.
std::uintptr_t allocator_proxy_callback(const void* allocator);
std::uintptr_t allocator_proxy_callback_context(const void* allocator);
std::uint64_t allocator_slot_hits(unsigned slot);
bool allocator_slot_first_call_seen(unsigned slot);
std::uint64_t allocator_table_overruns();
// Command list proxies created by device slot 73 (CreateCommandListX):
bool is_command_list_proxy(const void* list);
ID3D12GraphicsCommandList* command_list_proxy_real_list(const void* list);
unsigned command_list_proxy_table_size();
std::uint64_t list_slot_hits(unsigned slot);
bool list_slot_first_call_seen(unsigned slot);

// Heap slots 0-2, 7 (GetDevice -> device proxy) and 9/10 (handle conversion).
bool heap_slot_overridden(unsigned slot);

// Console list slot mapping: returns PC vtable slot or ~0u if console-only / unforwarded.
inline constexpr unsigned kListNotForwarded = ~0u;
inline constexpr unsigned console_to_pc_list_slot(unsigned slot) {
    if (slot <= 26) return slot;               // PC order, measured
    if (slot <= 30) return kListNotForwarded;  // console-only extras
    if (slot <= 59) return slot - 4;           // PC chain, shifted +4
    if (slot == 60) return 58;                 // EndEvent (no console SetMarker/BeginEvent)
    if (slot == 61) return 59;                 // ExecuteIndirect (Package)
    // Slot 85 is CopyTextureRegionX in the newer console layout; in the older one the tail sits two slots earlier
    // (list_tail_slot_name), and 85 is another method. The retail title has not been seen calling 85.
    if (slot == 85) return 16;                 // CopyTextureRegionX -> CopyTextureRegion
    if (slot == 139) return 10;                // Reset(allocator, pso), as the retail title calls it (measured, Pipodi)
    if (slot == 146) return 80;                // D3D12_BARRIER_GROUP (Package; PC GCL7)
    return kListNotForwarded;
}

// Console list memory writes and waits (docs/d3d12_xs_console_abi.md §5.5; the runtime side is
// d3d12_xs_list_values.h). `write32(address, value, flush, flags)` and `write64(address, value, flush, flags)` store a
// value once the work before them has run (write64 flag 1 stores the GPU timestamp, 2 the system one);
// `wait32/wait64(address, comparison, reference, ?)` hold the queue until memory compares true. The newer layout
// (this package's title: the list tail sits two slots later) has them at 71-74, the older at 69-72; a layout not
// detected yet is taken as the newer one, as for ExecuteIndirect at 61.
enum class ConsoleValueOp : unsigned char { none, write32, write64, wait32, wait64 };
inline constexpr ConsoleValueOp console_value_op(unsigned slot, unsigned char console_layout) {
    const unsigned first = console_layout == kConsoleLayoutOlder ? 69 : 71;
    if (slot < first || slot > first + 3) return ConsoleValueOp::none;
    return static_cast<ConsoleValueOp>(slot - first + 1);
}
// A wait's comparison codes are the D3D12_COMPARISON_FUNC values, memory OP reference: 2 less, 3 equal, 4 less or
// equal, 5 greater, 6 not equal, 7 greater or equal (the title's, with a rising reference). 8 (always), 1 (never)
// and any other code count as met: a wait nothing can meet would only hold the queue to its timeout. A 32-bit wait
// compares the low 32 bits of both.
inline constexpr bool console_value_met(std::uint32_t comparison, std::uint64_t value, std::uint64_t reference,
                                        bool wide) {
    if (!wide) {
        value &= 0xffffffffu;
        reference &= 0xffffffffu;
    }
    switch (comparison) {
    case 2: return value < reference;
    case 3: return value == reference;
    case 4: return value <= reference;
    case 5: return value > reference;
    case 6: return value != reference;
    case 7: return value >= reference;
    default: return true;
    }
}
struct ListValueStats {
    std::uint64_t writes_recorded, writes_landed, writes_failed;
    std::uint64_t waits_recorded, waits_met_at_submit, waits_relayed;  // relayed: a GPU wait on a write in flight
    std::uint64_t gates_held, gates_met, gates_timed_out;               // gate: opened from the CPU
    std::uint64_t cuts, cuts_deferred, cut_failures;                    // a list split at a wait after work
    std::uint64_t gates_short, demotions, late_met;  // short timeouts, every wait demoted, short waits met later
    std::uint64_t bundle_ops;                         // writes and waits on bundles, not modelled
    std::uint64_t releases, release_us, release_us_max, pass_us_max;  // the releaser's signals, the pump's passes
    std::uint64_t release_stalls;                     // gate Signals still running after a second
    std::uint64_t remaps_now, remaps_deferred, remaps_failed, remap_wait_us_max;  // queue slot 31
    std::uint64_t stamps_gpu, stamps_cpu;  // write64 flags 1/2 landed with the GPU's timestamp, the CPU's clock
    std::uint64_t labels_handed;           // slot 71/72 writes given to LEONIDA_GPU_LABELS (lists not tracked here)
    std::uint64_t writes_arena, writes_title_pages;  // landed in the arena, through xmem's mapping (in writes_landed)
};
ListValueStats list_value_stats();
void reset_list_value_stats();
// How long a queue stands at an unmet wait before it runs on (LEONIDA_VALUE_WAIT_TIMEOUT_MS, default 200 ms).
void set_list_value_wait_timeout_ms(unsigned ms);
// Test helper: LEONIDA_LIST_VALUES as if set to 0 (false) or unset (true), for lists made afterwards.
void set_list_values_enabled(bool on);
// Before the process exits: the threads that land list writes and open gates finish and end (up to `ms`).
void stop_list_values(unsigned ms);
// Test helper: true once every write landed and every gate opened, waiting up to `ms`.
bool list_values_idle(unsigned ms);
// Test helper: the real lists a list proxy's recording is in (1 + its cuts), 0 for anything else.
unsigned list_proxy_part_count(const void* list);

// Number of live references on a device proxy. Test helper for the heap's GetDevice/lifetime checks. References the
// layer under the proxy holds for background work of its own are waited out first (set_reference_settler), so the
// count is the callers'.
unsigned long device_proxy_ref_count(const void* device);

// The layer under the proxy may keep an object, and through it the device, for work it started itself in the
// background: the Vulkan layer's pipeline builds hold their pipeline state until the worker has run them. `settle`
// returns when no such work is pending. device_proxy_ref_count calls it before it reads the count; nothing else does.
using ReferenceSettler = void (*)();
void set_reference_settler(ReferenceSettler settle);

// Per-slot call counters and first-call flags for the device and heap proxies. Slot 0 also counts the
// QueryInterface call made while a proxy is created, so tests reset the counters after creation.
// reset_device_proxy_stats also resets the queue and fence counters.
// The hit counters (device, heap, queue, fence and allocator) are for tests, and they are off at the start.
// A test must read a counter (any *_slot_hits function) or call reset_device_proxy_stats BEFORE the calls that it
// counts. Calls made before that are not counted (the counter stays 0); they set only the first-call flags.
std::uint64_t device_slot_hits(unsigned slot);
bool device_slot_first_call_seen(unsigned slot);
std::uint64_t heap_slot_hits(unsigned slot);
bool heap_slot_first_call_seen(unsigned slot);
void reset_device_proxy_stats();

// Decoded input element from an Xbox slot-57 pipeline blob (188 B in the first 3 boot graphics PSOs).
struct XboxPipelineInputElement {
    std::string semantic_name;
    UINT semantic_index = 0;
    DXGI_FORMAT format = DXGI_FORMAT_UNKNOWN;
    UINT input_slot = 0;
    UINT aligned_byte_offset = 0;
};

// Defensively parses the input-layout elements from an Xbox slot-57 pipeline blob (+0x4c element count, +0x54 array
// of NUL-terminated ASCII semantic name + 7 u32s). Stops cleanly before reading out of bounds when the last 1..4 B
// of the trailing 28-byte block are truncated at `blob_size`. Returns true when the header is valid.
bool parse_xbox_pipeline_blob_input_layout(const void* blob, std::size_t blob_size,
                                           std::vector<XboxPipelineInputElement>& out_elements);

// Looks up the stage ("vs", "ps", "ms", "gs") of a bare RDNA2 program by its SHA-256 (`fspx_code_sha256`) in the
// configured shader library (`LEONIDA_SHADERS`, including `fspx_gfx/index.json`). Returns an empty string on miss.
std::string lookup_xbox_fspx_program_stage(const void* program_bytes, std::size_t program_size);

}  // namespace rt::d3d12_xs
