// The D3D12 tests link gta6rt_d3d12_com without the xmem shim, but the device's compute input probe (LOCAL
// DIAGNOSTIC, shims/d3d12_compute_input_probe.h) asks xmem for the other addresses of a page. A test process has
// no xmem mappings, so the answer is none.
#include <cstdint>
#include <vector>

namespace rt::xmem {
void geometry_aliases(std::uintptr_t, std::vector<std::uintptr_t>&) {}
}  // namespace rt::xmem
