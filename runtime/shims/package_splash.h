#pragma once

#include <filesystem>
#include <d3d12.h>

namespace rt::package_splash {

// Called before the title starts. An absent or disabled shell visual leaves presentation to the title.
void configure(const std::filesystem::path& package);
bool configured();
bool paint_window(HWND window);
ID3D12Resource* create_texture(ID3D12Device* device, UINT width, UINT height);

}  // namespace rt::package_splash
