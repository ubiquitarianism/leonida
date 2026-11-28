// The positive-only cadence oracle in vk_presenter_sync_test.cpp. Included after its driver counters.
#pragma once

namespace sparse_test {
namespace fs = std::filesystem;
using RGB = std::array<int, 3>;
constexpr RGB red{255, 0, 0}, blue{0, 0, 255}, black{0, 0, 0};

// Own a newly created owner/SYSTEM-only directory. Never reuse or recursively remove an existing tree.
struct Folder {
    fs::path path;
    bool create(const char* tag) {
        auto candidate = fs::temp_directory_path() / (std::string("leonida-sparse-") + tag + "-" +
            std::to_string(GetCurrentProcessId()) + "-" + std::to_string(GetTickCount64()));
        PSECURITY_DESCRIPTOR sd = nullptr;
        if (!ConvertStringSecurityDescriptorToSecurityDescriptorW(L"D:P(A;OICI;FA;;;OW)(A;OICI;FA;;;SY)",
                                                                  SDDL_REVISION_1, &sd, nullptr)) return false;
        SECURITY_ATTRIBUTES sa{sizeof(sa), sd, FALSE};
        const bool fresh = CreateDirectoryW(candidate.c_str(), &sa) != FALSE;
        LocalFree(sd);
        if (fresh) path = std::move(candidate);
        return fresh;
    }
    ~Folder() {
        if (path.empty()) return;
        std::error_code ec;
        for (auto name : {"index.txt", "frame_00001.ppm", "frame_00004.ppm"}) fs::remove(path / name, ec);
        fs::remove(path, ec);  // leave unknown contents alone on failure
    }
};
const char* sdr_format(VkFormat f) {
    switch (f) {
    case VK_FORMAT_B8G8R8A8_UNORM: return "VK_FORMAT_B8G8R8A8_UNORM";
    case VK_FORMAT_B8G8R8A8_SRGB: return "VK_FORMAT_B8G8R8A8_SRGB";
    case VK_FORMAT_R8G8B8A8_UNORM: return "VK_FORMAT_R8G8B8A8_UNORM";
    case VK_FORMAT_R8G8B8A8_SRGB: return "VK_FORMAT_R8G8B8A8_SRGB";
    default: return nullptr;
    }
}
bool disk_matches(const fs::path& dir, VkExtent2D extent, VkFormat fmt, RGB fourth) {
    std::set<fs::path> names;
    for (const auto& e : fs::directory_iterator(dir)) {
        if (!fs::is_regular_file(e.symlink_status())) return false;
        names.insert(e.path().filename());
    }
    if (names != std::set<fs::path>{"index.txt", "frame_00001.ppm", "frame_00004.ppm"} || !sdr_format(fmt)) return false;
    const auto w=extent.width, h=extent.height;
    if (!w || !h || w>1024 || h>1024) return false;
    const std::string header="P6\n"+std::to_string(w)+" "+std::to_string(h)+"\n255\n";
    std::ifstream index(dir/"index.txt");
    for (const int n : {1,4}) {
        const RGB color = n==1 ? red : fourth;
        const fs::path file=dir/(n==1 ? "frame_00001.ppm":"frame_00004.ppm");
        if (fs::file_size(file)!=header.size()+3ull*w*h) return false;
        std::string expected=header;
        expected.reserve(header.size()+3ull*w*h);
        for (size_t i=0; i<size_t(w)*h; ++i) for (int c:color) expected.push_back(static_cast<char>(c));
        std::ifstream img(file,std::ios::binary);
        std::string got(expected.size(),'\0');
        if (!img.read(got.data(),static_cast<std::streamsize>(got.size())) || got!=expected) return false;
        const auto row="frame "+std::to_string(n)+" format "+std::to_string(int(fmt))+" ("+sdr_format(fmt)+") extent "+
            std::to_string(w)+"x"+std::to_string(h)+" mean rgb "+std::to_string(color[0])+".00 "+
            std::to_string(color[1])+".00 "+std::to_string(color[2])+".00";
        if (!std::getline(index,got) || got!=row) return false;
    }
    return index.peek()==std::char_traits<char>::eof();
}

int run(vktest::Device& d, HWND window) {
    using namespace leonida;
    VulkanContext context(driver::get_device,d.device,true);
    VulkanQueue queue(context,d.queue,d.family);
    EXPECT(context.loaded() && queue.ok());
    const auto surface_fn=reinterpret_cast<PFN_vkCreateWin32SurfaceKHR>(d.instance_proc(d.instance,"vkCreateWin32SurfaceKHR"));
    EXPECT(surface_fn);
    // Two immutable RGBA8 test sources, filled on this queue once. present_blank supplies the third color.
    for (ResourceId id : {1u,2u}) {
        ResourceBinding rb{};
        rb.width=64;rb.height=64;rb.format=VK_FORMAT_R8G8B8A8_UNORM;rb.depth=rb.mip_levels=rb.array_layers=1;
        rb.samples=1;rb.color_attachment=true;rb.sampled=true;rb.aspects=VK_IMAGE_ASPECT_COLOR_BIT;
        rb.image=d.image(rb.format,rb.width,rb.height,1,VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT |
                         VK_IMAGE_USAGE_TRANSFER_DST_BIT | VK_IMAGE_USAGE_TRANSFER_SRC_BIT | VK_IMAGE_USAGE_SAMPLED_BIT);
        EXPECT(rb.image);
        context.add_resource(id,rb,VK_IMAGE_LAYOUT_UNDEFINED);
        CommandList list;
        EXPECT(list.transition({id,0,d3d12_state::common,d3d12_state::render_target}));
        EXPECT(list.clear_render_target({id,0,0,1},{id==1?1.f:0.f,0,id==2?1.f:0.f,1}));
        EXPECT(list.transition({id,0,d3d12_state::render_target,d3d12_state::common}) && list.close());
        const CommandList* input=&list;
        EXPECT(queue.execute({&input,1})==ReplayResult::success);
    }
    EXPECT(queue.idle());
    EXPECT(!_putenv_s("LEONIDA_CAPTURE_EVERY","3") && !_putenv_s("LEONIDA_CAPTURE_LIMIT","2") &&
           !_putenv_s("LEONIDA_CAPTURE_SOURCE","0"));
    enum class Phase { sparse, final_pending, continuous, splash, composed };
    for (Phase p : {Phase::sparse,Phase::final_pending,Phase::continuous,Phase::splash,Phase::composed}) {
        Folder folder;
        EXPECT(folder.create(p==Phase::sparse?"sparse":p==Phase::final_pending?"pending":p==Phase::continuous?"explicit":
                             p==Phase::splash?"splash":"composed"));
        EXPECT(!_putenv_s("LEONIDA_CAPTURE",folder.path.string().c_str()));
        VkExtent2D extent{};VkFormat fmt{};
        {
            VkWin32SurfaceCreateInfoKHR info{VK_STRUCTURE_TYPE_WIN32_SURFACE_CREATE_INFO_KHR};
            info.hinstance=GetModuleHandleW(nullptr);info.hwnd=window;
            VkSurfaceKHR surface{};EXPECT(surface_fn(d.instance,&info,nullptr,&surface)==VK_SUCCESS);
            VulkanPresenter present(context,driver::get_instance,d.instance,d.physical,d.queue,d.family,surface);
            EXPECT(present.ok());
            if (p==Phase::continuous) EXPECT(present.enable_capture());
            const unsigned last = p==Phase::final_pending?4:p==Phase::splash?7:8;
            for (unsigned n=1; n<=last; ++n) {
                const bool disk=n==1 || n==4;
                const bool replaced=p==Phase::splash && n>=2 && n<=4;
                const bool handoff=p==Phase::splash && n==5;
                const RGB expected=n==1 || replaced ? red : n==4 || handoff ? blue : n%2 ? red : black;
                driver::capture_copies=0; driver::capture_waw_barriers=0; driver::saw_capture_waw=false;
                driver::observe_capture=true;
                if (p==Phase::splash && n==1) {
                    EXPECT(present.present_splash(1,std::make_shared<int>(0)));
                } else if (replaced || expected==black) EXPECT(present.present_blank(true)==VK_SUCCESS);
                else if (p==Phase::composed) {
                    const auto before=present.composed_presents();
                    PresentPlanes planes{};
                    planes.plane[0].source=expected==red?1:2;
                    EXPECT(present.present_planes(planes,true)==VK_SUCCESS && present.composed_presents()==before+1);
                } else EXPECT(present.present(expected==red?1:2,true)==VK_SUCCESS);
                driver::observe_capture=false;
                EXPECT(present.presented()==n);
                const unsigned copies=replaced?2u : disk || p==Phase::continuous || handoff ? 1u : 0u;
                EXPECT(driver::capture_copies==copies && driver::capture_waw_barriers==copies &&
                       !driver::saw_capture_waw);  // also reject a WAW on a frame with no copy
                std::vector<std::uint8_t> pixels{77};VkExtent2D e{17,18};VkFormat f=VK_FORMAT_UNDEFINED;
                EXPECT(present.read_back(pixels,&e,&f)==(copies!=0));
                if (copies) {
                    if (n==1) {extent=e;fmt=f;}
                    EXPECT(e.width==extent.width && e.height==extent.height && f==fmt && sdr_format(f));
                    const bool bgra=f==VK_FORMAT_B8G8R8A8_UNORM || f==VK_FORMAT_B8G8R8A8_SRGB;
                    EXPECT(presenter_capture_matches_rgb(pixels,e.width,e.height,bgra,expected,0));
                } else EXPECT(pixels.size()==1 && pixels[0]==77 && e.width==17 && e.height==18 && f==VK_FORMAT_UNDEFINED);
            }
            if (p==Phase::final_pending) {
                EXPECT(fs::exists(folder.path/"frame_00001.ppm") && !fs::exists(folder.path/"frame_00004.ppm"));
            }
        }  // pending final frame 4 is written here, without another present
        EXPECT(disk_matches(folder.path,extent,fmt,p==Phase::splash?red:blue));
    }
    // Splash-only storage is not an ongoing readback request. No disk path, no enable_capture().
    EXPECT(_putenv_s("LEONIDA_CAPTURE", "") == 0);
    {
        VkWin32SurfaceCreateInfoKHR info{VK_STRUCTURE_TYPE_WIN32_SURFACE_CREATE_INFO_KHR};
        info.hinstance = GetModuleHandleW(nullptr);
        info.hwnd = window;
        VkSurfaceKHR surface{};
        EXPECT(surface_fn(d.instance, &info, nullptr, &surface) == VK_SUCCESS);
        VulkanPresenter present(context, driver::get_instance, d.instance, d.physical, d.queue, d.family, surface);
        EXPECT(present.ok());
        auto owner = std::make_shared<int>(0);
        std::weak_ptr<int> held = owner;
        for (unsigned n = 1; n <= 5; ++n) {
            driver::capture_copies = 0;
            driver::capture_waw_barriers = 0;
            driver::saw_capture_waw = false;
            driver::observe_capture = true;
            if (n == 1) EXPECT(present.present_splash(1, std::move(owner)));   // show the red shell
            else if (n == 3 || n == 5) EXPECT(present.present(2, true) == VK_SUCCESS);  // blue title
            else EXPECT(present.present_blank(true) == VK_SUCCESS);
            driver::observe_capture = false;

            // Initial shell: zero copies. Held black: inspect, then overwrite with the red shell. First blue:
            // inspect once and release. Subsequent black and blue are ordinary frames, with no copies.
            const unsigned expected_copies = n == 2 ? 2 : n == 3 ? 1 : 0;
            EXPECT(present.presented() == n && driver::capture_copies == expected_copies &&
                   driver::capture_waw_barriers == expected_copies && !driver::saw_capture_waw);
            EXPECT(held.expired() == (n >= 3));
            std::vector<std::uint8_t> pixels{77};
            VkExtent2D extent{17, 18};
            VkFormat format = VK_FORMAT_UNDEFINED;
            if (n == 2) {
                EXPECT(present.read_back(pixels, &extent, &format) && sdr_format(format));
                const bool bgra = format == VK_FORMAT_B8G8R8A8_UNORM || format == VK_FORMAT_B8G8R8A8_SRGB;
                EXPECT(presenter_capture_matches_rgb(pixels, extent.width, extent.height, bgra, red, 0));
            } else {
                EXPECT(!present.read_back(pixels, &extent, &format));
                EXPECT(pixels.size() == 1 && pixels[0] == 77 && extent.width == 17 && extent.height == 18 &&
                       format == VK_FORMAT_UNDEFINED);
            }
        }
    }
    EXPECT(queue.idle());context.remove_resource(1);context.remove_resource(2);
    std::puts("sparse capture: frames 1/4, cap, last-fence drain, explicit readback, disk and splash-only handoffs verified");
    return 0;
}
}  // namespace sparse_test
