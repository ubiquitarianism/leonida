// Small optimal images share memory until their deferred image destruction has run.
#define VK_NO_PROTOTYPES
#include "../src/vulkan_title_image_pool.h"

#include <cstdio>

namespace {
unsigned allocated, freed, destroyed, next_type;
bool fail_alloc, fail_bind;
VkDeviceSize last_offset;
VKAPI_ATTR VkResult VKAPI_CALL bind_image(VkDevice, VkImage, VkDeviceMemory, VkDeviceSize offset) {
    last_offset = offset;
    return fail_bind ? VK_ERROR_OUT_OF_DEVICE_MEMORY : VK_SUCCESS;
}
VKAPI_ATTR void VKAPI_CALL destroy_image(VkDevice, VkImage, const VkAllocationCallbacks*) { ++destroyed; }
VKAPI_ATTR void VKAPI_CALL free_memory(VkDevice, VkDeviceMemory, const VkAllocationCallbacks*) { ++freed; }
#define CHECK(cond) do { if (!(cond)) { std::fprintf(stderr, "failed at line %d: %s\n", __LINE__, #cond); return 1; } } while (false)
}

int main() {
    leonida::TitleImagePool pool;
    VkMemoryRequirements req{800, 2048, 1};
    unsigned next_image = 0;
    const auto make = [&] {
        const auto image = (VkImage)(std::uintptr_t(++next_image));
        return pool.bind(VK_NULL_HANDLE, image, req, bind_image, destroy_image, free_memory,
                         [](VkDeviceSize, std::uint32_t& type) {
                             type = next_type;
                             return fail_alloc ? VkDeviceMemory{} : (VkDeviceMemory)(std::uintptr_t(++allocated));
                         });
    };
    auto a = make();
    CHECK(a && last_offset == 0);
    auto b = make();
    CHECK(b && last_offset == 2048 && allocated == 1 && b->page == a->page);
    // Simulate removing the first image from replay, then draining its deferred deletion later.
    auto retired = std::move(a);
    auto c = make();
    CHECK(c && last_offset == 4096 && destroyed == 0);
    retired.reset();
    CHECK(destroyed == 1 && freed == 0);
    fail_bind = true;
    CHECK(!make() && last_offset == 0 && destroyed == 1);
    fail_bind = false;
    a = make();
    CHECK(a && last_offset == 0 && allocated == 1);
    req.memoryTypeBits = 2;
    next_type = 1;
    auto other = make();
    CHECK(other && other->page != a->page && allocated == 2);
    b.reset(); c.reset(); a.reset(); other.reset();
    CHECK(freed == 2 && destroyed == 5);

    req.size = 2u << 20;
    std::vector<std::unique_ptr<leonida::PooledTitleImage>> live;
    for (int i = 0; i < 5; ++i) live.push_back(make());
    CHECK(live.front() && live.back() && allocated == 4 && last_offset == 0);
    CHECK(live[0]->page == live[3]->page && live[4]->page != live[3]->page);
    live.clear();
    CHECK(freed == 4 && destroyed == 10);
    fail_alloc = true;
    CHECK(!make() && allocated == 4 && freed == 4);
    fail_alloc = false;
    req.size++;
    CHECK(!make() && allocated == 4);
    req = {701440, 512, 2};
    a = make(); b = make();
    CHECK(a && b && a->page == b->page && last_offset % req.alignment == 0);
    CHECK(last_offset >= req.size && last_offset <= req.size + req.size / 8);
    std::puts("title image pool alignment, type separation, retirement, reuse and fallback OK");
}
