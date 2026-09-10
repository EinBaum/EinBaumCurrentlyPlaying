// Vulkan plumbing: instance/device bootstrap, swapchain, render targets, pipelines, and GPU
// resource helpers (buffers, textures, meshes).
#include "core/renderer.hpp"
#include "core/renderer_internal.hpp"
#include "core/fatal.hpp"
#include "platform/platform.hpp"
#include <algorithm>
#include <array>
#include <cstddef>
#include <cstring>
#include <format>
#include <limits>

static constexpr uint32_t mesh_vert_spv[] =
#include "shaders/mesh_vert.inl"
;
static constexpr uint32_t mesh_frag_spv[] =
#include "shaders/mesh_frag.inl"
;
static constexpr uint32_t mesh_tesc_spv[] =
#include "shaders/mesh_tesc.inl"
;
static constexpr uint32_t mesh_tese_spv[] =
#include "shaders/mesh_tese.inl"
;
static constexpr uint32_t wash_shadow_comp_spv[] =
#include "shaders/wash_shadow_comp.inl"
;
static constexpr uint32_t shadow_vert_spv[] =
#include "shaders/shadow_vert.inl"
;
static constexpr uint32_t shadow_frag_spv[] =
#include "shaders/shadow_frag.inl"
;
static constexpr uint32_t shadow_tese_spv[] =
#include "shaders/shadow_tese.inl"
;

void vkCheck(VkResult r, std::source_location loc) {
    if (r != VK_SUCCESS)
        fatal(std::format("Vulkan call failed (VkResult {}) at {}:{}", static_cast<int>(r), loc.file_name(), loc.line()));
}

namespace {

VkShaderModule makeShaderModule(VkDevice device, const uint32_t* code, size_t bytes) {
    VkShaderModuleCreateInfo ci{VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO};
    ci.codeSize = bytes;
    ci.pCode = code;
    VkShaderModule m = VK_NULL_HANDLE;
    vkCheck(vkCreateShaderModule(device, &ci, nullptr, &m));
    return m;
}

VkPipelineShaderStageCreateInfo shaderStage(VkShaderStageFlagBits stage, VkShaderModule mod) {
    VkPipelineShaderStageCreateInfo ci{VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO};
    ci.stage = stage;
    ci.module = mod;
    ci.pName = "main";
    return ci;
}

}  // namespace

void Renderer::initVulkan(PlatformWindow& window) {
    VkApplicationInfo app{VK_STRUCTURE_TYPE_APPLICATION_INFO};
    app.pApplicationName = "EinBaumCurrentlyPlaying";
    app.apiVersion = VK_API_VERSION_1_2;
    std::vector<const char*> exts = window.requiredInstanceExtensions();
    VkInstanceCreateInfo ici{VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO};
    ici.pApplicationInfo = &app;
    ici.enabledExtensionCount = static_cast<uint32_t>(exts.size());
    ici.ppEnabledExtensionNames = exts.data();
    vkCheck(vkCreateInstance(&ici, nullptr, &instance_));

    surface_ = window.createSurface(instance_);
    fallbackExtent_ = window.defaultExtent();

    uint32_t nDev = 0;
    vkEnumeratePhysicalDevices(instance_, &nDev, nullptr);
    std::vector<VkPhysicalDevice> devs(nDev);
    vkEnumeratePhysicalDevices(instance_, &nDev, devs.data());
    for (auto d : devs) {
        uint32_t nq = 0;
        vkGetPhysicalDeviceQueueFamilyProperties(d, &nq, nullptr);
        std::vector<VkQueueFamilyProperties> qf(nq);
        vkGetPhysicalDeviceQueueFamilyProperties(d, &nq, qf.data());
        for (uint32_t i = 0; i < nq; ++i) {
            VkBool32 present = VK_FALSE;
            vkGetPhysicalDeviceSurfaceSupportKHR(d, i, surface_, &present);
            if ((qf[i].queueFlags & VK_QUEUE_GRAPHICS_BIT) && present) { phys_ = d; queueFamily_ = i; break; }
        }
        if (phys_) break;
    }
    if (!phys_)
        fatal("No suitable GPU: need a Vulkan 1.2 device with a graphics+present queue.");

    float prio = 1.0f;
    VkDeviceQueueCreateInfo qci{VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO};
    qci.queueFamilyIndex = queueFamily_;
    qci.queueCount = 1;
    qci.pQueuePriorities = &prio;

    std::vector<const char*> devExts = {VK_KHR_SWAPCHAIN_EXTENSION_NAME};

    VkPhysicalDeviceFeatures feats{};
    feats.tessellationShader = VK_TRUE;

    VkDeviceCreateInfo dci{VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO};
    dci.queueCreateInfoCount = 1;
    dci.pQueueCreateInfos = &qci;
    dci.enabledExtensionCount = static_cast<uint32_t>(devExts.size());
    dci.ppEnabledExtensionNames = devExts.data();
    dci.pEnabledFeatures = &feats;
    vkCheck(vkCreateDevice(phys_, &dci, nullptr, &device_));
    vkGetDeviceQueue(device_, queueFamily_, 0, &queue_);
    pickSampleCount();

    VkCommandPoolCreateInfo pci{VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO};
    pci.flags = VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT;
    pci.queueFamilyIndex = queueFamily_;
    vkCheck(vkCreateCommandPool(device_, &pci, nullptr, &cmdPool_));

    VkSemaphoreCreateInfo semci{VK_STRUCTURE_TYPE_SEMAPHORE_CREATE_INFO};
    vkCheck(vkCreateSemaphore(device_, &semci, nullptr, &semAcquire_));
    vkCheck(vkCreateSemaphore(device_, &semci, nullptr, &semRender_));
    VkFenceCreateInfo fci{VK_STRUCTURE_TYPE_FENCE_CREATE_INFO};
    fci.flags = VK_FENCE_CREATE_SIGNALED_BIT;
    vkCheck(vkCreateFence(device_, &fci, nullptr, &fence_));

    VkSamplerCreateInfo smp{VK_STRUCTURE_TYPE_SAMPLER_CREATE_INFO};
    smp.magFilter = smp.minFilter = VK_FILTER_LINEAR;
    smp.mipmapMode = VK_SAMPLER_MIPMAP_MODE_LINEAR;
    // U mirror-repeats so a horizontal sample coordinate carried past [0,1] tiles seamlessly; V/W
    // clamp because the album quad and cover-fit wash never sample outside [0,1] vertically.
    smp.addressModeU = VK_SAMPLER_ADDRESS_MODE_MIRRORED_REPEAT;
    smp.addressModeV = smp.addressModeW = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE;
    smp.borderColor = VK_BORDER_COLOR_FLOAT_TRANSPARENT_BLACK;
    smp.maxLod = VK_LOD_CLAMP_NONE;
    vkCheck(vkCreateSampler(device_, &smp, nullptr, &sampler_));

    VkDescriptorSetLayoutBinding b{};
    b.binding = 0;
    b.descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
    b.descriptorCount = 1;
    b.stageFlags = VK_SHADER_STAGE_FRAGMENT_BIT;
    VkDescriptorSetLayoutCreateInfo dli{VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO};
    dli.bindingCount = 1;
    dli.pBindings = &b;
    vkCheck(vkCreateDescriptorSetLayout(device_, &dli, nullptr, &dsetLayout_));

    VkDescriptorPoolSize ps{VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, 16};
    VkDescriptorPoolCreateInfo dpi{VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO};
    dpi.flags = VK_DESCRIPTOR_POOL_CREATE_FREE_DESCRIPTOR_SET_BIT;
    dpi.maxSets = 16;
    dpi.poolSizeCount = 1;
    dpi.pPoolSizes = &ps;
    vkCheck(vkCreateDescriptorPool(device_, &dpi, nullptr, &dsetPool_));
}

uint32_t Renderer::findMemoryType(uint32_t typeBits, VkMemoryPropertyFlags props) {
    VkPhysicalDeviceMemoryProperties mp;
    vkGetPhysicalDeviceMemoryProperties(phys_, &mp);
    for (uint32_t i = 0; i < mp.memoryTypeCount; ++i)
        if ((typeBits & (1u << i)) && (mp.memoryTypes[i].propertyFlags & props) == props)
            return i;
    fatal("No Vulkan memory type matches the required properties.");
}

void Renderer::createBuffer(VkDeviceSize size, VkBufferUsageFlags usage,
                            VkMemoryPropertyFlags props, VkBuffer& buf, VkDeviceMemory& mem) {
    VkBufferCreateInfo bci{VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO};
    bci.size = size;
    bci.usage = usage;
    bci.sharingMode = VK_SHARING_MODE_EXCLUSIVE;
    vkCheck(vkCreateBuffer(device_, &bci, nullptr, &buf));
    VkMemoryRequirements req;
    vkGetBufferMemoryRequirements(device_, buf, &req);
    VkMemoryAllocateInfo ai{VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO};
    ai.allocationSize = req.size;
    ai.memoryTypeIndex = findMemoryType(req.memoryTypeBits, props);
    vkCheck(vkAllocateMemory(device_, &ai, nullptr, &mem));
    vkBindBufferMemory(device_, buf, mem, 0);
}

void Renderer::allocBindImageMemory(VkImage image, VkDeviceMemory& mem) {
    VkMemoryRequirements req;
    vkGetImageMemoryRequirements(device_, image, &req);
    VkMemoryAllocateInfo ai{VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO};
    ai.allocationSize = req.size;
    ai.memoryTypeIndex = findMemoryType(req.memoryTypeBits, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT);
    vkCheck(vkAllocateMemory(device_, &ai, nullptr, &mem));
    vkBindImageMemory(device_, image, mem, 0);
}

void Renderer::submitNow(const std::function<void(VkCommandBuffer)>& rec) {
    VkCommandBufferAllocateInfo ai{VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO};
    ai.commandPool = cmdPool_;
    ai.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
    ai.commandBufferCount = 1;
    VkCommandBuffer cb;
    vkAllocateCommandBuffers(device_, &ai, &cb);
    VkCommandBufferBeginInfo bi{VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO};
    bi.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
    vkBeginCommandBuffer(cb, &bi);
    rec(cb);
    vkEndCommandBuffer(cb);
    VkSubmitInfo si{VK_STRUCTURE_TYPE_SUBMIT_INFO};
    si.commandBufferCount = 1;
    si.pCommandBuffers = &cb;
    vkQueueSubmit(queue_, 1, &si, VK_NULL_HANDLE);
    vkQueueWaitIdle(queue_);
    vkFreeCommandBuffers(device_, cmdPool_, 1, &cb);
}

Texture Renderer::createTextureRGBA(const uint8_t* rgba, int w, int h, bool mips) {
    Texture t;
    uint32_t mipLevels = 1;
    if (mips) { int m = std::max(w, h); while (m > 1) { m >>= 1; mipLevels++; } }
    VkDeviceSize size = static_cast<VkDeviceSize>(w) * h * 4;

    VkBuffer staging; VkDeviceMemory stagingMem;
    createBuffer(size, VK_BUFFER_USAGE_TRANSFER_SRC_BIT,
                 VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT,
                 staging, stagingMem);
    void* p; vkMapMemory(device_, stagingMem, 0, size, 0, &p);
    std::memcpy(p, rgba, size);
    vkUnmapMemory(device_, stagingMem);

    VkImageCreateInfo ici{VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO};
    ici.imageType = VK_IMAGE_TYPE_2D;
    ici.format = VK_FORMAT_R8G8B8A8_UNORM;
    ici.extent = {static_cast<uint32_t>(w), static_cast<uint32_t>(h), 1};
    ici.mipLevels = mipLevels;
    ici.arrayLayers = 1;
    ici.samples = VK_SAMPLE_COUNT_1_BIT;
    ici.tiling = VK_IMAGE_TILING_OPTIMAL;
    ici.usage = VK_IMAGE_USAGE_TRANSFER_DST_BIT | VK_IMAGE_USAGE_SAMPLED_BIT |
                (mipLevels > 1 ? VK_IMAGE_USAGE_TRANSFER_SRC_BIT : 0);  // each level blits into the next during mip-gen
    ici.sharingMode = VK_SHARING_MODE_EXCLUSIVE;
    ici.initialLayout = VK_IMAGE_LAYOUT_UNDEFINED;
    vkCheck(vkCreateImage(device_, &ici, nullptr, &t.image));
    allocBindImageMemory(t.image, t.mem);

    VkImageViewCreateInfo vci{VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO};
    vci.image = t.image;
    vci.viewType = VK_IMAGE_VIEW_TYPE_2D;
    vci.format = VK_FORMAT_R8G8B8A8_UNORM;
    vci.subresourceRange = {VK_IMAGE_ASPECT_COLOR_BIT, 0, mipLevels, 0, 1};
    vkCheck(vkCreateImageView(device_, &vci, nullptr, &t.view));

    VkDescriptorSetAllocateInfo dsa{VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO};
    dsa.descriptorPool = dsetPool_;
    dsa.descriptorSetCount = 1;
    dsa.pSetLayouts = &dsetLayout_;
    vkCheck(vkAllocateDescriptorSets(device_, &dsa, &t.dset));
    VkDescriptorImageInfo dii{sampler_, t.view, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL};
    VkWriteDescriptorSet wds{VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET};
    wds.dstSet = t.dset;
    wds.dstBinding = 0;
    wds.descriptorCount = 1;
    wds.descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
    wds.pImageInfo = &dii;
    vkUpdateDescriptorSets(device_, 1, &wds, 0, nullptr);

    t.valid = true;

    // Queue the staging copy for the frame command buffer; init-time callers that lack a frame CB
    // will flush pendingUploads_ synchronously via submitNow before returning from init().
    pendingUploads_.push_back({staging, stagingMem, t.image, mipLevels, w, h});

    return t;
}

// Record the buffer→image copy and mip-gen barriers into the given command buffer.  The staging
// resources are moved to inFlightStagings_ and freed after the next fence wait.
void Renderer::recordTextureUpload(VkCommandBuffer cb, StagedUpload& su) {
    VkImageMemoryBarrier br{VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER};
    br.srcQueueFamilyIndex = br.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
    br.image = su.image;
    br.subresourceRange = {VK_IMAGE_ASPECT_COLOR_BIT, 0, su.mipLevels, 0, 1};
    br.oldLayout = VK_IMAGE_LAYOUT_UNDEFINED;
    br.newLayout = VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL;
    br.srcAccessMask = 0;
    br.dstAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT;
    vkCmdPipelineBarrier(cb, VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT, VK_PIPELINE_STAGE_TRANSFER_BIT,
                         0, 0, nullptr, 0, nullptr, 1, &br);
    VkBufferImageCopy region{};
    region.imageSubresource = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 0, 1};
    region.imageExtent = {static_cast<uint32_t>(su.w), static_cast<uint32_t>(su.h), 1};
    vkCmdCopyBufferToImage(cb, su.staging, su.image, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, 1, &region);

    int32_t mw = su.w, mh = su.h;
    for (uint32_t i = 1; i < su.mipLevels; ++i) {
        VkImageMemoryBarrier b{VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER};
        b.srcQueueFamilyIndex = b.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
        b.image = su.image;
        b.subresourceRange = {VK_IMAGE_ASPECT_COLOR_BIT, i - 1, 1, 0, 1};
        b.oldLayout = VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL;
        b.newLayout = VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL;
        b.srcAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT;
        b.dstAccessMask = VK_ACCESS_TRANSFER_READ_BIT;
        vkCmdPipelineBarrier(cb, VK_PIPELINE_STAGE_TRANSFER_BIT, VK_PIPELINE_STAGE_TRANSFER_BIT,
                             0, 0, nullptr, 0, nullptr, 1, &b);
        int32_t nw = mw > 1 ? mw / 2 : 1, nh = mh > 1 ? mh / 2 : 1;
        VkImageBlit blit{};
        blit.srcSubresource = {VK_IMAGE_ASPECT_COLOR_BIT, i - 1, 0, 1};
        blit.srcOffsets[1] = {mw, mh, 1};
        blit.dstSubresource = {VK_IMAGE_ASPECT_COLOR_BIT, i, 0, 1};
        blit.dstOffsets[1] = {nw, nh, 1};
        vkCmdBlitImage(cb, su.image, VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL,
                       su.image, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, 1, &blit, VK_FILTER_LINEAR);
        mw = nw; mh = nh;
    }

    // To SHADER_READ_ONLY: levels 0..n-2 are TRANSFER_SRC, the last is TRANSFER_DST.
    if (su.mipLevels > 1) {
        br.subresourceRange = {VK_IMAGE_ASPECT_COLOR_BIT, 0, su.mipLevels - 1, 0, 1};
        br.oldLayout = VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL;
        br.newLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
        br.srcAccessMask = VK_ACCESS_TRANSFER_READ_BIT;
        br.dstAccessMask = VK_ACCESS_SHADER_READ_BIT;
        vkCmdPipelineBarrier(cb, VK_PIPELINE_STAGE_TRANSFER_BIT, VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT,
                             0, 0, nullptr, 0, nullptr, 1, &br);
    }
    br.subresourceRange = {VK_IMAGE_ASPECT_COLOR_BIT, su.mipLevels - 1, 1, 0, 1};
    br.oldLayout = VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL;
    br.newLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
    br.srcAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT;
    br.dstAccessMask = VK_ACCESS_SHADER_READ_BIT;
    vkCmdPipelineBarrier(cb, VK_PIPELINE_STAGE_TRANSFER_BIT, VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT,
                         0, 0, nullptr, 0, nullptr, 1, &br);

    inFlightStagings_.push_back(su);
    su = StagedUpload{};  // ownership transferred
}

void Renderer::destroyTexture(Texture& t) {
    if (!t.valid) return;
    if (t.dset) vkFreeDescriptorSets(device_, dsetPool_, 1, &t.dset);
    if (t.view) vkDestroyImageView(device_, t.view, nullptr);
    if (t.image) vkDestroyImage(device_, t.image, nullptr);
    if (t.mem) vkFreeMemory(device_, t.mem, nullptr);
    t = Texture{};
}

// Queue a texture for destruction after the next fence wait, avoiding vkDeviceWaitIdle.
void Renderer::deferDestroyTexture(Texture& t) {
    if (!t.valid) return;
    pendingDestroyTextures_.push_back(t);
    t = Texture{};  // caller no longer owns the handles
}

// Host-visible vertex + index buffers: the meshes are tiny and static once built, so mapped
// coherent memory avoids a staging copy per glyph.
Mesh Renderer::createMesh(const std::vector<Vertex3>& verts, const std::vector<uint32_t>& indices) {
    Mesh m;
    m.indexCount = static_cast<uint32_t>(indices.size());
    m.vertexCount = static_cast<uint32_t>(verts.size());
    if (verts.empty() || indices.empty()) return m;
    VkDeviceSize vbytes = verts.size() * sizeof(Vertex3);
    VkDeviceSize ibytes = indices.size() * sizeof(uint32_t);
    const auto hostFlags = VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT;
    createBuffer(vbytes, VK_BUFFER_USAGE_VERTEX_BUFFER_BIT, hostFlags, m.vbo, m.vboMem);
    createBuffer(ibytes, VK_BUFFER_USAGE_INDEX_BUFFER_BIT, hostFlags, m.ibo, m.iboMem);
    void* p;
    vkMapMemory(device_, m.vboMem, 0, vbytes, 0, &p);
    std::memcpy(p, verts.data(), vbytes);
    vkUnmapMemory(device_, m.vboMem);
    vkMapMemory(device_, m.iboMem, 0, ibytes, 0, &p);
    std::memcpy(p, indices.data(), ibytes);
    vkUnmapMemory(device_, m.iboMem);
    return m;
}

void Renderer::destroyMesh(Mesh& m) {
    if (m.vbo) vkDestroyBuffer(device_, m.vbo, nullptr);
    if (m.ibo) vkDestroyBuffer(device_, m.ibo, nullptr);
    if (m.vboMem) vkFreeMemory(device_, m.vboMem, nullptr);
    if (m.iboMem) vkFreeMemory(device_, m.iboMem, nullptr);
    m = Mesh{};
}

// Sample count and depth format are fixed, not queried: the render pass, targets, and pipelines
// are built for exactly VK_SAMPLE_COUNT_2_BIT and D32, so a GPU lacking either crashes rather
// than falling back.
void Renderer::pickSampleCount() {
    samples_ = VK_SAMPLE_COUNT_2_BIT;
    depthFormat_ = VK_FORMAT_D32_SFLOAT;
}

void Renderer::createRenderTargets() {
    auto makeTarget = [&](VkFormat fmt, VkImageUsageFlags usage, VkImageAspectFlags aspect,
                          VkImage& img, VkDeviceMemory& mem, VkImageView& view) {
        VkImageCreateInfo ici{VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO};
        ici.imageType = VK_IMAGE_TYPE_2D;
        ici.format = fmt;
        ici.extent = {extent_.width, extent_.height, 1};
        ici.mipLevels = 1;
        ici.arrayLayers = 1;
        ici.samples = samples_;
        ici.tiling = VK_IMAGE_TILING_OPTIMAL;
        ici.usage = usage;
        ici.sharingMode = VK_SHARING_MODE_EXCLUSIVE;
        ici.initialLayout = VK_IMAGE_LAYOUT_UNDEFINED;
        vkCheck(vkCreateImage(device_, &ici, nullptr, &img));
        allocBindImageMemory(img, mem);
        VkImageViewCreateInfo vci{VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO};
        vci.image = img;
        vci.viewType = VK_IMAGE_VIEW_TYPE_2D;
        vci.format = fmt;
        vci.subresourceRange = {aspect, 0, 1, 0, 1};
        vkCheck(vkCreateImageView(device_, &vci, nullptr, &view));
    };
    makeTarget(format_, VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT | VK_IMAGE_USAGE_TRANSIENT_ATTACHMENT_BIT,
               VK_IMAGE_ASPECT_COLOR_BIT, msaaImage_, msaaMem_, msaaView_);
    makeTarget(depthFormat_, VK_IMAGE_USAGE_DEPTH_STENCIL_ATTACHMENT_BIT | VK_IMAGE_USAGE_TRANSIENT_ATTACHMENT_BIT,
               VK_IMAGE_ASPECT_DEPTH_BIT, depthImage_, depthMem_, depthView_);
}

void Renderer::destroyRenderTargets() {
    if (msaaView_) vkDestroyImageView(device_, msaaView_, nullptr);
    if (msaaImage_) vkDestroyImage(device_, msaaImage_, nullptr);
    if (msaaMem_) vkFreeMemory(device_, msaaMem_, nullptr);
    if (depthView_) vkDestroyImageView(device_, depthView_, nullptr);
    if (depthImage_) vkDestroyImage(device_, depthImage_, nullptr);
    if (depthMem_) vkFreeMemory(device_, depthMem_, nullptr);
    msaaView_ = VK_NULL_HANDLE; msaaImage_ = VK_NULL_HANDLE; msaaMem_ = VK_NULL_HANDLE;
    depthView_ = VK_NULL_HANDLE; depthImage_ = VK_NULL_HANDLE; depthMem_ = VK_NULL_HANDLE;
}

void Renderer::createSwapchain() {
    VkSurfaceCapabilitiesKHR caps;
    vkGetPhysicalDeviceSurfaceCapabilitiesKHR(phys_, surface_, &caps);
    extent_ = caps.currentExtent;
    if (extent_.width == std::numeric_limits<uint32_t>::max()) {
        // No fixed surface size (Wayland): use the window's requested client size, clamped to the
        // range the surface allows.
        extent_ = fallbackExtent_;
        extent_.width = std::clamp(extent_.width, caps.minImageExtent.width, caps.maxImageExtent.width);
        extent_.height = std::clamp(extent_.height, caps.minImageExtent.height, caps.maxImageExtent.height);
    }

    // Hardcoded, not queried: the UNORM format (not _SRGB) means shader output bytes reach the
    // surface without gamma conversion, which the chroma-key compositing requires.
    format_ = VK_FORMAT_B8G8R8A8_UNORM;
    VkColorSpaceKHR colorSpace = VK_COLOR_SPACE_SRGB_NONLINEAR_KHR;

    uint32_t imgCount = caps.minImageCount + 1;
    if (caps.maxImageCount && imgCount > caps.maxImageCount) imgCount = caps.maxImageCount;

    VkSwapchainCreateInfoKHR sci{VK_STRUCTURE_TYPE_SWAPCHAIN_CREATE_INFO_KHR};
    sci.surface = surface_;
    sci.minImageCount = imgCount;
    sci.imageFormat = format_;
    sci.imageColorSpace = colorSpace;
    sci.imageExtent = extent_;
    sci.imageArrayLayers = 1;
    sci.imageUsage = VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT;
    sci.imageSharingMode = VK_SHARING_MODE_EXCLUSIVE;
    sci.preTransform = caps.currentTransform;
    sci.compositeAlpha = VK_COMPOSITE_ALPHA_OPAQUE_BIT_KHR;
    sci.presentMode = VK_PRESENT_MODE_FIFO_KHR;
    sci.clipped = VK_TRUE;
    vkCheck(vkCreateSwapchainKHR(device_, &sci, nullptr, &swapchain_));

    uint32_t n = 0;
    vkGetSwapchainImagesKHR(device_, swapchain_, &n, nullptr);
    images_.resize(n);
    vkGetSwapchainImagesKHR(device_, swapchain_, &n, images_.data());

    createRenderTargets();

    std::array<VkAttachmentDescription, 3> att{};
    att[0].format = format_;
    att[0].samples = samples_;
    att[0].loadOp = VK_ATTACHMENT_LOAD_OP_CLEAR;
    att[0].storeOp = VK_ATTACHMENT_STORE_OP_DONT_CARE;
    att[0].stencilLoadOp = VK_ATTACHMENT_LOAD_OP_DONT_CARE;
    att[0].stencilStoreOp = VK_ATTACHMENT_STORE_OP_DONT_CARE;
    att[0].initialLayout = VK_IMAGE_LAYOUT_UNDEFINED;
    att[0].finalLayout = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL;
    att[1].format = depthFormat_;
    att[1].samples = samples_;
    att[1].loadOp = VK_ATTACHMENT_LOAD_OP_CLEAR;
    att[1].storeOp = VK_ATTACHMENT_STORE_OP_DONT_CARE;
    att[1].stencilLoadOp = VK_ATTACHMENT_LOAD_OP_DONT_CARE;
    att[1].stencilStoreOp = VK_ATTACHMENT_STORE_OP_DONT_CARE;
    att[1].initialLayout = VK_IMAGE_LAYOUT_UNDEFINED;
    att[1].finalLayout = VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL;
    att[2].format = format_;
    att[2].samples = VK_SAMPLE_COUNT_1_BIT;
    att[2].loadOp = VK_ATTACHMENT_LOAD_OP_DONT_CARE;
    att[2].storeOp = VK_ATTACHMENT_STORE_OP_STORE;
    att[2].stencilLoadOp = VK_ATTACHMENT_LOAD_OP_DONT_CARE;
    att[2].stencilStoreOp = VK_ATTACHMENT_STORE_OP_DONT_CARE;
    att[2].initialLayout = VK_IMAGE_LAYOUT_UNDEFINED;
    att[2].finalLayout = VK_IMAGE_LAYOUT_PRESENT_SRC_KHR;

    VkAttachmentReference colorRef{0, VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL};
    VkAttachmentReference depthRef{1, VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL};
    VkAttachmentReference resolveRef{2, VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL};
    VkSubpassDescription sub{};
    sub.pipelineBindPoint = VK_PIPELINE_BIND_POINT_GRAPHICS;
    sub.colorAttachmentCount = 1;
    sub.pColorAttachments = &colorRef;
    sub.pDepthStencilAttachment = &depthRef;
    sub.pResolveAttachments = &resolveRef;

    VkSubpassDependency dep{};
    dep.srcSubpass = VK_SUBPASS_EXTERNAL;
    dep.dstSubpass = 0;
    dep.srcStageMask = VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT | VK_PIPELINE_STAGE_EARLY_FRAGMENT_TESTS_BIT;
    dep.dstStageMask = VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT | VK_PIPELINE_STAGE_EARLY_FRAGMENT_TESTS_BIT;
    dep.dstAccessMask = VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT | VK_ACCESS_DEPTH_STENCIL_ATTACHMENT_WRITE_BIT;
    VkRenderPassCreateInfo rpci{VK_STRUCTURE_TYPE_RENDER_PASS_CREATE_INFO};
    rpci.attachmentCount = static_cast<uint32_t>(att.size());
    rpci.pAttachments = att.data();
    rpci.subpassCount = 1;
    rpci.pSubpasses = &sub;
    rpci.dependencyCount = 1;
    rpci.pDependencies = &dep;
    vkCheck(vkCreateRenderPass(device_, &rpci, nullptr, &renderPass_));

    views_.resize(n);
    framebuffers_.resize(n);
    for (uint32_t i = 0; i < n; ++i) {
        VkImageViewCreateInfo vci{VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO};
        vci.image = images_[i];
        vci.viewType = VK_IMAGE_VIEW_TYPE_2D;
        vci.format = format_;
        vci.subresourceRange = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1};
        vkCheck(vkCreateImageView(device_, &vci, nullptr, &views_[i]));
        std::array<VkImageView, 3> fbAtt{msaaView_, depthView_, views_[i]};
        VkFramebufferCreateInfo fbci{VK_STRUCTURE_TYPE_FRAMEBUFFER_CREATE_INFO};
        fbci.renderPass = renderPass_;
        fbci.attachmentCount = static_cast<uint32_t>(fbAtt.size());
        fbci.pAttachments = fbAtt.data();
        fbci.width = extent_.width;
        fbci.height = extent_.height;
        fbci.layers = 1;
        vkCheck(vkCreateFramebuffer(device_, &fbci, nullptr, &framebuffers_[i]));
    }

    cmds_.resize(n);
    VkCommandBufferAllocateInfo cbai{VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO};
    cbai.commandPool = cmdPool_;
    cbai.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
    cbai.commandBufferCount = n;
    vkCheck(vkAllocateCommandBuffers(device_, &cbai, cmds_.data()));

    // The half-res shadow target tracks the swapchain extent. washOccPass_ is created before the
    // first swapchain, so the occ framebuffer attaches here. writeWashShadowDescriptors no-ops on
    // the first call (compute pipeline sets do not exist yet); init writes them once the pipeline
    // is built, and every later recreate rewrites them to the new view.
    createWashShadowImage();
    writeWashShadowDescriptors();
}

void Renderer::destroySwapchain() {
    vkDeviceWaitIdle(device_);
    if (!cmds_.empty()) vkFreeCommandBuffers(device_, cmdPool_, static_cast<uint32_t>(cmds_.size()), cmds_.data());
    cmds_.clear();
    for (auto fb : framebuffers_) vkDestroyFramebuffer(device_, fb, nullptr);
    for (auto v : views_) vkDestroyImageView(device_, v, nullptr);
    destroyRenderTargets();
    destroyWashShadowImage();
    if (renderPass_) vkDestroyRenderPass(device_, renderPass_, nullptr);
    if (swapchain_) vkDestroySwapchainKHR(device_, swapchain_, nullptr);
    framebuffers_.clear(); views_.clear();
    renderPass_ = VK_NULL_HANDLE; swapchain_ = VK_NULL_HANDLE;
}

void Renderer::makeMeshPipeline() {
    std::array<VkDescriptorSetLayoutBinding, 2> camBindings{};
    camBindings[0].binding = 0;
    camBindings[0].descriptorType = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER;
    camBindings[0].descriptorCount = 1;
    // Vertex/eval read viewProj (and the shadow pass reads tip lights to project onto the card);
    // the compute pass reads the tip lights and card half-height to filter the half-res wash shadow.
    camBindings[0].stageFlags = VK_SHADER_STAGE_VERTEX_BIT | VK_SHADER_STAGE_TESSELLATION_EVALUATION_BIT |
                                VK_SHADER_STAGE_FRAGMENT_BIT | VK_SHADER_STAGE_COMPUTE_BIT;
    // binding 1: the half-res wash shadow image, sampled only by the Wash fragment path.
    camBindings[1].binding = 1;
    camBindings[1].descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
    camBindings[1].descriptorCount = 1;
    camBindings[1].stageFlags = VK_SHADER_STAGE_FRAGMENT_BIT;
    VkDescriptorSetLayoutCreateInfo cli{VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO};
    cli.bindingCount = static_cast<uint32_t>(camBindings.size());
    cli.pBindings = camBindings.data();
    vkCheck(vkCreateDescriptorSetLayout(device_, &cli, nullptr, &camLayout_));

    createBuffer(sizeof(CameraUBO), VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT,
                 VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT,
                 camUbo_, camUboMem_);
    vkMapMemory(device_, camUboMem_, 0, sizeof(CameraUBO), 0, &camUboMapped_);

    std::array<VkDescriptorPoolSize, 2> cps{{{VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER, 1},
                                             {VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, 1}}};
    VkDescriptorPoolCreateInfo cpi{VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO};
    cpi.maxSets = 1;
    cpi.poolSizeCount = static_cast<uint32_t>(cps.size());
    cpi.pPoolSizes = cps.data();
    vkCheck(vkCreateDescriptorPool(device_, &cpi, nullptr, &camPool_));
    VkDescriptorSetAllocateInfo dsa{VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO};
    dsa.descriptorPool = camPool_;
    dsa.descriptorSetCount = 1;
    dsa.pSetLayouts = &camLayout_;
    vkCheck(vkAllocateDescriptorSets(device_, &dsa, &camSet_));
    VkDescriptorBufferInfo dbi{camUbo_, 0, sizeof(CameraUBO)};
    VkWriteDescriptorSet wds{VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET};
    wds.dstSet = camSet_;
    wds.dstBinding = 0;
    wds.descriptorCount = 1;
    wds.descriptorType = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER;
    wds.pBufferInfo = &dbi;
    vkUpdateDescriptorSets(device_, 1, &wds, 0, nullptr);

    VkPushConstantRange pcr{VK_SHADER_STAGE_VERTEX_BIT | VK_SHADER_STAGE_TESSELLATION_CONTROL_BIT |
                            VK_SHADER_STAGE_TESSELLATION_EVALUATION_BIT | VK_SHADER_STAGE_FRAGMENT_BIT,
                            0, sizeof(MeshPush)};
    std::array<VkDescriptorSetLayout, 2> setLayouts{dsetLayout_, camLayout_};
    VkPipelineLayoutCreateInfo pli{VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO};
    pli.setLayoutCount = static_cast<uint32_t>(setLayouts.size());
    pli.pSetLayouts = setLayouts.data();
    pli.pushConstantRangeCount = 1;
    pli.pPushConstantRanges = &pcr;
    vkCheck(vkCreatePipelineLayout(device_, &pli, nullptr, &meshPipeLayout_));

    VkShaderModule vert = makeShaderModule(device_, mesh_vert_spv, sizeof(mesh_vert_spv));
    VkShaderModule frag = makeShaderModule(device_, mesh_frag_spv, sizeof(mesh_frag_spv));
    std::array<VkPipelineShaderStageCreateInfo, 2> stages{
        shaderStage(VK_SHADER_STAGE_VERTEX_BIT, vert),
        shaderStage(VK_SHADER_STAGE_FRAGMENT_BIT, frag),
    };

    VkVertexInputBindingDescription bind{0, sizeof(Vertex3), VK_VERTEX_INPUT_RATE_VERTEX};
    std::array<VkVertexInputAttributeDescription, 3> attrs{{
        {0, 0, VK_FORMAT_R32G32B32_SFLOAT, offsetof(Vertex3, pos)},
        {1, 0, VK_FORMAT_R32G32B32_SFLOAT, offsetof(Vertex3, normal)},
        {2, 0, VK_FORMAT_R32G32_SFLOAT, offsetof(Vertex3, uv)},
    }};
    VkPipelineVertexInputStateCreateInfo vin{VK_STRUCTURE_TYPE_PIPELINE_VERTEX_INPUT_STATE_CREATE_INFO};
    vin.vertexBindingDescriptionCount = 1;
    vin.pVertexBindingDescriptions = &bind;
    vin.vertexAttributeDescriptionCount = static_cast<uint32_t>(attrs.size());
    vin.pVertexAttributeDescriptions = attrs.data();

    VkPipelineInputAssemblyStateCreateInfo ia{VK_STRUCTURE_TYPE_PIPELINE_INPUT_ASSEMBLY_STATE_CREATE_INFO};
    ia.topology = VK_PRIMITIVE_TOPOLOGY_TRIANGLE_LIST;

    VkPipelineViewportStateCreateInfo vp{VK_STRUCTURE_TYPE_PIPELINE_VIEWPORT_STATE_CREATE_INFO};
    vp.viewportCount = 1;
    vp.scissorCount = 1;

    VkPipelineRasterizationStateCreateInfo rs{VK_STRUCTURE_TYPE_PIPELINE_RASTERIZATION_STATE_CREATE_INFO};
    rs.polygonMode = VK_POLYGON_MODE_FILL;
    // Glyphs are double-sided and lit from per-vertex normals, so both faces must rasterize.
    rs.cullMode = VK_CULL_MODE_NONE;
    rs.frontFace = VK_FRONT_FACE_COUNTER_CLOCKWISE;
    rs.lineWidth = 1.0f;

    VkPipelineMultisampleStateCreateInfo ms{VK_STRUCTURE_TYPE_PIPELINE_MULTISAMPLE_STATE_CREATE_INFO};
    ms.rasterizationSamples = samples_;

    VkPipelineDepthStencilStateCreateInfo dss{VK_STRUCTURE_TYPE_PIPELINE_DEPTH_STENCIL_STATE_CREATE_INFO};
    dss.depthTestEnable = VK_TRUE;
    dss.depthWriteEnable = VK_TRUE;
    dss.depthCompareOp = VK_COMPARE_OP_LESS_OR_EQUAL;

    VkPipelineColorBlendAttachmentState cba{};
    cba.blendEnable = VK_TRUE;
    cba.srcColorBlendFactor = VK_BLEND_FACTOR_SRC_ALPHA;
    cba.dstColorBlendFactor = VK_BLEND_FACTOR_ONE_MINUS_SRC_ALPHA;
    cba.colorBlendOp = VK_BLEND_OP_ADD;
    cba.srcAlphaBlendFactor = VK_BLEND_FACTOR_ONE;
    cba.dstAlphaBlendFactor = VK_BLEND_FACTOR_ONE_MINUS_SRC_ALPHA;
    cba.alphaBlendOp = VK_BLEND_OP_ADD;
    cba.colorWriteMask = VK_COLOR_COMPONENT_R_BIT | VK_COLOR_COMPONENT_G_BIT |
                         VK_COLOR_COMPONENT_B_BIT | VK_COLOR_COMPONENT_A_BIT;
    VkPipelineColorBlendStateCreateInfo cb{VK_STRUCTURE_TYPE_PIPELINE_COLOR_BLEND_STATE_CREATE_INFO};
    cb.attachmentCount = 1;
    cb.pAttachments = &cba;

    std::array<VkDynamicState, 2> dyn{VK_DYNAMIC_STATE_VIEWPORT, VK_DYNAMIC_STATE_SCISSOR};
    VkPipelineDynamicStateCreateInfo ds{VK_STRUCTURE_TYPE_PIPELINE_DYNAMIC_STATE_CREATE_INFO};
    ds.dynamicStateCount = static_cast<uint32_t>(dyn.size());
    ds.pDynamicStates = dyn.data();

    VkGraphicsPipelineCreateInfo gp{VK_STRUCTURE_TYPE_GRAPHICS_PIPELINE_CREATE_INFO};
    gp.stageCount = static_cast<uint32_t>(stages.size());
    gp.pStages = stages.data();
    gp.pVertexInputState = &vin;
    gp.pInputAssemblyState = &ia;
    gp.pViewportState = &vp;
    gp.pRasterizationState = &rs;
    gp.pMultisampleState = &ms;
    gp.pDepthStencilState = &dss;
    gp.pColorBlendState = &cb;
    gp.pDynamicState = &ds;
    gp.layout = meshPipeLayout_;
    gp.renderPass = renderPass_;
    gp.subpass = 0;
    vkCheck(vkCreateGraphicsPipelines(device_, VK_NULL_HANDLE, 1, &gp, nullptr, &meshPipeline_));

    VkShaderModule tesc = makeShaderModule(device_, mesh_tesc_spv, sizeof(mesh_tesc_spv));
    VkShaderModule tese = makeShaderModule(device_, mesh_tese_spv, sizeof(mesh_tese_spv));
    std::array<VkPipelineShaderStageCreateInfo, 4> tstages{
        shaderStage(VK_SHADER_STAGE_VERTEX_BIT, vert),
        shaderStage(VK_SHADER_STAGE_TESSELLATION_CONTROL_BIT, tesc),
        shaderStage(VK_SHADER_STAGE_TESSELLATION_EVALUATION_BIT, tese),
        shaderStage(VK_SHADER_STAGE_FRAGMENT_BIT, frag),
    };

    VkPipelineInputAssemblyStateCreateInfo tia{VK_STRUCTURE_TYPE_PIPELINE_INPUT_ASSEMBLY_STATE_CREATE_INFO};
    tia.topology = VK_PRIMITIVE_TOPOLOGY_PATCH_LIST;
    VkPipelineTessellationStateCreateInfo tess{VK_STRUCTURE_TYPE_PIPELINE_TESSELLATION_STATE_CREATE_INFO};
    tess.patchControlPoints = 3;

    VkGraphicsPipelineCreateInfo tgp = gp;
    tgp.stageCount = static_cast<uint32_t>(tstages.size());
    tgp.pStages = tstages.data();
    tgp.pInputAssemblyState = &tia;
    tgp.pTessellationState = &tess;
    vkCheck(vkCreateGraphicsPipelines(device_, VK_NULL_HANDLE, 1, &tgp, nullptr, &tessPipeline_));

    vkDestroyShaderModule(device_, vert, nullptr);
    vkDestroyShaderModule(device_, frag, nullptr);
    vkDestroyShaderModule(device_, tesc, nullptr);
    vkDestroyShaderModule(device_, tese, nullptr);
}

// Hard-occlusion render pass. Created before the first swapchain so createWashShadowImage can
// attach a framebuffer; the pass outlives swapchain recreates.
void Renderer::createWashOccPass() {
    VkAttachmentDescription att{};
    att.format = VK_FORMAT_R8G8B8A8_UNORM;
    att.samples = VK_SAMPLE_COUNT_1_BIT;
    att.loadOp = VK_ATTACHMENT_LOAD_OP_CLEAR;
    att.storeOp = VK_ATTACHMENT_STORE_OP_STORE;
    att.stencilLoadOp = VK_ATTACHMENT_LOAD_OP_DONT_CARE;
    att.stencilStoreOp = VK_ATTACHMENT_STORE_OP_DONT_CARE;
    att.initialLayout = VK_IMAGE_LAYOUT_UNDEFINED;
    att.finalLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
    VkAttachmentReference colorRef{0, VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL};
    VkSubpassDescription sub{};
    sub.pipelineBindPoint = VK_PIPELINE_BIND_POINT_GRAPHICS;
    sub.colorAttachmentCount = 1;
    sub.pColorAttachments = &colorRef;
    VkSubpassDependency dep{};
    dep.srcSubpass = 0;
    dep.dstSubpass = VK_SUBPASS_EXTERNAL;
    dep.srcStageMask = VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT;
    dep.dstStageMask = VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT;
    dep.srcAccessMask = VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT;
    dep.dstAccessMask = VK_ACCESS_SHADER_READ_BIT;
    VkRenderPassCreateInfo rpci{VK_STRUCTURE_TYPE_RENDER_PASS_CREATE_INFO};
    rpci.attachmentCount = 1;
    rpci.pAttachments = &att;
    rpci.subpassCount = 1;
    rpci.pSubpasses = &sub;
    rpci.dependencyCount = 1;
    rpci.pDependencies = &dep;
    vkCheck(vkCreateRenderPass(device_, &rpci, nullptr, &washOccPass_));
}

// Graphics pipelines that project occluders onto the card plane into washOcc*. Built after
// makeMeshPipeline (reuses camLayout_) and createWashOccPass.
void Renderer::makeShadowPipeline() {
    VkPushConstantRange pcr{VK_SHADER_STAGE_VERTEX_BIT | VK_SHADER_STAGE_TESSELLATION_CONTROL_BIT |
                            VK_SHADER_STAGE_TESSELLATION_EVALUATION_BIT | VK_SHADER_STAGE_FRAGMENT_BIT,
                            0, sizeof(MeshPush)};
    VkPipelineLayoutCreateInfo pli{VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO};
    pli.setLayoutCount = 1;
    pli.pSetLayouts = &camLayout_;
    pli.pushConstantRangeCount = 1;
    pli.pPushConstantRanges = &pcr;
    vkCheck(vkCreatePipelineLayout(device_, &pli, nullptr, &shadowPipeLayout_));

    VkShaderModule vert = makeShaderModule(device_, shadow_vert_spv, sizeof(shadow_vert_spv));
    VkShaderModule frag = makeShaderModule(device_, shadow_frag_spv, sizeof(shadow_frag_spv));
    std::array<VkPipelineShaderStageCreateInfo, 2> stages{
        shaderStage(VK_SHADER_STAGE_VERTEX_BIT, vert),
        shaderStage(VK_SHADER_STAGE_FRAGMENT_BIT, frag),
    };

    VkVertexInputBindingDescription bind{0, sizeof(Vertex3), VK_VERTEX_INPUT_RATE_VERTEX};
    std::array<VkVertexInputAttributeDescription, 3> attrs{{
        {0, 0, VK_FORMAT_R32G32B32_SFLOAT, offsetof(Vertex3, pos)},
        {1, 0, VK_FORMAT_R32G32B32_SFLOAT, offsetof(Vertex3, normal)},
        {2, 0, VK_FORMAT_R32G32_SFLOAT, offsetof(Vertex3, uv)},
    }};
    VkPipelineVertexInputStateCreateInfo vin{VK_STRUCTURE_TYPE_PIPELINE_VERTEX_INPUT_STATE_CREATE_INFO};
    vin.vertexBindingDescriptionCount = 1;
    vin.pVertexBindingDescriptions = &bind;
    vin.vertexAttributeDescriptionCount = static_cast<uint32_t>(attrs.size());
    vin.pVertexAttributeDescriptions = attrs.data();

    VkPipelineInputAssemblyStateCreateInfo ia{VK_STRUCTURE_TYPE_PIPELINE_INPUT_ASSEMBLY_STATE_CREATE_INFO};
    ia.topology = VK_PRIMITIVE_TOPOLOGY_TRIANGLE_LIST;

    VkPipelineViewportStateCreateInfo vp{VK_STRUCTURE_TYPE_PIPELINE_VIEWPORT_STATE_CREATE_INFO};
    vp.viewportCount = 1;
    vp.scissorCount = 1;

    VkPipelineRasterizationStateCreateInfo rs{VK_STRUCTURE_TYPE_PIPELINE_RASTERIZATION_STATE_CREATE_INFO};
    rs.polygonMode = VK_POLYGON_MODE_FILL;
    rs.cullMode = VK_CULL_MODE_NONE;
    rs.frontFace = VK_FRONT_FACE_COUNTER_CLOCKWISE;
    rs.lineWidth = 1.0f;

    VkPipelineMultisampleStateCreateInfo ms{VK_STRUCTURE_TYPE_PIPELINE_MULTISAMPLE_STATE_CREATE_INFO};
    ms.rasterizationSamples = VK_SAMPLE_COUNT_1_BIT;

    VkPipelineDepthStencilStateCreateInfo dss{VK_STRUCTURE_TYPE_PIPELINE_DEPTH_STENCIL_STATE_CREATE_INFO};

    // MIN against a clear of 1: an occluder writes (1 - fade) in its light's channel and 1
    // elsewhere, so overlapping letters keep the channel at the darkest contribution without
    // wiping the other light.
    VkPipelineColorBlendAttachmentState cba{};
    cba.blendEnable = VK_TRUE;
    cba.colorBlendOp = VK_BLEND_OP_MIN;
    cba.alphaBlendOp = VK_BLEND_OP_MIN;
    cba.srcColorBlendFactor = VK_BLEND_FACTOR_ONE;
    cba.dstColorBlendFactor = VK_BLEND_FACTOR_ONE;
    cba.srcAlphaBlendFactor = VK_BLEND_FACTOR_ONE;
    cba.dstAlphaBlendFactor = VK_BLEND_FACTOR_ONE;
    cba.colorWriteMask = VK_COLOR_COMPONENT_R_BIT | VK_COLOR_COMPONENT_G_BIT |
                         VK_COLOR_COMPONENT_B_BIT | VK_COLOR_COMPONENT_A_BIT;
    VkPipelineColorBlendStateCreateInfo cb{VK_STRUCTURE_TYPE_PIPELINE_COLOR_BLEND_STATE_CREATE_INFO};
    cb.attachmentCount = 1;
    cb.pAttachments = &cba;

    std::array<VkDynamicState, 2> dyn{VK_DYNAMIC_STATE_VIEWPORT, VK_DYNAMIC_STATE_SCISSOR};
    VkPipelineDynamicStateCreateInfo ds{VK_STRUCTURE_TYPE_PIPELINE_DYNAMIC_STATE_CREATE_INFO};
    ds.dynamicStateCount = static_cast<uint32_t>(dyn.size());
    ds.pDynamicStates = dyn.data();

    VkGraphicsPipelineCreateInfo gp{VK_STRUCTURE_TYPE_GRAPHICS_PIPELINE_CREATE_INFO};
    gp.stageCount = static_cast<uint32_t>(stages.size());
    gp.pStages = stages.data();
    gp.pVertexInputState = &vin;
    gp.pInputAssemblyState = &ia;
    gp.pViewportState = &vp;
    gp.pRasterizationState = &rs;
    gp.pMultisampleState = &ms;
    gp.pDepthStencilState = &dss;
    gp.pColorBlendState = &cb;
    gp.pDynamicState = &ds;
    gp.layout = shadowPipeLayout_;
    gp.renderPass = washOccPass_;
    gp.subpass = 0;
    vkCheck(vkCreateGraphicsPipelines(device_, VK_NULL_HANDLE, 1, &gp, nullptr, &shadowPipeline_));

    VkShaderModule tesc = makeShaderModule(device_, mesh_tesc_spv, sizeof(mesh_tesc_spv));
    VkShaderModule tese = makeShaderModule(device_, shadow_tese_spv, sizeof(shadow_tese_spv));
    std::array<VkPipelineShaderStageCreateInfo, 4> tstages{
        shaderStage(VK_SHADER_STAGE_VERTEX_BIT, vert),
        shaderStage(VK_SHADER_STAGE_TESSELLATION_CONTROL_BIT, tesc),
        shaderStage(VK_SHADER_STAGE_TESSELLATION_EVALUATION_BIT, tese),
        shaderStage(VK_SHADER_STAGE_FRAGMENT_BIT, frag),
    };
    VkPipelineInputAssemblyStateCreateInfo tia{VK_STRUCTURE_TYPE_PIPELINE_INPUT_ASSEMBLY_STATE_CREATE_INFO};
    tia.topology = VK_PRIMITIVE_TOPOLOGY_PATCH_LIST;
    VkPipelineTessellationStateCreateInfo tess{VK_STRUCTURE_TYPE_PIPELINE_TESSELLATION_STATE_CREATE_INFO};
    tess.patchControlPoints = 3;
    VkGraphicsPipelineCreateInfo tgp = gp;
    tgp.stageCount = static_cast<uint32_t>(tstages.size());
    tgp.pStages = tstages.data();
    tgp.pInputAssemblyState = &tia;
    tgp.pTessellationState = &tess;
    vkCheck(vkCreateGraphicsPipelines(device_, VK_NULL_HANDLE, 1, &tgp, nullptr, &shadowTessPipeline_));

    vkDestroyShaderModule(device_, vert, nullptr);
    vkDestroyShaderModule(device_, frag, nullptr);
    vkDestroyShaderModule(device_, tesc, nullptr);
    vkDestroyShaderModule(device_, tese, nullptr);
}

// Compute pipeline filling the half-res wash shadow image. Set 0 is the output storage image plus
// the hard-occlusion sampler; set 1 reuses the camera UBO so range matches the fragment path.
void Renderer::makeWashShadowPipeline() {
    std::array<VkDescriptorSetLayoutBinding, 2> sb{};
    sb[0].binding = 0;
    sb[0].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_IMAGE;
    sb[0].descriptorCount = 1;
    sb[0].stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;
    sb[1].binding = 1;
    sb[1].descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
    sb[1].descriptorCount = 1;
    sb[1].stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;
    VkDescriptorSetLayoutCreateInfo sli{VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO};
    sli.bindingCount = static_cast<uint32_t>(sb.size());
    sli.pBindings = sb.data();
    vkCheck(vkCreateDescriptorSetLayout(device_, &sli, nullptr, &washStoreLayout_));

    std::array<VkDescriptorPoolSize, 2> sps{{{VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, 1},
                                             {VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, 1}}};
    VkDescriptorPoolCreateInfo spi{VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO};
    spi.maxSets = 1;
    spi.poolSizeCount = static_cast<uint32_t>(sps.size());
    spi.pPoolSizes = sps.data();
    vkCheck(vkCreateDescriptorPool(device_, &spi, nullptr, &washStorePool_));
    VkDescriptorSetAllocateInfo dsa{VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO};
    dsa.descriptorPool = washStorePool_;
    dsa.descriptorSetCount = 1;
    dsa.pSetLayouts = &washStoreLayout_;
    vkCheck(vkAllocateDescriptorSets(device_, &dsa, &washStoreSet_));

    // Linear + clamp so the Wash fragment bilinearly upsamples the half-res term without wrapping
    // at the card edges.
    VkSamplerCreateInfo smp{VK_STRUCTURE_TYPE_SAMPLER_CREATE_INFO};
    smp.magFilter = smp.minFilter = VK_FILTER_LINEAR;
    smp.mipmapMode = VK_SAMPLER_MIPMAP_MODE_NEAREST;
    smp.addressModeU = smp.addressModeV = smp.addressModeW = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE;
    vkCheck(vkCreateSampler(device_, &smp, nullptr, &washSampler_));

    // PCF taps that walk off the card must read fully-lit, not a clamped edge of an occluder.
    VkSamplerCreateInfo osmp = smp;
    osmp.addressModeU = osmp.addressModeV = osmp.addressModeW = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_BORDER;
    osmp.borderColor = VK_BORDER_COLOR_FLOAT_OPAQUE_WHITE;
    vkCheck(vkCreateSampler(device_, &osmp, nullptr, &washOccSampler_));

    std::array<VkDescriptorSetLayout, 2> setLayouts{washStoreLayout_, camLayout_};
    VkPipelineLayoutCreateInfo pli{VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO};
    pli.setLayoutCount = static_cast<uint32_t>(setLayouts.size());
    pli.pSetLayouts = setLayouts.data();
    vkCheck(vkCreatePipelineLayout(device_, &pli, nullptr, &washShadowPipeLayout_));

    VkShaderModule comp = makeShaderModule(device_, wash_shadow_comp_spv, sizeof(wash_shadow_comp_spv));
    VkComputePipelineCreateInfo cpci{VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO};
    cpci.stage = shaderStage(VK_SHADER_STAGE_COMPUTE_BIT, comp);
    cpci.layout = washShadowPipeLayout_;
    vkCheck(vkCreateComputePipelines(device_, VK_NULL_HANDLE, 1, &cpci, nullptr, &washShadowPipeline_));
    vkDestroyShaderModule(device_, comp, nullptr);
}

void Renderer::createWashOccFramebuffer() {
    if (!washOccPass_ || !washOccView_) return;
    if (washOccFb_) {
        vkDestroyFramebuffer(device_, washOccFb_, nullptr);
        washOccFb_ = VK_NULL_HANDLE;
    }
    VkFramebufferCreateInfo fbci{VK_STRUCTURE_TYPE_FRAMEBUFFER_CREATE_INFO};
    fbci.renderPass = washOccPass_;
    fbci.attachmentCount = 1;
    fbci.pAttachments = &washOccView_;
    fbci.width = washShadowExtent_.width;
    fbci.height = washShadowExtent_.height;
    fbci.layers = 1;
    vkCheck(vkCreateFramebuffer(device_, &fbci, nullptr, &washOccFb_));
}

void Renderer::createWashShadowImage() {
    washShadowExtent_ = {(extent_.width + 1) / 2, (extent_.height + 1) / 2};

    auto makeImg = [&](VkFormat fmt, VkImageUsageFlags usage, VkImage& img, VkDeviceMemory& mem, VkImageView& view) {
        VkImageCreateInfo ici{VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO};
        ici.imageType = VK_IMAGE_TYPE_2D;
        ici.format = fmt;
        ici.extent = {washShadowExtent_.width, washShadowExtent_.height, 1};
        ici.mipLevels = 1;
        ici.arrayLayers = 1;
        ici.samples = VK_SAMPLE_COUNT_1_BIT;
        ici.tiling = VK_IMAGE_TILING_OPTIMAL;
        ici.usage = usage;
        ici.sharingMode = VK_SHARING_MODE_EXCLUSIVE;
        ici.initialLayout = VK_IMAGE_LAYOUT_UNDEFINED;
        vkCheck(vkCreateImage(device_, &ici, nullptr, &img));
        allocBindImageMemory(img, mem);
        VkImageViewCreateInfo vci{VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO};
        vci.image = img;
        vci.viewType = VK_IMAGE_VIEW_TYPE_2D;
        vci.format = fmt;
        vci.subresourceRange = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1};
        vkCheck(vkCreateImageView(device_, &vci, nullptr, &view));
    };
    makeImg(VK_FORMAT_R16G16B16A16_SFLOAT, VK_IMAGE_USAGE_STORAGE_BIT | VK_IMAGE_USAGE_SAMPLED_BIT,
            washShadowImage_, washShadowMem_, washShadowView_);
    makeImg(VK_FORMAT_R8G8B8A8_UNORM, VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT | VK_IMAGE_USAGE_SAMPLED_BIT,
            washOccImage_, washOccMem_, washOccView_);

    // Soft target to GENERAL once: the compute pass writes it and the fragment samples it, both
    // from GENERAL, so the per-frame command buffer only needs the compute->fragment memory barrier.
    // The occ target is transitioned by its render pass (UNDEFINED -> COLOR -> SHADER_READ).
    submitNow([&](VkCommandBuffer cb) {
        VkImageMemoryBarrier br{VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER};
        br.srcQueueFamilyIndex = br.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
        br.image = washShadowImage_;
        br.subresourceRange = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1};
        br.oldLayout = VK_IMAGE_LAYOUT_UNDEFINED;
        br.newLayout = VK_IMAGE_LAYOUT_GENERAL;
        vkCmdPipelineBarrier(cb, VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
                             0, 0, nullptr, 0, nullptr, 1, &br);
    });
    createWashOccFramebuffer();
}

void Renderer::destroyWashShadowImage() {
    if (washOccFb_) vkDestroyFramebuffer(device_, washOccFb_, nullptr);
    if (washOccView_) vkDestroyImageView(device_, washOccView_, nullptr);
    if (washOccImage_) vkDestroyImage(device_, washOccImage_, nullptr);
    if (washOccMem_) vkFreeMemory(device_, washOccMem_, nullptr);
    washOccFb_ = VK_NULL_HANDLE; washOccView_ = VK_NULL_HANDLE;
    washOccImage_ = VK_NULL_HANDLE; washOccMem_ = VK_NULL_HANDLE;
    if (washShadowView_) vkDestroyImageView(device_, washShadowView_, nullptr);
    if (washShadowImage_) vkDestroyImage(device_, washShadowImage_, nullptr);
    if (washShadowMem_) vkFreeMemory(device_, washShadowMem_, nullptr);
    washShadowView_ = VK_NULL_HANDLE; washShadowImage_ = VK_NULL_HANDLE; washShadowMem_ = VK_NULL_HANDLE;
}

void Renderer::writeWashShadowDescriptors() {
    if (!camSet_ || !washStoreSet_ || !washShadowView_ || !washOccView_ || !washOccSampler_) return;
    VkDescriptorImageInfo store{VK_NULL_HANDLE, washShadowView_, VK_IMAGE_LAYOUT_GENERAL};
    VkDescriptorImageInfo samp{washSampler_, washShadowView_, VK_IMAGE_LAYOUT_GENERAL};
    VkDescriptorImageInfo occ{washOccSampler_, washOccView_, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL};
    std::array<VkWriteDescriptorSet, 3> w{};
    w[0].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
    w[0].dstSet = washStoreSet_;
    w[0].dstBinding = 0;
    w[0].descriptorCount = 1;
    w[0].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_IMAGE;
    w[0].pImageInfo = &store;
    w[1].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
    w[1].dstSet = camSet_;
    w[1].dstBinding = 1;
    w[1].descriptorCount = 1;
    w[1].descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
    w[1].pImageInfo = &samp;
    w[2].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
    w[2].dstSet = washStoreSet_;
    w[2].dstBinding = 1;
    w[2].descriptorCount = 1;
    w[2].descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
    w[2].pImageInfo = &occ;
    vkUpdateDescriptorSets(device_, static_cast<uint32_t>(w.size()), w.data(), 0, nullptr);
}
