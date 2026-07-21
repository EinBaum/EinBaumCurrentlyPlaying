// Acceleration structures for the ray-traced shadows: per-mesh BLAS builds (batched per track
// change) and the per-frame scene TLAS.
#include "core/renderer.hpp"
#include "core/renderer_internal.hpp"
#include <algorithm>
#include <cstring>

VkDeviceAddress Renderer::bufferAddr(VkBuffer b) const {
    VkBufferDeviceAddressInfo info{VK_STRUCTURE_TYPE_BUFFER_DEVICE_ADDRESS_INFO};
    info.buffer = b;
    return pfnGetBufferDeviceAddress_(device_, &info);
}

void Renderer::loadRayTracingFns() {
    auto load = [&](const char* nm) { return vkGetDeviceProcAddr(device_, nm); };
    pfnGetBufferDeviceAddress_ = reinterpret_cast<PFN_vkGetBufferDeviceAddressKHR>(load("vkGetBufferDeviceAddress"));
    pfnCreateAS_ = reinterpret_cast<PFN_vkCreateAccelerationStructureKHR>(load("vkCreateAccelerationStructureKHR"));
    pfnDestroyAS_ = reinterpret_cast<PFN_vkDestroyAccelerationStructureKHR>(load("vkDestroyAccelerationStructureKHR"));
    pfnGetASBuildSizes_ = reinterpret_cast<PFN_vkGetAccelerationStructureBuildSizesKHR>(load("vkGetAccelerationStructureBuildSizesKHR"));
    pfnCmdBuildAS_ = reinterpret_cast<PFN_vkCmdBuildAccelerationStructuresKHR>(load("vkCmdBuildAccelerationStructuresKHR"));
    pfnGetASDeviceAddr_ = reinterpret_cast<PFN_vkGetAccelerationStructureDeviceAddressKHR>(load("vkGetAccelerationStructureDeviceAddressKHR"));
}

void Renderer::prepareMeshBlas(Mesh& m) {
    if (m.blas || m.indexCount == 0) return;
    PendingBlas p;
    p.geom = {VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_GEOMETRY_KHR};
    p.geom.geometryType = VK_GEOMETRY_TYPE_TRIANGLES_KHR;
    p.geom.flags = VK_GEOMETRY_OPAQUE_BIT_KHR;
    auto& tri = p.geom.geometry.triangles;
    tri.sType = VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_GEOMETRY_TRIANGLES_DATA_KHR;
    tri.vertexFormat = VK_FORMAT_R32G32B32_SFLOAT;
    tri.vertexData.deviceAddress = bufferAddr(m.vbo);
    tri.vertexStride = sizeof(Vertex3);
    tri.maxVertex = m.vertexCount - 1;
    tri.indexType = VK_INDEX_TYPE_UINT32;
    tri.indexData.deviceAddress = bufferAddr(m.ibo);

    p.bgi = {VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_BUILD_GEOMETRY_INFO_KHR};
    p.bgi.type = VK_ACCELERATION_STRUCTURE_TYPE_BOTTOM_LEVEL_KHR;
    p.bgi.flags = VK_BUILD_ACCELERATION_STRUCTURE_PREFER_FAST_TRACE_BIT_KHR;
    p.bgi.mode = VK_BUILD_ACCELERATION_STRUCTURE_MODE_BUILD_KHR;
    p.bgi.geometryCount = 1;
    p.bgi.pGeometries = &p.geom;

    p.primCount = m.indexCount / 3;
    VkAccelerationStructureBuildSizesInfoKHR sizes{VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_BUILD_SIZES_INFO_KHR};
    pfnGetASBuildSizes_(device_, VK_ACCELERATION_STRUCTURE_BUILD_TYPE_DEVICE_KHR, &p.bgi, &p.primCount, &sizes);

    createBuffer(sizes.accelerationStructureSize,
                 VK_BUFFER_USAGE_ACCELERATION_STRUCTURE_STORAGE_BIT_KHR | VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT,
                 VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT, m.blasBuf, m.blasMem);
    VkAccelerationStructureCreateInfoKHR aci{VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_CREATE_INFO_KHR};
    aci.buffer = m.blasBuf;
    aci.size = sizes.accelerationStructureSize;
    aci.type = VK_ACCELERATION_STRUCTURE_TYPE_BOTTOM_LEVEL_KHR;
    vkCheck(pfnCreateAS_(device_, &aci, nullptr, &m.blas));

    createBuffer(sizes.buildScratchSize,
                 VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT,
                 VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT, p.scratch, p.scratchMem);
    p.bgi.dstAccelerationStructure = m.blas;
    p.bgi.scratchData.deviceAddress = bufferAddr(p.scratch);

    VkAccelerationStructureDeviceAddressInfoKHR adi{VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_DEVICE_ADDRESS_INFO_KHR};
    adi.accelerationStructure = m.blas;
    m.blasAddr = pfnGetASDeviceAddr_(device_, &adi);

    pendingBlas_.push_back(p);
}

void Renderer::flushPendingBlas() {
    if (pendingBlas_.empty()) return;
    const uint32_t n = static_cast<uint32_t>(pendingBlas_.size());
    std::vector<VkAccelerationStructureBuildGeometryInfoKHR> bgis;
    std::vector<VkAccelerationStructureBuildRangeInfoKHR> ranges;
    std::vector<const VkAccelerationStructureBuildRangeInfoKHR*> pRanges;
    bgis.reserve(n); ranges.reserve(n); pRanges.reserve(n);
    for (PendingBlas& p : pendingBlas_) {
        p.bgi.pGeometries = &p.geom;   // pushing relocated pendingBlas_; re-point before recording
        bgis.push_back(p.bgi);
        ranges.push_back({.primitiveCount = p.primCount});
    }
    for (const VkAccelerationStructureBuildRangeInfoKHR& r : ranges) pRanges.push_back(&r);

    submitNow([&](VkCommandBuffer cb) { pfnCmdBuildAS_(cb, n, bgis.data(), pRanges.data()); });

    for (PendingBlas& p : pendingBlas_) {
        vkDestroyBuffer(device_, p.scratch, nullptr);
        vkFreeMemory(device_, p.scratchMem, nullptr);
    }
    pendingBlas_.clear();
}

void Renderer::ensureMeshBlas(Mesh& m) {
    prepareMeshBlas(m);
    flushPendingBlas();
}

void Renderer::initSceneTlas() {
    VkDescriptorSetLayoutBinding asb{};
    asb.binding = 0;
    asb.descriptorType = VK_DESCRIPTOR_TYPE_ACCELERATION_STRUCTURE_KHR;
    asb.descriptorCount = 1;
    // The half-res shadow compute pass traces against this TLAS; the graphics pipeline keeps it bound too.
    asb.stageFlags = VK_SHADER_STAGE_FRAGMENT_BIT | VK_SHADER_STAGE_COMPUTE_BIT;
    VkDescriptorSetLayoutCreateInfo dli{VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO};
    dli.bindingCount = 1;
    dli.pBindings = &asb;
    vkCheck(vkCreateDescriptorSetLayout(device_, &dli, nullptr, &sceneAsLayout_));

    VkDescriptorPoolSize ps{VK_DESCRIPTOR_TYPE_ACCELERATION_STRUCTURE_KHR, 1};
    VkDescriptorPoolCreateInfo dpi{VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO};
    dpi.maxSets = 1;
    dpi.poolSizeCount = 1;
    dpi.pPoolSizes = &ps;
    vkCheck(vkCreateDescriptorPool(device_, &dpi, nullptr, &sceneAsPool_));
    VkDescriptorSetAllocateInfo dsa{VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO};
    dsa.descriptorPool = sceneAsPool_;
    dsa.descriptorSetCount = 1;
    dsa.pSetLayouts = &sceneAsLayout_;
    vkCheck(vkAllocateDescriptorSets(device_, &dsa, &sceneAsSet_));

    const VkDeviceSize instBytes = kMaxSceneInstances * sizeof(VkAccelerationStructureInstanceKHR);
    createBuffer(instBytes,
                 VK_BUFFER_USAGE_ACCELERATION_STRUCTURE_BUILD_INPUT_READ_ONLY_BIT_KHR |
                 VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT,
                 VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT,
                 sceneInstBuf_, sceneInstMem_);
    vkMapMemory(device_, sceneInstMem_, 0, instBytes, 0, &sceneInstMapped_);

    VkAccelerationStructureGeometryKHR tgeom{VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_GEOMETRY_KHR};
    tgeom.geometryType = VK_GEOMETRY_TYPE_INSTANCES_KHR;
    tgeom.flags = VK_GEOMETRY_OPAQUE_BIT_KHR;
    tgeom.geometry.instances.sType = VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_GEOMETRY_INSTANCES_DATA_KHR;
    tgeom.geometry.instances.data.deviceAddress = bufferAddr(sceneInstBuf_);

    VkAccelerationStructureBuildGeometryInfoKHR tbgi{VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_BUILD_GEOMETRY_INFO_KHR};
    tbgi.type = VK_ACCELERATION_STRUCTURE_TYPE_TOP_LEVEL_KHR;
    tbgi.flags = VK_BUILD_ACCELERATION_STRUCTURE_PREFER_FAST_TRACE_BIT_KHR;
    tbgi.mode = VK_BUILD_ACCELERATION_STRUCTURE_MODE_BUILD_KHR;
    tbgi.geometryCount = 1;
    tbgi.pGeometries = &tgeom;

    const uint32_t maxPrim = kMaxSceneInstances;
    VkAccelerationStructureBuildSizesInfoKHR tsizes{VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_BUILD_SIZES_INFO_KHR};
    pfnGetASBuildSizes_(device_, VK_ACCELERATION_STRUCTURE_BUILD_TYPE_DEVICE_KHR, &tbgi, &maxPrim, &tsizes);

    createBuffer(tsizes.accelerationStructureSize,
                 VK_BUFFER_USAGE_ACCELERATION_STRUCTURE_STORAGE_BIT_KHR | VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT,
                 VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT, sceneTlasBuf_, sceneTlasMem_);
    VkAccelerationStructureCreateInfoKHR taci{VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_CREATE_INFO_KHR};
    taci.buffer = sceneTlasBuf_;
    taci.size = tsizes.accelerationStructureSize;
    taci.type = VK_ACCELERATION_STRUCTURE_TYPE_TOP_LEVEL_KHR;
    vkCheck(pfnCreateAS_(device_, &taci, nullptr, &sceneTlas_));

    createBuffer(tsizes.buildScratchSize,
                 VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT,
                 VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT, sceneScratchBuf_, sceneScratchMem_);
    sceneScratchAddr_ = bufferAddr(sceneScratchBuf_);

    VkWriteDescriptorSetAccelerationStructureKHR wasi{VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET_ACCELERATION_STRUCTURE_KHR};
    wasi.accelerationStructureCount = 1;
    wasi.pAccelerationStructures = &sceneTlas_;
    VkWriteDescriptorSet wds{VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET};
    wds.pNext = &wasi;
    wds.dstSet = sceneAsSet_;
    wds.dstBinding = 0;
    wds.descriptorCount = 1;
    wds.descriptorType = VK_DESCRIPTOR_TYPE_ACCELERATION_STRUCTURE_KHR;
    vkUpdateDescriptorSets(device_, 1, &wds, 0, nullptr);
}

void Renderer::buildSceneTlas(VkCommandBuffer cb, const std::vector<VkAccelerationStructureInstanceKHR>& insts) {
    const uint32_t n = static_cast<uint32_t>(std::min<size_t>(insts.size(), kMaxSceneInstances));
    if (n) std::memcpy(sceneInstMapped_, insts.data(), n * sizeof(VkAccelerationStructureInstanceKHR));

    VkAccelerationStructureGeometryKHR tgeom{VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_GEOMETRY_KHR};
    tgeom.geometryType = VK_GEOMETRY_TYPE_INSTANCES_KHR;
    tgeom.flags = VK_GEOMETRY_OPAQUE_BIT_KHR;
    tgeom.geometry.instances.sType = VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_GEOMETRY_INSTANCES_DATA_KHR;
    tgeom.geometry.instances.data.deviceAddress = bufferAddr(sceneInstBuf_);

    VkAccelerationStructureBuildGeometryInfoKHR tbgi{VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_BUILD_GEOMETRY_INFO_KHR};
    tbgi.type = VK_ACCELERATION_STRUCTURE_TYPE_TOP_LEVEL_KHR;
    tbgi.flags = VK_BUILD_ACCELERATION_STRUCTURE_PREFER_FAST_TRACE_BIT_KHR;
    tbgi.mode = VK_BUILD_ACCELERATION_STRUCTURE_MODE_BUILD_KHR;
    tbgi.geometryCount = 1;
    tbgi.pGeometries = &tgeom;
    tbgi.dstAccelerationStructure = sceneTlas_;
    tbgi.scratchData.deviceAddress = sceneScratchAddr_;

    VkAccelerationStructureBuildRangeInfoKHR range{.primitiveCount = n};
    const VkAccelerationStructureBuildRangeInfoKHR* pRange = &range;
    pfnCmdBuildAS_(cb, 1, &tbgi, &pRange);

    VkMemoryBarrier mb{VK_STRUCTURE_TYPE_MEMORY_BARRIER};
    mb.srcAccessMask = VK_ACCESS_ACCELERATION_STRUCTURE_WRITE_BIT_KHR;
    mb.dstAccessMask = VK_ACCESS_ACCELERATION_STRUCTURE_READ_BIT_KHR;
    vkCmdPipelineBarrier(cb, VK_PIPELINE_STAGE_ACCELERATION_STRUCTURE_BUILD_BIT_KHR,
                         VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT | VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT,
                         0, 1, &mb, 0, nullptr, 0, nullptr);
}
