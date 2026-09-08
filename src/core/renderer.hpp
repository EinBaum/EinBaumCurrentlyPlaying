#pragma once
#include <vulkan/vulkan.h>
#include <array>
#include <functional>
#include <map>
#include <memory>
#include <utility>
#include <vector>
#include "core/art_decoder.hpp"
#include "core/font.hpp"
#include "core/math3d.hpp"
#include "core/theme.hpp"
#include "core/track.hpp"

class PlatformWindow;

struct Texture {
    VkImage image = VK_NULL_HANDLE;
    VkDeviceMemory mem = VK_NULL_HANDLE;
    VkImageView view = VK_NULL_HANDLE;
    VkDescriptorSet dset = VK_NULL_HANDLE;
    bool valid = false;
};

// A staging buffer and mip count recorded by createTextureRGBA but not yet submitted to the GPU.
// recordTextureUpload() records the copy + mip-gen into a command buffer; the staging resources
// are freed after the next fence wait.
struct StagedUpload {
    VkBuffer staging = VK_NULL_HANDLE;
    VkDeviceMemory stagingMem = VK_NULL_HANDLE;
    VkImage image = VK_NULL_HANDLE;
    uint32_t mipLevels = 1;
    int w = 0, h = 0;
};

struct Mesh {
    VkBuffer vbo = VK_NULL_HANDLE, ibo = VK_NULL_HANDLE;
    VkDeviceMemory vboMem = VK_NULL_HANDLE, iboMem = VK_NULL_HANDLE;
    uint32_t indexCount = 0;
    uint32_t vertexCount = 0;
    VkAccelerationStructureKHR blas = VK_NULL_HANDLE;
    VkBuffer blasBuf = VK_NULL_HANDLE;
    VkDeviceMemory blasMem = VK_NULL_HANDLE;
    VkDeviceAddress blasAddr = 0;
};

struct GlyphInstance {
    const Mesh* mesh = nullptr;
    vec3 pos;            // baseline-left origin, world units
    float scale = 0.0f;  // em size in world units
    vec4 color;
};

struct GpuGlyph {
    const Mesh* mesh = nullptr;   // null for whitespace
    float advanceEm = 0.0f;       // pen advance, em units
};

enum class Line : uint8_t { Title, Artist, Count };

// Scroll state for one text line too long to fit its column; start is latched on the first draw
// frame via armed.
struct Marquee {
    bool active = false;
    bool armed = false;
    // Outgoing burning copy: it holds at the scroll offset the burn caught it at, column-clipped.
    struct Burn {
        bool on = false;
        float shift = 0.0f;   // frozen scroll offset, world units
        float period = 0.0f;  // width + gap at freeze time, world units
    } burn;
    double start = 0.0;    // steady-clock seconds the scroll is measured from
    float width = 0.0f;    // laid line width, world units
    float gap = 0.0f;      // blank gap before the wrapped repeat, world units
    float speed = 0.0f;    // world units per second
};

// Top-level on-screen lifecycle. Pause and seek modulate Active rather than being states here;
// Transitioning is the song-to-song cross-dissolve, mutually exclusive with NoMedia.
enum class PlaybackState : uint8_t { NoMedia, Active, Transitioning };

class Renderer {
public:
    void init(PlatformWindow& window);
    void shutdown();
    void setTrack(const Track& t);
    void refreshArt(const Track& t);
    void draw(const Track& t, double nowSteady);
    void setShowFps(bool v) { showFps_ = v; }
    [[nodiscard]] bool animating() const {
        return state_ == PlaybackState::Transitioning || seekGliding_ || playFading_ || artWait_.pending;
    }
    [[nodiscard]] bool textScrolling() const {
        return mq(Line::Title).active || mq(Line::Artist).active;
    }

private:
    // --- Vulkan device, swapchain, and frame plumbing (renderer_vk.cpp) ---
    VkInstance instance_ = VK_NULL_HANDLE;
    VkSurfaceKHR surface_ = VK_NULL_HANDLE;
    VkPhysicalDevice phys_ = VK_NULL_HANDLE;
    uint32_t queueFamily_ = 0;
    VkDevice device_ = VK_NULL_HANDLE;
    VkQueue queue_ = VK_NULL_HANDLE;
    VkSwapchainKHR swapchain_ = VK_NULL_HANDLE;
    VkFormat format_ = VK_FORMAT_B8G8R8A8_UNORM;
    VkExtent2D extent_{};
    // Client size requested by the platform window, used when the surface reports no fixed
    // currentExtent (Wayland lets the client choose; Win32 pins it to the HWND).
    VkExtent2D fallbackExtent_{};
    std::vector<VkImage> images_;
    std::vector<VkImageView> views_;
    std::vector<VkFramebuffer> framebuffers_;
    VkRenderPass renderPass_ = VK_NULL_HANDLE;
    VkCommandPool cmdPool_ = VK_NULL_HANDLE;
    std::vector<VkCommandBuffer> cmds_;
    VkSemaphore semAcquire_ = VK_NULL_HANDLE, semRender_ = VK_NULL_HANDLE;
    VkFence fence_ = VK_NULL_HANDLE;

    // MSAA color + depth targets (shared across framebuffers; one frame in flight). The MSAA color
    // resolves into the swapchain image; depth is multisampled and discarded.
    VkSampleCountFlagBits samples_ = VK_SAMPLE_COUNT_1_BIT;
    VkFormat depthFormat_ = VK_FORMAT_D32_SFLOAT;
    VkImage msaaImage_ = VK_NULL_HANDLE, depthImage_ = VK_NULL_HANDLE;
    VkDeviceMemory msaaMem_ = VK_NULL_HANDLE, depthMem_ = VK_NULL_HANDLE;
    VkImageView msaaView_ = VK_NULL_HANDLE, depthView_ = VK_NULL_HANDLE;

    VkDescriptorSetLayout dsetLayout_ = VK_NULL_HANDLE;  // set 0: combined image sampler
    VkDescriptorPool dsetPool_ = VK_NULL_HANDLE;
    VkSampler sampler_ = VK_NULL_HANDLE;

    VkPipeline meshPipeline_ = VK_NULL_HANDLE;
    // Tessellated variant (PATCH_LIST + TCS/TES) for the progress-bar rod; shares meshPipeLayout_
    // and the vertex/fragment stages.
    VkPipeline tessPipeline_ = VK_NULL_HANDLE;
    VkPipelineLayout meshPipeLayout_ = VK_NULL_HANDLE;
    VkDescriptorSetLayout camLayout_ = VK_NULL_HANDLE;   // set 1: camera UBO
    VkDescriptorPool camPool_ = VK_NULL_HANDLE;
    VkDescriptorSet camSet_ = VK_NULL_HANDLE;
    VkBuffer camUbo_ = VK_NULL_HANDLE;
    VkDeviceMemory camUboMem_ = VK_NULL_HANDLE;
    void* camUboMapped_ = nullptr;

    // Half-resolution wash shadow term from wash_shadow.comp (R = key, G/B = tip lights). Half the
    // swapchain extent, rebuilt on resize; stays in GENERAL layout (compute writes, fragment samples).
    VkPipeline washShadowPipeline_ = VK_NULL_HANDLE;
    VkPipelineLayout washShadowPipeLayout_ = VK_NULL_HANDLE;
    VkDescriptorSetLayout washStoreLayout_ = VK_NULL_HANDLE;  // compute set 0: the output storage image
    VkDescriptorPool washStorePool_ = VK_NULL_HANDLE;
    VkDescriptorSet washStoreSet_ = VK_NULL_HANDLE;
    VkSampler washSampler_ = VK_NULL_HANDLE;
    VkImage washShadowImage_ = VK_NULL_HANDLE;
    VkDeviceMemory washShadowMem_ = VK_NULL_HANDLE;
    VkImageView washShadowView_ = VK_NULL_HANDLE;
    VkExtent2D washShadowExtent_{};

    // --- Ray-traced shadow scene (renderer_raytracing.cpp) ---
    // Scene acceleration structure (set 2), rebuilt each frame from the drawn occluder meshes; the
    // TLAS/scratch/instance buffers are sized once for the worst-case instance count.
    static constexpr uint32_t kMaxSceneInstances = 1024;
    VkDescriptorSetLayout sceneAsLayout_ = VK_NULL_HANDLE;  // set 2: scene TLAS
    VkDescriptorPool sceneAsPool_ = VK_NULL_HANDLE;
    VkDescriptorSet sceneAsSet_ = VK_NULL_HANDLE;
    VkAccelerationStructureKHR sceneTlas_ = VK_NULL_HANDLE;
    VkBuffer sceneTlasBuf_ = VK_NULL_HANDLE, sceneInstBuf_ = VK_NULL_HANDLE, sceneScratchBuf_ = VK_NULL_HANDLE;
    VkDeviceMemory sceneTlasMem_ = VK_NULL_HANDLE, sceneInstMem_ = VK_NULL_HANDLE, sceneScratchMem_ = VK_NULL_HANDLE;
    void* sceneInstMapped_ = nullptr;
    VkDeviceAddress sceneScratchAddr_ = 0;
    // Occluder instances from the last TLAS build; an identical set next frame skips the rebuild.
    // tlasBuilt_ forces the first build, since initSceneTlas creates the TLAS without building it.
    std::vector<VkAccelerationStructureInstanceKHR> lastBuiltInsts_;
    bool tlasBuilt_ = false;

    // First-seen glyph BLAS builds queued during layout and recorded in one submit by flushPendingBlas.
    // bgi.pGeometries must be re-pointed at geom at flush: pushing relocates the vector.
    struct PendingBlas {
        VkAccelerationStructureGeometryKHR geom{};
        VkAccelerationStructureBuildGeometryInfoKHR bgi{};
        uint32_t primCount = 0;
        VkBuffer scratch = VK_NULL_HANDLE;
        VkDeviceMemory scratchMem = VK_NULL_HANDLE;
    };
    std::vector<PendingBlas> pendingBlas_;
    // Scratch from flushPendingBlas(cb); freed after the frame fence (one in-flight frame).
    struct GpuScratch {
        VkBuffer buf = VK_NULL_HANDLE;
        VkDeviceMemory mem = VK_NULL_HANDLE;
    };
    std::vector<GpuScratch> inFlightBlasScratch_;

    // VK_KHR_acceleration_structure entry points, resolved at runtime by loadRayTracingFns.
    PFN_vkGetBufferDeviceAddressKHR pfnGetBufferDeviceAddress_ = nullptr;
    PFN_vkCreateAccelerationStructureKHR pfnCreateAS_ = nullptr;
    PFN_vkDestroyAccelerationStructureKHR pfnDestroyAS_ = nullptr;
    PFN_vkGetAccelerationStructureBuildSizesKHR pfnGetASBuildSizes_ = nullptr;
    PFN_vkCmdBuildAccelerationStructuresKHR pfnCmdBuildAS_ = nullptr;
    PFN_vkGetAccelerationStructureDeviceAddressKHR pfnGetASDeviceAddr_ = nullptr;

    // --- Scene resources and per-track state (renderer.cpp) ---
    Mesh unitQuad_;   // [0,1]x[0,1] quad
    Mesh barRod_;     // unit cylinder: x in [0,1], radius 1
    // Keyed by (FontFace::handle, codepoint). glyphCache_ holds the uploaded GPU mesh; advanceCache_
    // holds every glyph's advance (including whitespace, which has no mesh), so an outline is
    // fetched and tessellated exactly once per face for the process lifetime.
    std::map<std::pair<const void*, uint32_t>, Mesh> glyphCache_;
    std::map<std::pair<const void*, uint32_t>, float> advanceCache_;

    Texture white_, album_;
    std::unique_ptr<FontFace> fontTitle_, fontTitleSmall_, fontArtist_, fontFps_;

    std::vector<GlyphInstance> titleGlyphs_, artistGlyphs_;

    RGBA albumAccent_{};
    bool hasAlbumAccent_ = false;
    float washDim_ = 1.0f;          // current cover's wash darkening factor (see computeAccent)

    // Cover decode runs on artDecoder_'s thread; draw() uploads the result. seq tags the newest
    // submit, pending holds until that result lands, reserved keeps the text column and bar sized
    // for the cover while it decodes (known from the art bytes alone, before any result).
    struct ArtWait { uint64_t seq = 0; bool pending = false; bool reserved = false; };
    ArtDecoder artDecoder_;
    ArtWait artWait_;

    // Textures whose destruction is deferred to the next fence wait, so setTrack() can hand off
    // textures without a vkDeviceWaitIdle.
    std::vector<Texture> pendingDestroyTextures_;

    // Staged texture uploads awaiting recording into the frame command buffer.
    std::vector<StagedUpload> pendingUploads_;
    // Staging resources whose GPU work has been recorded but not yet completed; freed after the
    // next fence wait.
    std::vector<StagedUpload> inFlightStagings_;

    // Cross-dissolve song slots. outgoing* is the song dissolving away; currentTrack_ the song
    // dissolving in and then shown; pendingTrack_ the latest change requested mid-dissolve
    // (overwritten by each further change, so a fast-skip burst collapses to the final song) and
    // promoted to its own fresh dissolve once the running one completes.
    RGBA outgoingAccent_{};
    bool outgoingHasAccent_ = false;
    float outgoingWashDim_ = 1.0f;      // keeps the outgoing cover's retained wash calibrated
    bool outgoingHadAlbum_ = false;     // a cover shifts the fill rod's left edge right past the art
    bool sameCover_ = false;            // incoming art is byte-identical to the outgoing: skip the cover handoff
    Texture outgoingAlbum_;
    Track currentTrack_;
    Track outgoingTrack_;
    struct PendingTrack { Track track; bool queued = false; };
    PendingTrack pendingTrack_;
    PlaybackState state_ = PlaybackState::NoMedia;
    // transitionStart_ is sampled from the draw clock; transitionArmed_ defers that latch to the
    // first draw frame.
    double transitionStart_ = 0.0;
    bool transitionArmed_ = false;

    // Seek glide. playOffset_ holds (shown - true) seconds and eases to zero over SEEK_GLIDE_SEC;
    // lastTrueLive_/lastNowSteady_ hold the prior frame's playhead and clock so the next frame can
    // detect the discontinuity a seek produces.
    double playOffset_ = 0.0;
    double glideFromOffset_ = 0.0;
    double glideStart_ = 0.0;
    bool seekGliding_ = false;
    bool havePlayhead_ = false;
    double lastTrueLive_ = 0.0;
    double lastNowSteady_ = 0.0;

    // Play/pause colour fade: playFade_ is the eased mix, 1 = playing, 0 = paused.
    float playFade_ = 1.0f;
    float playFadeFrom_ = 1.0f;
    float playFadeTarget_ = 1.0f;
    double playFadeStart_ = 0.0;
    bool playFading_ = false;

    // Outgoing title/artist moved here by beginTextBurn on a song change; they disintegrate over
    // the transition timer. A line whose text is unchanged is left out and does not animate:
    // titleChanged_/artistChanged_ gate both its burn-out and its materialize-in.
    std::vector<GlyphInstance> outgoingTitleGlyphs_, outgoingArtistGlyphs_;
    RGBA outgoingEmberAccent_{};   // ember tint of the burn
    bool titleChanged_ = false;
    bool artistChanged_ = false;

    // Both lines share one horizontal column (only baselines differ), so the scissor, wash band,
    // and cast-shadow clip are a single band whose edges live here, not per line.
    std::array<Marquee, static_cast<size_t>(Line::Count)> mq_;
    float colLeft_ = 0.0f, colRight_ = 0.0f;

    [[nodiscard]] Marquee& mq(Line l) { return mq_[static_cast<size_t>(l)]; }
    [[nodiscard]] const Marquee& mq(Line l) const { return mq_[static_cast<size_t>(l)]; }

    // Frame-rate readout: an exponential moving average of 1/frametime with a 0.25 s time constant.
    double fpsLastSteady_ = 0.0;
    double fpsSmoothed_ = 0.0;
    bool showFps_ = false;

    // renderer_vk.cpp: device bootstrap, swapchain, pipelines, GPU resources.
    void initVulkan(PlatformWindow& window);
    void pickSampleCount();
    void createSwapchain();
    void destroySwapchain();
    void createRenderTargets();
    void destroyRenderTargets();
    void makeMeshPipeline();
    void makeWashShadowPipeline();
    void createWashShadowImage();
    void destroyWashShadowImage();
    void writeWashShadowDescriptors();
    [[nodiscard]] uint32_t findMemoryType(uint32_t typeBits, VkMemoryPropertyFlags props);
    void createBuffer(VkDeviceSize size, VkBufferUsageFlags usage,
                      VkMemoryPropertyFlags props, VkBuffer& buf, VkDeviceMemory& mem);
    void allocBindImageMemory(VkImage image, VkDeviceMemory& mem);
    void submitNow(const std::function<void(VkCommandBuffer)>& rec);
    [[nodiscard]] Texture createTextureRGBA(const uint8_t* rgba, int w, int h, bool mips = false);
    void recordTextureUpload(VkCommandBuffer cb, StagedUpload& su);
    void destroyTexture(Texture& t);
    void deferDestroyTexture(Texture& t);
    [[nodiscard]] Mesh createMesh(const std::vector<Vertex3>& verts, const std::vector<uint32_t>& indices);
    void destroyMesh(Mesh& m);

    // renderer_raytracing.cpp: acceleration structures.
    void loadRayTracingFns();
    void ensureMeshBlas(Mesh& m);
    void prepareMeshBlas(Mesh& m);
    void flushPendingBlas();
    void flushPendingBlas(VkCommandBuffer cb);
    void initSceneTlas();
    void buildSceneTlas(VkCommandBuffer cb, const std::vector<VkAccelerationStructureInstanceKHR>& insts);
    [[nodiscard]] VkDeviceAddress bufferAddr(VkBuffer b) const;

    // renderer.cpp: text layout and track state.
    [[nodiscard]] GpuGlyph glyphGpuMesh(const FontFace& f, uint32_t cp);
    void layoutLine(std::vector<GlyphInstance>& out, const std::wstring& s, FontFace& f,
                    float emWorld, float penX, float baselineY, float z, vec4 color, float maxX);
    [[nodiscard]] float measureLine(const FontFace& f, const std::wstring& s, float emWorld);
    [[nodiscard]] float armLine(Marquee& m, float laidW, float colW, float emWorld, float maxX);
    void buildCurrent(const Track& t, bool redecode = false);
    void layoutText(const Track& t);
    void beginTextBurn(bool burnTitle, bool burnArtist);
    float advancePlayPauseFade(bool playing, double nowSteady);
};
