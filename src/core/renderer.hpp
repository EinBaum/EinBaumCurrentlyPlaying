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
    double start = 0.0;    // steady-clock seconds the scroll is measured from
    float width = 0.0f;    // laid line width, world units
    float gap = 0.0f;      // blank gap before the wrapped repeat, world units
    float speed = 0.0f;    // world units per second
};

// The copy of a text line that fades out during a song change. A scrolling line holds at the
// offset the change caught it at, drawn as two wrapped column-clipped copies like the live line.
struct OutgoingLine {
    bool changed = false;  // incoming text differs: this copy fades out, then the new line fades in
    std::vector<GlyphInstance> glyphs;
    bool scrolling = false;
    float shift = 0.0f;    // frozen scroll offset, world units
    float period = 0.0f;   // width + gap at freeze time, world units
};

// Top-level on-screen lifecycle. Pause and seek modulate Active rather than being states here;
// Transitioning is the song-to-song text fade, mutually exclusive with NoMedia.
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

    // Wash-plane shadows, half the swapchain extent, rebuilt on resize. Occluders are projected
    // onto the card into washOcc* (occlusion 0..1 per light channel, MIN blend). wash_shadow.comp
    // PCF-filters that into washShadow* (R = key, G = tip light), which stays in GENERAL
    // (compute writes, fragment samples). washOccPass_ outlives the swapchain; the framebuffer
    // is rebuilt with the images.
    VkPipeline washShadowPipeline_ = VK_NULL_HANDLE;
    VkPipelineLayout washShadowPipeLayout_ = VK_NULL_HANDLE;
    VkDescriptorSetLayout washStoreLayout_ = VK_NULL_HANDLE;  // compute set 0: output + hard occlusion
    VkDescriptorPool washStorePool_ = VK_NULL_HANDLE;
    VkDescriptorSet washStoreSet_ = VK_NULL_HANDLE;
    VkSampler washSampler_ = VK_NULL_HANDLE;
    VkImage washShadowImage_ = VK_NULL_HANDLE;
    VkDeviceMemory washShadowMem_ = VK_NULL_HANDLE;
    VkImageView washShadowView_ = VK_NULL_HANDLE;
    VkExtent2D washShadowExtent_{};
    VkRenderPass washOccPass_ = VK_NULL_HANDLE;
    VkFramebuffer washOccFb_ = VK_NULL_HANDLE;
    VkPipeline shadowPipeline_ = VK_NULL_HANDLE;
    VkPipeline shadowTessPipeline_ = VK_NULL_HANDLE;
    VkPipelineLayout shadowPipeLayout_ = VK_NULL_HANDLE;
    VkSampler washOccSampler_ = VK_NULL_HANDLE;
    VkImage washOccImage_ = VK_NULL_HANDLE;
    VkDeviceMemory washOccMem_ = VK_NULL_HANDLE;
    VkImageView washOccView_ = VK_NULL_HANDLE;

    // --- Scene resources and per-track state (renderer.cpp) ---
    Mesh unitQuad_;   // [0,1]x[0,1] quad
    Mesh barRod_;     // unit cylinder: x in [0,1], radius 1
    // Keyed by (FontFace::handle, codepoint). glyphCache_ holds the uploaded GPU mesh; advanceCache_
    // holds every glyph's advance (including whitespace, which has no mesh), so an outline is
    // fetched and tessellated exactly once per face for the process lifetime.
    std::map<std::pair<const void*, uint32_t>, Mesh> glyphCache_;
    std::map<std::pair<const void*, uint32_t>, float> advanceCache_;

    Texture white_, album_;
    Texture outgoingAlbum_;             // cover fading out during a song change
    float outgoingWashDim_ = 1.0f;
    RGBA outgoingAccent_{};             // fill colour of the song fading out
    bool outgoingHasAccent_ = false;
    bool outgoingHadAlbum_ = false;     // outgoing fill rod is inset past the cover

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

    // currentTrack_ is the song fading in (or shown). outgoingTrack_ is the song whose playhead
    // and fill colour still show during fade-out. A skip mid-fade retargets currentTrack_ in place
    // so the incoming cover and text are the latest song, not one already skipped past.
    Track currentTrack_;
    Track outgoingTrack_;
    PlaybackState state_ = PlaybackState::NoMedia;
    // transitionStart_ is sampled from the draw clock; transitionArmed_ defers that latch to the
    // first draw frame.
    double transitionStart_ = 0.0;
    bool transitionArmed_ = false;
    // Opacity the outgoing song fades out from. 1 after a normal change. Below 1 when a skip
    // caught the previous incoming song partway through its fade-in and made it the outgoing one.
    float fadeOutFrom_ = 1.0f;

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

    // Per-line outgoing copy, latched by latchOutgoingLine on a song change. A line whose text is
    // unchanged is not latched and stays solid through the change.
    std::array<OutgoingLine, static_cast<size_t>(Line::Count)> outgoing_;
    bool coverChanged_ = false;         // art bytes differ: fade the square and its wash

    // Both lines share one horizontal column (only baselines differ), so the scissor, wash band,
    // and cast-shadow clip are a single band whose edges live here, not per line.
    std::array<Marquee, static_cast<size_t>(Line::Count)> mq_;
    float colLeft_ = 0.0f, colRight_ = 0.0f;

    [[nodiscard]] Marquee& mq(Line l) { return mq_[static_cast<size_t>(l)]; }
    [[nodiscard]] const Marquee& mq(Line l) const { return mq_[static_cast<size_t>(l)]; }
    [[nodiscard]] OutgoingLine& outgoing(Line l) { return outgoing_[static_cast<size_t>(l)]; }
    [[nodiscard]] std::vector<GlyphInstance>& glyphs(Line l) { return l == Line::Title ? titleGlyphs_ : artistGlyphs_; }
    [[nodiscard]] static const std::wstring& lineText(const Track& t, Line l) { return l == Line::Title ? t.title : t.artist; }

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
    void createWashOccPass();
    void makeMeshPipeline();
    void makeShadowPipeline();
    void makeWashShadowPipeline();
    void createWashShadowImage();
    void destroyWashShadowImage();
    void createWashOccFramebuffer();
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

    // renderer.cpp: text layout and track state.
    [[nodiscard]] GpuGlyph glyphGpuMesh(const FontFace& f, uint32_t cp);
    void layoutLine(std::vector<GlyphInstance>& out, const std::wstring& s, FontFace& f,
                    float emWorld, float penX, float baselineY, float z, vec4 color, float maxX);
    [[nodiscard]] float measureLine(const FontFace& f, const std::wstring& s, float emWorld);
    [[nodiscard]] float armLine(Marquee& m, float laidW, float colW, float emWorld, float maxX);
    void buildCurrent(const Track& t, bool redecode = false);
    void layoutText(const Track& t);
    void latchOutgoingLine(Line l);
    void retargetIncoming(const Track& t);
    float advancePlayPauseFade(bool playing, double nowSteady);
};
