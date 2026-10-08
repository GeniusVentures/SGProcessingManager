#pragma once
#include <vulkan/vulkan.h>
#include <functional>
#include <map>
#include <optional>
#include <string>
#include <vector>
#include "processing_processor.hpp"
#include <RenderShaderUniform.hpp>
#include <Parameter.hpp>
#include <Stage.hpp>
#include <IndexType.hpp>
#include <RenderTarget.hpp>
#include <PipelineState.hpp>
#include <VertexLayoutEntry.hpp>
#include <TextureFilter.hpp>

namespace sgns::sgprocessing
{
    class RenderProcessor : public ProcessingProcessor
    {
    public:
        RenderProcessor() {}
        ~RenderProcessor() override = default;

        ProcessingResult StartProcessing( std::vector<std::vector<uint8_t>> &chunkhashes,
                           const sgns::IoDeclaration         &proc,
                           std::vector<char>                 &imageData,
                           std::vector<char>                 &modelFile,
                           const std::vector<sgns::Parameter> *parameters,
                           const ExecutionContext            &execCtx ) override;

        /// Device-type filter (DISCRETE_GPU/INTEGRATED_GPU only). Public so
        /// vulkan_gpu_probe.cpp's HasUsableVulkanDevice() can reuse the exact
        /// same predicate instead of duplicating it (avoids drift risk).
        static bool IsAcceptable( VkPhysicalDeviceType type );

        /// Lazy-init Vulkan instance/device (idempotent, double-checked locking).
        /// Public so CapabilityValidator::BuildSnapshot can ensure the device exists
        /// before querying its properties (D-10).
        bool InitializeContext();

        /// Returns the physical device handle after InitializeContext() has succeeded.
        /// Returns VK_NULL_HANDLE if context not yet initialized.
        VkPhysicalDevice GetPhysicalDevice() const { return m_physicalDevice; }

    private:
        /// One parsed SPIR-V shader stage, inverted from ProcessingManager.cpp's
        /// SerializeCompiledStages( stages, entryPoints ) wire format (plan 03-01).
        struct ParsedStage
        {
            sgns::Stage           stage;
            std::string           entry_point;
            std::vector<uint32_t> spirv;
        };

        /// Result of resolving a render pass's declared uniforms (D-29/D-30):
        /// packed bytes plus the push-constant-vs-descriptor-set decision.
        struct ResolvedUniforms
        {
            std::vector<uint8_t> packedBytes;
            bool                 pushConstant = true;
        };

        static VkDeviceSize LargestDeviceLocalHeap( VkPhysicalDevice device );

        /// Exact byte-for-byte inverse of ProcessingManager.cpp's
        /// SerializeCompiledStages( stages, entryPoints ). Bounds-checks every
        /// read against modelFile.size() -- never reads past the end of a
        /// malformed/truncated buffer.
        static bool ParseCompiledStages( const std::vector<char>  &modelFile,
                                          std::vector<ParsedStage> &outStages,
                                          ProcessingResult         &errorOut );

        /// Exact byte-for-byte inverse of ProcessingManager.cpp's
        /// SerializeRenderPassConfig(...). This is the ONLY method that ever
        /// produces sgns::RenderTarget/PipelineState/VertexLayoutEntry/uniform-map
        /// instances inside RenderProcessor, and the only source of
        /// outDataTransformCount -- RenderProcessor has no other path to a
        /// Pass/RenderShaderConfig object at all. outHasTextureBuffer/
        /// outTextureWidth/outTextureHeight/outTextureBytes carry the trailing
        /// texture_buffer section (Phase 17, D-05) -- populated here but not yet
        /// consumed until Wave 3's Plan 17-04 (BuildPipeline's descriptor binding,
        /// UploadBuffers' image upload).
        static bool ParseRenderPassConfig(
            const std::vector<char>                                            &imageData,
            sgns::RenderTarget                                                 &outTarget,
            boost::optional<sgns::PipelineState>                               &outPipelineState,
            std::vector<sgns::VertexLayoutEntry>                               &outVertexLayout,
            boost::optional<std::map<std::string, sgns::RenderShaderUniform>>  &outUniforms,
            std::vector<uint8_t>                                               &outVertexBytes,
            bool                                                                &outHasIndex,
            sgns::IndexType                                                    &outIndexType,
            std::vector<uint8_t>                                               &outIndexBytes,
            uint32_t                                                           &outDataTransformCount,
            bool                                                                &outHasTextureBuffer,
            uint32_t                                                            &outTextureWidth,
            uint32_t                                                            &outTextureHeight,
            std::vector<uint8_t>                                               &outTextureBytes,
            ProcessingResult                                                   &errorOut );

        /// Resolves each declared uniform's value -- either a literal
        /// RenderShaderUniform.value or a parameter:-sourced value read from
        /// `parameters` -- and packs the resolved bytes per each uniform's
        /// declared DataType, 16-byte-aligned per uniform (std430-avoidance,
        /// see 03-03-PLAN.md objective). Sets pushConstant per D-29's fixed
        /// 128-byte threshold and D-30's all-or-nothing rule.
        static bool ResolveUniforms(
            const boost::optional<std::map<std::string, sgns::RenderShaderUniform>> &uniforms,
            const std::vector<sgns::Parameter>                                      *parameters,
            ResolvedUniforms                                                        &outResolved,
            ProcessingResult                                                        &errorOut );

        /// Constructs a ProcessingResult populated via the given stage/message
        /// (D-25/D-26). Does NOT call RunTeardown() itself -- every caller must
        /// call RunTeardown() immediately before or after, per this plan's
        /// Task 2 convention. Static -- needs no instance state, so the
        /// also-static ParseCompiledStages()/ParseRenderPassConfig()/
        /// ResolveUniforms() can call it directly, alongside the non-static
        /// CheckFormatSupport()/CreateBufferDedicated()/CreateImageDedicated().
        static ProcessingResult MakeError( sgns::sgprocessing::ProcessingErrorStage stage, const std::string &message );

        /// Queries vkGetPhysicalDeviceFormatProperties and checks that
        /// requiredFeature is present in optimalTilingFeatures (RESEARCH.md
        /// Pitfall 7) -- fails with a structured FORMAT_UNSUPPORTED error
        /// naming the specific format, rather than letting image/render-pass
        /// creation fail with an opaque VkResult or misbehave silently.
        bool CheckFormatSupport( VkFormat format, VkFormatFeatureFlagBits requiredFeature, ProcessingResult &errorOut );

        /// Allocates a VkBuffer with its own dedicated VkDeviceMemory
        /// allocation (D-18/D-19), sized exactly to the buffer's memory
        /// requirements -- no sub-allocation. Registers automatic teardown
        /// via PushTeardown() on success; destroys the buffer itself (but not
        /// via the teardown stack, since it isn't registered yet) on a
        /// partial-failure path (D-24).
        bool CreateBufferDedicated( VkDeviceSize          size,
                                     VkBufferUsageFlags    usage,
                                     VkMemoryPropertyFlags properties,
                                     VkBuffer              &outBuffer,
                                     VkDeviceMemory        &outMemory,
                                     ProcessingResult      &errorOut );

        /// Allocates a VkImage with its own dedicated VkDeviceMemory
        /// allocation (D-18/D-19), identical in shape to CreateBufferDedicated.
        bool CreateImageDedicated( const VkImageCreateInfo &imageInfo,
                                    VkMemoryPropertyFlags   properties,
                                    VkImage                 &outImage,
                                    VkDeviceMemory          &outMemory,
                                    ProcessingResult        &errorOut );

        /// Sane, conservative maximum render_target width/height (Security Domain
        /// V5's DoS concern -- the schema only enforces minimum:1, no maximum). 8192
        /// is a generous-but-bounded default; no specific value is mandated by
        /// REQUIREMENTS.md/CONTEXT.md.
        static constexpr uint32_t kMaxRenderDimension = 8192;

        /// Builds the offscreen VkRenderPass (color+depth, explicit CLEAR load ops
        /// on both, VK_SAMPLE_COUNT_1_BIT unconditionally per DETV-02). Bounds-checks
        /// target.get_width()/get_height() against kMaxRenderDimension and format-
        /// support-checks both formats via CheckFormatSupport() (RESEARCH.md Pitfall
        /// 7) before creating anything. Sets m_renderWidth/m_renderHeight for
        /// plan 03-04 Task 2's BuildPipeline() to consume for its fixed viewport.
        bool BuildRenderPass( const sgns::RenderTarget &target, ProcessingResult &errorOut );

        /// Builds the offscreen VkFramebuffer: a color+depth VkImage/VkImageView
        /// pair (each image via CreateImageDedicated(), DEVICE_LOCAL) referencing
        /// m_renderPass. Must be called after BuildRenderPass() succeeds.
        bool BuildFramebuffer( const sgns::RenderTarget &target, ProcessingResult &errorOut );

        /// Builds the complete graphics pipeline: one VkShaderModule/shader-stage
        /// per parsed stage (real per-stage entry point, never a hard-coded
        /// "main"), fixed (never dynamic) pipeline state from pipelineState (or
        /// schema-documented defaults when absent), the auto-computed vertex
        /// input binding/attributes from vertexLayout, and a pipeline layout
        /// branching on uniforms.pushConstant (D-29/D-30's all-or-nothing rule).
        /// hasTexture (Phase 17 Wave 3, D-05) independently forces a descriptor
        /// set to exist (binding=1 combined-image-sampler) even when uniforms
        /// alone would have used push constants -- binding=0's uniform-buffer
        /// entry is only added when uniforms also route through a descriptor set.
        bool BuildPipeline( const std::vector<ParsedStage>             &stages,
                             const std::vector<sgns::VertexLayoutEntry> &vertexLayout,
                             const boost::optional<sgns::PipelineState> &pipelineState,
                             const ResolvedUniforms                     &uniforms,
                             bool                                        hasTexture,
                             ProcessingResult                           &errorOut );

        /// Validates vertexBytes.size() % stride == 0 and (if hasIndex)
        /// indexBytes.size() % index-type-byte-size == 0 BEFORE any buffer is
        /// created (closes T-03-03-02 -- out-of-bounds vkCmdDraw(Indexed) read),
        /// then uploads vertex/index/uniform bytes into dedicated HOST_VISIBLE|
        /// HOST_COHERENT buffers via direct vkMapMemory/memcpy/vkUnmapMemory (D-20/
        /// D-21 -- no staging+device-local path, no manual flush). Only allocates
        /// m_uniformBuffer (descriptor-set path) when uniforms.pushConstant is
        /// false; the push-constant path needs no VkBuffer (bytes copied directly
        /// from ResolvedUniforms::packedBytes at record time via
        /// RecordAndSubmit()/vkCmdPushConstants).
        bool UploadBuffers( const std::vector<uint8_t> &vertexBytes,
                             bool                        hasIndex,
                             sgns::IndexType             indexType,
                             const std::vector<uint8_t> &indexBytes,
                             uint32_t                    stride,
                             const ResolvedUniforms      &uniforms,
                             ProcessingResult            &errorOut );

        /// Phase 17 Wave 3 (D-05): the genuinely new Vulkan work texturing needs --
        /// staging buffer (HOST_VISIBLE|HOST_COHERENT, mirrors UploadBuffers'
        /// existing memcpy-into-staging-buffer style) -> device-local sampled
        /// VkImage (via the existing CreateImageDedicated()) -> VkImageView ->
        /// VkSampler -> a binding=1 combined-image-sampler descriptor-set write
        /// (mirrors UploadBuffers' existing binding=0 uniform-buffer write).
        /// Validates textureBytes.size() == width*height*4 BEFORE any GPU
        /// resource is created (T-17-09, RESOURCE_RESOLUTION on mismatch). Must
        /// be called after BuildPipeline() has already created m_descriptorSet
        /// (BuildPipeline()'s hasTexture=true path). Sets m_hasTexture/
        /// m_textureWidth/m_textureHeight on success -- consumed by
        /// RecordAndSubmit()'s pre-render-pass upload barrier/copy/barrier
        /// sequence.
        bool UploadTexture( const std::vector<uint8_t> &textureBytes,
                             uint32_t                    width,
                             uint32_t                    height,
                             sgns::TextureFilter         filter,
                             ProcessingResult           &errorOut );

        /// Records and submits ONE command buffer: begin render pass (clears from
        /// target.get_clear_color()/get_clear_depth()) -> bind pipeline/vertex/
        /// index buffers -> push constants or bind descriptor set -> draw(Indexed)
        /// -> end render pass -> (Pitfall 4) record the vkCmdCopyImageToBuffer
        /// readback copy INTO THIS SAME command buffer, immediately after
        /// vkCmdEndRenderPass and before vkEndCommandBuffer -- no second command
        /// buffer/submission, no extra image-layout-transition barrier (the render pass's
        /// color attachment finalLayout is already VK_IMAGE_LAYOUT_TRANSFER_SRC_
        /// OPTIMAL, plan 03-04) -- then vkQueueSubmit and a synchronous
        /// vkDeviceWaitIdle (D-23). Allocates m_stagingBuffer/m_stagingMemory
        /// (the readback destination Readback() later maps) as part of recording
        /// this copy.
        bool RecordAndSubmit( const sgns::RenderTarget &target, ProcessingResult &errorOut );

        /// Maps m_stagingBuffer (already populated by RecordAndSubmit()'s
        /// vkCmdCopyImageToBuffer + vkDeviceWaitIdle) and copies its bytes into
        /// outBytes -- no vkInvalidateMappedMemoryRanges call (HOST_COHERENT,
        /// D-20). Must be called after RecordAndSubmit() succeeds.
        bool Readback( const sgns::RenderTarget &target, std::vector<uint8_t> &outBytes, ProcessingResult &errorOut );

        /// Bytes per pixel for a given color attachment format. RGBA8 -> 4,
        /// RGB8 -> 3.
        static uint32_t ColorFormatByteSize( sgns::ColorFormat fmt );

        static VkFormat ToVkFormat( sgns::ColorFormat fmt );
        static VkFormat ToVkFormat( sgns::DepthFormat fmt );
        static VkFormat ToVkFormat( sgns::VertexLayoutFormat fmt );
        static VkPrimitiveTopology ToVkTopology( sgns::Topology t );
        static VkCullModeFlags ToVkCullMode( sgns::CullMode c );
        static VkFrontFace ToVkFrontFace( sgns::FrontFace f );
        static VkBool32 ToVkBool( sgns::DepthTest d );
        static VkBlendFactor ToVkBlendFactor( sgns::BlendFactor f );

        /// Byte size of a single scalar vertex-attribute component (this plan's
        /// documented scalar-component reading of vertex_layout -- see
        /// 03-04-PLAN.md's objective). FLOAT32/INT32 -> 4, FLOAT16 -> 2.
        static uint32_t VertexFormatByteSize( sgns::VertexLayoutFormat f );

        VkInstance m_instance{VK_NULL_HANDLE};
        VkPhysicalDevice m_physicalDevice{VK_NULL_HANDLE};
        VkDevice m_device{VK_NULL_HANDLE};
        VkQueue m_queue{VK_NULL_HANDLE};
        /// Graphics queue family index InitializeContext() resolved for m_queue --
        /// stored so RecordAndSubmit()'s VkCommandPool creation reuses the same
        /// already-selected graphics queue family instead of re-running device
        /// queue-family selection.
        uint32_t m_queueFamilyIndex{0};
        bool m_contextInitialized{false};

        /// render_target width/height, set by BuildRenderPass() after its bounds
        /// check succeeds -- consumed by Task 2's BuildPipeline() for its fixed
        /// (never a runtime-settable pipeline attribute, per D-22) viewport/
        /// scissor, since VkGraphicsPipelineCreateInfo requires a concrete
        /// VkPipelineViewportStateCreateInfo when no dynamic viewport/scissor
        /// state is used.
        uint32_t m_renderWidth{0};
        uint32_t m_renderHeight{0};

        VkRenderPass   m_renderPass{VK_NULL_HANDLE};
        VkFramebuffer  m_framebuffer{VK_NULL_HANDLE};
        VkImage        m_colorImage{VK_NULL_HANDLE}, m_depthImage{VK_NULL_HANDLE};
        VkImageView    m_colorView{VK_NULL_HANDLE}, m_depthView{VK_NULL_HANDLE};
        VkDeviceMemory m_colorMemory{VK_NULL_HANDLE}, m_depthMemory{VK_NULL_HANDLE};

        VkPipelineLayout      m_pipelineLayout{VK_NULL_HANDLE};
        VkPipeline            m_pipeline{VK_NULL_HANDLE};
        VkDescriptorSetLayout m_descriptorSetLayout{VK_NULL_HANDLE};
        VkDescriptorPool      m_descriptorPool{VK_NULL_HANDLE};
        VkDescriptorSet       m_descriptorSet{VK_NULL_HANDLE};

        VkBuffer       m_vertexBuffer{VK_NULL_HANDLE}, m_indexBuffer{VK_NULL_HANDLE};
        VkBuffer       m_uniformBuffer{VK_NULL_HANDLE}, m_stagingBuffer{VK_NULL_HANDLE};
        VkDeviceMemory m_vertexMemory{VK_NULL_HANDLE}, m_indexMemory{VK_NULL_HANDLE};
        VkDeviceMemory m_uniformMemory{VK_NULL_HANDLE}, m_stagingMemory{VK_NULL_HANDLE};

        /// Texture upload path state (Phase 17 Wave 3, D-05 texturing): staging
        /// buffer -> device-local sampled VkImage -> VkImageView -> VkSampler,
        /// bound at descriptor set binding=1. Populated by UploadTexture(),
        /// consumed by RecordAndSubmit()'s upload barrier/copy sequence.
        VkBuffer       m_textureStagingBuffer{VK_NULL_HANDLE};
        VkDeviceMemory m_textureStagingMemory{VK_NULL_HANDLE};
        VkImage        m_textureImage{VK_NULL_HANDLE};
        VkDeviceMemory m_textureMemory{VK_NULL_HANDLE};
        VkImageView    m_textureView{VK_NULL_HANDLE};
        VkSampler      m_textureSampler{VK_NULL_HANDLE};
        bool           m_hasTexture{false};
        uint32_t       m_textureWidth{0}, m_textureHeight{0};

        VkCommandPool   m_commandPool{VK_NULL_HANDLE};
        VkCommandBuffer m_commandBuffer{VK_NULL_HANDLE};

        bool            m_hasIndexBuffer{false};
        sgns::IndexType m_indexType{sgns::IndexType::UINT32};
        uint32_t        m_vertexCount{0};
        uint32_t        m_indexCount{0};

        /// Set by UploadBuffers() from the ResolvedUniforms passed into it --
        /// RecordAndSubmit()'s declared signature (target, errorOut) carries no
        /// uniform data of its own, so the push-constant bytes/decision must be
        /// stored here for RecordAndSubmit()'s vkCmdPushConstants call. The
        /// descriptor-set path needs no equivalent member: m_descriptorSet
        /// (already built by BuildPipeline()) is bound directly.
        bool                 m_usePushConstant{false};
        std::vector<uint8_t> m_pushConstantBytes;
    };
}
