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
                           const std::vector<sgns::Parameter> *parameters ) override;

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

        bool InitializeContext();

        static bool IsAcceptable( VkPhysicalDeviceType type );

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
        /// Pass/RenderShaderConfig object at all.
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

        /// Appends a teardown action to the ordered teardown stack (D-22/D-24).
        void PushTeardown( std::function<void()> fn );

        /// Invokes every entry in m_teardown in reverse order (rbegin()/rend()),
        /// then clears the stack. The single, reused-by-every-later-plan
        /// mechanism satisfying D-22/D-24's "always destroy whatever was
        /// already created" rule.
        void RunTeardown();

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
        bool BuildPipeline( const std::vector<ParsedStage>             &stages,
                             const std::vector<sgns::VertexLayoutEntry> &vertexLayout,
                             const boost::optional<sgns::PipelineState> &pipelineState,
                             const ResolvedUniforms                     &uniforms,
                             ProcessingResult                           &errorOut );

        static VkFormat ToVkFormat( sgns::ColorFormat fmt );
        static VkFormat ToVkFormat( sgns::DepthFormat fmt );
        static VkFormat ToVkFormat( sgns::VertexLayoutFormat fmt );
        static VkPrimitiveTopology ToVkTopology( sgns::Topology t );
        static VkCullModeFlags ToVkCullMode( sgns::CullMode c );
        static VkFrontFace ToVkFrontFace( sgns::FrontFace f );
        static VkBool32 ToVkBool( sgns::DepthTest d );

        /// Byte size of a single scalar vertex-attribute component (this plan's
        /// documented scalar-component reading of vertex_layout -- see
        /// 03-04-PLAN.md's objective). FLOAT32/INT32 -> 4, FLOAT16 -> 2.
        static uint32_t VertexFormatByteSize( sgns::VertexLayoutFormat f );

        VkInstance m_instance{VK_NULL_HANDLE};
        VkPhysicalDevice m_physicalDevice{VK_NULL_HANDLE};
        VkDevice m_device{VK_NULL_HANDLE};
        VkQueue m_queue{VK_NULL_HANDLE};
        bool m_contextInitialized{false};

        /// Ordered teardown stack (D-22/D-24) -- every per-job Vulkan object
        /// this plan (and every later plan in this phase) allocates pushes its
        /// own destroy lambda here; RunTeardown() unwinds in reverse order.
        std::vector<std::function<void()>> m_teardown;

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
    };
}
