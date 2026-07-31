#pragma once
#include <vulkan/vulkan.h>
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

        VkInstance m_instance{VK_NULL_HANDLE};
        VkPhysicalDevice m_physicalDevice{VK_NULL_HANDLE};
        VkDevice m_device{VK_NULL_HANDLE};
        VkQueue m_queue{VK_NULL_HANDLE};
        bool m_contextInitialized{false};
    };
}
