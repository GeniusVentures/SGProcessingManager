#include "processors/processing_processor_mnn_llm.hpp"

// RETIRED PROCESSOR (elmbridge Phase 3, D-04): MNN_Llm is a fail-closed shim.
//
// The temp-dir materializer (MaterializeModelToTempDir) and LoadModel are
// DELETED -- zero temp-dir materializers remain in SGProcessingManager, and
// the Phase 2 content-addressed ELM cache (ElmModelCache) is the single
// model materialization point. The DataType::LLM two-buffer job shape
// (promptData + modelFile bytes) has no production caller: ELM jobs carry
// no passes[] and execute via ElmProcessor::StartProcessingElm with the
// cache, and non-ELM jobs never use DataType::LLM today. This shim exists
// so a legacy two-buffer call fails closed with a structured
// RESOURCE_RESOLUTION error instead of linking away.
//
// This TU performs no MNN calls, holds no locks (neither VulkanInitMutex
// nor LlmLoadMutex), and includes no MNN headers.

namespace sgns::sgprocessing
{
    ProcessingResult MNN_Llm::StartProcessing( std::vector<std::vector<uint8_t>> &chunkhashes,
                                                const sgns::IoDeclaration         &proc,
                                                std::vector<char>                 &promptData,
                                                std::vector<char>                 &modelFile,
                                                const std::vector<sgns::Parameter> *parameters,
                                                const ExecutionContext            &execCtx )
    {
        (void) chunkhashes;
        (void) proc;
        (void) promptData;
        (void) parameters;

        // Check cancellation before doing any work -- the historical first
        // statement, preserved byte-for-byte so a job cancelled prior to
        // dispatch still short-circuits with a structured CANCELLED result.
        if ( execCtx.cancelToken.IsCancelled() )
        {
            return ProcessingResult{ {}, nullptr, {},
                ProcessingError{ ProcessingErrorStage::CANCELLED, "LLM pass cancelled" } };
        }

        // Empty-model check, preserved byte-for-byte: an empty buffer is still
        // a RESOURCE_RESOLUTION failure (and is what both surviving
        // mnn_llm_test legs exercise).
        if ( modelFile.empty() )
        {
            m_logger->error( "MNN LLM: no model file provided" );
            return ProcessingResult{ {}, nullptr, {},
                ProcessingError{ ProcessingErrorStage::RESOURCE_RESOLUTION,
                    "MNN LLM model failed to load: empty model buffer" } };
        }

        // Any non-empty model buffer hits the retirement: this processor no
        // longer loads models by any path. ELM jobs execute via the ELM
        // processor + content-addressed cache instead.
        m_logger->error( "MNN_Llm: retired processor invoked with a model buffer" );
        return ProcessingResult{ {}, nullptr, {},
            ProcessingError{ ProcessingErrorStage::RESOURCE_RESOLUTION,
                "MNN_Llm processor retired (elmbridge Phase 3 D-04): DataType::LLM "
                "two-buffer jobs are no longer supported; ELM jobs execute via the "
                "ELM processor + content-addressed cache" } };
    }
}

