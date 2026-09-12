/**
* Header for the ELM processor -- one causal-LM work item end-to-end from a
* pinned Phase 2 cache bundle to a complete result envelope (elmbridge Phase 3,
* GEN-01/GEN-02/GEN-03/RES-01).
*
* MNN::Transformer::Llm is only forward-declared here (the
* processing_processor_mnn_llm.hpp include-isolation pattern) so consumers of
* this header -- notably ProcessingManager.hpp, which must name the concrete
* class to register its factory -- never need <llm/llm.hpp> at their
* translation unit. Only processing_processor_elm.cpp's gated half names an
* MNN type.
*
* ENTRY CONVENTION (Q3 resolution): ELM work items enter via
* StartProcessingElm -- standalone-testable per the phase context; Phase 4's
* splitter constructs the sgns::Elm from subtask JSON, resolves input_uri
* transport, and passes the production cache + validator. The base
* StartProcessing override ALWAYS fails closed: the two-buffer contract
* cannot carry an ELM work item.
*/
#ifndef SGPROCMGR_PROCESSING_PROCESSOR_ELM_HPP
#define SGPROCMGR_PROCESSING_PROCESSOR_ELM_HPP

#include <cstdint>
#include <memory>
#include <string>
#include <vector>

#include "processing_processor.hpp"

#include <SGNSProcMain.hpp>

namespace MNN
{
    namespace Transformer
    {
        class Llm;
    } // namespace Transformer
} // namespace MNN

namespace sgns
{
    class Elm;
}

namespace sgns::elmruntime
{
    class ElmModelCache;
}

namespace sgns::sgprocessing
{
    class CapabilityValidator;

    class ElmProcessor : public ProcessingProcessor
    {
    public:
        /** Create an ELM processor
        */
        ElmProcessor()
        {
        }

        ~ElmProcessor() override
        {
        };

        /** Start processing -- base 6-arg contract, ALWAYS fails closed.
        * The two-buffer (promptData + modelFile) contract cannot carry an ELM
        * work item: ELM jobs have no passes[] and execute via StartProcessingElm
        * (Phase 4 wires grid routing).
        * @param chunkhashes - Reference to vector to store chunk hashes
        * @param proc - Input/output declaration with processing parameters
        * @param promptData - Input prompt text (ignored)
        * @param modelFile - Legacy model-file buffer (rejected)
        * @param parameters - Processing parameters (ignored)
        * @param execCtx - Execution context (cancel token checked first)
        * @return a RESOURCE_RESOLUTION error result naming the ELM entry point
        */
        ProcessingResult StartProcessing( std::vector<std::vector<uint8_t>> &chunkhashes,
                           const sgns::IoDeclaration         &proc,
                           std::vector<char>                 &promptData,
                           std::vector<char>                 &modelFile,
                           const std::vector<sgns::Parameter> *parameters,
                           const ExecutionContext            &execCtx ) override;

        /** Execute one ELM (causal-LM) work item end-to-end: cache acquire ->
        * resource preflight -> locked session create (VulkanInitMutex around
        * createLLM, LlmLoadMutex around load) -> set_config + dump_config
        * assert BEFORE load -> streambuf-wrapped response with stop-string +
        * cancel polling -> LlmContext count reconciliation -> envelope.
        *
        * Both parameter seams:
        *   promptText   - the fully rendered prompt string. Phase 4's submit
        *                  wiring resolves input_uri transport and passes the
        *                  resolved string; this processor never fetches.
        *   stopStrings  - the D-05/D-06/D-07 stop list. The Phase 1
        *                  gnus-processing-schema.json ElmGeneration carries NO
        *                  stop field (verified: only max_output_tokens /
        *                  temperature / top_p / seed), so stop strings CANNOT
        *                  ride the job JSON today -- job-JSON carriage is the
        *                  Phase 4 schema amendment (STATE.md TODO escalation).
        *                  Standalone callers and tests pass the list directly;
        *                  an absent list is an empty vector (no stop-string
        *                  scanning).
        *
        * @param chunkhashes - Reference to vector to store the result hash
        * @param promptText - the fully rendered prompt string
        * @param stopStrings - stop strings excluded from envelope text (D-07)
        * @param elm - the validated work item (quicktype-generated type)
        * @param execCtx - Execution context (cancel token, budgets, progress)
        * @param cache - the Phase 2 content-addressed model cache (the ONLY
        *                model source; Acquire pin held for the whole call)
        * @param capabilityValidator - for the CheckElmResources preflight;
        *                nullptr skips the check (test seam -- logged)
        * @return a ProcessingResult whose output buffer carries the JSON
        *         envelope (work_item_id, text, counts, finish_reason,
        *         model_manifest_hash, error detail on error)
        */
        ProcessingResult StartProcessingElm( std::vector<std::vector<uint8_t>> &chunkhashes,
                           const std::string                 &promptText,
                           const std::vector<std::string>    &stopStrings,
                           const sgns::Elm                   &elm,
                           const ExecutionContext            &execCtx,
                           std::shared_ptr<sgns::elmruntime::ElmModelCache> cache,
                           CapabilityValidator               *capabilityValidator );
    };

} // namespace sgns::sgprocessing

#endif // SGPROCMGR_PROCESSING_PROCESSOR_ELM_HPP
