/**
* Header file for the RETIRED MNN_Llm processor (elmbridge Phase 3, D-04).
*
* MNN_Llm is now a fail-closed shim: its DataType::LLM two-buffer job shape
* (promptData + modelFile bytes) has no production caller -- ELM jobs carry
* no passes[] and never route here, and non-ELM jobs never use DataType::LLM
* today -- and the temp-dir materializer it used to own is DELETED (zero
* temp-dir materializers remain in the tree; the Phase 2 content-addressed
* ELM cache is the single model materialization point). This shim exists so
* a legacy two-buffer DataType::LLM call fails closed with a structured
* RESOURCE_RESOLUTION error instead of linking away.
*
* MNN::Transformer::Llm remains only forward-declared here (its historical
* include-isolation pattern) so consumers of this header -- notably
* ProcessingManager.hpp, which must name the concrete class for factory
* registration -- never need <llm/llm.hpp> at their translation unit. The
* shim TU itself no longer includes llm.hpp at all: it performs no MNN
* calls and takes no locks.
*
* @author Justin Church
*/
#pragma once
#include <cstdint>
#include <memory>
#include <string>
#include <vector>

#include "processing_processor.hpp"

namespace MNN
{
    namespace Transformer
    {
        class Llm;
    } // namespace Transformer
} // namespace MNN

namespace sgns::sgprocessing
{
    class MNN_Llm : public ProcessingProcessor
    {
    public:
        /** Create the retired-LLM shim
        */
        MNN_Llm()
        {
        }

        ~MNN_Llm() override
        {
        };

        /** Start processing -- RETIRED, always fails closed (elmbridge Phase 3 D-04).
        * The pre-cancel check and the empty-model check remain; any non-empty
        * model buffer returns a structured RESOURCE_RESOLUTION retirement
        * error. ELM jobs execute via ElmProcessor::StartProcessingElm with the
        * content-addressed cache (see processing_processor_elm.hpp).
        * @param chunkhashes - Reference to vector to store chunk hashes
        * @param proc - Input/output declaration with processing parameters
        * @param promptData - Input prompt text as character vector (ignored)
        * @param modelFile - Legacy model-file buffer (any non-empty value is
        *                   rejected with the retirement message)
        */
        ProcessingResult StartProcessing( std::vector<std::vector<uint8_t>> &chunkhashes,
                           const sgns::IoDeclaration         &proc,
                           std::vector<char>                 &promptData,
                           std::vector<char>                 &modelFile,
                           const std::vector<sgns::Parameter> *parameters,
                           const ExecutionContext            &execCtx ) override;
    };

}
