/**
* Header file for processing autoregressive LLM text generation using MNN's
* native MNN::Transformer::Llm API (PROC-01, Phase 04-sgprocessing-integration).
*
* MNN::Transformer::Llm is only forward-declared here (matching
* GNUS-NEO-SWARM/src/core/engine/mnn_inference_engine.hpp's own established
* pattern for this exact type) so that consumers of this header -- notably
* ProcessingManager.hpp, which must name the concrete MNN_Llm class to
* register its factory -- never need <llm/llm.hpp> to be available at
* their translation unit. Only processing_processor_mnn_llm.cpp needs the
* real header, since only it calls into MNN::Transformer::Llm's API.
*
* @author Justin Church
*/
#pragma once
#include <cmath>
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
        /** Create an LLM processor
        */
        MNN_Llm()
        {
        }

        ~MNN_Llm() override
        {
        };

        /** Start processing data -- autoregressive LLM text generation.
        * @param chunkhashes - Reference to vector to store chunk hashes
        * @param proc - Input/output declaration with processing parameters
        * @param promptData - Input prompt text as character vector
        * @param modelFile - MNN LLM model file data (materialized to a temp
        *                    directory before MNN::Transformer::Llm::createLLM(),
        *                    which -- unlike MNN::Interpreter::createFromBuffer --
        *                    requires a directory path, not an in-memory buffer)
        */
        ProcessingResult StartProcessing( std::vector<std::vector<uint8_t>> &chunkhashes,
                           const sgns::IoDeclaration         &proc,
                           std::vector<char>                 &promptData,
                           std::vector<char>                 &modelFile,
                           const std::vector<sgns::Parameter> *parameters,
                           const ExecutionContext            &execCtx ) override;

    private:
        /** Materializes modelFile bytes to a fresh temp directory and loads them via
        * MNN::Transformer::Llm::createLLM()/load(), taking VulkanInitMutex() around
        * that call to preserve the Vulkan coexistence contract every other MNN
        * processor in this codebase already follows.
        * @param modelFileBytes - Raw MNN LLM model bytes
        * @return A loaded MNN::Transformer::Llm* (caller takes ownership, must
        *         destroy via MNN::Transformer::Llm::destroy()), or nullptr on any
        *         failure (empty bytes, materialize failure, createLLM/load failure).
        */
        MNN::Transformer::Llm *LoadModel( const std::vector<uint8_t> &modelFileBytes );
    };

}
