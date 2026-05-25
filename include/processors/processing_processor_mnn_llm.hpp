/**
 * MNN LLM processor for autoregressive text generation
 *
 * Unlike single-pass processors (MNN_Float, MNN_String, etc.), this processor
 * runs an autoregressive generation loop: tokenize → forward → sample → repeat.
 * Input data is space-separated token IDs. Output is generated token IDs as int32 bytes.
 *
 * Generation parameters are read from the JSON "parameters" array:
 *   - maxTokens (int, default 512)
 *   - temperature (float, default 0.7)
 *   - topP (float, default 0.9)
 *   - topK (int, default 40)
 *   - eosTokenId (int, default 2)
 */
#pragma once

#include <memory>
#include <vector>

#include <MNN/Interpreter.hpp>
#include "processing_processor.hpp"

namespace sgns::sgprocessing
{
    class MNN_LLM : public ProcessingProcessor
    {
    public:
        MNN_LLM() = default;
        ~MNN_LLM() override = default;

        ProcessingResult StartProcessing(
            std::vector<std::vector<uint8_t>>  &chunkhashes,
            const sgns::IoDeclaration          &proc,
            std::vector<char>                  &inputData,
            std::vector<char>                  &modelFile,
            const std::vector<sgns::Parameter> *parameters ) override;

    private:
        /// Run a single forward pass, return output tensor (logits)
        std::unique_ptr<MNN::Tensor> Forward(
            const std::vector<int32_t> &input_ids,
            std::vector<uint8_t>       &modelFileBytes,
            int                         seq_len );

        /// Sample next token from logits using temperature + top-k + top-p
        int32_t SampleToken( const float *logits, int vocab_size,
                             float temperature, float top_p, int top_k ) const;
    };
}
