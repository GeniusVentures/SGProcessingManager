/**
 * MNN processor for FP4_ULTRA quantized input data
 *
 * Handles input data in FP4_ULTRA format (4-bit NF4-style quantization).
 * The processor dequantizes packed nibbles + per-macroblock scales to FLOAT32,
 * then runs standard windowed MNN inference (same pattern as MNN_Float).
 *
 * FP4_ULTRA data layout:
 *   [packed_nibbles: (num_elements+1)/2 bytes] [scales: num_macroblocks * 4 bytes]
 *
 * Each macroblock is 64x64 = 4096 elements with one float32 scale factor.
 * Nibbles index into a 16-entry NF4 lookup table in [-1, 1].
 */
#pragma once

#include <memory>
#include <vector>

#include <MNN/Interpreter.hpp>
#include "processing_processor.hpp"

namespace sgns::sgprocessing
{
    class MNN_FP4Ultra : public ProcessingProcessor
    {
    public:
        MNN_FP4Ultra() = default;
        ~MNN_FP4Ultra() override = default;

        ProcessingResult StartProcessing(
            std::vector<std::vector<uint8_t>>  &chunkhashes,
            const sgns::IoDeclaration          &proc,
            std::vector<char>                  &fp4Data,
            std::vector<char>                  &modelFile,
            const std::vector<sgns::Parameter> *parameters ) override;

    private:
        /// Dequantize FP4 packed nibbles + scales to FLOAT32
        std::vector<float> DequantizeFP4( const std::vector<char> &packed,
                                           size_t                   num_elements );

        /// Run single-pass MNN inference on float data
        std::unique_ptr<MNN::Tensor> Process( const std::vector<float> &floatData,
                                               std::vector<uint8_t>    &modelFileBytes,
                                               int                      length );
    };
}
