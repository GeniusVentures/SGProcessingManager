/**
* Header file for processing tensor inputs using MNN
*/
#pragma once

#include <memory>
#include <vector>

#include <MNN/Interpreter.hpp>
#include "processing_processor.hpp"

namespace sgns::sgprocessing
{
    class MNN_Tensor : public ProcessingProcessor
    {
    public:
        MNN_Tensor() = default;
        ~MNN_Tensor() override = default;

        ProcessingResult StartProcessing( std::vector<std::vector<uint8_t>> &chunkhashes,
                                           const sgns::IoDeclaration         &proc,
                                           std::vector<char>                 &tensorData,
                                           std::vector<char>                 &modelFile,
                                           const std::vector<sgns::Parameter> *parameters ) override;

    private:
        std::unique_ptr<MNN::Tensor> Process( const std::vector<float> &signalData,
                                               std::vector<uint8_t>    &modelFile,
                                               int                      length );
    };
}
