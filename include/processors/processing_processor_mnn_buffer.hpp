/**
* Header file for processing buffer inputs using MNN
*/
#pragma once

#include <memory>
#include <vector>

#include <MNN/Interpreter.hpp>
#include "processing_processor.hpp"

namespace sgns::sgprocessing
{
    class MNN_Buffer : public ProcessingProcessor
    {
    public:
        MNN_Buffer() = default;
        ~MNN_Buffer() override = default;

        ProcessingResult StartProcessing( std::vector<std::vector<uint8_t>> &chunkhashes,
                                           const sgns::IoDeclaration         &proc,
                                           std::vector<char>                 &bufferData,
                                           std::vector<char>                 &modelFile,
                                           const std::vector<sgns::Parameter> *parameters ) override;

    private:
        std::unique_ptr<MNN::Tensor> Process( const std::vector<float> &signalData,
                                               std::vector<uint8_t>    &modelFile,
                                               int                      length );
    };
}
