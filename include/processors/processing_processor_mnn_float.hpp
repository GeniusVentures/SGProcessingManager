/**
* Header file for processing float inputs using MNN
*/
#pragma once

#include <memory>
#include <vector>

#include <MNN/Interpreter.hpp>
#include "processing_processor.hpp"

namespace sgns::sgprocessing
{
    class MNN_Float : public ProcessingProcessor
    {
    public:
        MNN_Float() = default;
        ~MNN_Float() override = default;

        ProcessingResult StartProcessing( std::vector<std::vector<uint8_t>> &chunkhashes,
                                           const sgns::IoDeclaration         &proc,
                                           std::vector<char>                 &floatData,
                                           std::vector<char>                 &modelFile,
                                           const std::vector<sgns::Parameter> *parameters ) override;

    private:
        std::unique_ptr<MNN::Tensor> Process( const std::vector<float> &signalData,
                                               std::vector<uint8_t>    &modelFile,
                                               int                      length );
    };
}
