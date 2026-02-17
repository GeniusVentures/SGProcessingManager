/**
* Header file for processing textureCube inputs using MNN
*/
#pragma once

#include <memory>
#include <vector>

#include <MNN/Interpreter.hpp>
#include "processing_processor.hpp"

namespace sgns::sgprocessing
{
    class MNN_TextureCube : public ProcessingProcessor
    {
    public:
        MNN_TextureCube() = default;
        ~MNN_TextureCube() override = default;

        ProcessingResult StartProcessing( std::vector<std::vector<uint8_t>> &chunkhashes,
                                           const sgns::IoDeclaration         &proc,
                                           std::vector<char>                 &cubeData,
                                           std::vector<char>                 &modelFile,
                                           const std::vector<sgns::Parameter> *parameters ) override;

    private:
        std::unique_ptr<MNN::Tensor> Process( const std::vector<float> &inputData,
                               std::vector<uint8_t>    &modelFile,
                               int                      width,
                               int                      height,
                               int                      channels,
                               bool                     inputIsInterleaved );
    };
}
