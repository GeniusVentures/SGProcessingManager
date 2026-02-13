/**
* Header file for processing 3D volume data using MNN
* @author Justin Church
*/
#pragma once
#include <cmath>
#include <memory>
#include <vector>
#include <string>

#include <MNN/Interpreter.hpp>
#include "processing_processor.hpp"
#define MNN_OPEN_TIME_TRACE
#include <MNN/AutoTime.hpp>

namespace sgns::sgprocessing
{
    using namespace MNN;

    class MNN_Volume : public ProcessingProcessor
    {
    public:
        /** Create a volume processor
        */
        MNN_Volume()
        {
        }

        ~MNN_Volume() override
        {
        };

        /** Start processing data
        * @param chunkhashes - Reference to vector to store chunk hashes
        * @param proc - Input declaration with processing parameters
        * @param volumeData - Input volume data as raw bytes
        * @param modelFile - MNN model file data
        */
        ProcessingResult StartProcessing( std::vector<std::vector<uint8_t>> &chunkhashes,
                           const sgns::IoDeclaration         &proc,
                           std::vector<char>                 &volumeData,
                           std::vector<char>                 &modelFile ) override;

    private:
        /** Run MNN processing on volume data
        * @param volumeData - Input volume data as float32 array
        * @param modelFile - MNN model file bytes
        * @param width - Volume width
        * @param height - Volume height
        * @param depth - Volume depth
        */
        std::unique_ptr<MNN::Tensor> Process( const std::vector<float> &volumeData,
                                               std::vector<uint8_t> &modelFile,
                                               const int width,
                                               const int height,
                                               const int depth );
    };

}
