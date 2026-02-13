/**
* Header file for processing text/string using MNN
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

    class MNN_String : public ProcessingProcessor
    {
    public:
        /** Create a string processor
        */
        MNN_String() 
        {
        }

        ~MNN_String() override
        {
        };

        /** Start processing data
        * @param chunkhashes - Reference to vector to store chunk hashes
        * @param proc - Input/output declaration with processing parameters
        * @param textData - Input text data as character vector
        * @param modelFile - MNN model file data
        */
        ProcessingResult StartProcessing( std::vector<std::vector<uint8_t>> &chunkhashes,
                                              const sgns::IoDeclaration         &proc,
                                              std::vector<char>                 &textData,
                                              std::vector<char>                 &modelFile ) override;

    private:
        /** Run MNN processing on text/string
        * @param tokenIds - Input token ids
        * @param modelFile - MNN model file bytes
        * @param maxLength - Maximum sequence length
        */
        std::unique_ptr<MNN::Tensor> Process( const std::vector<int32_t> &tokenIds, 
                               std::vector<uint8_t> &modelFile,
                               const int maxLength );
    };

}
