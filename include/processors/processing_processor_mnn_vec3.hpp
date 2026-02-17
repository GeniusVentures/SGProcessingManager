#ifndef SG_PROCESSING_MNN_VEC3_HPP
#define SG_PROCESSING_MNN_VEC3_HPP

#include <MNN/Interpreter.hpp>
#include <memory>
#include <vector>
#include "processing_processor.hpp"

namespace sgns::sgprocessing
{
    class MNN_Vec3 : public ProcessingProcessor
    {
    public:
        virtual ~MNN_Vec3() = default;

        ProcessingResult StartProcessing( std::vector<std::vector<uint8_t>>     &chunkhashes,
                                          const sgns::IoDeclaration              &proc,
                                          std::vector<char>                      &vec3Data,
                                          std::vector<char>                      &modelFile,
                                          const std::vector<sgns::Parameter>     *parameters )
            override;

    private:
        std::unique_ptr<MNN::Tensor> Process( const std::vector<float> &input, 
                                              std::vector<uint8_t> &model,
                                              int length );
    };
}

#endif
