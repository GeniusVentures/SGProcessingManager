#include "processors/processing_processor_mnn_ml.hpp"
#include "datasplitter/ImageSplitter.hpp"
#include <openssl/sha.h> 

namespace sgns::processing
{
    using namespace MNN;

    std::vector<uint8_t> MNN_ML::StartProcessing( std::vector<std::vector<uint8_t>> &chunkhashes,
                                                  const sgns::IoDeclaration         &proc,
                                                  std::vector<char>                 &imageData,
                                                  std::vector<char>                 &modelFile )
    {
        std::vector<uint8_t> modelFile_bytes;
        modelFile_bytes.assign(modelFile.begin(), modelFile.end());

            //Get stride data
        std::vector<uint8_t> subTaskResultHash(SHA256_DIGEST_LENGTH);

            auto          dataindex           = 0;
            //auto          basechunk           = subTask.chunkstoprocess( 0 );
            //bool          isValidationSubTask = ( subTask.subtaskid() == "subtask_validation" );

            
            for ( int chunkIdx = 0; chunkIdx < proc.get_dimensions().value().get_chunk_count().value(); ++chunkIdx )
            {
            }
            return subTaskResultHash;
    }

    std::unique_ptr<MNN::Tensor> MNN_ML::Process(const std::vector<uint8_t>& imgdata, 
                                                         std::vector<uint8_t>& modelFile, 
                                                         const int channels, 
                                                         const int origwidth,
                                                         const int origheight, 
                                                         const std::string filename) 
    {
        auto outputHost = std::make_unique<MNN::Tensor>();
        return outputHost;
    }

}
