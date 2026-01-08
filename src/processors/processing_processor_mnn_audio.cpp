#include "processors/processing_processor_mnn_audio.hpp"
#include "datasplitter/ImageSplitter.hpp"
#include <openssl/sha.h> // For SHA256_DIGEST_LENGTH

//#define STB_IMAGE_IMPLEMENTATION
//#define STB_IMAGE_WRITE_IMPLEMENTATION
//#include "stb_image.h"
//#include "stb_image_write.h"

namespace sgns::sgprocessing
{
    using namespace MNN;

    std::vector<uint8_t> MNN_Audio::StartProcessing( std::vector<std::vector<uint8_t>> &chunkhashes,
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

            auto totalChunks = proc.get_dimensions().value().get_chunk_count().value();
            m_progress = 0.0f; // Reset progress at start
            
            for ( int chunkIdx = 0; chunkIdx < totalChunks; ++chunkIdx )
            {
                // Update progress after each chunk
                m_progress = std::round(((chunkIdx + 1) * 100.0f / totalChunks) * 100.0f) / 100.0f;
            }
            return subTaskResultHash;
    }

    std::unique_ptr<MNN::Tensor> MNN_Audio::Process(const std::vector<uint8_t>& imgdata, 
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
