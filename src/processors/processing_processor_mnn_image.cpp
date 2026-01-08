#include "processors/processing_processor_mnn_image.hpp"
#include "datasplitter/ImageSplitter.hpp"
#include <functional>
#include <thread>
#include <openssl/sha.h> // For SHA256_DIGEST_LENGTH
#include "util/sha256.hpp"
#include "util/InputTypes.hpp"

//#define STB_IMAGE_IMPLEMENTATION
//#define STB_IMAGE_WRITE_IMPLEMENTATION
//#include "stb_image.h"
//#include "stb_image_write.h"

namespace sgns::sgprocessing
{
    using namespace MNN;

    std::vector<uint8_t> MNN_Image::StartProcessing( std::vector<std::vector<uint8_t>> &chunkhashes,
                                                     const sgns::IoDeclaration         &proc,
                                                     std::vector<char>                 &imageData,
                                                     std::vector<char>                 &modelFile )
    {
        std::vector<uint8_t> modelFile_bytes;
        modelFile_bytes.assign(modelFile.begin(), modelFile.end());

            //Get stride data
        std::vector<uint8_t> subTaskResultHash(SHA256_DIGEST_LENGTH);
        auto                 block_len             = proc.get_dimensions().value().get_block_len().value();
        auto                 block_line_stride     = proc.get_dimensions().value().get_block_line_stride().value();
        auto                 block_stride          = proc.get_dimensions().value().get_block_stride().value();
        auto                 chunk_line_stride     = proc.get_dimensions().value().get_chunk_line_stride().value();
        auto                 chunk_offset          = proc.get_dimensions().value().get_chunk_offset().value();
        auto                 chunk_stride          = proc.get_dimensions().value().get_chunk_stride().value();
        auto                 chunk_subchunk_height = proc.get_dimensions().value().get_chunk_subchunk_height().value();
        auto                 chunk_subchunk_width  = proc.get_dimensions().value().get_chunk_subchunk_width().value();
        auto                 chunk_count           = proc.get_dimensions().value().get_chunk_count().value();
        auto                 format                = proc.get_format().value();
        //Make sure channels is valid.
        auto                 maybe_channels = sgns::sgprocessing::InputTypes::GetImageChannels( proc.get_format().value() );
        m_logger->debug( "Channels to process {}", maybe_channels.value() );
        if ( !maybe_channels )
        {
            return std::vector<uint8_t>();
        }
        auto channels = maybe_channels.value();        

        //for ( auto image : *imageData_ )
        //{
            std::vector<uint8_t> output(imageData.size());
            std::transform(imageData.begin(), imageData.end(), output.begin(),
                            []( char c ) { return static_cast<uint8_t>( c ); } );
            //ImageSplitter animageSplit( output, task.block_line_stride(), task.block_stride(), task.block_len() );
            ImageSplitter animageSplit(output, block_line_stride, block_stride, block_len, channels);
            auto          dataindex           = 0;
            ImageSplitter ChunkSplit( animageSplit.GetPart( dataindex ), chunk_line_stride, chunk_stride,
                                      animageSplit.GetPartHeightActual( dataindex ) / chunk_subchunk_height *
                                            chunk_line_stride, channels);
            
            auto totalChunks = proc.get_dimensions().value().get_chunk_count().value();
            m_progress = 0.0f; // Reset progress at start
            
            for ( int chunkIdx = 0; chunkIdx < totalChunks; ++chunkIdx )
            {
                m_logger->info( "Chunk IDX {} Total {}",
                                chunkIdx,
                                totalChunks );
                std::vector<uint8_t> shahash( SHA256_DIGEST_LENGTH );

                // Chunk result hash should be calculated
                size_t chunkHash = 0;

                auto procresults =
                    Process( ChunkSplit.GetPart( chunkIdx ), modelFile_bytes, channels, ChunkSplit.GetPartWidthActual( chunkIdx ),
                                ChunkSplit.GetPartHeightActual( chunkIdx ) );

                const float *data     = procresults->host<float>();
                size_t       dataSize = procresults->elementSize() * sizeof( float );
                shahash               = sgprocmanagersha::sha256( data, dataSize );
                std::string hashString( shahash.begin(), shahash.end() );
                chunkhashes.push_back( shahash );

                std::string combinedHash = std::string(subTaskResultHash.begin(), subTaskResultHash.end()) + hashString;
                subTaskResultHash = sgprocmanagersha::sha256( combinedHash.c_str(), combinedHash.length() );
                
                // Update progress: round to 2 decimal places
                m_progress = std::round(((chunkIdx + 1) * 100.0f / totalChunks) * 100.0f) / 100.0f;
                
                std::this_thread::sleep_for(std::chrono::milliseconds(100));
            }
            return subTaskResultHash;
        //}
        //return subTaskResultHash;
    }

    std::unique_ptr<MNN::Tensor> MNN_Image::Process(const std::vector<uint8_t>& imgdata, 
                                                         std::vector<uint8_t>& modelFile, 
                                                         const int channels, 
                                                         const int origwidth,
                                                         const int origheight, 
                                                         const std::string filename) 
    {
        std::vector<uint8_t> ret_vect(imgdata);

        // Get Target Width
        const int targetWidth = static_cast<int>((float)origwidth / (float)OUTPUT_STRIDE) * OUTPUT_STRIDE + 1;
        const int targetHeight = static_cast<int>((float)origheight / (float)OUTPUT_STRIDE) * OUTPUT_STRIDE + 1;

        // Scale
        CV::Point scale;
        scale.fX = (float)origwidth / (float)targetWidth;
        scale.fY = (float)origheight / (float)targetHeight;

        // Create net and session
        const void* buffer = static_cast<const void*>( modelFile.data() );
        auto mnnNet = std::shared_ptr<MNN::Interpreter>( MNN::Interpreter::createFromBuffer( buffer, modelFile.size() ) );

        //auto backendConfig           = new MNN::BackendConfig();
        //backendConfig->power         = MNN::BackendConfig::Power_Low;
        //backendConfig->queuePriority = 0.1f;

        MNN::ScheduleConfig netConfig;
        netConfig.type      = MNN_FORWARD_VULKAN;
        netConfig.numThread = 4;
        netConfig.mode = 0;
        //netConfig.backendConfig = backendConfig;
        auto session        = mnnNet->createSession( netConfig );

        auto input = mnnNet->getSessionInput( session, nullptr );

        if ( input->elementSize() <= 4 )
        {
            mnnNet->resizeTensor( input, { 1, 3, targetHeight, targetWidth } );
            mnnNet->resizeSession( session );
        }

        // Preprocess input image
        {
            const float              means[3] = { 127.5f, 127.5f, 127.5f };
            const float              norms[3] = { 2.0f / 255.0f, 2.0f / 255.0f, 2.0f / 255.0f };
            CV::ImageProcess::Config preProcessConfig;
            ::memcpy( preProcessConfig.mean, means, sizeof( means ) );
            ::memcpy( preProcessConfig.normal, norms, sizeof( norms ) );
            preProcessConfig.sourceFormat = CV::RGBA;

            if (channels == 3)
            {
                preProcessConfig.sourceFormat = CV::RGB;
            }
            preProcessConfig.destFormat = CV::RGB;
            preProcessConfig.filterType = CV::BILINEAR;

            auto       pretreat = std::shared_ptr<CV::ImageProcess>( CV::ImageProcess::create( preProcessConfig ) );
            CV::Matrix trans;

            // Dst -> [0, 1]
            trans.postScale( 1.0 / targetWidth, 1.0 / targetHeight );
            //[0, 1] -> Src
            trans.postScale( origwidth, origheight );

            pretreat->setMatrix( trans );
            pretreat->convert( ret_vect.data(), origwidth, origheight, 0, input );
        }

        // Log preprocessed input tensor data hash
        {
            const float *inputData     = input->host<float>();
            size_t       inputDataSize = input->elementSize() * sizeof( float );
        }

        {
            AUTOTIME;
            mnnNet->runSession( session );
        }

        auto outputTensor = mnnNet->getSessionOutput( session, nullptr );
        auto outputHost   = std::make_unique<MNN::Tensor>( outputTensor, MNN::Tensor::CAFFE );
        outputTensor->copyToHostTensor( outputHost.get() );

        return outputHost;
    }

}
