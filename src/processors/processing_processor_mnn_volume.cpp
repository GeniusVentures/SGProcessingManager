#include "processors/processing_processor_mnn_volume.hpp"
#include <functional>
#include <thread>
#include <sstream>
#include <algorithm>
#include <cstring>
#include <openssl/sha.h> // For SHA256_DIGEST_LENGTH
#include "util/sha256.hpp"

namespace sgns::sgprocessing
{
    using namespace MNN;

    namespace
    {
        std::string FormatTensorShape( const MNN::Tensor &tensor )
        {
            std::ostringstream out;
            const int dims = tensor.dimensions();
            out << "[";
            for ( int i = 0; i < dims; ++i )
            {
                if ( i > 0 )
                {
                    out << ", ";
                }
                out << tensor.length( i );
            }
            out << "]";
            return out.str();
        }
    }

    std::vector<uint8_t> MNN_Volume::StartProcessing( std::vector<std::vector<uint8_t>> &chunkhashes,
                                                       const sgns::IoDeclaration         &proc,
                                                       std::vector<char>                 &volumeData,
                                                       std::vector<char>                 &modelFile )
    {
        std::vector<uint8_t> modelFile_bytes;
        modelFile_bytes.assign(modelFile.begin(), modelFile.end());

        std::vector<uint8_t> subTaskResultHash(SHA256_DIGEST_LENGTH);

        if ( !proc.get_dimensions() || !proc.get_dimensions()->get_width() ||
             !proc.get_dimensions()->get_height() || !proc.get_dimensions()->get_chunk_count() )
        {
            m_logger->error( "Texture3D input missing width/height/chunk_count" );
            return std::vector<uint8_t>();
        }

        const int width  = static_cast<int>( proc.get_dimensions()->get_width().value() );
        const int height = static_cast<int>( proc.get_dimensions()->get_height().value() );
        const int depth  = static_cast<int>( proc.get_dimensions()->get_chunk_count().value() );

        const int patchWidth = proc.get_dimensions()->get_chunk_subchunk_width().value_or( width );
        const int patchHeight = proc.get_dimensions()->get_chunk_subchunk_height().value_or( height );
        const int patchDepth = proc.get_dimensions()->get_block_len().value_or( depth );

        const int strideX = proc.get_dimensions()->get_chunk_stride().value_or( patchWidth );
        const int strideY = proc.get_dimensions()->get_chunk_line_stride().value_or( patchHeight );
        const int strideZ = proc.get_dimensions()->get_block_stride().value_or( patchDepth );

        if ( patchWidth <= 0 || patchHeight <= 0 || patchDepth <= 0 ||
             strideX <= 0 || strideY <= 0 || strideZ <= 0 )
        {
            m_logger->error( "Invalid patch size or stride values for texture3D" );
            return std::vector<uint8_t>();
        }

        const size_t expectedBytes = static_cast<size_t>( width ) * height * depth * sizeof( float );
        if ( volumeData.size() < expectedBytes )
        {
            m_logger->error( "Texture3D input size {} bytes is smaller than expected {} bytes",
                             volumeData.size(),
                             expectedBytes );
            return std::vector<uint8_t>();
        }

        std::vector<float> volumeFloats;
        volumeFloats.resize( static_cast<size_t>( width ) * height * depth );
        std::memcpy( volumeFloats.data(), volumeData.data(), expectedBytes );

        m_logger->info( "Processing volume input: {}x{}x{} ({} floats)",
                        width,
                        height,
                        depth,
                        volumeFloats.size() );

        m_logger->info( "Patch size: {}x{}x{} | Stride: {}x{}x{}",
                patchWidth,
                patchHeight,
                patchDepth,
                strideX,
                strideY,
                strideZ );

        m_progress = 0.0f;

        std::vector<uint8_t> shahash( SHA256_DIGEST_LENGTH );

        size_t patchIndex = 0;
        for ( int z = 0; z < depth; z += strideZ )
        {
            for ( int y = 0; y < height; y += strideY )
            {
                for ( int x = 0; x < width; x += strideX )
                {
                    std::vector<float> patch;
                    patch.resize( static_cast<size_t>( patchWidth ) * patchHeight * patchDepth, 0.0f );

                    for ( int dz = 0; dz < patchDepth; ++dz )
                    {
                        const int srcZ = z + dz;
                        if ( srcZ >= depth )
                        {
                            continue;
                        }
                        for ( int dy = 0; dy < patchHeight; ++dy )
                        {
                            const int srcY = y + dy;
                            if ( srcY >= height )
                            {
                                continue;
                            }
                            for ( int dx = 0; dx < patchWidth; ++dx )
                            {
                                const int srcX = x + dx;
                                if ( srcX >= width )
                                {
                                    continue;
                                }
                                const size_t srcIndex =
                                    static_cast<size_t>( srcZ ) * height * width +
                                    static_cast<size_t>( srcY ) * width +
                                    static_cast<size_t>( srcX );
                                const size_t dstIndex =
                                    static_cast<size_t>( dz ) * patchHeight * patchWidth +
                                    static_cast<size_t>( dy ) * patchWidth +
                                    static_cast<size_t>( dx );
                                patch[dstIndex] = volumeFloats[srcIndex];
                            }
                        }
                    }

                    auto procresults = Process( patch, modelFile_bytes, patchWidth, patchHeight, patchDepth );
                    const float *data     = procresults->host<float>();
                    size_t       dataSize = procresults->elementSize() * sizeof( float );

                    if ( patchIndex == 0 )
                    {
                        std::ostringstream sample;
                        size_t sampleCount = std::min<size_t>( 16, procresults->elementSize() );
                        sample << "Output sample (first " << sampleCount << "): ";
                        for ( size_t i = 0; i < sampleCount; ++i )
                        {
                            if ( i > 0 )
                            {
                                sample << ", ";
                            }
                            sample << data[i];
                        }
                        m_logger->info( "{}", sample.str() );
                    }

                    shahash = sgprocmanagersha::sha256( data, dataSize );
                    std::string hashString( shahash.begin(), shahash.end() );
                    chunkhashes.push_back( shahash );

                    std::string combinedHash = std::string(subTaskResultHash.begin(), subTaskResultHash.end()) + hashString;
                    subTaskResultHash = sgprocmanagersha::sha256( combinedHash.c_str(), combinedHash.length() );

                    ++patchIndex;
                }
            }
        }

        m_progress = 100.0f;

        m_logger->info( "Volume processing complete" );

        return subTaskResultHash;
    }

    std::unique_ptr<MNN::Tensor> MNN_Volume::Process( const std::vector<float> &volumeData,
                                                        std::vector<uint8_t> &modelFile,
                                                        const int width,
                                                        const int height,
                                                        const int depth )
    {
        m_logger->info( "Creating MNN interpreter from model file" );

        auto interpreter = std::shared_ptr<MNN::Interpreter>(
            MNN::Interpreter::createFromBuffer(modelFile.data(), modelFile.size())
        );

        if (!interpreter) {
            m_logger->error( "Failed to create MNN interpreter" );
            return std::make_unique<MNN::Tensor>();
        }

        MNN::ScheduleConfig config;
        config.type = MNN_FORWARD_VULKAN;
        config.numThread = 4;

        auto session = interpreter->createSession(config);
        if (!session) {
            m_logger->error( "Failed to create MNN session" );
            return std::make_unique<MNN::Tensor>();
        }

        auto inputTensors = interpreter->getSessionInputAll(session);
        m_logger->info( "Model has {} input tensor(s)", inputTensors.size() );

        for (const auto& inputPair : inputTensors) {
            m_logger->info( "Input '{}': shape {}", 
                           inputPair.first,
                           FormatTensorShape( *inputPair.second ) );
        }

        for (const auto& inputPair : inputTensors) {
            auto tensor = inputPair.second;
            if (tensor->elementSize() <= 4) {
                m_logger->info( "Resizing '{}' to [1, 1, {}, {}, {}]", inputPair.first, depth, height, width );
                interpreter->resizeTensor( tensor, { 1, 1, depth, height, width } );
            }
        }
        interpreter->resizeSession( session );

        for (const auto& inputPair : inputTensors) {
            m_logger->info( "After resize '{}': shape {}", 
                           inputPair.first,
                           FormatTensorShape( *inputPair.second ) );
        }

        for (const auto& inputPair : inputTensors) {
            auto tensor = inputPair.second;
            MNN::Tensor inputTensorUser(tensor, tensor->getDimensionType());

            auto inputData = inputTensorUser.host<float>();
            const size_t elementCount = inputTensorUser.elementSize();
            const size_t expectedElements = static_cast<size_t>( width ) * height * depth;
            if ( elementCount != expectedElements )
            {
                m_logger->warn( "Input tensor element count {} does not match expected volume size {}",
                                elementCount,
                                expectedElements );
            }
            const size_t copyCount = std::min( elementCount, volumeData.size() );
            for ( size_t i = 0; i < copyCount; ++i )
            {
                inputData[i] = volumeData[i];
            }
            for ( size_t i = copyCount; i < elementCount; ++i )
            {
                inputData[i] = 0.0f;
            }

            tensor->copyFromHostTensor(&inputTensorUser);
            m_logger->info( "Filled '{}' with {} elements", inputPair.first, inputTensorUser.elementSize() );
        }

        m_logger->info( "Running MNN inference" );
        interpreter->runSession(session);

        auto outputTensor = interpreter->getSessionOutput(session, nullptr);
        if (!outputTensor) {
            m_logger->error( "Failed to get output tensor" );
            return std::make_unique<MNN::Tensor>();
        }

        m_logger->info( "Output tensor shape: {}", FormatTensorShape( *outputTensor ) );

        auto outputHost = std::make_unique<MNN::Tensor>(outputTensor, outputTensor->getDimensionType());
        outputTensor->copyToHostTensor(outputHost.get());

        m_logger->info( "MNN inference complete" );

        return outputHost;
    }
}
