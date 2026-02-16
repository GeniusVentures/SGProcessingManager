#include "processors/processing_processor_mnn_int.hpp"

#include <algorithm>
#include <cstdint>
#include <cstring>
#include <limits>
#include <openssl/sha.h>
#include "util/sha256.hpp"

namespace sgns::sgprocessing
{
    using namespace MNN;

    namespace
    {
        std::vector<int> ComputeWindowStarts( int length, int roi, int stride )
        {
            std::vector<int> starts;
            if ( length <= roi )
            {
                starts.push_back( 0 );
                return starts;
            }

            const int step = std::max( 1, stride );
            for ( int pos = 0; pos <= length - roi; pos += step )
            {
                starts.push_back( pos );
            }

            const int last = length - roi;
            if ( starts.empty() || starts.back() != last )
            {
                starts.push_back( last );
            }

            return starts;
        }

        struct OutputLayout
        {
            int channels = 1;
            int length = 1;
            bool length_is_first_spatial = false;
        };

        OutputLayout GetOutputLayout( const MNN::Tensor &tensor )
        {
            OutputLayout layout;
            const int dims = tensor.dimensions();
            const auto dimType = tensor.getDimensionType();

            if ( dims == 4 )
            {
                if ( dimType == MNN::Tensor::CAFFE )
                {
                    layout.channels = tensor.length( 1 );
                    const int h = tensor.length( 2 );
                    const int w = tensor.length( 3 );
                    layout.length = std::max( h, w );
                    layout.length_is_first_spatial = ( h >= w );
                }
                else
                {
                    layout.channels = tensor.length( 3 );
                    const int h = tensor.length( 1 );
                    const int w = tensor.length( 2 );
                    layout.length = std::max( h, w );
                    layout.length_is_first_spatial = ( h >= w );
                }
            }
            else if ( dims == 3 )
            {
                if ( dimType == MNN::Tensor::CAFFE )
                {
                    layout.channels = tensor.length( 1 );
                    layout.length = tensor.length( 2 );
                }
                else
                {
                    layout.channels = tensor.length( 2 );
                    layout.length = tensor.length( 1 );
                }
            }
            else if ( dims == 2 )
            {
                layout.channels = 1;
                layout.length = tensor.length( 1 );
            }
            else
            {
                layout.channels = 1;
                layout.length = static_cast<int>( tensor.elementSize() );
            }

            return layout;
        }

        size_t OutputIndex1D( const MNN::Tensor &tensor, const OutputLayout &layout, int c, int i )
        {
            const int dims = tensor.dimensions();
            const auto dimType = tensor.getDimensionType();

            if ( dims == 3 )
            {
                if ( dimType == MNN::Tensor::CAFFE )
                {
                    return static_cast<size_t>( c ) * layout.length + static_cast<size_t>( i );
                }
                return static_cast<size_t>( i ) * layout.channels + static_cast<size_t>( c );
            }

            return static_cast<size_t>( i );
        }
    }

    ProcessingResult MNN_Int::StartProcessing( std::vector<std::vector<uint8_t>> &chunkhashes,
                                                const sgns::IoDeclaration         &proc,
                                                std::vector<char>                 &intData,
                                                std::vector<char>                 &modelFile,
                                                const std::vector<sgns::Parameter> *parameters )
    {
        (void)parameters;
        std::vector<uint8_t> modelFileBytes;
        modelFileBytes.assign( modelFile.begin(), modelFile.end() );

        if ( !proc.get_dimensions() || !proc.get_dimensions()->get_width() )
        {
            m_logger->error( "Int input missing width" );
            return ProcessingResult{};
        }

        const int length = static_cast<int>( proc.get_dimensions()->get_width().value() );
        const int patchLength = proc.get_dimensions()->get_block_len().value_or( length );
        const int stride = proc.get_dimensions()->get_chunk_stride().value_or( patchLength );

        if ( length <= 0 || patchLength <= 0 || stride <= 0 )
        {
            m_logger->error( "Invalid int length/patch/stride values" );
            return ProcessingResult{};
        }

        const auto format = proc.get_format().value_or( sgns::InputFormat::INT32 );
        if ( format != sgns::InputFormat::INT32 && format != sgns::InputFormat::INT16 &&
             format != sgns::InputFormat::INT8 )
        {
            m_logger->error( "Int supports INT32/INT16/INT8 formats only" );
            return ProcessingResult{};
        }

        const size_t expectedElements = static_cast<size_t>( length );
        const size_t bytesPerElement = ( format == sgns::InputFormat::INT32 ) ? sizeof( int32_t )
                                      : ( format == sgns::InputFormat::INT16 ) ? sizeof( int16_t )
                                      : sizeof( int8_t );
        const size_t expectedBytes = expectedElements * bytesPerElement;
        if ( intData.size() < expectedBytes )
        {
            m_logger->error( "Int input size {} bytes is smaller than expected {} bytes",
                             intData.size(),
                             expectedBytes );
            return ProcessingResult{};
        }

        std::vector<float> signalValues;
        signalValues.resize( expectedElements );
        if ( format == sgns::InputFormat::INT32 )
        {
            const auto *src = reinterpret_cast<const int32_t *>( intData.data() );
            for ( size_t i = 0; i < expectedElements; ++i )
            {
                signalValues[i] = static_cast<float>( src[i] );
            }
        }
        else if ( format == sgns::InputFormat::INT16 )
        {
            const auto *src = reinterpret_cast<const int16_t *>( intData.data() );
            for ( size_t i = 0; i < expectedElements; ++i )
            {
                signalValues[i] = static_cast<float>( src[i] );
            }
        }
        else
        {
            const auto *src = reinterpret_cast<const int8_t *>( intData.data() );
            for ( size_t i = 0; i < expectedElements; ++i )
            {
                signalValues[i] = static_cast<float>( src[i] );
            }
        }

        m_logger->info( "Processing int input length: {} | patch: {} | stride: {}", length, patchLength, stride );

        std::vector<uint8_t> subTaskResultHash( SHA256_DIGEST_LENGTH );
        const auto starts = ComputeWindowStarts( length, patchLength, stride );

        int outputChannels = 0;
        int outputLength = patchLength;
        OutputLayout outputLayout;
        std::vector<float> stitchedOutput;
        std::vector<float> stitchedWeights;

        for ( int start : starts )
        {
            std::vector<float> patch;
            patch.resize( static_cast<size_t>( patchLength ), 0.0f );
            for ( int i = 0; i < patchLength; ++i )
            {
                const int srcIndex = start + i;
                if ( srcIndex >= length )
                {
                    break;
                }
                patch[static_cast<size_t>( i )] = signalValues[static_cast<size_t>( srcIndex )];
            }

            auto procresults = Process( patch, modelFileBytes, patchLength );
            const float *data = procresults->host<float>();
            size_t dataSize = procresults->elementSize() * sizeof( float );

            if ( outputChannels == 0 )
            {
                outputLayout = GetOutputLayout( *procresults );
                outputChannels = outputLayout.channels;
                outputLength = outputLayout.length;

                stitchedOutput.assign( static_cast<size_t>( outputChannels ) * length, 0.0f );
                stitchedWeights.assign( static_cast<size_t>( length ), 0.0f );
            }

            if ( outputLength == patchLength )
            {
                for ( int i = 0; i < patchLength; ++i )
                {
                    const int outIndex = start + i;
                    if ( outIndex >= length )
                    {
                        break;
                    }
                    
                    for ( int c = 0; c < outputChannels; ++c )
                    {
                        const size_t srcIdx = OutputIndex1D( *procresults, outputLayout, c, i );
                        const size_t dstIdx = static_cast<size_t>( c * length + outIndex );

                        stitchedOutput[dstIdx] += data[srcIdx];
                    }

                    stitchedWeights[static_cast<size_t>( outIndex )] += 1.0f;
                }
            }

            auto hash = sgprocmanagersha::sha256( data , dataSize );
            chunkhashes.emplace_back( hash.begin(), hash.end() );
        }

        for ( size_t idx = 0; idx < stitchedOutput.size(); ++idx )
        {
            const int spatialIdx = static_cast<int>( idx % length );
            const float weight = stitchedWeights[static_cast<size_t>( spatialIdx )];
            if ( weight > 0.0f )
            {
                stitchedOutput[idx] /= weight;
            }
        }

        std::string stitchedStr( reinterpret_cast<const char *>( stitchedOutput.data() ),
                                  stitchedOutput.size() * sizeof( float ) );
        subTaskResultHash = sgprocmanagersha::sha256( stitchedStr.c_str(), stitchedStr.size() );

        m_progress = 100.0f;

        ProcessingResult result;
        result.hash = subTaskResultHash;

        if ( !stitchedOutput.empty() )
        {
            const size_t byteCount = stitchedOutput.size() * sizeof( float );
            std::vector<char> outputBytes( byteCount );
            std::memcpy( outputBytes.data(), stitchedOutput.data(), byteCount );

            result.output_buffers =
                std::make_shared<std::pair<std::vector<std::string>, std::vector<std::vector<char>>>>();
            result.output_buffers->first.push_back( "" );
            result.output_buffers->second.push_back( std::move( outputBytes ) );
        }

        m_logger->info( "Int processing complete" );
        return result;
    }

    std::unique_ptr<MNN::Tensor> MNN_Int::Process( const std::vector<float> &signalData,
                                                     std::vector<uint8_t>    &modelFile,
                                                     int                      length )
    {
        auto interpreter = std::unique_ptr<MNN::Interpreter>( MNN::Interpreter::createFromBuffer( modelFile.data(), modelFile.size() ) );
        if ( !interpreter )
        {
            m_logger->error( "Failed to create MNN interpreter from buffer" );
            return nullptr;
        }

        MNN::ScheduleConfig config;
        config.type = MNN_FORWARD_CPU;
        config.numThread = 4;
        config.backendConfig = nullptr;

        auto session = interpreter->createSession( config );
        if ( !session )
        {
            m_logger->error( "Failed to create MNN session" );
            return nullptr;
        }

        auto inputTensor = interpreter->getSessionInput( session, nullptr );
        if ( !inputTensor )
        {
            m_logger->error( "Failed to get input tensor" );
            return nullptr;
        }

        MNN::Tensor inputTensorUser( inputTensor, inputTensor->getDimensionType() );
        auto inputPtr = inputTensorUser.host<float>();
        std::memcpy( inputPtr, signalData.data(), length * sizeof( float ) );
        inputTensor->copyFromHostTensor( &inputTensorUser );

        interpreter->runSession( session );

        auto outputTensor = interpreter->getSessionOutput( session, nullptr );
        if ( !outputTensor )
        {
            m_logger->error( "Failed to get output tensor" );
            return nullptr;
        }

        MNN::Tensor::DimensionType outputDimType = outputTensor->getDimensionType();
        auto outputUserTensor = std::make_unique<MNN::Tensor>( outputTensor, outputDimType );
        outputTensor->copyToHostTensor( outputUserTensor.get() );

        return outputUserTensor;
    }
}
