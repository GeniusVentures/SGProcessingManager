#include "processors/processing_processor_mnn_buffer.hpp"

#include <algorithm>
#include <cstdint>
#include <cstring>
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

            if ( dims == 4 )
            {
                if ( dimType == MNN::Tensor::CAFFE )
                {
                    const int h = tensor.length( 2 );
                    const int w = tensor.length( 3 );
                    const int hIndex = layout.length_is_first_spatial ? i : 0;
                    const int wIndex = layout.length_is_first_spatial ? 0 : i;
                    return ( static_cast<size_t>( c ) * h + static_cast<size_t>( hIndex ) ) * w + static_cast<size_t>( wIndex );
                }
                const int h = tensor.length( 1 );
                const int w = tensor.length( 2 );
                const int hIndex = layout.length_is_first_spatial ? i : 0;
                const int wIndex = layout.length_is_first_spatial ? 0 : i;
                return ( static_cast<size_t>( hIndex ) * w + static_cast<size_t>( wIndex ) ) * layout.channels + static_cast<size_t>( c );
            }

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

    ProcessingResult MNN_Buffer::StartProcessing( std::vector<std::vector<uint8_t>> &chunkhashes,
                                                  const sgns::IoDeclaration         &proc,
                                                  std::vector<char>                 &bufferData,
                                                  std::vector<char>                 &modelFile,
                                                  const std::vector<sgns::Parameter> *parameters )
    {
        (void)parameters;
        std::vector<uint8_t> modelFileBytes;
        modelFileBytes.assign( modelFile.begin(), modelFile.end() );

        if ( !proc.get_dimensions() || !proc.get_dimensions()->get_width() )
        {
            m_logger->error( "Buffer input missing width" );
            return ProcessingResult{};
        }

        const int length = static_cast<int>( proc.get_dimensions()->get_width().value() );
        const int patchLength = proc.get_dimensions()->get_block_len().value_or( length );
        const int stride = proc.get_dimensions()->get_chunk_stride().value_or( patchLength );

        if ( length <= 0 || patchLength <= 0 || stride <= 0 )
        {
            m_logger->error( "Invalid buffer length/patch/stride values" );
            return ProcessingResult{};
        }

        const auto format = proc.get_format().value_or( sgns::InputFormat::INT8 );
        if ( format != sgns::InputFormat::INT8 )
        {
            m_logger->error( "Buffer supports INT8 format only" );
            return ProcessingResult{};
        }

        const size_t expectedElements = static_cast<size_t>( length );
        const size_t expectedBytes = expectedElements * sizeof( int8_t );
        if ( bufferData.size() < expectedBytes )
        {
            m_logger->error( "Buffer input size {} bytes is smaller than expected {} bytes",
                             bufferData.size(),
                             expectedBytes );
            return ProcessingResult{};
        }

        std::vector<float> signalValues;
        signalValues.resize( expectedElements );
        const auto *src = reinterpret_cast<const int8_t *>( bufferData.data() );
        for ( size_t i = 0; i < expectedElements; ++i )
        {
            signalValues[i] = static_cast<float>( src[i] );
        }

        m_logger->info( "Processing buffer input length: {} | patch: {} | stride: {}", length, patchLength, stride );

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

                    const size_t weightIndex = static_cast<size_t>( outIndex );
                    stitchedWeights[weightIndex] += 1.0f;

                    for ( int c = 0; c < outputChannels; ++c )
                    {
                        const size_t srcIndex = OutputIndex1D( *procresults, outputLayout, c, i );
                        const size_t dstIndex = static_cast<size_t>( c ) * length + static_cast<size_t>( outIndex );
                        stitchedOutput[dstIndex] += data[srcIndex];
                    }
                }
            }

            std::vector<uint8_t> shahash = sgprocmanagersha::sha256( data, dataSize );
            std::string hashString( shahash.begin(), shahash.end() );
            chunkhashes.push_back( shahash );

            std::string combinedHash = std::string( subTaskResultHash.begin(), subTaskResultHash.end() ) + hashString;
            subTaskResultHash = sgprocmanagersha::sha256( combinedHash.c_str(), combinedHash.length() );
        }

        if ( !stitchedOutput.empty() )
        {
            for ( int c = 0; c < outputChannels; ++c )
            {
                for ( int i = 0; i < length; ++i )
                {
                    const size_t weightIndex = static_cast<size_t>( i );
                    if ( stitchedWeights[weightIndex] <= 0.0f )
                    {
                        continue;
                    }
                    const size_t dstIndex = static_cast<size_t>( c ) * length + static_cast<size_t>( i );
                    stitchedOutput[dstIndex] /= stitchedWeights[weightIndex];
                }
            }
        }

        m_progress = 100.0f;

        ProcessingResult result;
        result.hash = subTaskResultHash;

        if ( !stitchedOutput.empty() )
        {
            const size_t byteCount = stitchedOutput.size() * sizeof( float );
            std::vector<char> outputBytes( byteCount );
            std::memcpy( outputBytes.data(), stitchedOutput.data(), byteCount );

            result.output_buffers = std::make_shared<std::pair<std::vector<std::string>, std::vector<std::vector<char>>>>();
            result.output_buffers->first.push_back( "" );
            result.output_buffers->second.push_back( std::move( outputBytes ) );
        }

        m_logger->info( "Buffer processing complete" );

        return result;
    }

    std::unique_ptr<MNN::Tensor> MNN_Buffer::Process( const std::vector<float> &signalData,
                                                      std::vector<uint8_t>    &modelFile,
                                                      int                      length )
    {
        auto interpreter = std::shared_ptr<MNN::Interpreter>(
            MNN::Interpreter::createFromBuffer( modelFile.data(), modelFile.size() ) );

        if ( !interpreter )
        {
            m_logger->error( "Failed to create MNN interpreter" );
            return std::make_unique<MNN::Tensor>();
        }

        MNN::ScheduleConfig config;
        config.type = MNN_FORWARD_CPU;
        config.numThread = 4;

        auto session = interpreter->createSession( config );
        if ( !session )
        {
            m_logger->error( "Failed to create MNN session" );
            return std::make_unique<MNN::Tensor>();
        }

        auto inputTensors = interpreter->getSessionInputAll( session );
        if ( inputTensors.empty() )
        {
            m_logger->error( "Model has no inputs" );
            return std::make_unique<MNN::Tensor>();
        }

        for ( const auto &inputPair : inputTensors )
        {
            auto tensor = inputPair.second;
            const int dims = tensor->dimensions();
            const auto dimType = tensor->getDimensionType();
            if ( tensor->elementSize() <= 4 )
            {
                if ( dims == 4 )
                {
                    if ( dimType == MNN::Tensor::TENSORFLOW )
                    {
                        interpreter->resizeTensor( tensor, { 1, length, 1, 1 } );
                    }
                    else
                    {
                        interpreter->resizeTensor( tensor, { 1, 1, 1, length } );
                    }
                }
                else if ( dims == 3 )
                {
                    if ( dimType == MNN::Tensor::TENSORFLOW )
                    {
                        interpreter->resizeTensor( tensor, { 1, length, 1 } );
                    }
                    else
                    {
                        interpreter->resizeTensor( tensor, { 1, 1, length } );
                    }
                }
                else if ( dims == 2 )
                {
                    interpreter->resizeTensor( tensor, { 1, length } );
                }
                else
                {
                    interpreter->resizeTensor( tensor, { 1, 1, 1, length } );
                }
            }
        }
        interpreter->resizeSession( session );

        for ( const auto &inputPair : inputTensors )
        {
            auto tensor = inputPair.second;
            MNN::Tensor inputTensorUser( tensor, tensor->getDimensionType() );

            auto inputData = inputTensorUser.host<float>();
            const size_t elementCount = inputTensorUser.elementSize();
            const size_t copyCount = std::min( elementCount, signalData.size() );
            for ( size_t i = 0; i < copyCount; ++i )
            {
                inputData[i] = signalData[i];
            }
            for ( size_t i = copyCount; i < elementCount; ++i )
            {
                inputData[i] = 0.0f;
            }

            tensor->copyFromHostTensor( &inputTensorUser );
        }

        interpreter->runSession( session );

        auto outputTensor = interpreter->getSessionOutput( session, nullptr );
        if ( !outputTensor )
        {
            m_logger->error( "Failed to get output tensor" );
            return std::make_unique<MNN::Tensor>();
        }

        auto outputHost = std::make_unique<MNN::Tensor>( outputTensor, MNN::Tensor::CAFFE );
        outputTensor->copyToHostTensor( outputHost.get() );

        return outputHost;
    }
}
