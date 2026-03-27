#include "processors/processing_processor_mnn_vec3.hpp"

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

        float HalfToFloat( uint16_t value )
        {
            const uint16_t sign = static_cast<uint16_t>( value >> 15 );
            const uint16_t exponent = static_cast<uint16_t>( ( value >> 10 ) & 0x1F );
            const uint16_t mantissa = static_cast<uint16_t>( value & 0x03FF );

            uint32_t sign32 = static_cast<uint32_t>( sign ) << 31;
            uint32_t exponent32 = 0;
            uint32_t mantissa32 = 0;

            if ( exponent == 0 )
            {
                if ( mantissa == 0 )
                {
                    exponent32 = 0;
                    mantissa32 = 0;
                }
                else
                {
                    int shift = 0;
                    uint16_t mant = mantissa;
                    while ( ( mant & 0x0400 ) == 0 )
                    {
                        mant <<= 1;
                        ++shift;
                    }
                    mant &= 0x03FF;
                    exponent32 = static_cast<uint32_t>( 127 - 15 - shift ) << 23;
                    mantissa32 = static_cast<uint32_t>( mant ) << 13;
                }
            }
            else if ( exponent == 31 )
            {
                exponent32 = 0xFFu << 23;
                mantissa32 = static_cast<uint32_t>( mantissa ) << 13;
            }
            else
            {
                exponent32 = static_cast<uint32_t>( exponent + ( 127 - 15 ) ) << 23;
                mantissa32 = static_cast<uint32_t>( mantissa ) << 13;
            }

            uint32_t bits = sign32 | exponent32 | mantissa32;
            float result = 0.0f;
            std::memcpy( &result, &bits, sizeof( result ) );
            return result;
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
                    return ( static_cast<size_t>( c ) * h + static_cast<size_t>( hIndex ) ) * w +
                        static_cast<size_t>( wIndex );
                }
                const int h = tensor.length( 1 );
                const int w = tensor.length( 2 );
                const int hIndex = layout.length_is_first_spatial ? i : 0;
                const int wIndex = layout.length_is_first_spatial ? 0 : i;
                return ( static_cast<size_t>( hIndex ) * w + static_cast<size_t>( wIndex ) ) *
                    layout.channels + static_cast<size_t>( c );
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

    ProcessingResult MNN_Vec3::StartProcessing( std::vector<std::vector<uint8_t>> &chunkhashes,
                                               const sgns::IoDeclaration         &proc,
                                               std::vector<char>                 &vec3Data,
                                               std::vector<char>                 &modelFile,
                                               const std::vector<sgns::Parameter> *parameters )
    {
        (void)parameters;
        std::vector<uint8_t> modelFileBytes;
        modelFileBytes.assign( modelFile.begin(), modelFile.end() );

        if ( !proc.get_dimensions() || !proc.get_dimensions()->get_width() )
        {
            m_logger->error( "Vec3 input missing width" );
            return ProcessingResult{};
        }

        const int vectorCount = static_cast<int>( proc.get_dimensions()->get_width().value() );
        const int patchVectors = proc.get_dimensions()->get_block_len().value_or( vectorCount );
        const int stride = proc.get_dimensions()->get_chunk_stride().value_or( patchVectors );

        if ( vectorCount <= 0 || patchVectors <= 0 || stride <= 0 )
        {
            m_logger->error( "Invalid vec3 length/patch/stride values" );
            return ProcessingResult{};
        }

        const auto format = proc.get_format().value_or( sgns::InputFormat::FLOAT32 );
        if ( format != sgns::InputFormat::FLOAT32 && format != sgns::InputFormat::FLOAT16 )
        {
            m_logger->error( "Vec3 supports FLOAT32/FLOAT16 formats only" );
            return ProcessingResult{};
        }

        const size_t expectedElements = static_cast<size_t>( vectorCount ) * 3;
        const size_t bytesPerElement = ( format == sgns::InputFormat::FLOAT16 ) ? sizeof( uint16_t ) :
            sizeof( float );
        const size_t expectedBytes = expectedElements * bytesPerElement;
        if ( vec3Data.size() < expectedBytes )
        {
            m_logger->error( "Vec3 input size {} bytes is smaller than expected {} bytes",
                             vec3Data.size(),
                             expectedBytes );
            return ProcessingResult{};
        }

        std::vector<float> signalValues;
        signalValues.resize( expectedElements );
        if ( format == sgns::InputFormat::FLOAT32 )
        {
            const auto *src = reinterpret_cast<const float *>( vec3Data.data() );
            std::memcpy( signalValues.data(), src, expectedBytes );
        }
        else
        {
            const auto *src = reinterpret_cast<const uint16_t *>( vec3Data.data() );
            for ( size_t i = 0; i < expectedElements; ++i )
            {
                signalValues[i] = HalfToFloat( src[i] );
            }
        }

        m_logger->info( "Processing vec3 input count: {} | patch: {} | stride: {}",
                        vectorCount,
                        patchVectors,
                        stride );

        std::vector<uint8_t> subTaskResultHash( SHA256_DIGEST_LENGTH );
        const auto starts = ComputeWindowStarts( vectorCount, patchVectors, stride );

        int outputChannels = 0;
        int outputLength = patchVectors;
        OutputLayout outputLayout;
        std::vector<float> stitchedOutput;
        std::vector<float> stitchedWeights;

        for ( int start : starts )
        {
            std::vector<float> patch;
            patch.resize( static_cast<size_t>( patchVectors ) * 3, 0.0f );

            for ( int c = 0; c < 3; ++c )
            {
                for ( int i = 0; i < patchVectors; ++i )
                {
                    const int vectorIndex = start + i;
                    if ( vectorIndex >= vectorCount )
                    {
                        break;
                    }
                    const size_t srcIndex = static_cast<size_t>( vectorIndex * 3 + c );
                    const size_t dstIndex = static_cast<size_t>( c * patchVectors + i );
                    patch[dstIndex] = signalValues[srcIndex];
                }
            }

            auto procresults = Process( patch, modelFileBytes, patchVectors * 3 );
            const float *data = procresults->host<float>();
            size_t dataSize = procresults->elementSize() * sizeof( float );

            if ( outputChannels == 0 )
            {
                outputLayout = GetOutputLayout( *procresults );
                outputChannels = outputLayout.channels;
                outputLength = outputLayout.length;

                stitchedOutput.assign( static_cast<size_t>( outputChannels ) * vectorCount, 0.0f );
                stitchedWeights.assign( static_cast<size_t>( vectorCount ), 0.0f );
            }

            if ( outputLength == patchVectors )
            {
                for ( int i = 0; i < patchVectors; ++i )
                {
                    const int outIndex = start + i;
                    if ( outIndex >= vectorCount )
                    {
                        break;
                    }

                    for ( int c = 0; c < outputChannels; ++c )
                    {
                        const size_t srcIdx = OutputIndex1D( *procresults, outputLayout, c, i );
                        const size_t dstIdx = static_cast<size_t>( outIndex ) * outputChannels +
                            static_cast<size_t>( c );

                        if ( stitchedWeights[outIndex] == 0.0f )
                        {
                            stitchedOutput[dstIdx] = data[srcIdx];
                        }
                        else
                        {
                            stitchedOutput[dstIdx] =
                                ( stitchedOutput[dstIdx] * stitchedWeights[outIndex] + data[srcIdx] ) /
                                ( stitchedWeights[outIndex] + 1.0f );
                        }
                    }
                    stitchedWeights[outIndex] += 1.0f;
                }
            }

            SHA256( reinterpret_cast<const unsigned char *>( data ), dataSize, 
                    subTaskResultHash.data() );
            chunkhashes.push_back( subTaskResultHash );
        }

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

        m_logger->info( "Vec3 processing complete" );
        return result;
    }

    std::unique_ptr<MNN::Tensor> MNN_Vec3::Process( const std::vector<float> &input,
                                                   std::vector<uint8_t> &model,
                                                   int length )
    {
        auto interpreter = std::unique_ptr<MNN::Interpreter>(
            MNN::Interpreter::createFromBuffer( model.data(), model.size() ) );
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

        const int vectorCount = length / 3;
        if ( vectorCount > 0 )
        {
            const auto dimType = inputTensor->getDimensionType();
            const int dims = inputTensor->dimensions();
            if ( dims == 3 )
            {
                if ( dimType == MNN::Tensor::CAFFE )
                {
                    interpreter->resizeTensor( inputTensor, { 1, 3, vectorCount } );
                }
                else
                {
                    interpreter->resizeTensor( inputTensor, { 1, vectorCount, 3 } );
                }
            }
            else if ( dims == 4 )
            {
                if ( dimType == MNN::Tensor::CAFFE )
                {
                    interpreter->resizeTensor( inputTensor, { 1, 3, vectorCount, 1 } );
                }
                else
                {
                    interpreter->resizeTensor( inputTensor, { 1, vectorCount, 1, 3 } );
                }
            }
            else if ( dims == 2 )
            {
                interpreter->resizeTensor( inputTensor, { 1, length } );
            }
            interpreter->resizeSession( session );
        }

        MNN::Tensor inputTensorUser( inputTensor, inputTensor->getDimensionType() );
        auto inputPtr = inputTensorUser.host<float>();
        std::memcpy( inputPtr, input.data(), static_cast<size_t>( length ) * sizeof( float ) );
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
