#include "processors/processing_processor_mnn_texture1d.hpp"

#include <algorithm>
#include <cctype>
#include <cstdint>
#include <cstring>
#include <sstream>
#include <thread>
#include <openssl/sha.h>
#include "util/sha256.hpp"

namespace sgns::sgprocessing
{
    using namespace MNN;

    namespace
    {
        enum class VolumeLayout
        {
            HWD,
            HDW,
            WHD,
            WDH,
            DHW,
            DWH
        };

        std::string ToUpperAscii( std::string value )
        {
            std::transform( value.begin(), value.end(), value.begin(),
                            []( unsigned char c ) { return static_cast<char>( std::toupper( c ) ); } );
            return value;
        }

        VolumeLayout ParseLayout( const std::vector<sgns::Parameter> *parameters, const std::string &inputName )
        {
            const std::vector<std::string> keys = {
                inputName + "Layout",
                inputName + "_layout",
                "volumeLayout",
                "layout"
            };

            if ( parameters )
            {
                for ( const auto &key : keys )
                {
                    auto it = std::find_if( parameters->begin(), parameters->end(),
                                            [&key]( const sgns::Parameter &param ) {
                                                return param.get_name() == key;
                                            } );
                    if ( it != parameters->end() && it->get_parameter_default().is_string() )
                    {
                        const std::string layout = ToUpperAscii( it->get_parameter_default().get<std::string>() );
                        if ( layout == "HWD" ) return VolumeLayout::HWD;
                        if ( layout == "HDW" ) return VolumeLayout::HDW;
                        if ( layout == "WHD" ) return VolumeLayout::WHD;
                        if ( layout == "WDH" ) return VolumeLayout::WDH;
                        if ( layout == "DHW" ) return VolumeLayout::DHW;
                        if ( layout == "DWH" ) return VolumeLayout::DWH;
                    }
                }
            }

            return VolumeLayout::HWD;
        }

        const char *LayoutToString( VolumeLayout layout )
        {
            switch ( layout )
            {
                case VolumeLayout::HWD:
                    return "HWD";
                case VolumeLayout::HDW:
                    return "HDW";
                case VolumeLayout::WHD:
                    return "WHD";
                case VolumeLayout::WDH:
                    return "WDH";
                case VolumeLayout::DHW:
                    return "DHW";
                case VolumeLayout::DWH:
                    return "DWH";
            }
            return "HWD";
        }

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

        size_t OutputIndex1D( int dims, int channels, int length, int c, int i )
        {
            if ( dims == 4 )
            {
                return static_cast<size_t>( c ) * length + static_cast<size_t>( i );
            }
            if ( dims == 3 )
            {
                return static_cast<size_t>( c ) * length + static_cast<size_t>( i );
            }
            return static_cast<size_t>( i );
        }
    }

    ProcessingResult MNN_Texture1D::StartProcessing( std::vector<std::vector<uint8_t>> &chunkhashes,
                                                     const sgns::IoDeclaration         &proc,
                                                     std::vector<char>                 &signalData,
                                                     std::vector<char>                 &modelFile,
                                                     const std::vector<sgns::Parameter> *parameters )
    {
        std::vector<uint8_t> modelFileBytes;
        modelFileBytes.assign( modelFile.begin(), modelFile.end() );

        if ( !proc.get_dimensions() || !proc.get_dimensions()->get_width() )
        {
            m_logger->error( "Texture1D input missing width" );
            return ProcessingResult{};
        }

        const int length = static_cast<int>( proc.get_dimensions()->get_width().value() );
        const int patchLength = proc.get_dimensions()->get_block_len().value_or( length );
        const int stride = proc.get_dimensions()->get_chunk_stride().value_or( patchLength );

        if ( length <= 0 || patchLength <= 0 || stride <= 0 )
        {
            m_logger->error( "Invalid texture1D length/patch/stride values" );
            return ProcessingResult{};
        }

        const auto format = proc.get_format().value_or( sgns::InputFormat::FLOAT32 );
        if ( format != sgns::InputFormat::FLOAT32 && format != sgns::InputFormat::FLOAT16 )
        {
            m_logger->error( "Texture1D supports FLOAT32/FLOAT16 formats only" );
            return ProcessingResult{};
        }

        const size_t expectedElements = static_cast<size_t>( length );
        const size_t bytesPerElement = ( format == sgns::InputFormat::FLOAT16 ) ? sizeof( uint16_t ) : sizeof( float );
        const size_t expectedBytes = expectedElements * bytesPerElement;
        if ( signalData.size() < expectedBytes )
        {
            m_logger->error( "Texture1D input size {} bytes is smaller than expected {} bytes",
                             signalData.size(),
                             expectedBytes );
            return ProcessingResult{};
        }

        const VolumeLayout layout = ParseLayout( parameters, proc.get_name() );
        m_logger->info( "Texture1D input format: {} | layout: {}",
                        format == sgns::InputFormat::FLOAT16 ? "FLOAT16" : "FLOAT32",
                        LayoutToString( layout ) );

        std::vector<float> signalValues;
        signalValues.resize( expectedElements );
        if ( format == sgns::InputFormat::FLOAT32 )
        {
            const auto *src = reinterpret_cast<const float *>( signalData.data() );
            std::memcpy( signalValues.data(), src, expectedBytes );
        }
        else
        {
            const auto *src = reinterpret_cast<const uint16_t *>( signalData.data() );
            for ( size_t i = 0; i < expectedElements; ++i )
            {
                signalValues[i] = HalfToFloat( src[i] );
            }
        }

        if ( layout != VolumeLayout::HWD )
        {
            m_logger->warn( "Texture1D layout '{}' is ignored; using linear order", LayoutToString( layout ) );
        }

        m_logger->info( "Processing texture1D input length: {} | patch: {} | stride: {}", length, patchLength, stride );

        std::vector<uint8_t> subTaskResultHash( SHA256_DIGEST_LENGTH );
        const auto starts = ComputeWindowStarts( length, patchLength, stride );

        size_t patchIndex = 0;
        int outputChannels = 0;
        int outputLength = patchLength;
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
                const int dims = procresults->dimensions();
                if ( dims == 4 )
                {
                    outputChannels = procresults->length( 1 );
                    outputLength = procresults->length( 3 );
                }
                else if ( dims == 3 )
                {
                    outputChannels = procresults->length( 1 );
                    outputLength = procresults->length( 2 );
                }
                else if ( dims == 2 )
                {
                    outputChannels = 1;
                    outputLength = procresults->length( 1 );
                }
                else
                {
                    outputChannels = 1;
                    outputLength = patchLength;
                }

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
                        const size_t srcIndex = OutputIndex1D( procresults->dimensions(), outputChannels, outputLength, c, i );
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

            ++patchIndex;
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

        m_logger->info( "Texture1D processing complete" );

        return result;
    }

    std::unique_ptr<MNN::Tensor> MNN_Texture1D::Process( const std::vector<float> &signalData,
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
        config.type = MNN_FORWARD_VULKAN;
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
            if ( tensor->elementSize() <= 4 )
            {
                if ( dims == 4 )
                {
                    interpreter->resizeTensor( tensor, { 1, 1, 1, length } );
                }
                else if ( dims == 3 )
                {
                    interpreter->resizeTensor( tensor, { 1, 1, length } );
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

        auto outputHost = std::make_unique<MNN::Tensor>( outputTensor, outputTensor->getDimensionType() );
        outputTensor->copyToHostTensor( outputHost.get() );

        return outputHost;
    }
}
