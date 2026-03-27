#include "processors/processing_processor_mnn_volume.hpp"
#include <functional>
#include <thread>
#include <sstream>
#include <sstream>
#include <algorithm>
#include <cstring>
#include <cstdint>
#include <cctype>
#include <cstdlib>
#include <openssl/sha.h> // For SHA256_DIGEST_LENGTH
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

        size_t LayoutIndex( VolumeLayout layout,
                            int h,
                            int w,
                            int d,
                            int height,
                            int width,
                            int depth )
        {
            switch ( layout )
            {
                case VolumeLayout::HWD:
                    return ( static_cast<size_t>( h ) * width + static_cast<size_t>( w ) ) * depth + static_cast<size_t>( d );
                case VolumeLayout::HDW:
                    return ( static_cast<size_t>( h ) * depth + static_cast<size_t>( d ) ) * width + static_cast<size_t>( w );
                case VolumeLayout::WHD:
                    return ( static_cast<size_t>( w ) * height + static_cast<size_t>( h ) ) * depth + static_cast<size_t>( d );
                case VolumeLayout::WDH:
                    return ( static_cast<size_t>( w ) * depth + static_cast<size_t>( d ) ) * height + static_cast<size_t>( h );
                case VolumeLayout::DHW:
                    return ( static_cast<size_t>( d ) * height + static_cast<size_t>( h ) ) * width + static_cast<size_t>( w );
                case VolumeLayout::DWH:
                    return ( static_cast<size_t>( d ) * width + static_cast<size_t>( w ) ) * height + static_cast<size_t>( h );
            }

            return ( static_cast<size_t>( h ) * width + static_cast<size_t>( w ) ) * depth + static_cast<size_t>( d );
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
    }

    ProcessingResult MNN_Volume::StartProcessing( std::vector<std::vector<uint8_t>> &chunkhashes,
                                                   const sgns::IoDeclaration         &proc,
                                                   std::vector<char>                 &volumeData,
                                                   std::vector<char>                 &modelFile,
                                                   const std::vector<sgns::Parameter> *parameters )
    {
        std::vector<uint8_t> modelFile_bytes;
        modelFile_bytes.assign(modelFile.begin(), modelFile.end());

        std::vector<uint8_t> subTaskResultHash(SHA256_DIGEST_LENGTH);

        if ( !proc.get_dimensions() || !proc.get_dimensions()->get_width() ||
             !proc.get_dimensions()->get_height() || !proc.get_dimensions()->get_chunk_count() )
        {
            m_logger->error( "Texture3D input missing width/height/chunk_count" );
            return ProcessingResult{};
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
            return ProcessingResult{};
        }

        const auto format = proc.get_format().value_or( sgns::InputFormat::FLOAT32 );
        const size_t expectedElements = static_cast<size_t>( width ) * height * depth;
        const size_t bytesPerElement = ( format == sgns::InputFormat::FLOAT16 ) ? sizeof( uint16_t ) : sizeof( float );
        const size_t expectedBytes = expectedElements * bytesPerElement;
        if ( volumeData.size() < expectedBytes )
        {
            m_logger->error( "Texture3D input size {} bytes is smaller than expected {} bytes",
                             volumeData.size(),
                             expectedBytes );
            return ProcessingResult{};
        }

        const VolumeLayout layout = ParseLayout( parameters, proc.get_name() );
        m_logger->info( "Texture3D input format: {} | layout: {}",
                format == sgns::InputFormat::FLOAT16 ? "FLOAT16" : "FLOAT32",
                LayoutToString( layout ) );

        std::vector<float> volumeFloats;
        volumeFloats.resize( expectedElements );

        if ( format == sgns::InputFormat::FLOAT32 && layout == VolumeLayout::HWD )
        {
            std::memcpy( volumeFloats.data(), volumeData.data(), expectedBytes );
        }
        else
        {
            if ( format == sgns::InputFormat::FLOAT32 )
            {
                const auto *src = reinterpret_cast<const float *>( volumeData.data() );
                for ( int h = 0; h < height; ++h )
                {
                    for ( int w = 0; w < width; ++w )
                    {
                        for ( int d = 0; d < depth; ++d )
                        {
                            const size_t srcIndex = LayoutIndex( layout, h, w, d, height, width, depth );
                            const size_t dstIndex = LayoutIndex( VolumeLayout::HWD, h, w, d, height, width, depth );
                            volumeFloats[dstIndex] = src[srcIndex];
                        }
                    }
                }
            }
            else if ( format == sgns::InputFormat::FLOAT16 )
            {
                const auto *src = reinterpret_cast<const uint16_t *>( volumeData.data() );
                for ( int h = 0; h < height; ++h )
                {
                    for ( int w = 0; w < width; ++w )
                    {
                        for ( int d = 0; d < depth; ++d )
                        {
                            const size_t srcIndex = LayoutIndex( layout, h, w, d, height, width, depth );
                            const size_t dstIndex = LayoutIndex( VolumeLayout::HWD, h, w, d, height, width, depth );
                            volumeFloats[dstIndex] = HalfToFloat( src[srcIndex] );
                        }
                    }
                }
            }
            else
            {
                m_logger->error( "Unsupported texture3D format for volume input" );
                return ProcessingResult{};
            }
        }

        m_logger->info( "Processing volume input (H,W,D): {}x{}x{} ({} floats)",
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

        const auto startsX = ComputeWindowStarts( width, patchWidth, strideX );
        const auto startsY = ComputeWindowStarts( height, patchHeight, strideY );
        const auto startsZ = ComputeWindowStarts( depth, patchDepth, strideZ );

        size_t patchIndex = 0;
        int outputChannels = 0;
        int outputHeight = patchHeight;
        int outputWidth = patchWidth;
        int outputDepth = patchDepth;
        std::vector<float> stitchedOutput;
        std::vector<float> stitchedWeights;
        for ( const int z : startsZ )
        {
            for ( const int y : startsY )
            {
                for ( const int x : startsX )
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
                                                                        ( static_cast<size_t>( srcY ) * width +
                                                                            static_cast<size_t>( srcX ) ) * depth +
                                                                        static_cast<size_t>( srcZ );
                                                                const size_t dstIndex =
                                                                        ( static_cast<size_t>( dy ) * patchWidth +
                                                                            static_cast<size_t>( dx ) ) * patchDepth +
                                                                        static_cast<size_t>( dz );
                                                                patch[dstIndex] = volumeFloats[srcIndex];
                                                        }
                        }
                    }

                    auto procresults = Process( patch, modelFile_bytes, patchWidth, patchHeight, patchDepth );
                    const float *data     = procresults->host<float>();
                    size_t       dataSize = procresults->elementSize() * sizeof( float );

                    if ( outputChannels == 0 )
                    {
                        const int dims = procresults->dimensions();
                        if ( dims >= 5 )
                        {
                            outputChannels = procresults->length( 1 );
                            outputHeight = procresults->length( 2 );
                            outputWidth = procresults->length( 3 );
                            outputDepth = procresults->length( 4 );
                        }
                        else if ( dims == 4 )
                        {
                            outputChannels = procresults->length( 0 );
                            outputHeight = procresults->length( 1 );
                            outputWidth = procresults->length( 2 );
                            outputDepth = procresults->length( 3 );
                        }
                        else
                        {
                            outputChannels = 1;
                            outputHeight = patchHeight;
                            outputWidth = patchWidth;
                            outputDepth = patchDepth;
                        }

                        stitchedOutput.assign( static_cast<size_t>( outputChannels ) * width * height * depth, 0.0f );
                        stitchedWeights.assign( static_cast<size_t>( width ) * height * depth, 0.0f );
                    }

                    if ( outputHeight == patchHeight && outputWidth == patchWidth && outputDepth == patchDepth )
                    {
                        for ( int dy = 0; dy < patchHeight; ++dy )
                        {
                            const int outY = y + dy;
                            if ( outY >= height )
                            {
                                continue;
                            }
                            for ( int dx = 0; dx < patchWidth; ++dx )
                            {
                                const int outX = x + dx;
                                if ( outX >= width )
                                {
                                    continue;
                                }
                                for ( int dz = 0; dz < patchDepth; ++dz )
                                {
                                    const int outZ = z + dz;
                                    if ( outZ >= depth )
                                    {
                                        continue;
                                    }

                                    const size_t weightIndex =
                                        ( static_cast<size_t>( outY ) * width + static_cast<size_t>( outX ) ) * depth +
                                        static_cast<size_t>( outZ );
                                    stitchedWeights[weightIndex] += 1.0f;

                                    for ( int c = 0; c < outputChannels; ++c )
                                    {
                                        const size_t srcIndex =
                                            ( ( static_cast<size_t>( c ) * outputHeight + static_cast<size_t>( dy ) ) *
                                              outputWidth + static_cast<size_t>( dx ) ) * outputDepth +
                                            static_cast<size_t>( dz );
                                        const size_t dstIndex =
                                            ( static_cast<size_t>( c ) * height + static_cast<size_t>( outY ) ) * width * depth +
                                            static_cast<size_t>( outX ) * depth +
                                            static_cast<size_t>( outZ );
                                        stitchedOutput[dstIndex] += data[srcIndex];
                                    }
                                }
                            }
                        }
                    }

                    if ( patchIndex == 0 )
                    {
                        {
                            std::ofstream inputDump( "first_patch_input.raw", std::ios::binary );
                            if ( inputDump.is_open() )
                            {
                                inputDump.write( reinterpret_cast<const char *>( patch.data() ),
                                                 static_cast<std::streamsize>( patch.size() * sizeof( float ) ) );
                                m_logger->info( "Wrote first patch input to first_patch_input.raw" );
                            }
                        }

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

                        {
                            std::ofstream outputDump( "first_patch_output.raw", std::ios::binary );
                            if ( outputDump.is_open() )
                            {
                                outputDump.write( reinterpret_cast<const char *>( data ),
                                                  static_cast<std::streamsize>( procresults->elementSize() * sizeof( float ) ) );
                                m_logger->info( "Wrote first patch output to first_patch_output.raw" );
                            }
                        }
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

        if ( !stitchedOutput.empty() )
        {
            for ( int c = 0; c < outputChannels; ++c )
            {
                for ( int y = 0; y < height; ++y )
                {
                    for ( int x = 0; x < width; ++x )
                    {
                        for ( int z = 0; z < depth; ++z )
                        {
                            const size_t weightIndex =
                                ( static_cast<size_t>( y ) * width + static_cast<size_t>( x ) ) * depth +
                                static_cast<size_t>( z );
                            if ( stitchedWeights[weightIndex] <= 0.0f )
                            {
                                continue;
                            }
                            const size_t dstIndex =
                                ( static_cast<size_t>( c ) * height + static_cast<size_t>( y ) ) * width * depth +
                                static_cast<size_t>( x ) * depth +
                                static_cast<size_t>( z );
                            stitchedOutput[dstIndex] /= stitchedWeights[weightIndex];
                        }
                    }
                }
            }

            std::ofstream outputVolume( "stitched_logits.raw", std::ios::binary );
            if ( outputVolume.is_open() )
            {
                outputVolume.write( reinterpret_cast<const char *>( stitchedOutput.data() ),
                                    static_cast<std::streamsize>( stitchedOutput.size() * sizeof( float ) ) );
                m_logger->info( "Wrote stitched logits to stitched_logits.raw" );
            }
        }

        m_logger->info( "Volume processing complete" );

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

        return result;
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
        m_logger->info( "Using MNN Vulkan backend" );
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
                m_logger->info( "Resizing '{}' to [1, 1, {}, {}, {}]", inputPair.first, height, width, depth );
                interpreter->resizeTensor( tensor, { 1, 1, height, width, depth } );
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
