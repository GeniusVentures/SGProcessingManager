#include "processors/processing_processor_mnn_texturecube.hpp"

#include <algorithm>
#include <cctype>
#include <cstdint>
#include <cstring>
#include <string>
#include <openssl/sha.h>
#include "datasplitter/ImageSplitter.hpp"
#include "util/InputTypes.hpp"
#include "util/sha256.hpp"

namespace sgns::sgprocessing
{
    using namespace MNN;

    namespace
    {
        enum class CubeLayout
        {
            FacesInOrder,
            Atlas3x2
        };

        std::string ToUpperAscii( std::string value )
        {
            std::transform( value.begin(), value.end(), value.begin(),
                            []( unsigned char c ) { return static_cast<char>( std::toupper( c ) ); } );
            return value;
        }

        CubeLayout ParseLayout( const std::vector<sgns::Parameter> *parameters, const std::string &inputName )
        {
            const std::vector<std::string> keys = {
                inputName + "Layout",
                inputName + "_layout",
                "cubeLayout",
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
                        if ( layout == "ATLAS" || layout == "ATLAS_3X2" || layout == "ATLAS3X2" )
                        {
                            return CubeLayout::Atlas3x2;
                        }
                    }
                }
            }

            return CubeLayout::FacesInOrder;
        }

        const char *LayoutToString( CubeLayout layout )
        {
            switch ( layout )
            {
                case CubeLayout::FacesInOrder:
                    return "faces_in_order";
                case CubeLayout::Atlas3x2:
                    return "atlas_3x2";
            }
            return "faces_in_order";
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

        bool HasAnyTexture2DChunkFields( const sgns::Dimensions &dimensions )
        {
            return dimensions.get_block_len() || dimensions.get_block_line_stride() || dimensions.get_block_stride() ||
                dimensions.get_chunk_line_stride() || dimensions.get_chunk_offset() || dimensions.get_chunk_stride() ||
                dimensions.get_chunk_subchunk_height() || dimensions.get_chunk_subchunk_width() ||
                dimensions.get_chunk_count();
        }

        bool HasAllTexture2DChunkFields( const sgns::Dimensions &dimensions )
        {
            return dimensions.get_block_len() && dimensions.get_block_line_stride() && dimensions.get_block_stride() &&
                dimensions.get_chunk_line_stride() && dimensions.get_chunk_offset() && dimensions.get_chunk_stride() &&
                dimensions.get_chunk_subchunk_height() && dimensions.get_chunk_subchunk_width() &&
                dimensions.get_chunk_count();
        }

        void AppendOutput( std::vector<float> &dest, const MNN::Tensor &tensor )
        {
            const float *data = tensor.host<float>();
            const size_t count = tensor.elementSize();
            dest.insert( dest.end(), data, data + count );
        }

        std::vector<uint8_t> ExtractFace( const std::vector<uint8_t> &atlas,
                                          int faceIndex,
                                          int faceWidth,
                                          int faceHeight,
                                          int channels )
        {
            const int atlasWidth = faceWidth * 3;
            const int atlasHeight = faceHeight * 2;
            const int faceX = ( faceIndex % 3 ) * faceWidth;
            const int faceY = ( faceIndex / 3 ) * faceHeight;

            std::vector<uint8_t> face;
            face.resize( static_cast<size_t>( faceWidth ) * faceHeight * channels );

            for ( int y = 0; y < faceHeight; ++y )
            {
                const int srcY = faceY + y;
                if ( srcY >= atlasHeight )
                {
                    break;
                }
                const size_t srcRow = static_cast<size_t>( srcY ) * atlasWidth * channels;
                const size_t dstRow = static_cast<size_t>( y ) * faceWidth * channels;
                const size_t srcOffset = srcRow + static_cast<size_t>( faceX ) * channels;
                std::memcpy( face.data() + dstRow, atlas.data() + srcOffset,
                             static_cast<size_t>( faceWidth ) * channels );
            }

            return face;
        }

        std::vector<float> ConvertImageToFloatsInterleaved( const std::vector<uint8_t> &image,
                                                            int width,
                                                            int height,
                                                            int channels )
        {
            const size_t total = static_cast<size_t>( width ) * height * channels;
            std::vector<float> output;
            output.resize( total, 0.0f );

            for ( int y = 0; y < height; ++y )
            {
                for ( int x = 0; x < width; ++x )
                {
                    const size_t base = ( static_cast<size_t>( y ) * width + x ) * channels;
                    for ( int c = 0; c < channels; ++c )
                    {
                        const float value = static_cast<float>( image[base + static_cast<size_t>( c )] );
                        output[base + static_cast<size_t>( c )] = value;
                    }
                }
            }

            return output;
        }

        std::vector<float> ConvertFloatImageToFloats( const std::vector<char> &image,
                                                      int width,
                                                      int height,
                                                      sgns::InputFormat format )
        {
            const size_t total = static_cast<size_t>( width ) * height;
            std::vector<float> output;
            output.resize( total, 0.0f );

            if ( format == sgns::InputFormat::FLOAT32 )
            {
                const auto *src = reinterpret_cast<const float *>( image.data() );
                std::memcpy( output.data(), src, total * sizeof( float ) );
            }
            else
            {
                const auto *src = reinterpret_cast<const uint16_t *>( image.data() );
                for ( size_t i = 0; i < total; ++i )
                {
                    output[i] = HalfToFloat( src[i] );
                }
            }

            return output;
        }

        std::vector<float> ConvertInterleavedToNCHW( const std::vector<float> &input,
                                                     int width,
                                                     int height,
                                                     int channels )
        {
            std::vector<float> output;
            output.resize( static_cast<size_t>( width ) * height * channels, 0.0f );

            for ( int y = 0; y < height; ++y )
            {
                for ( int x = 0; x < width; ++x )
                {
                    const size_t base = ( static_cast<size_t>( y ) * width + x ) * channels;
                    for ( int c = 0; c < channels; ++c )
                    {
                        const size_t dstIndex = static_cast<size_t>( c ) * width * height +
                            static_cast<size_t>( y ) * width + x;
                        output[dstIndex] = input[base + static_cast<size_t>( c )];
                    }
                }
            }

            return output;
        }
    }

    ProcessingResult MNN_TextureCube::StartProcessing( std::vector<std::vector<uint8_t>> &chunkhashes,
                                                       const sgns::IoDeclaration         &proc,
                                                       std::vector<char>                 &cubeData,
                                                       std::vector<char>                 &modelFile,
                                                       const std::vector<sgns::Parameter> *parameters )
    {
        std::vector<uint8_t> modelFileBytes;
        modelFileBytes.assign( modelFile.begin(), modelFile.end() );

        if ( !proc.get_dimensions() || !proc.get_dimensions()->get_width() || !proc.get_dimensions()->get_height() )
        {
            m_logger->error( "TextureCube input missing width/height" );
            return ProcessingResult{};
        }

        const int faceWidth = static_cast<int>( proc.get_dimensions()->get_width().value() );
        const int faceHeight = static_cast<int>( proc.get_dimensions()->get_height().value() );

        const auto format = proc.get_format().value_or( sgns::InputFormat::RGB8 );
        if ( format != sgns::InputFormat::RGB8 && format != sgns::InputFormat::RGBA8 &&
             format != sgns::InputFormat::FLOAT32 && format != sgns::InputFormat::FLOAT16 )
        {
            m_logger->error( "TextureCube supports RGB8/RGBA8/FLOAT32/FLOAT16 formats only" );
            return ProcessingResult{};
        }

        const CubeLayout layout = ParseLayout( parameters, proc.get_name() );
        m_logger->info( "TextureCube layout: {}", LayoutToString( layout ) );

        bool isImageFormat = ( format == sgns::InputFormat::RGB8 || format == sgns::InputFormat::RGBA8 );
        int channels = 1;
        if ( isImageFormat )
        {
            auto maybeChannels = sgns::sgprocessing::InputTypes::GetImageChannels( format );
            if ( !maybeChannels )
            {
                m_logger->error( "TextureCube image format has no channel mapping" );
                return ProcessingResult{};
            }
            channels = maybeChannels.value();
        }

        const size_t bytesPerElement = ( format == sgns::InputFormat::FLOAT16 ) ? sizeof( uint16_t ) :
            ( format == sgns::InputFormat::FLOAT32 ) ? sizeof( float ) : sizeof( uint8_t );
        const size_t faceElements = static_cast<size_t>( faceWidth ) * faceHeight * ( isImageFormat ? channels : 1 );
        const size_t faceBytes = faceElements * bytesPerElement;
        const size_t expectedBytes = faceBytes * 6;
        if ( cubeData.size() < expectedBytes )
        {
            m_logger->error( "TextureCube input size {} bytes is smaller than expected {} bytes",
                             cubeData.size(),
                             expectedBytes );
            return ProcessingResult{};
        }

        const auto &dimensions = proc.get_dimensions().value();
        const bool hasChunkFields = HasAnyTexture2DChunkFields( dimensions );
        if ( hasChunkFields && !HasAllTexture2DChunkFields( dimensions ) )
        {
            m_logger->error( "TextureCube chunking requires full texture2D chunk parameters" );
            return ProcessingResult{};
        }
        if ( hasChunkFields && !isImageFormat )
        {
            m_logger->warn( "TextureCube chunking is ignored for float formats" );
        }

        std::vector<uint8_t> cubeBytes;
        cubeBytes.reserve( cubeData.size() );
        cubeBytes.assign( cubeData.begin(), cubeData.end() );

        std::vector<std::vector<uint8_t>> faces;
        faces.reserve( 6 );

        if ( layout == CubeLayout::FacesInOrder )
        {
            for ( int face = 0; face < 6; ++face )
            {
                const size_t offset = static_cast<size_t>( face ) * faceBytes;
                faces.emplace_back( cubeBytes.begin() + offset, cubeBytes.begin() + offset + faceBytes );
            }
        }
        else
        {
            if ( !isImageFormat )
            {
                m_logger->error( "TextureCube atlas layout requires RGB/RGBA formats" );
                return ProcessingResult{};
            }
            for ( int face = 0; face < 6; ++face )
            {
                faces.push_back( ExtractFace( cubeBytes, face, faceWidth, faceHeight, channels ) );
            }
        }

        std::vector<uint8_t> subTaskResultHash( SHA256_DIGEST_LENGTH );
        std::vector<float> outputFloats;
        size_t totalChunks = 0;

        for ( int faceIndex = 0; faceIndex < 6; ++faceIndex )
        {
            const auto &face = faces[faceIndex];

            if ( hasChunkFields && isImageFormat )
            {
                const auto block_len = dimensions.get_block_len().value();
                const auto block_line_stride = dimensions.get_block_line_stride().value();
                const auto block_stride = dimensions.get_block_stride().value();
                const auto chunk_line_stride = dimensions.get_chunk_line_stride().value();
                const auto chunk_offset = dimensions.get_chunk_offset().value();
                const auto chunk_stride = dimensions.get_chunk_stride().value();
                const auto chunk_subchunk_height = dimensions.get_chunk_subchunk_height().value();
                const auto chunk_subchunk_width = dimensions.get_chunk_subchunk_width().value();

                ImageSplitter faceSplitter( face, block_line_stride, block_stride, block_len, channels );
                const int chunkCount = static_cast<int>( dimensions.get_chunk_count().value() );

                ImageSplitter chunkSplitter( faceSplitter.GetPart( 0 ),
                                             chunk_line_stride,
                                             chunk_stride,
                                             faceSplitter.GetPartHeightActual( 0 ) / chunk_subchunk_height *
                                                 chunk_line_stride,
                                             channels );

                for ( int chunkIdx = 0; chunkIdx < chunkCount; ++chunkIdx )
                {
                    const auto chunkData = chunkSplitter.GetPart( chunkIdx );
                    const int chunkWidth = chunkSplitter.GetPartWidthActual( chunkIdx );
                    const int chunkHeight = chunkSplitter.GetPartHeightActual( chunkIdx );

                    auto interpreter = std::unique_ptr<MNN::Interpreter>(
                        MNN::Interpreter::createFromBuffer( modelFileBytes.data(), modelFileBytes.size() ) );
                    if ( !interpreter )
                    {
                        m_logger->error( "Failed to create MNN interpreter from buffer" );
                        return ProcessingResult{};
                    }

                    MNN::ScheduleConfig config;
                    config.type = MNN_FORWARD_CPU;
                    config.numThread = 4;
                    config.backendConfig = nullptr;

                    auto session = interpreter->createSession( config );
                    if ( !session )
                    {
                        m_logger->error( "Failed to create MNN session" );
                        return ProcessingResult{};
                    }

                    auto inputTensor = interpreter->getSessionInput( session, nullptr );
                    if ( !inputTensor )
                    {
                        m_logger->error( "Failed to get input tensor" );
                        return ProcessingResult{};
                    }

                    const auto dimType = inputTensor->getDimensionType();
                    if ( dimType == MNN::Tensor::TENSORFLOW )
                    {
                        interpreter->resizeTensor( inputTensor, { 1, chunkHeight, chunkWidth, channels } );
                    }
                    else
                    {
                        interpreter->resizeTensor( inputTensor, { 1, channels, chunkHeight, chunkWidth } );
                    }
                    interpreter->resizeSession( session );

                    const auto inputInterleaved = ConvertImageToFloatsInterleaved( chunkData, chunkWidth, chunkHeight, channels );
                    const auto inputFloats = ( dimType == MNN::Tensor::TENSORFLOW ) ? inputInterleaved :
                        ConvertInterleavedToNCHW( inputInterleaved, chunkWidth, chunkHeight, channels );
                    MNN::Tensor inputUser( inputTensor, dimType );
                    std::memcpy( inputUser.host<float>(), inputFloats.data(), inputFloats.size() * sizeof( float ) );
                    inputTensor->copyFromHostTensor( &inputUser );

                    interpreter->runSession( session );

                    auto outputTensor = interpreter->getSessionOutput( session, nullptr );
                    if ( !outputTensor )
                    {
                        m_logger->error( "Failed to get output tensor" );
                        return ProcessingResult{};
                    }

                    auto outputUserTensor = std::make_unique<MNN::Tensor>( outputTensor, MNN::Tensor::CAFFE );
                    outputTensor->copyToHostTensor( outputUserTensor.get() );

                    const float *data = outputUserTensor->host<float>();
                    const size_t dataSize = outputUserTensor->elementSize() * sizeof( float );

                    auto hash = sgprocmanagersha::sha256( data, dataSize );
                    chunkhashes.emplace_back( hash.begin(), hash.end() );
                    std::string combinedHash = std::string( subTaskResultHash.begin(), subTaskResultHash.end() ) +
                        std::string( hash.begin(), hash.end() );
                    subTaskResultHash = sgprocmanagersha::sha256( combinedHash.c_str(), combinedHash.length() );

                    AppendOutput( outputFloats, *outputUserTensor );
                    ++totalChunks;
                }
            }
            else
            {
                std::vector<float> inputFloats;
                if ( isImageFormat )
                {
                    inputFloats = ConvertImageToFloatsInterleaved( face, faceWidth, faceHeight, channels );
                }
                else
                {
                    std::vector<char> floatFace;
                    floatFace.assign( face.begin(), face.end() );
                    inputFloats = ConvertFloatImageToFloats( floatFace, faceWidth, faceHeight, format );
                }

                auto outputTensor = Process( inputFloats, modelFileBytes, faceWidth, faceHeight, channels, true );
                if ( !outputTensor )
                {
                    m_logger->error( "Failed to process textureCube face" );
                    return ProcessingResult{};
                }

                const float *data = outputTensor->host<float>();
                const size_t dataSize = outputTensor->elementSize() * sizeof( float );

                auto hash = sgprocmanagersha::sha256( data, dataSize );
                chunkhashes.emplace_back( hash.begin(), hash.end() );
                std::string combinedHash = std::string( subTaskResultHash.begin(), subTaskResultHash.end() ) +
                    std::string( hash.begin(), hash.end() );
                subTaskResultHash = sgprocmanagersha::sha256( combinedHash.c_str(), combinedHash.length() );

                AppendOutput( outputFloats, *outputTensor );
                ++totalChunks;
            }
        }

        m_progress = 100.0f;

        ProcessingResult result;
        result.hash = subTaskResultHash;

        if ( !outputFloats.empty() )
        {
            const size_t byteCount = outputFloats.size() * sizeof( float );
            std::vector<char> outputBytes( byteCount );
            std::memcpy( outputBytes.data(), outputFloats.data(), byteCount );

            result.output_buffers =
                std::make_shared<std::pair<std::vector<std::string>, std::vector<std::vector<char>>>>();
            result.output_buffers->first.push_back( "" );
            result.output_buffers->second.push_back( std::move( outputBytes ) );
        }

        m_logger->info( "TextureCube processing complete ({} chunks)", totalChunks );
        return result;
    }

    std::unique_ptr<MNN::Tensor> MNN_TextureCube::Process( const std::vector<float> &inputData,
                                                           std::vector<uint8_t>    &modelFile,
                                                           int                      width,
                                                           int                      height,
                                                           int                      channels,
                                                           bool                     inputIsInterleaved )
    {
        auto interpreter = std::unique_ptr<MNN::Interpreter>(
            MNN::Interpreter::createFromBuffer( modelFile.data(), modelFile.size() ) );
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

        const auto dimType = inputTensor->getDimensionType();
        if ( dimType == MNN::Tensor::TENSORFLOW )
        {
            interpreter->resizeTensor( inputTensor, { 1, height, width, channels } );
        }
        else
        {
            interpreter->resizeTensor( inputTensor, { 1, channels, height, width } );
        }
        interpreter->resizeSession( session );

        std::vector<float> reordered;
        const std::vector<float> *srcData = &inputData;
        if ( inputIsInterleaved && dimType != MNN::Tensor::TENSORFLOW && channels > 1 )
        {
            reordered = ConvertInterleavedToNCHW( inputData, width, height, channels );
            srcData = &reordered;
        }

        MNN::Tensor inputUser( inputTensor, dimType );
        const size_t copyCount = std::min( static_cast<size_t>( inputUser.elementSize() ), srcData->size() );
        auto inputPtr = inputUser.host<float>();
        std::memset( inputPtr, 0, inputUser.elementSize() * sizeof( float ) );
        std::memcpy( inputPtr, srcData->data(), copyCount * sizeof( float ) );
        inputTensor->copyFromHostTensor( &inputUser );

        interpreter->runSession( session );

        auto outputTensor = interpreter->getSessionOutput( session, nullptr );
        if ( !outputTensor )
        {
            m_logger->error( "Failed to get output tensor" );
            return nullptr;
        }

        auto outputUserTensor = std::make_unique<MNN::Tensor>( outputTensor, MNN::Tensor::CAFFE );
        outputTensor->copyToHostTensor( outputUserTensor.get() );

        return outputUserTensor;
    }
}
