/**
 * MNN processor for FP4_ULTRA quantized input data
 *
 * Workflow:
 *   1. Dequantize FP4_ULTRA packed data → FLOAT32
 *   2. Run windowed MNN inference (same pattern as MNN_Float)
 *   3. Stitch overlapping windows, hash each chunk for proof-of-work
 *
 * FP4_ULTRA data layout in the input buffer:
 *   [packed_nibbles: ceil(num_elements/2) bytes] [scales: num_macroblocks * sizeof(float)]
 *
 * Each macroblock covers 4096 elements (64×64) with one scale factor.
 * Nibble values index into a 16-entry NF4 symmetric lookup table.
 */

#include "processors/processing_processor_mnn_fp4ultra.hpp"

#include <algorithm>
#include <cmath>
#include <cstring>
#include <openssl/sha.h>
#include "util/sha256.hpp"

namespace sgns::sgprocessing
{
    using namespace MNN;

    namespace
    {
        /// Macroblock size: 64 rows × 64 cols = 4096 elements
        static constexpr size_t kMacroblockSize = 64 * 64;

        /// NF4-style symmetric lookup table: 16 representable values in [-1, 1]
        static constexpr float kFP4LUT[16] = {
            -1.0f,    -0.6962f, -0.5251f, -0.3949f,
            -0.2844f, -0.1848f, -0.0911f,  0.0f,
             0.0796f,  0.1609f,  0.2461f,  0.3379f,
             0.4407f,  0.5626f,  0.7230f,  1.0f
        };

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
                starts.push_back( pos );

            const int last = length - roi;
            if ( starts.empty() || starts.back() != last )
                starts.push_back( last );

            return starts;
        }
    }

    std::vector<float> MNN_FP4Ultra::DequantizeFP4( const std::vector<char> &packed,
                                                     size_t                   num_elements )
    {
        // Layout: [packed_nibbles | scales_as_float32]
        const size_t packed_bytes    = ( num_elements + 1 ) / 2;
        const size_t num_macroblocks = ( num_elements + kMacroblockSize - 1 ) / kMacroblockSize;
        const size_t scales_bytes    = num_macroblocks * sizeof( float );

        if ( packed.size() < packed_bytes + scales_bytes )
        {
            m_logger->error( "FP4_ULTRA data too small: {} bytes, need at least {} + {} = {}",
                             packed.size(), packed_bytes, scales_bytes, packed_bytes + scales_bytes );
            return {};
        }

        // Read per-macroblock scales from after the packed nibbles
        std::vector<float> scales( num_macroblocks );
        std::memcpy( scales.data(), packed.data() + packed_bytes, scales_bytes );

        // Dequantize each element
        std::vector<float> output( num_elements );
        for ( size_t i = 0; i < num_elements; ++i )
        {
            const size_t byte_idx = i / 2;
            uint8_t nibble;
            if ( i % 2 == 0 )
                nibble = ( static_cast<uint8_t>( packed[byte_idx] ) >> 4 ) & 0x0F;
            else
                nibble = static_cast<uint8_t>( packed[byte_idx] ) & 0x0F;

            const size_t mb_idx = i / kMacroblockSize;
            const float  scale  = ( mb_idx < scales.size() ) ? scales[mb_idx] : 1.0f;
            output[i] = kFP4LUT[nibble] * scale;
        }

        m_logger->info( "FP4_ULTRA dequantized {} elements from {} macroblocks",
                        num_elements, num_macroblocks );
        return output;
    }

    ProcessingResult MNN_FP4Ultra::StartProcessing(
        std::vector<std::vector<uint8_t>>  &chunkhashes,
        const sgns::IoDeclaration          &proc,
        std::vector<char>                  &fp4Data,
        std::vector<char>                  &modelFile,
        const std::vector<sgns::Parameter> *parameters )
    {
        (void)parameters;
        std::vector<uint8_t> modelBytes( modelFile.begin(), modelFile.end() );

        if ( !proc.get_dimensions() || !proc.get_dimensions()->get_width() )
        {
            m_logger->error( "FP4_ULTRA input missing width dimension" );
            return ProcessingResult{};
        }

        const int length      = static_cast<int>( proc.get_dimensions()->get_width().value() );
        const int patchLength = static_cast<int>(
            proc.get_dimensions()->get_block_len().value_or( length ) );
        const int stride = static_cast<int>(
            proc.get_dimensions()->get_chunk_stride().value_or( patchLength ) );

        if ( length <= 0 || patchLength <= 0 || stride <= 0 )
        {
            m_logger->error( "FP4_ULTRA: invalid length/patch/stride values" );
            return ProcessingResult{};
        }

        // Dequantize FP4 → FLOAT32
        std::vector<float> floatValues = DequantizeFP4( fp4Data, static_cast<size_t>( length ) );
        if ( floatValues.empty() )
        {
            m_logger->error( "FP4_ULTRA: dequantization failed" );
            return ProcessingResult{};
        }

        m_logger->info( "FP4_ULTRA processing: length={} patch={} stride={}", length, patchLength, stride );

        // -----------------------------------------------------------------
        // Windowed inference (same pattern as MNN_Float)
        // -----------------------------------------------------------------
        std::vector<uint8_t> subTaskResultHash( SHA256_DIGEST_LENGTH, 0 );
        const auto starts = ComputeWindowStarts( length, patchLength, stride );

        std::vector<float> stitchedOutput( length, 0.0f );
        std::vector<float> stitchedWeights( length, 0.0f );

        for ( int start : starts )
        {
            std::vector<float> patch( static_cast<size_t>( patchLength ), 0.0f );
            for ( int i = 0; i < patchLength; ++i )
            {
                const int srcIndex = start + i;
                if ( srcIndex >= length ) break;
                patch[static_cast<size_t>( i )] = floatValues[static_cast<size_t>( srcIndex )];
            }

            auto procresults = Process( patch, modelBytes, patchLength );
            if ( !procresults )
            {
                m_logger->error( "FP4_ULTRA: MNN inference failed for window at {}", start );
                continue;
            }

            const float *data     = procresults->host<float>();
            size_t       dataSize = procresults->elementSize() * sizeof( float );

            // Stitch output (overlap-add)
            const int outputLength = std::min( patchLength,
                                               static_cast<int>( procresults->elementSize() ) );
            for ( int i = 0; i < outputLength; ++i )
            {
                const int outIndex = start + i;
                if ( outIndex >= length ) break;
                stitchedOutput[outIndex]  += data[i];
                stitchedWeights[outIndex] += 1.0f;
            }

            // Hash this chunk
            auto hash = sgprocmanagersha::sha256( data, dataSize );
            chunkhashes.push_back( hash );
        }

        // Normalize overlapping regions
        for ( int i = 0; i < length; ++i )
        {
            if ( stitchedWeights[i] > 0.0f )
                stitchedOutput[i] /= stitchedWeights[i];
        }

        // Final hash
        std::string stitchedStr( reinterpret_cast<const char *>( stitchedOutput.data() ),
                                  stitchedOutput.size() * sizeof( float ) );
        subTaskResultHash = sgprocmanagersha::sha256( stitchedStr.c_str(), stitchedStr.size() );

        m_progress = 100.0f;
        m_logger->info( "FP4_ULTRA processing complete" );

        // Build result
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

        return result;
    }

    std::unique_ptr<MNN::Tensor> MNN_FP4Ultra::Process(
        const std::vector<float> &floatData,
        std::vector<uint8_t>     &modelFileBytes,
        int                       length )
    {
        auto interpreter = std::unique_ptr<MNN::Interpreter>(
            MNN::Interpreter::createFromBuffer( modelFileBytes.data(), modelFileBytes.size() ) );
        if ( !interpreter )
        {
            m_logger->error( "FP4_ULTRA: Failed to create MNN interpreter" );
            return nullptr;
        }

        MNN::ScheduleConfig config;
        config.type      = MNN_FORWARD_VULKAN;
        config.numThread = 4;
        config.backendConfig = nullptr;

        auto session = interpreter->createSession( config );
        if ( !session )
        {
            m_logger->error( "FP4_ULTRA: Failed to create MNN session" );
            return nullptr;
        }

        auto inputTensor = interpreter->getSessionInput( session, nullptr );
        if ( !inputTensor )
        {
            m_logger->error( "FP4_ULTRA: Failed to get input tensor" );
            return nullptr;
        }

        MNN::Tensor inputTensorUser( inputTensor, inputTensor->getDimensionType() );
        auto inputPtr = inputTensorUser.host<float>();
        std::memcpy( inputPtr, floatData.data(), static_cast<size_t>( length ) * sizeof( float ) );
        inputTensor->copyFromHostTensor( &inputTensorUser );

        interpreter->runSession( session );

        auto outputTensor = interpreter->getSessionOutput( session, nullptr );
        if ( !outputTensor )
        {
            m_logger->error( "FP4_ULTRA: Failed to get output tensor" );
            return nullptr;
        }

        auto outputUserTensor = std::make_unique<MNN::Tensor>( outputTensor, outputTensor->getDimensionType() );
        outputTensor->copyToHostTensor( outputUserTensor.get() );
        return outputUserTensor;
    }
}
