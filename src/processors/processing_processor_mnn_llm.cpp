/**
 * MNN LLM processor — autoregressive text generation
 *
 * This processor implements token-by-token generation:
 *   1. Parse input as space-separated token IDs
 *   2. Forward pass through MNN model → logits
 *   3. Sample next token (temperature + top-k + top-p)
 *   4. Append token, repeat until EOS or max_tokens
 *   5. Output generated token IDs as raw int32 bytes
 *   6. Chain SHA256 hashes per step for proof-of-work
 */

#include "processors/processing_processor_mnn_llm.hpp"

#include <algorithm>
#include <cmath>
#include <cstring>
#include <numeric>
#include <random>
#include <sstream>
#include <openssl/sha.h>
#include "util/sha256.hpp"

namespace sgns::sgprocessing
{
    using namespace MNN;

    ProcessingResult MNN_LLM::StartProcessing(
        std::vector<std::vector<uint8_t>>  &chunkhashes,
        const sgns::IoDeclaration          &proc,
        std::vector<char>                  &inputData,
        std::vector<char>                  &modelFile,
        const std::vector<sgns::Parameter> *parameters )
    {
        (void)proc;
        std::vector<uint8_t> modelBytes( modelFile.begin(), modelFile.end() );
        std::string inputText( inputData.begin(), inputData.end() );

        // -----------------------------------------------------------------
        // Extract generation parameters from JSON parameters array
        // -----------------------------------------------------------------
        int   max_tokens   = 512;
        float temperature  = 0.7f;
        float top_p        = 0.9f;
        int   top_k        = 40;
        int   eos_token_id = 2;

        if ( parameters )
        {
            for ( const auto &p : *parameters )
            {
                const auto &def = p.get_parameter_default();
                if ( p.get_name() == "maxTokens" && def.is_number_integer() )
                    max_tokens = static_cast<int>( def.get<int64_t>() );
                else if ( p.get_name() == "temperature" && def.is_number() )
                    temperature = static_cast<float>( def.get<double>() );
                else if ( p.get_name() == "topP" && def.is_number() )
                    top_p = static_cast<float>( def.get<double>() );
                else if ( p.get_name() == "topK" && def.is_number_integer() )
                    top_k = static_cast<int>( def.get<int64_t>() );
                else if ( p.get_name() == "eosTokenId" && def.is_number_integer() )
                    eos_token_id = static_cast<int>( def.get<int64_t>() );
            }
        }

        // -----------------------------------------------------------------
        // Parse input as space-separated token IDs
        // -----------------------------------------------------------------
        std::vector<int32_t> token_ids;
        {
            std::istringstream stream( inputText );
            int64_t val;
            while ( stream >> val )
                token_ids.push_back( static_cast<int32_t>( val ) );
        }

        if ( token_ids.empty() )
        {
            m_logger->error( "LLM processor: no token IDs in input data" );
            return ProcessingResult{};
        }

        m_logger->info( "LLM autoregressive generation: {} input tokens, max_new_tokens={}",
                        token_ids.size(), max_tokens );

        // -----------------------------------------------------------------
        // Autoregressive generation loop
        // -----------------------------------------------------------------
        std::vector<int32_t> generated;
        std::vector<uint8_t> subTaskResultHash( SHA256_DIGEST_LENGTH, 0 );

        for ( int step = 0; step < max_tokens; ++step )
        {
            int seq_len = static_cast<int>( token_ids.size() );
            auto logits_tensor = Forward( token_ids, modelBytes, seq_len );

            if ( !logits_tensor || logits_tensor->elementSize() == 0 )
            {
                m_logger->error( "LLM forward pass failed at step {}", step );
                break;
            }

            const float *logits     = logits_tensor->host<float>();
            int          vocab_size = logits_tensor->elementSize();

            // For causal LLM with output shape [1, seq_len, vocab_size],
            // take logits from the last sequence position
            int dims = logits_tensor->dimensions();
            if ( dims >= 3 )
            {
                int out_seq = logits_tensor->length( 1 );
                vocab_size  = logits_tensor->length( 2 );
                logits      = logits + ( out_seq - 1 ) * vocab_size;
            }
            else if ( dims == 2 )
            {
                // Shape [seq_len, vocab_size] — take last row
                int out_seq = logits_tensor->length( 0 );
                vocab_size  = logits_tensor->length( 1 );
                logits      = logits + ( out_seq - 1 ) * vocab_size;
            }

            int32_t next_token = SampleToken( logits, vocab_size, temperature, top_p, top_k );

            // Chain hash for proof-of-work
            auto step_hash = sgprocmanagersha::sha256( logits,
                                                       static_cast<size_t>( vocab_size ) * sizeof( float ) );
            chunkhashes.push_back( step_hash );

            std::string combined( subTaskResultHash.begin(), subTaskResultHash.end() );
            combined.append( step_hash.begin(), step_hash.end() );
            subTaskResultHash = sgprocmanagersha::sha256( combined.c_str(), combined.size() );

            // Check EOS
            if ( next_token == eos_token_id )
            {
                m_logger->info( "LLM generation: EOS at step {}", step );
                break;
            }

            token_ids.push_back( next_token );
            generated.push_back( next_token );

            m_progress = static_cast<float>( step + 1 ) * 100.0f / static_cast<float>( max_tokens );
        }

        m_progress = 100.0f;
        m_logger->info( "LLM generation complete: {} tokens generated", generated.size() );

        // -----------------------------------------------------------------
        // Build result: generated token IDs as raw int32 bytes
        // -----------------------------------------------------------------
        ProcessingResult result;
        result.hash = subTaskResultHash;

        if ( !generated.empty() )
        {
            const size_t byteCount = generated.size() * sizeof( int32_t );
            std::vector<char> outputBytes( byteCount );
            std::memcpy( outputBytes.data(), generated.data(), byteCount );

            result.output_buffers =
                std::make_shared<std::pair<std::vector<std::string>, std::vector<std::vector<char>>>>();
            result.output_buffers->first.push_back( "" );
            result.output_buffers->second.push_back( std::move( outputBytes ) );
        }

        return result;
    }

    std::unique_ptr<MNN::Tensor> MNN_LLM::Forward(
        const std::vector<int32_t> &input_ids,
        std::vector<uint8_t>       &modelFileBytes,
        int                         seq_len )
    {
        auto interpreter = std::unique_ptr<MNN::Interpreter>(
            MNN::Interpreter::createFromBuffer( modelFileBytes.data(), modelFileBytes.size() ) );
        if ( !interpreter )
        {
            m_logger->error( "LLM: Failed to create MNN interpreter" );
            return nullptr;
        }

        MNN::ScheduleConfig config;
        config.type      = MNN_FORWARD_VULKAN;
        config.numThread = 4;

        auto session = interpreter->createSession( config );
        if ( !session )
        {
            m_logger->error( "LLM: Failed to create MNN session" );
            return nullptr;
        }

        auto inputTensor = interpreter->getSessionInput( session, nullptr );
        if ( !inputTensor )
        {
            m_logger->error( "LLM: Failed to get input tensor" );
            return nullptr;
        }

        // Resize input to [1, seq_len]
        interpreter->resizeTensor( inputTensor, { 1, seq_len } );
        interpreter->resizeSession( session );

        // Fill input with token IDs
        MNN::Tensor inputUser( inputTensor, inputTensor->getDimensionType() );
        auto *ptr = inputUser.host<int32_t>();
        for ( int i = 0; i < seq_len; ++i )
            ptr[i] = ( i < static_cast<int>( input_ids.size() ) ) ? input_ids[i] : 0;
        inputTensor->copyFromHostTensor( &inputUser );

        // Run forward pass
        interpreter->runSession( session );

        auto outputTensor = interpreter->getSessionOutput( session, nullptr );
        if ( !outputTensor )
        {
            m_logger->error( "LLM: Failed to get output tensor" );
            return nullptr;
        }

        auto outputHost = std::make_unique<MNN::Tensor>( outputTensor, outputTensor->getDimensionType() );
        outputTensor->copyToHostTensor( outputHost.get() );
        return outputHost;
    }

    int32_t MNN_LLM::SampleToken( const float *logits, int vocab_size,
                                    float temperature, float top_p, int top_k ) const
    {
        // Build scored pairs: (logit/temperature, token_id)
        std::vector<std::pair<float, int>> scored( vocab_size );
        const float temp = std::max( temperature, 0.01f );
        for ( int i = 0; i < vocab_size; ++i )
            scored[i] = { logits[i] / temp, i };

        // Top-K: keep only the K highest-scoring tokens
        int k = std::min( top_k, vocab_size );
        std::partial_sort( scored.begin(), scored.begin() + k, scored.end(),
                           []( const auto &a, const auto &b ) { return a.first > b.first; } );
        scored.resize( k );

        // Softmax over top-K
        float max_val = scored[0].first;
        float sum     = 0.0f;
        for ( auto &s : scored )
        {
            s.first = std::exp( s.first - max_val );
            sum += s.first;
        }
        for ( auto &s : scored )
            s.first /= sum;

        // Top-P (nucleus): keep tokens until cumulative probability >= top_p
        float cumulative = 0.0f;
        int   cutoff     = 0;
        for ( ; cutoff < static_cast<int>( scored.size() ); ++cutoff )
        {
            cumulative += scored[cutoff].first;
            if ( cumulative >= top_p )
            {
                ++cutoff;
                break;
            }
        }
        scored.resize( cutoff );

        // Re-normalize and sample
        sum = 0.0f;
        for ( auto &s : scored )
            sum += s.first;

        static thread_local std::mt19937 rng( std::random_device{}() );
        std::uniform_real_distribution<float> dist( 0.0f, sum );
        float r = dist( rng );

        float acc = 0.0f;
        for ( const auto &s : scored )
        {
            acc += s.first;
            if ( acc >= r )
                return static_cast<int32_t>( s.second );
        }
        return static_cast<int32_t>( scored.back().second );
    }
}
