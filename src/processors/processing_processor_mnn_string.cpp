#include "processors/processing_processor_mnn_string.hpp"
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
        bool TryParseTokenIds( const std::string &text, std::vector<int32_t> &tokenIds )
        {
            std::istringstream stream( text );
            int64_t            value = 0;
            while ( stream >> value )
            {
                tokenIds.push_back( static_cast<int32_t>( value ) );
            }

            if ( tokenIds.empty() )
            {
                return false;
            }

            stream.clear();
            stream >> std::ws;
            return stream.eof();
        }

        std::string ToLowerAscii( std::string value )
        {
            std::transform( value.begin(), value.end(), value.begin(),
                            []( unsigned char c ) { return static_cast<char>( std::tolower( c ) ); } );
            return value;
        }
    }

    ProcessingResult MNN_String::StartProcessing( std::vector<std::vector<uint8_t>> &chunkhashes,
                                                   const sgns::IoDeclaration         &proc,
                                                   std::vector<char>                 &textData,
                                                   std::vector<char>                 &modelFile,
                                                   const std::vector<sgns::Parameter> *parameters )
    {
        (void)parameters;
        std::vector<uint8_t> modelFile_bytes;
        modelFile_bytes.assign(modelFile.begin(), modelFile.end());

        std::vector<uint8_t> subTaskResultHash(SHA256_DIGEST_LENGTH);
        
        // Convert text data to string
        std::string inputText( textData.begin(), textData.end() );

        m_logger->info( "Processing text input: {}", inputText );
        
        // For string inputs, we process as a single "chunk"
        m_progress = 0.0f;
        
        std::vector<uint8_t> shahash( SHA256_DIGEST_LENGTH );
        
        // Default max length (could be extracted from parameters)
        int maxLength = 128;

        std::vector<int32_t> tokenIds;
        bool                 parsedTokenIds = TryParseTokenIds( inputText, tokenIds );
        if ( parsedTokenIds )
        {
            if ( static_cast<int>( tokenIds.size() ) > maxLength )
            {
                tokenIds.resize( maxLength );
            }
            m_logger->info( "Parsed {} token id(s) from input", tokenIds.size() );
        }
        else
        {
            m_logger->info( "Input is not token ids; using character codes as fallback" );
            tokenIds.reserve( std::min( static_cast<int>( inputText.size() ), maxLength ) );
            for ( size_t i = 0; i < inputText.size() && static_cast<int>( i ) < maxLength; ++i )
            {
                tokenIds.push_back( static_cast<int32_t>( static_cast<unsigned char>( inputText[i] ) ) );
            }
        }

        auto procresults = Process( tokenIds, modelFile_bytes, maxLength );
        
        const float *data     = procresults->host<float>();
        size_t       dataSize = procresults->elementSize() * sizeof( float );
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
        shahash               = sgprocmanagersha::sha256( data, dataSize );
        std::string hashString( shahash.begin(), shahash.end() );
        chunkhashes.push_back( shahash );
        
        std::string combinedHash = std::string(subTaskResultHash.begin(), subTaskResultHash.end()) + hashString;
        subTaskResultHash = sgprocmanagersha::sha256( combinedHash.c_str(), combinedHash.length() );
        
        m_progress = 100.0f;
        
        m_logger->info( "String processing complete" );
        
        ProcessingResult result;
        result.hash = subTaskResultHash;

        if ( procresults && procresults->elementSize() > 0 )
        {
            const size_t byteCount = procresults->elementSize() * sizeof( float );
            std::vector<char> outputBytes( byteCount );
            std::memcpy( outputBytes.data(), data, byteCount );

            result.output_buffers = std::make_shared<std::pair<std::vector<std::string>, std::vector<std::vector<char>>>>();
            result.output_buffers->first.push_back( "" );
            result.output_buffers->second.push_back( std::move( outputBytes ) );
        }

        return result;
    }

    std::unique_ptr<MNN::Tensor> MNN_String::Process( const std::vector<int32_t> &tokenIds, 
                                                        std::vector<uint8_t> &modelFile,
                                                        const int maxLength ) 
    {
        m_logger->info( "Creating MNN interpreter from model file" );
        
        // Create MNN interpreter from model bytes
        auto interpreter = std::shared_ptr<MNN::Interpreter>(
            MNN::Interpreter::createFromBuffer(modelFile.data(), modelFile.size())
        );
        
        if (!interpreter) {
            m_logger->error( "Failed to create MNN interpreter" );
            return std::make_unique<MNN::Tensor>();
        }
        
        // Configure session
        MNN::ScheduleConfig config;
        config.type = MNN_FORWARD_VULKAN;  // Use Vulkan backend as requested
        config.numThread = 4;
        
        auto session = interpreter->createSession(config);
        if (!session) {
            m_logger->error( "Failed to create MNN session" );
            return std::make_unique<MNN::Tensor>();
        }
        
        // Get all input tensors - BERT models typically have multiple inputs
        auto inputTensors = interpreter->getSessionInputAll(session);
        m_logger->info( "Model has {} input tensor(s)", inputTensors.size() );
        
        // Log all input tensor names and shapes
        for (const auto& inputPair : inputTensors) {
            m_logger->info( "Input '{}': shape {}x{}x{}x{}", 
                           inputPair.first,
                           inputPair.second->batch(), 
                           inputPair.second->channel(),
                           inputPair.second->height(), 
                           inputPair.second->width() );
        }
        
        // Get the first input tensor (usually input_ids for BERT)
        auto inputTensor = interpreter->getSessionInput(session, nullptr);
        if (!inputTensor) {
            m_logger->error( "Failed to get input tensor" );
            return std::make_unique<MNN::Tensor>();
        }
        
        // Resize all input tensors to [batch=1, sequence_length=maxLength]
        // BERT models expect: input_ids, attention_mask, token_type_ids (all same shape)
        for (const auto& inputPair : inputTensors) {
            auto tensor = inputPair.second;
            if (tensor->elementSize() <= 4) {
                m_logger->info( "Resizing '{}' to [1, {}]", inputPair.first, maxLength );
                interpreter->resizeTensor( tensor, { 1, maxLength } );
            }
        }
        interpreter->resizeSession( session );
        
        // Log shapes after resize
        for (const auto& inputPair : inputTensors) {
            m_logger->info( "After resize '{}': shape {}x{}x{}x{}", 
                           inputPair.first,
                           inputPair.second->batch(), 
                           inputPair.second->channel(),
                           inputPair.second->height(), 
                           inputPair.second->width() );
        }
        
        // Fill input tensors with dummy data
        // In production, you would use a proper BERT tokenizer
        for (const auto& inputPair : inputTensors) {
            auto tensor = inputPair.second;
            MNN::Tensor inputTensorUser(tensor, tensor->getDimensionType());
            
            auto inputData = inputTensorUser.host<int32_t>();
            for (int i = 0; i < inputTensorUser.elementSize(); ++i) {
                std::string inputName = ToLowerAscii( inputPair.first );
                if ( inputName.find( "input_ids" ) != std::string::npos )
                {
                    inputData[i] = ( i < static_cast<int>( tokenIds.size() ) ) ? tokenIds[i] : 0;
                }
                else if ( inputName.find( "attention_mask" ) != std::string::npos )
                {
                    inputData[i] = ( i < static_cast<int>( tokenIds.size() ) ) ? 1 : 0;
                }
                else if ( inputName.find( "token_type_ids" ) != std::string::npos )
                {
                    inputData[i] = 0;
                }
                else
                {
                    inputData[i] = ( i < static_cast<int>( tokenIds.size() ) ) ? tokenIds[i] : 0;
                }
            }
            
            tensor->copyFromHostTensor(&inputTensorUser);
            m_logger->info( "Filled '{}' with {} elements", inputPair.first, inputTensorUser.elementSize() );
        }
        
        // Run inference
        m_logger->info( "Running MNN inference" );
        interpreter->runSession(session);
        
        // Get output tensor
        auto outputTensor = interpreter->getSessionOutput(session, nullptr);
        if (!outputTensor) {
            m_logger->error( "Failed to get output tensor" );
            return std::make_unique<MNN::Tensor>();
        }
        
        m_logger->info( "Output tensor shape: {}x{}x{}x{}", 
                       outputTensor->batch(), 
                       outputTensor->channel(),
                       outputTensor->height(), 
                       outputTensor->width() );
        
        // Create host tensor for output
        auto outputHost = std::make_unique<MNN::Tensor>(outputTensor, outputTensor->getDimensionType());
        outputTensor->copyToHostTensor(outputHost.get());
        
        m_logger->info( "MNN inference complete" );
        
        return outputHost;
    }
}
