#include "processors/processing_processor_mnn_string.hpp"
#include <functional>
#include <thread>
#include <openssl/sha.h> // For SHA256_DIGEST_LENGTH
#include "util/sha256.hpp"

namespace sgns::sgprocessing
{
    using namespace MNN;

    std::vector<uint8_t> MNN_String::StartProcessing( std::vector<std::vector<uint8_t>> &chunkhashes,
                                                       const sgns::IoDeclaration         &proc,
                                                       std::vector<char>                 &textData,
                                                       std::vector<char>                 &modelFile )
    {
        std::vector<uint8_t> modelFile_bytes;
        modelFile_bytes.assign(modelFile.begin(), modelFile.end());

        std::vector<uint8_t> subTaskResultHash(SHA256_DIGEST_LENGTH);
        
        // Convert text data to string
        std::string inputText(textData.begin(), textData.end());
        
        m_logger->info( "Processing text input: {}", inputText );
        
        // For string inputs, we process as a single "chunk"
        m_progress = 0.0f;
        
        std::vector<uint8_t> shahash( SHA256_DIGEST_LENGTH );
        
        // Default max length (could be extracted from parameters)
        int maxLength = 128;
        
        auto procresults = Process( inputText, modelFile_bytes, maxLength );
        
        const float *data     = procresults->host<float>();
        size_t       dataSize = procresults->elementSize() * sizeof( float );
        shahash               = sgprocmanagersha::sha256( data, dataSize );
        std::string hashString( shahash.begin(), shahash.end() );
        chunkhashes.push_back( shahash );
        
        std::string combinedHash = std::string(subTaskResultHash.begin(), subTaskResultHash.end()) + hashString;
        subTaskResultHash = sgprocmanagersha::sha256( combinedHash.c_str(), combinedHash.length() );
        
        m_progress = 100.0f;
        
        m_logger->info( "String processing complete" );
        
        return subTaskResultHash;
    }

    std::unique_ptr<MNN::Tensor> MNN_String::Process( const std::string &textData, 
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
                // Fill input_ids with character codes, others with 1s (attention mask)
                if (inputPair.first.find("input") != std::string::npos || 
                    inputPair.first.find("ids") != std::string::npos) {
                    inputData[i] = (i < textData.length()) ? static_cast<int32_t>(textData[i]) : 0;
                } else {
                    inputData[i] = (i < textData.length()) ? 1 : 0; // attention mask
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
