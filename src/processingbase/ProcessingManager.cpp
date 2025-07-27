#include <processingbase/ProcessingManager.hpp>
#include <Generators.hpp>
#include <datasplitter/ImageSplitter.hpp>
#include "FileManager.hpp"


OUTCOME_CPP_DEFINE_CATEGORY_3( sgns::sgprocessing, ProcessingManager::Error, e )
{
    switch ( e )
    {
        case sgns::sgprocessing::ProcessingManager::Error::PROCESS_INFO_MISSING:
            return "Processing information missing on JSON file";
        case sgns::sgprocessing::ProcessingManager::Error::INVALID_JSON:
            return "Json cannot be parsed";
        case sgns::sgprocessing::ProcessingManager::Error::INVALID_BLOCK_PARAMETERS:
            return "Json missing block params";
        case sgns::sgprocessing::ProcessingManager::Error::NO_PROCESSOR:
            return "Json missing processor";
        case sgns::sgprocessing::ProcessingManager::Error::MISSING_INPUT:
            return "Input missing";
    }
    return "Unknown error";
}

namespace sgns::sgprocessing
{

    ProcessingManager::~ProcessingManager() {}

    outcome::result<std::shared_ptr<ProcessingManager>> ProcessingManager::Create( const std::string &jsondata )
    {
        auto instance = std::shared_ptr<ProcessingManager>( new ProcessingManager() );
        OUTCOME_TRY( instance->Init( jsondata ) );
        return instance;
    }

    outcome::result<void> ProcessingManager::Init( const std::string &jsondata )
    {
        m_processor = nullptr;
        //Register Processors
        RegisterProcessorFactory( 11, [] { return std::make_unique<sgprocessing::MNN_Image>(); } );

        //Parse Json
        auto                 data = nlohmann::json::parse( jsondata );
        //This will check required fields inherently.
        try
        {
            sgns::from_json( data, processing_ );
        }
        catch ( const nlohmann::json::exception &e )
        {
            return outcome::failure( Error::INVALID_JSON );
        }
        if (!CheckProcessValidity())
        {
            return outcome::failure( Error::INVALID_JSON );
        }
        const auto &inputs = processing_.get_inputs();
        for ( size_t i = 0; i < inputs.size(); ++i )
        {
            std::string sourceKey = "input:" + inputs[i].get_name();
            m_inputMap[sourceKey] = i;
        } 
        // Successful parse
        return outcome::success();
    }

    outcome::result<void> ProcessingManager::CheckProcessValidity()
    {
        for (auto& pass : processing_.get_passes())
        {
            //Check optional params if needed
            switch(pass.get_type())
            {
                case PassType::INFERENCE:
                {
                    if ( !pass.get_model() )
                    {
                        m_logger->error( "Inference json has no model" );
                        return outcome::failure( Error::PROCESS_INFO_MISSING );
                    }
                    break;
                }
                case PassType::COMPUTE:
                    break;
                case PassType::DATA_TRANSFORM:
                    break;
                case PassType::RENDER:
                    break;
                case PassType::RETRAIN:
                    break;
                default:
                    m_logger->error( "Somehow pass has no type" );
                    return outcome::failure( Error::PROCESS_INFO_MISSING );
            }
            
            
        }
        //Check Input optionals
        for (auto& input : processing_.get_inputs())
        {
            switch (input.get_type())
            {
                case DataType::BOOL:
                    break;
                case DataType::BUFFER:
                    break;
                case DataType::FLOAT:
                    break;
                case DataType::INT:
                    break;
                case DataType::MAT2:
                    break;
                case DataType::MAT3:
                    break;
                case DataType::MAT4:
                    break;
                case DataType::STRING:
                    break;
                case DataType::TENSOR:
                    break;
                case DataType::TEXTURE1_D:
                    break;
                case DataType::TEXTURE1_D_ARRAY:
                    break;
                case DataType::TEXTURE2_D:
                {
                    if ( !input.get_dimensions() )
                    {
                        m_logger->error( "Texture2d type has no dimensions" );
                        return outcome::failure( Error::PROCESS_INFO_MISSING );
                    }
                    else
                    {
                        auto dimensions = input.get_dimensions().value();
                        //We need these dimensions
                        if ( !dimensions.get_block_len() || !dimensions.get_block_line_stride() )
                        {
                            m_logger->error( "Texture2d type has no block len or block line stride" );
                            return outcome::failure( Error::PROCESS_INFO_MISSING );
                        }
                        uint64_t block_len         = dimensions.get_block_len().value();
                        uint64_t block_line_stride = dimensions.get_block_line_stride().value();

                        // Ensure block_len is evenly divisible by block_line_stride
                        if ( block_line_stride == 0 || ( block_len % block_line_stride ) != 0 )
                        {
                            m_logger->error( "Texture2d type has dimensions not divisible" );
                            return outcome::failure( Error::INVALID_BLOCK_PARAMETERS );
                        }
                        
                        break;
                    }
                }
                case DataType::TEXTURE2_D_ARRAY:
                    break;
                case DataType::TEXTURE3_D:
                    break;
                case DataType::TEXTURE3_D_ARRAY:
                    break;
                case DataType::TEXTURE_CUBE:
                    break;
                case DataType::VEC2:
                    break;
                case DataType::VEC3:
                    break;
                case DataType::VEC4:
                    break;
                default:
                    return outcome::failure( Error::PROCESS_INFO_MISSING );
            }
        }
        //Check Output optionals. Anything to do here?
        for (auto& output : processing_.get_outputs())
        {

        }

        return outcome::success();
    }

    outcome::result<uint64_t> ProcessingManager::ParseBlockSize()
    {
        uint64_t block_total_len = 0;
        for (auto& input : processing_.get_inputs())
        {
            block_total_len += input.get_dimensions().value().get_block_len().value();
        }
        return block_total_len;
    }

    outcome::result<std::vector<uint8_t>> ProcessingManager::Process( std::shared_ptr<boost::asio::io_context> ioc,
                                                                      std::vector<std::vector<uint8_t>> &chunkhashes,
                                                                      int                                pass )
    {
        auto buffers = GetCidForProc( ioc, pass );
        SetProcessorByName( static_cast<int>(processing_.get_inputs()[pass].get_type()) );
        auto process = m_processor->StartProcessing( chunkhashes,
                                                     processing_.get_inputs()[pass],
                                                     *buffers->second,
                                                     *buffers->first );
        return process;
    }

    std::shared_ptr<std::pair<std::shared_ptr<std::vector<char>>, std::shared_ptr<std::vector<char>>>>
    ProcessingManager::GetCidForProc(
        std::shared_ptr<boost::asio::io_context> ioc, int pass )
    {
        boost::asio::io_context::executor_type                                   executor = ioc->get_executor();
        boost::asio::executor_work_guard<boost::asio::io_context::executor_type> workGuard( executor );

        auto mainbuffers =
            std::make_shared<std::pair<std::shared_ptr<std::vector<char>>, std::shared_ptr<std::vector<char>>>>(
                std::make_shared<std::vector<char>>(),
                std::make_shared<std::vector<char>>() );

        //Set processor or fail.
        if ( !SetProcessorByName( static_cast<int>( processing_.get_inputs()[pass].get_type() ) ) )
        {
            std::cerr << "No processor available for this type:"
                      << static_cast<int>( processing_.get_inputs()[pass].get_type() ) << std::endl;
            return mainbuffers;
        }

        std::string modelFile = processing_.get_passes()[pass].get_model().value().get_source_uri_param();

        std::string image = processing_.get_inputs()[pass].get_source_uri_param();

        //Init Loaders
        FileManager::GetInstance().InitializeSingletons();
        //Get Model
        string modelURL = modelFile;
        GetSubCidForProc( ioc, modelURL, mainbuffers->first );

        string imageUrl = image;
        GetSubCidForProc( ioc, imageUrl, mainbuffers->second );
        //Run IO
        ioc->reset();
        ioc->run();

        return mainbuffers;
    }

    sgns::SgnsProcessing ProcessingManager::GetProcessingData()
    {
        return processing_;
    }

    outcome::result<size_t> ProcessingManager::GetInputIndex(std::string &input)
    {
        auto it = m_inputMap.find( input );
        if ( it != m_inputMap.end() )
        {
            return it->second;
        }
        return outcome::failure( Error::MISSING_INPUT );
    }

    void ProcessingManager::GetSubCidForProc( std::shared_ptr<boost::asio::io_context> ioc,
                                               std::string                              url,
                                               std::shared_ptr<std::vector<char>>       results )
    {
        //std::pair<std::vector<std::string>, std::vector<std::vector<char>>> results;
        auto modeldata = FileManager::GetInstance().LoadASync(
            url,
            false,
            false,
            ioc,
            []( const sgns::AsyncError::CustomResult &status )
            {
                if ( status.has_value() )
                {
                    std::cout << "Success: " << status.value().message << std::endl;
                }
                else
                {
                    std::cout << "Error: " << status.error() << std::endl;
                }
            },
            [results]( std::shared_ptr<std::pair<std::vector<std::string>, std::vector<std::vector<char>>>> buffers )
            {
                if ( results && buffers )
                {
                    results->insert( results->end(), buffers->second[0].begin(), buffers->second[0].end() );
                }
            },
            "file" );
    }
}
