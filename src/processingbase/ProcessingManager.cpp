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
        case sgns::sgprocessing::ProcessingManager::Error::INPUT_UNAVAIL:
            return "Could not get input from source";
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
        auto isvalid = CheckProcessValidity();
        if ( !isvalid )
        {
            return isvalid.error();
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
                        if ( !dimensions.get_block_len() || !dimensions.get_block_line_stride() || !dimensions.get_width() || !dimensions.get_height() || !dimensions.get_block_stride() || !dimensions.get_chunk_line_stride() || 
                            !dimensions.get_chunk_offset() || !dimensions.get_chunk_stride() || !dimensions.get_chunk_subchunk_height() || !dimensions.get_chunk_subchunk_width() )
                        {
                            m_logger->error( "Texture2d type missing dimension values" );
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

                        if (!dimensions.get_chunk_count())
                        {
                            m_logger->error( "Texture2d type has no chunk count" );
                            return outcome::failure( Error::PROCESS_INFO_MISSING );
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
        auto     passes          = processing_.get_passes();
        for ( const auto &pass : passes )
        {
            auto input_nodes = pass.get_model().value().get_input_nodes();
            for ( auto &model : input_nodes )
            {
                auto index = GetInputIndex( model.get_source().value() );
                if (!index)
                {
                    return index.error();
                }
                block_total_len +=
                    processing_.get_inputs()[index.value()].get_dimensions().value().get_block_len().value();
            }
        }
        return block_total_len;
    }

    outcome::result<std::vector<uint8_t>> ProcessingManager::Process( std::shared_ptr<boost::asio::io_context> ioc,
                                                                      std::vector<std::vector<uint8_t>> &chunkhashes,
                                                                      sgns::ModelNode                    &model )
    {
        //Get input index
        auto modelname = model.get_source().value();
        auto index     = GetInputIndex( modelname );
        if (!index)
        {
            return outcome::failure( Error::MISSING_INPUT );
        }
        auto maybe_buffers = GetCidForProc( ioc, model );
        if (!maybe_buffers)
        {
            return maybe_buffers.error();
        }
        auto buffers = maybe_buffers.value();
        if (!SetProcessorByName(static_cast<int>(processing_.get_inputs()[index.value()].get_type())))
        {
            return outcome::failure( Error::NO_PROCESSOR );
        }
        auto process = m_processor->StartProcessing( chunkhashes,
                                                     processing_.get_inputs()[index.value()],
                                                     *buffers->second,
                                                     *buffers->first );
        return process;
    }

    outcome::result <
        std::shared_ptr<std::pair<std::shared_ptr<std::vector<char>>, std::shared_ptr<std::vector<char>>>>>
    ProcessingManager::GetCidForProc( std::shared_ptr<boost::asio::io_context> ioc, sgns::ModelNode &model )
    {
        auto modelname = model.get_source().value();
        auto index     = GetInputIndex( modelname );
        if ( !index )
        {
            return outcome::failure( Error::MISSING_INPUT );
        }
        boost::asio::io_context::executor_type                                   executor = ioc->get_executor();
        boost::asio::executor_work_guard<boost::asio::io_context::executor_type> workGuard( executor );

        auto mainbuffers =
            std::make_shared<std::pair<std::shared_ptr<std::vector<char>>, std::shared_ptr<std::vector<char>>>>(
                std::make_shared<std::vector<char>>(),
                std::make_shared<std::vector<char>>() );

        std::string modelFile = processing_.get_passes()[index.value()].get_model().value().get_source_uri_param();

        std::string image = processing_.get_inputs()[index.value()].get_source_uri_param();

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

        if ( mainbuffers == nullptr )
        {
            return outcome::failure( Error::INPUT_UNAVAIL );
        }
        if ( mainbuffers->first->size() <= 0 || mainbuffers->second->size() <= 0 )
        {
            return outcome::failure( Error::INPUT_UNAVAIL );
        }

        return mainbuffers;
    }

    sgns::SgnsProcessing ProcessingManager::GetProcessingData()
    {
        return processing_;
    }

    outcome::result<size_t> ProcessingManager::GetInputIndex( const std::string &input )
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
