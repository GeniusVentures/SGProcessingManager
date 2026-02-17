#include <processingbase/ProcessingManager.hpp>
#include <Generators.hpp>
#include <datasplitter/ImageSplitter.hpp>
#include "FileManager.hpp"
#include "URLStringUtil.h"


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
    namespace
    {
        bool IsUrl( const std::string &value )
        {
            return value.find( "://" ) != std::string::npos;
        }

        bool EndsWithSlash( const std::string &value )
        {
            if ( value.empty() )
            {
                return false;
            }
            const char last = value.back();
            return last == '/' || last == '\\';
        }

        bool UrlHasExtension( const std::string &value )
        {
            std::string prefix;
            std::string base;
            std::string extension;
            if ( !getURLComponents( value, prefix, base, extension ) )
            {
                return false;
            }
            return !extension.empty();
        }
    }

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
        RegisterProcessorFactory( 7, [] { return std::make_unique<sgprocessing::MNN_String>(); } );
        RegisterProcessorFactory( 0, [] { return std::make_unique<sgprocessing::MNN_Bool>(); } );
        RegisterProcessorFactory( 1, [] { return std::make_unique<sgprocessing::MNN_Buffer>(); } );
        RegisterProcessorFactory( 2, [] { return std::make_unique<sgprocessing::MNN_Float>(); } );
        RegisterProcessorFactory( 3, [] { return std::make_unique<sgprocessing::MNN_Int>(); } );
        RegisterProcessorFactory( 4, [] { return std::make_unique<sgprocessing::MNN_Mat2>(); } );
        RegisterProcessorFactory( 9, [] { return std::make_unique<sgprocessing::MNN_Texture1D>(); } );
        RegisterProcessorFactory( 13, [] { return std::make_unique<sgprocessing::MNN_Volume>(); } );

        //Parse Json
        //This will check required fields inherently.
        try
        {
            auto data = nlohmann::json::parse( jsondata );
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
                {
                    if ( !input.get_dimensions() || !input.get_dimensions()->get_width() )
                    {
                        m_logger->error( "Bool type missing width" );
                        return outcome::failure( Error::PROCESS_INFO_MISSING );
                    }

                    if ( input.get_format() )
                    {
                        const auto format = input.get_format().value();
                        if ( format != sgns::InputFormat::FLOAT32 && format != sgns::InputFormat::FLOAT16 &&
                             format != sgns::InputFormat::INT8 )
                        {
                            m_logger->error( "Bool type supports FLOAT32/FLOAT16/INT8 formats only" );
                            return outcome::failure( Error::PROCESS_INFO_MISSING );
                        }
                    }
                    else
                    {
                        m_logger->warn( "Bool input missing format; defaulting to FLOAT32" );
                    }
                    break;
                }
                case DataType::BUFFER:
                {
                    if ( !input.get_dimensions() || !input.get_dimensions()->get_width() )
                    {
                        m_logger->error( "Buffer type missing width" );
                        return outcome::failure( Error::PROCESS_INFO_MISSING );
                    }

                    if ( input.get_format() )
                    {
                        const auto format = input.get_format().value();
                        if ( format != sgns::InputFormat::INT8 )
                        {
                            m_logger->error( "Buffer type supports INT8 format only" );
                            return outcome::failure( Error::PROCESS_INFO_MISSING );
                        }
                    }
                    else
                    {
                        m_logger->warn( "Buffer input missing format; defaulting to INT8" );
                    }
                    break;
                }
                case DataType::FLOAT:
                {
                    if ( !input.get_dimensions() || !input.get_dimensions()->get_width() )
                    {
                        m_logger->error( "Float type missing width" );
                        return outcome::failure( Error::PROCESS_INFO_MISSING );
                    }

                    if ( input.get_format() )
                    {
                        const auto format = input.get_format().value();
                        if ( format != sgns::InputFormat::FLOAT32 && format != sgns::InputFormat::FLOAT16 )
                        {
                            m_logger->error( "Float type supports FLOAT32/FLOAT16 formats only" );
                            return outcome::failure( Error::PROCESS_INFO_MISSING );
                        }
                    }
                    else
                    {
                        m_logger->warn( "Float input missing format; defaulting to FLOAT32" );
                    }
                    break;
                }
                case DataType::INT:
                {
                    if ( !input.get_dimensions() || !input.get_dimensions()->get_width() )
                    {
                        m_logger->error( "Int type missing width" );
                        return outcome::failure( Error::PROCESS_INFO_MISSING );
                    }

                    if ( input.get_format() )
                    {
                        const auto format = input.get_format().value();
                        if ( format != sgns::InputFormat::INT32 && format != sgns::InputFormat::INT16 &&
                             format != sgns::InputFormat::INT8 )
                        {
                            m_logger->error( "Int type supports INT32/INT16/INT8 formats only" );
                            return outcome::failure( Error::PROCESS_INFO_MISSING );
                        }
                    }
                    else
                    {
                        m_logger->warn( "Int input missing format; defaulting to INT32" );
                    }
                    break;
                }
                case DataType::MAT2:
                {
                    if ( !input.get_dimensions() || !input.get_dimensions()->get_width() )
                    {
                        m_logger->error( "Mat2 type missing width" );
                        return outcome::failure( Error::PROCESS_INFO_MISSING );
                    }

                    if ( input.get_format() )
                    {
                        const auto format = input.get_format().value();
                        if ( format != sgns::InputFormat::FLOAT32 && format != sgns::InputFormat::FLOAT16 )
                        {
                            m_logger->error( "Mat2 type supports FLOAT32/FLOAT16 formats only" );
                            return outcome::failure( Error::PROCESS_INFO_MISSING );
                        }
                    }
                    else
                    {
                        m_logger->warn( "Mat2 input missing format; defaulting to FLOAT32" );
                    }
                    break;
                }
                case DataType::MAT3:
                    break;
                case DataType::MAT4:
                    break;
                case DataType::STRING:
                {
                    if ( !processing_.get_parameters() )
                    {
                        m_logger->error( "String input missing parameters" );
                        return outcome::failure( Error::PROCESS_INFO_MISSING );
                    }

                    const auto params = processing_.get_parameters().value();
                    auto find_param = [&params]( const std::string &name ) -> const sgns::Parameter * {
                        for ( const auto &param : params )
                        {
                            if ( param.get_name() == name )
                            {
                                return &param;
                            }
                        }
                        return nullptr;
                    };

                    const auto *tokenizer_mode = find_param( "tokenizerMode" );
                    if ( !tokenizer_mode || tokenizer_mode->get_type() != sgns::ParameterType::STRING )
                    {
                        m_logger->error( "String input missing tokenizerMode parameter" );
                        return outcome::failure( Error::PROCESS_INFO_MISSING );
                    }

                    std::string mode;
                    const auto &mode_default = tokenizer_mode->get_parameter_default();
                    if ( mode_default.is_string() )
                    {
                        mode = mode_default.get<std::string>();
                    }
                    else
                    {
                        m_logger->error( "tokenizerMode default must be a string" );
                        return outcome::failure( Error::PROCESS_INFO_MISSING );
                    }

                    if ( mode == "raw_text" )
                    {
                        const auto *vocab_uri = find_param( "vocabUri" );
                        if ( !vocab_uri || vocab_uri->get_type() != sgns::ParameterType::URI )
                        {
                            m_logger->error( "raw_text tokenizer mode requires vocabUri parameter" );
                            return outcome::failure( Error::PROCESS_INFO_MISSING );
                        }
                    }
                    break;
                }
                case DataType::TENSOR:
                    break;
                case DataType::TEXTURE1_D:
                {
                    if ( !input.get_dimensions() )
                    {
                        m_logger->error( "Texture1d type has no dimensions" );
                        return outcome::failure( Error::PROCESS_INFO_MISSING );
                    }

                    auto dimensions = input.get_dimensions().value();
                    if ( !dimensions.get_width() )
                    {
                        m_logger->error( "Texture1d type missing width" );
                        return outcome::failure( Error::PROCESS_INFO_MISSING );
                    }

                    if ( input.get_format() )
                    {
                        const auto format = input.get_format().value();
                        if ( format != sgns::InputFormat::FLOAT32 && format != sgns::InputFormat::FLOAT16 )
                        {
                            m_logger->error( "Texture1d type supports FLOAT32/FLOAT16 formats only" );
                            return outcome::failure( Error::PROCESS_INFO_MISSING );
                        }
                    }
                    else
                    {
                        m_logger->warn( "Texture1d input missing format; defaulting to FLOAT32" );
                    }
                    break;
                }
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
                {
                    if ( !input.get_dimensions() )
                    {
                        m_logger->error( "Texture3d type has no dimensions" );
                        return outcome::failure( Error::PROCESS_INFO_MISSING );
                    }

                    auto dimensions = input.get_dimensions().value();
                    if ( !dimensions.get_width() || !dimensions.get_height() || !dimensions.get_chunk_count() )
                    {
                        m_logger->error( "Texture3d type missing width/height/chunk_count" );
                        return outcome::failure( Error::PROCESS_INFO_MISSING );
                    }

                    if ( !dimensions.get_chunk_subchunk_width() || !dimensions.get_chunk_subchunk_height() ||
                         !dimensions.get_block_len() )
                    {
                        m_logger->error( "Texture3d type missing patch size parameters" );
                        return outcome::failure( Error::PROCESS_INFO_MISSING );
                    }

                    if ( input.get_format() )
                    {
                        const auto format = input.get_format().value();
                        if ( format != sgns::InputFormat::FLOAT32 && format != sgns::InputFormat::FLOAT16 )
                        {
                            m_logger->error( "Texture3d type supports FLOAT32/FLOAT16 formats only" );
                            return outcome::failure( Error::PROCESS_INFO_MISSING );
                        }
                    }
                    else
                    {
                        m_logger->warn( "Texture3d input missing format; defaulting to FLOAT32" );
                    }
                    break;
                }
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
        const auto *parameters = processing_.get_parameters() ? &processing_.get_parameters().value() : nullptr;
        auto processResult = m_processor->StartProcessing( chunkhashes,
                                   processing_.get_inputs()[index.value()],
                                   *buffers->second,
                                   *buffers->first,
                                   parameters );

        const auto &outputs = processing_.get_outputs();
        if ( processResult.output_buffers && !outputs.empty() )
        {
            const auto &bufferNames = processResult.output_buffers->first;
            const auto &bufferData = processResult.output_buffers->second;

            if ( !bufferData.empty() )
            {
                FileManager::GetInstance().InitializeSingletons();
                bool hasSaves = false;

                for ( size_t outputIndex = 0; outputIndex < outputs.size(); ++outputIndex )
                {
                    const auto &output = outputs[outputIndex];
                    const auto &outputUrl = output.get_source_uri_param();
                    if ( outputUrl.empty() )
                    {
                        continue;
                    }
                    if ( !IsUrl( outputUrl ) )
                    {
                        m_logger->warn( "Output source_uri_param '{}' is not a URL; skipping save", outputUrl );
                        continue;
                    }

                    const size_t dataIndex = ( bufferData.size() == outputs.size() ) ? outputIndex : 0;
                    if ( dataIndex >= bufferData.size() )
                    {
                        continue;
                    }

                    const size_t nameIndex = ( bufferNames.size() == outputs.size() ) ? outputIndex : 0;
                    std::string outputFileName;
                    if ( !UrlHasExtension( outputUrl ) )
                    {
                        std::string baseName;
                        if ( nameIndex < bufferNames.size() && !bufferNames[nameIndex].empty() )
                        {
                            baseName = bufferNames[nameIndex];
                        }
                        else
                        {
                            baseName = output.get_name() + ".raw";
                        }

                        if ( EndsWithSlash( outputUrl ) )
                        {
                            outputFileName = baseName;
                        }
                        else
                        {
                            outputFileName = "/" + baseName;
                        }
                    }

                    auto saveBuffers = std::make_shared<std::pair<std::vector<std::string>, std::vector<std::vector<char>>>>();
                    saveBuffers->first.push_back( outputFileName );
                    saveBuffers->second.push_back( bufferData[dataIndex] );

                    FileManager::GetInstance().SaveASync(
                        outputUrl,
                        outcome::success( saveBuffers ),
                        ioc,
                        [this, outputUrl]( const FileManager::ResultType &result ) {
                            if ( !result )
                            {
                                m_logger->error( "Failed to save output to {}: {}", outputUrl, result.error().message() );
                            }
                        } );
                    hasSaves = true;
                }

                if ( hasSaves )
                {
                    ioc->reset();
                    ioc->run();
                }
            }
        }

        return processResult.hash;
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
        m_logger->info( "Model Input URL: {}", modelFile );
        m_logger->info( "Data Input URL: {}", image );
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
            [this, results]( outcome::result<std::shared_ptr<std::pair<std::vector<std::string>, std::vector<std::vector<char>>>>> buffers )
            {
                if (buffers)
                {
                    if ( results )
                    {
                        results->insert( results->end(),
                                         buffers.value()->second[0].begin(),
                                         buffers.value()->second[0].end() );
                    }
                }
                else
                {
                    m_logger->error( "Failed to obtain processing source: {}", buffers.error().message() );
                }

            },
            "file" );
    }
}
