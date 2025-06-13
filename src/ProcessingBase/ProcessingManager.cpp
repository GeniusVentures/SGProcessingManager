#include <ProcessingBase/ProcessingManager.hpp>
#include <SGNSProcMain.hpp>
#include <Generators.hpp>

using namespace BOOST_OUTCOME_V2_NAMESPACE;

OUTCOME_CPP_DEFINE_CATEGORY_3( sgns, ProcessingManager::Error, e )
{
    switch ( e )
    {
        case sgns::ProcessingManager::Error::PROCESS_INFO_MISSING:
            return "Processing information missing on JSON file";
        case sgns::ProcessingManager::Error::INVALID_JSON:
            return "Json cannot be parsed";
        case sgns::ProcessingManager::Error::INVALID_BLOCK_PARAMETERS:
            return "Json missing block params";
        case sgns::ProcessingManager::Error::NO_PROCESSOR:
            return "Json missing processor";
    }
    return "Unknown error";
}

namespace sgns
{
    ProcessingManager::ProcessingManager() {

    }

    ProcessingManager::~ProcessingManager() {}

    outcome::result<void> ProcessingManager::CheckProcessValidity( const std::string &jsondata )
    {
        auto data = nlohmann::json::parse( jsondata );
        sgns::SgnsProcessing processing;
        //This will check required fields inherently.
        try
        {
            sgns::from_json( data, processing );
        }
        catch (const nlohmann::json::exception& e)
        {
            return outcome::failure( Error::INVALID_JSON );
        }
        for (auto& pass : processing.get_passes())
        {
            //Check optional params if needed
            switch(pass.get_type())
            {
                case PassType::INFERENCE:
                {
                    if ( !pass.get_model() )
                    {
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
                    return outcome::failure( Error::PROCESS_INFO_MISSING );
            }
            
            
        }
        //Check Input optionals
        for (auto& input : processing.get_inputs())
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
                        return outcome::failure( Error::PROCESS_INFO_MISSING );
                    }
                    else
                    {
                        auto dimensions = input.get_dimensions().value();
                        //We need these dimensions
                        if ( !dimensions.get_block_len() || !dimensions.get_block_line_stride() )
                        {
                            return outcome::failure( Error::PROCESS_INFO_MISSING );

                            uint64_t block_len         = dimensions.get_block_len().value();
                            uint64_t block_line_stride = dimensions.get_block_line_stride().value();

                            // Ensure block_len is evenly divisible by block_line_stride
                            if ( block_line_stride == 0 || ( block_len % block_line_stride ) != 0 )
                            {
                                return outcome::failure( Error::INVALID_BLOCK_PARAMETERS );
                            }
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
        for (auto& output : processing.get_outputs())
        {

        }

        return outcome::success();
    }
}
