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
                    if (!pass.get_model())
                    {
                        return outcome::failure( Error::PROCESS_INFO_MISSING );
                    }
                case PassType::COMPUTE:
                case PassType::DATA_TRANSFORM:
                case PassType::RENDER:
                case PassType::RETRAIN:
                default:
                    return outcome::failure( Error::PROCESS_INFO_MISSING );
            }
            
            
        }

        //// Extract input array
        //const auto &inputArray = document["input"];
        //if ( inputArray.Size() == 0 )
        //{
        //    return outcome::failure( Error::PROCESS_INFO_MISSING );
        //}

        //// Validate each input entry
        //for ( auto &input : inputArray.GetArray() )
        //{
        //    if ( !input.IsObject() )
        //    {
        //        return outcome::failure( Error::PROCESS_INFO_MISSING );
        //    }

        //    if ( !input.HasMember( "block_len" ) || !input["block_len"].IsUint64() )
        //    {
        //        return outcome::failure( Error::PROCESS_INFO_MISSING );
        //    }

        //    if ( !input.HasMember( "block_line_stride" ) || !input["block_line_stride"].IsUint64() )
        //    {
        //        return outcome::failure( Error::PROCESS_INFO_MISSING );
        //    }

        //    uint64_t block_len         = input["block_len"].GetUint64();
        //    uint64_t block_line_stride = input["block_line_stride"].GetUint64();

        //    // Ensure block_len is evenly divisible by block_line_stride
        //    if ( block_line_stride == 0 || ( block_len % block_line_stride ) != 0 )
        //    {
        //        return outcome::failure( Error::INVALID_BLOCK_PARAMETERS );
        //    }
        //}

        return outcome::success();
    }
}
