#ifndef PROCESSING_MANAGER_HPP_
#define PROCESSING_MANAGER_HPP_

#include <outcome/sgprocmgr-outcome.hpp>
#include <util/sgprocmgr-logger.hpp>

namespace sgns
{
    using namespace BOOST_OUTCOME_V2_NAMESPACE;

    // Move enum to namespace level


    class ProcessingManager
    {
    public:
        ProcessingManager();
        ~ProcessingManager();
        enum class Error
        {
            PROCESS_INFO_MISSING     = 1,
            INVALID_JSON             = 2,
            INVALID_BLOCK_PARAMETERS = 3,
            NO_PROCESSOR             = 4,
        };

        outcome::result<uint64_t> ParseBlockSize( const std::string &json_data );
        outcome::result<void>        CheckProcessValidity( const std::string &jsondata );

    private:
        sgns::sgprocmanager::Logger m_logger = sgns::sgprocmanager::createLogger( "GlobalDB" );
    };
}

#endif
