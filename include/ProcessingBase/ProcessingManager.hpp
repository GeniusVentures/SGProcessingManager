#ifndef PROCESSING_MANAGER_HPP_
#define PROCESSING_MANAGER_HPP_

#include <outcome/sgprocmgr-outcome.hpp>
#include <util/sgprocmgr-logger.hpp>
#include <SGNSProcMain.hpp>

namespace sgns
{
    using namespace BOOST_OUTCOME_V2_NAMESPACE;

    // Move enum to namespace level


    class ProcessingManager
    {
    public:
        ~ProcessingManager();
        enum class Error
        {
            PROCESS_INFO_MISSING     = 1,
            INVALID_JSON             = 2,
            INVALID_BLOCK_PARAMETERS = 3,
            NO_PROCESSOR             = 4,
        };
        static outcome::result<std::shared_ptr<ProcessingManager>> Create( const std::string &jsondata );

        outcome::result<uint64_t> ParseBlockSize();
        outcome::result<void>        CheckProcessValidity();
        outcome::result<void>     Process();

    private:
        outcome::result<void>       Init( const std::string &jsondata ); 
        ProcessingManager()                  = default;
        sgns::sgprocmanager::Logger m_logger = sgns::sgprocmanager::createLogger( "GlobalDB" );
        sgns::SgnsProcessing        processing_;
    };
}

#endif
