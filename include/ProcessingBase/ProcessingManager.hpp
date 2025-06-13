#ifndef PROCESSING_MANAGER_HPP_
#define PROCESSING_MANAGER_HPP_

#include <outcome/outcome.hpp>

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

        outcome::result<std::string> ProcessImage( const std::string &jsondata );
        outcome::result<void>        CheckProcessValidity( const std::string &jsondata );

    private:
    };
}

#endif
