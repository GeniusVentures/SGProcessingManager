#ifndef PROCESSING_MANAGER_HPP_
#define PROCESSING_MANAGER_HPP_

#include <outcome/sgprocmgr-outcome.hpp>
#include <util/sgprocmgr-logger.hpp>
#include <SGNSProcMain.hpp>
#include <Processors/processing_processor.hpp>
#include <boost/asio/io_context.hpp>

namespace sgns
{
    using namespace BOOST_OUTCOME_V2_NAMESPACE;

    // Move enum to namespace level
    using ProcessingProcessor = sgns::processing::ProcessingProcessor;

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
        outcome::result<void>     Process( std::shared_ptr<boost::asio::io_context> ioc );

        /** Register an available processor
        * @param name - Name of processor
        * @param factoryFunction - Pointer to processor
        */
        void RegisterProcessorFactory( const int                                    &name,
                                       std::function<std::unique_ptr<ProcessingProcessor>()> factoryFunction )
        {
            m_processorFactories[name] = std::move( factoryFunction );
        }

    private:
        ProcessingManager() = default;
        outcome::result<void>       Init( const std::string &jsondata ); 
        std::shared_ptr<std::pair<std::shared_ptr<std::vector<char>>, std::shared_ptr<std::vector<char>>>>
             GetCidForProc( std::shared_ptr<boost::asio::io_context> ioc, int pass );
        void GetSubCidForProc( std::shared_ptr<boost::asio::io_context> ioc,
                                                  std::string                              url,
                                                  std::shared_ptr<std::vector<char>>       results );

        bool SetProcessorByName( const int &name )
        {
            auto factoryFunction = m_processorFactories.find( name );
            if ( factoryFunction != m_processorFactories.end() )
            {
                m_processor = factoryFunction->second();
                return true;
            }
            std::cerr << "Unknown processor name: " << name << std::endl;
            return false;
        }
        
        sgns::sgprocmanager::Logger m_logger = sgns::sgprocmanager::createLogger( "GlobalDB" );
        sgns::SgnsProcessing        processing_;
        std::unique_ptr<ProcessingProcessor> m_processor;
        std::unordered_map<int, std::function<std::unique_ptr<ProcessingProcessor>()>> m_processorFactories;
    };
}

#endif
