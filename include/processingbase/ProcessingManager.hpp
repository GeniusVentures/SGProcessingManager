#ifndef PROCESSING_MANAGER_HPP_
#define PROCESSING_MANAGER_HPP_

#include <outcome/sgprocmgr-outcome.hpp>
#include <util/sgprocmgr-logger.hpp>
#include <SGNSProcMain.hpp>
#include <processors/processing_processor_mnn_image.hpp>
#include <processors/processing_processor_mnn_string.hpp>
#include <processors/processing_processor_mnn_volume.hpp>
#include <boost/asio/io_context.hpp>
#include <iostream>



namespace sgns::sgprocessing
{
    // Move enum to namespace level
    using ProcessingProcessor = sgns::sgprocessing::ProcessingProcessor;


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
            MISSING_INPUT            = 5,
            INPUT_UNAVAIL            = 6,
        };
        static outcome::result<std::shared_ptr<ProcessingManager>> Create( const std::string &jsondata );

        outcome::result<uint64_t> ParseBlockSize();
        outcome::result<void>        CheckProcessValidity();
        outcome::result<std::vector<uint8_t>> Process( std::shared_ptr<boost::asio::io_context> ioc,
                                                       std::vector<std::vector<uint8_t>>       &chunkhashes,
                                                       sgns::ModelNode                          &model );

        /** Register an available processor
        * @param name - Name of processor
        * @param factoryFunction - Pointer to processor
        */
        void RegisterProcessorFactory( const int                                    &name,
                                       std::function<std::unique_ptr<ProcessingProcessor>()> factoryFunction )
        {
            m_processorFactories[name] = std::move( factoryFunction );
        }

        /** Get Processing Data item which can be used to access any processing data, inputs, or params.
        */
        sgns::SgnsProcessing GetProcessingData();

        /** Get input map Index
        */
        outcome::result<size_t> GetInputIndex( const std::string &input );

        /** Get current processing progress
        * @return Progress percentage (0.0 to 100.0)
        */
        float GetProgress() const
        {
            if (m_processor) {
                return m_processor->GetProgress();
            }
            return 0.0f;
        }

    private:
        ProcessingManager() = default;
        outcome::result<void>       Init( const std::string &jsondata ); 
        outcome::result<std::shared_ptr<std::pair<std::shared_ptr<std::vector<char>>, std::shared_ptr<std::vector<char>>>>>
             GetCidForProc( std::shared_ptr<boost::asio::io_context> ioc, sgns::ModelNode &model );
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
        
        sgns::sgprocmanager::Logger m_logger = sgns::sgprocmanager::createLogger( "SGProcessingManager" );
        sgns::SgnsProcessing        processing_;
        std::unique_ptr<ProcessingProcessor> m_processor;
        std::unordered_map<int, std::function<std::unique_ptr<ProcessingProcessor>()>> m_processorFactories;
        std::unordered_map<std::string, size_t>                                        m_inputMap;
    };
}

#endif
