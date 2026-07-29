#ifndef PROCESSING_MANAGER_HPP_
#define PROCESSING_MANAGER_HPP_

#include <outcome/sgprocmgr-outcome.hpp>
#include <util/sgprocmgr-logger.hpp>
#include <SGNSProcMain.hpp>

#include <processors/processing_processor_mnn_image.hpp>
#include <processors/processing_processor_mnn_string.hpp>
#include <processors/processing_processor_mnn_volume.hpp>
#include <processors/processing_processor_mnn_texture1d.hpp>
#include <processors/processing_processor_mnn_mat2.hpp>
#include <processors/processing_processor_mnn_mat3.hpp>
#include <processors/processing_processor_mnn_mat4.hpp>
#include <processors/processing_processor_mnn_vec2.hpp>
#include <processors/processing_processor_mnn_vec3.hpp>
#include <processors/processing_processor_mnn_vec4.hpp>
#include <processors/processing_processor_mnn_tensor.hpp>
#include <processors/processing_processor_mnn_texturecube.hpp>
#include <processors/processing_processor_mnn_bool.hpp>
#include <processors/processing_processor_mnn_buffer.hpp>
#include <processors/processing_processor_mnn_float.hpp>
#include <processors/processing_processor_mnn_int.hpp>
#include <processors/processing_processor_render.hpp>
#include <boost/asio/io_context.hpp>
#include <iostream>
#include <Generators.hpp>

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

        outcome::result<uint64_t>             ParseBlockSize();
        outcome::result<void>                 CheckProcessValidity();
        outcome::result<std::vector<uint8_t>> Process( std::shared_ptr<boost::asio::io_context> ioc,
                                                       std::vector<std::vector<uint8_t>>       &chunkhashes,
                                                       sgns::ModelNode                         &model,
                                                       std::vector<std::string>                &output_locations );

        /** Register an available processor keyed by DataType
         * @param name - DataType cast to int
         * @param factoryFunction - Pointer to processor
         */
        void RegisterProcessorFactory( const int                                            &name,
                                       std::function<std::unique_ptr<ProcessingProcessor>()> factoryFunction )
        {
            m_processorFactories[name] = std::move( factoryFunction );
        }

        /** Register an available processor keyed by PassType
         * @param type - PassType enum
         * @param factoryFunction - Pointer to processor
         */
        void RegisterPassProcessorFactory( PassType                                          type,
                                           std::function<std::unique_ptr<ProcessingProcessor>()> factoryFunction )
        {
            m_passFactories[type] = std::move( factoryFunction );
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
            if ( m_processor )
            {
                return m_processor->GetProgress();
            }
            return 0.0f;
        }

        /**
         * @brief       Checks whether a processing json is valid by attempting to create a ProcessingManager instance with it
         * @param[in]   jsondata JSON string containing the processing data to validate.
         * @return      True if the processing is valid and a ProcessingManager instance can be created, false otherwise.
         */
        static bool IsProcessingValid( const std::string &jsondata );
        /**
         * @brief       Checks if a json encoded data contains a valid ModelNode structure by attempting to parse it.
         * @param[in]   jsondata JSON string containing the ModelNode data to validate.
         * @return      True if the json can be parsed into a ModelNode, false otherwise.
         */
        static bool IsProcessingModelValid( const std::string &jsondata );

        /**
         * @brief       Gets the ModelNode structure parsed from a json string.
         * @param[in]   jsondata JSON string containing the ModelNode data to parse
         * @return      The ModelNode parsed from the json string, or an error if the json is invalid or the ModelNode structure cannot be parsed.
         */
        static outcome::result<sgns::ModelNode> GetModelNodeFromJson( const std::string &jsondata );

    private:
        ProcessingManager() = default;
        outcome::result<void> Init( const std::string &jsondata );
        outcome::result<
            std::shared_ptr<std::pair<std::shared_ptr<std::vector<char>>, std::shared_ptr<std::vector<char>>>>>
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

        bool SetProcessorByPassType( PassType type )
        {
            auto factoryFunction = m_passFactories.find( type );
            if ( factoryFunction != m_passFactories.end() )
            {
                m_processor = factoryFunction->second();
                return true;
            }
            std::cerr << "Unknown pass type: " << static_cast<int>( type ) << std::endl;
            return false;
        }

        struct PassTypeHash
        {
            size_t operator()( PassType p ) const { return static_cast<size_t>( p ); }
        };

        sgns::sgprocmanager::Logger          m_logger = sgns::sgprocmanager::createLogger( "SGProcessingManager" );
        sgns::SgnsProcessing                 processing_;
        std::unique_ptr<ProcessingProcessor> m_processor;
        std::unordered_map<int, std::function<std::unique_ptr<ProcessingProcessor>()>>     m_processorFactories;
        std::unordered_map<PassType, std::function<std::unique_ptr<ProcessingProcessor>()>, PassTypeHash> m_passFactories;
        std::unordered_map<std::string, size_t>                                            m_inputMap;
    };
}

#endif
