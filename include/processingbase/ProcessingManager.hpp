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
#include <capability/capability_validator.hpp>
#include <execution/execution_context.hpp>
#include <artifacts/artifact_types.hpp>
#include <artifacts/execution_manifest.hpp>
#include <boost/asio/io_context.hpp>
#include <iostream>
#include <Generators.hpp>

namespace sgns::sgprocessing
{
    // Move enum to namespace level
    using ProcessingProcessor = sgns::sgprocessing::ProcessingProcessor;

    /// Executor registry entry wrapping a processor factory and checkpoint support flag (D-20).
    struct ExecutorRegistryEntry
    {
        std::function<std::unique_ptr<ProcessingProcessor>()> factory;
        bool supports_checkpointing = false;
    };

    /// Structured output from Process() — typed artifact records + execution manifest (Phase 08, ARTF-01/02/04).
    struct ProcessOutput
    {
        std::vector<Artifact> artifacts;       ///< One Artifact per output buffer (ARTF-01, ARTF-02)
        ExecutionManifest    manifest;         ///< Full execution manifest (ARTF-04)
        std::vector<uint8_t> combinedHash;     ///< SHA-256 of serialized manifest (for backward compat)

        // Backward-compatible accessors — delegate to combinedHash so existing callers
        // that treat the Process() return as std::vector<uint8_t> continue to compile (D-10).
        size_t size()  const { return combinedHash.size(); }
        bool   empty() const { return combinedHash.empty(); }
        auto   begin() const { return combinedHash.begin(); }
        auto   end()   const { return combinedHash.end(); }
        auto   begin()       { return combinedHash.begin(); }
        auto   end()         { return combinedHash.end(); }
    };

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
            SHADER_COMPILE_FAILED    = 7,
            SPIRV_VALIDATION_FAILED  = 8,
            PROCESSING_FAILED        = 9,
        };
        static outcome::result<std::shared_ptr<ProcessingManager>> Create( const std::string &jsondata );

        outcome::result<uint64_t>             ParseBlockSize();
        outcome::result<void>                 CheckProcessValidity();
        outcome::result<ProcessOutput> Process( std::shared_ptr<boost::asio::io_context> ioc,
                                                std::vector<std::vector<uint8_t>>       &chunkhashes,
                                                sgns::ModelNode                         &model,
                                                std::vector<std::string>                &output_locations );

        /** Pre-execution capability gate (D-02, D-19).
         * Validates whether this node can execute the given pass — checks PassType
         * registration, Vulkan limits, MNN model compatibility, GPU memory, and disk
         * space against the cached startup snapshot. Caller's responsibility to call
         * this before Process(); Process() trusts the caller validated.
         * @param pass     — the job pass definition to validate
         * @param callback — invoked with CanExecuteResult
         */
        void CanExecute( const sgns::Pass                         &pass,
                         sgns::sgprocessing::CanExecuteCallback callback );

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
         * @param supportsCheckpointing - Whether this executor supports checkpoint/resume (D-20)
         */
        void RegisterPassProcessorFactory( PassType                                          type,
                                           std::function<std::unique_ptr<ProcessingProcessor>()> factoryFunction,
                                           bool                                              supportsCheckpointing = false )
        {
            m_passFactories[type] = { std::move( factoryFunction ), supportsCheckpointing };
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
                m_processor = factoryFunction->second.factory();
                return true;
            }
            std::cerr << "Unknown pass type: " << static_cast<int>( type ) << std::endl;
            return false;
        }

        sgns::sgprocmanager::Logger          m_logger = sgns::sgprocmanager::createLogger( "SGProcessingManager" );
        sgns::SgnsProcessing                 processing_;
        std::unique_ptr<ProcessingProcessor> m_processor;
        std::unordered_map<int, std::function<std::unique_ptr<ProcessingProcessor>()>>                    m_processorFactories;
        std::unordered_map<PassType, ExecutorRegistryEntry, PassTypeHash>                                m_passFactories;
        std::unordered_map<std::string, size_t>                                                           m_inputMap;
        std::unique_ptr<CapabilityValidator>                                                              m_capabilityValidator;
    };
}

OUTCOME_HPP_DECLARE_ERROR_2( sgns::sgprocessing, ProcessingManager::Error );

#endif
