/**
* Header file for base class for processors. Derived classes will handle processing various
* types of AI/ML processing as needed. Give this to a ProcessingCoreImpl.
* @author Justin Church
*/
#ifndef PROCESSING_PROCESSOR_HPP
#define PROCESSING_PROCESSOR_HPP

#include <cmath>
#include <memory>
#include <optional>
#include <string>
#include <utility>
#include <vector>
#include <SGNSProcMain.hpp>
#include <util/sgprocmgr-logger.hpp>

namespace sgns::sgprocessing
{
    /// Per-stage failure classification for structured processor errors (D-25/D-26).
    /// Plain, non-outcome::result enum -- StartProcessing()'s return type stays the
    /// concrete ProcessingResult, so the OUTCOME_HPP_DECLARE_ERROR_2 machinery is
    /// unnecessary here.
    enum class ProcessingErrorStage
    {
        UNSPECIFIED = 0,
        CONTEXT_INIT_FAILED,
        RESOURCE_RESOLUTION,
        BUFFER_ALLOCATION,
        IMAGE_ALLOCATION,
        FORMAT_UNSUPPORTED,
        SHADER_MODULE_CREATION,
        PIPELINE_CREATION,
        RENDER_PASS_CREATION,
        DRAW_SUBMISSION,
        READBACK,
        DATA_TRANSFORM_UNSUPPORTED
    };

    /// Structured, per-stage processor failure detail (D-25/D-26). Carries the
    /// failing VkResult/context as a plain message string.
    struct ProcessingError
    {
        ProcessingErrorStage stage = ProcessingErrorStage::UNSPECIFIED;
        std::string          message;
    };

    struct ProcessingResult
    {
        std::vector<uint8_t> hash;
        std::shared_ptr<std::pair<std::vector<std::string>, std::vector<std::vector<char>>>> output_buffers;
        /// Output locations for each saved result (file paths, IPFS CIDs, URLs, etc.)
        std::vector<std::string> output_locations;
        /// Structured per-stage failure detail (D-25/D-26). Empty/unset on success.
        std::optional<ProcessingError> error;
    };

    class ProcessingProcessor
    {
    public:
        virtual ~ProcessingProcessor() = default;

        /** Start processing data
        * @param result - Reference to result item to set hashes to
        * @param task - Reference to task to get image split data
        * @param subTask - Reference to subtask to get chunk data from
        */
        virtual ProcessingResult StartProcessing( std::vector<std::vector<uint8_t>> &chunkhashes,
                               const sgns::IoDeclaration         &proc,
                               std::vector<char>                 &imageData,
                               std::vector<char>                 &modelFile,
                               const std::vector<sgns::Parameter> *parameters ) = 0;

        /** Set data for processor
        * @param buffers - Data containing file name and data pair lists.
        */
        //virtual void SetData(std::shared_ptr<std::pair<std::vector<std::string>, std::vector<std::vector<char>>>> buffers) = 0;
        
        /** Get current processing progress
        * @return Progress percentage (0.0 to 100.0)
        */
        virtual float GetProgress() const { return m_progress; }

    protected:
        std::atomic<float> m_progress{0.0f}; // Progress percentage
        sgns::sgprocmanager::Logger m_logger = sgns::sgprocmanager::createLogger( "SGProcessor" );
    };
}

#endif 