/**
* Header file for base class for processors. Derived classes will handle processing various
* types of AI/ML processing as needed. Give this to a ProcessingCoreImpl.
* @author Justin Church
*/
#ifndef PROCESSING_PROCESSOR_HPP
#define PROCESSING_PROCESSOR_HPP

#include <cmath>
#include <memory>
#include <vector>
#include <SGNSProcMain.hpp>
#include <util/sgprocmgr-logger.hpp>

namespace sgns::processing
{
    class ProcessingProcessor
    {
    public:
        virtual ~ProcessingProcessor() = default;

        /** Start processing data
        * @param result - Reference to result item to set hashes to
        * @param task - Reference to task to get image split data
        * @param subTask - Reference to subtask to get chunk data from
        */
        virtual std::vector<uint8_t> StartProcessing( std::vector<std::vector<uint8_t>> &chunkhashes,
                                                      const sgns::IoDeclaration               &proc,
                                                      std::vector<char>                 &imageData,
                                                      std::vector<char>                 &modelFile ) = 0;

        /** Set data for processor
        * @param buffers - Data containing file name and data pair lists.
        */
        //virtual void SetData(std::shared_ptr<std::pair<std::vector<std::string>, std::vector<std::vector<char>>>> buffers) = 0;
        sgns::sgprocmanager::Logger m_logger = sgns::sgprocmanager::createLogger( "SGProcessingManager" );
    };
}

#endif 