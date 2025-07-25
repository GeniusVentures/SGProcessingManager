/**
* Header file for processing posenet using MNN
* @author Justin Church
*/
#pragma once
#include <cmath>
#include <memory>
#include <vector>

#include <MNN/ImageProcess.hpp>
#include <MNN/Interpreter.hpp>
#include "processing_processor.hpp"
#define MNN_OPEN_TIME_TRACE
#include <MNN/AutoTime.hpp>

//Defines
#define MODEL_IMAGE_SIZE 513
#define OUTPUT_STRIDE 16

#define MAX_POSE_DETECTIONS 10
#define NUM_KEYPOINTS 17
#define SCORE_THRESHOLD 0.5
#define MIN_POSE_SCORE 0.25
#define NMS_RADIUS 20
#define LOCAL_MAXIMUM_RADIUS 1

#define OFFSET_NODE_NAME "offset_2"
#define DISPLACE_FWD_NODE_NAME "displacement_fwd_2"
#define DISPLACE_BWD_NODE_NAME "displacement_bwd_2"
#define HEATMAPS "heatmap"

#define CIRCLE_RADIUS 3

namespace sgns::sgprocessing
{
    using namespace MNN;

    class MNN_Image : public ProcessingProcessor
    {
    public:
        /** Create a posenet processor
        */
        MNN_Image() 
            //imageData_( std::make_unique<std::vector<std::vector<char>>>() ),
            //modelFile_( std::make_unique<std::vector<uint8_t>>() )
        {
        }

        ~MNN_Image() override{
            //stbi_image_free(imageData_);
        };
        /** Start processing data
        * @param result - Reference to result item to set hashes to
        * @param task - Reference to task to get image split data
        * @param subTask - Reference to subtask to get chunk data from
        */
        std::vector<uint8_t> StartProcessing( std::vector<std::vector<uint8_t>> &chunkhashes,
                                              const sgns::IoDeclaration               &proc,
                                              std::vector<char>                 &imageData,
                                              std::vector<char>                 &modelFile ) override;

        /** Set data for processor
        * @param buffers - Data containing file name and data pair lists.
        */
        //void SetData(
        //    std::shared_ptr<std::pair<std::vector<std::string>, std::vector<std::vector<char>>>> buffers ) override;

    private:
        /** Run MNN processing on image
        * @param imgdata - RGBA image bytes
        * @param origwidth - Width of image
        * @param origheight - Height of image
        */
        std::unique_ptr<MNN::Tensor> Process( const std::vector<uint8_t> &imgdata, 
                                                std::vector<uint8_t> &modelFile, 
                                                const int channels, 
                                                const int origwidth, 
                                                const int origheight,
                                                const std::string filename = "" );

        //std::unique_ptr<std::vector<std::vector<char>>> imageData_;
        //std::unique_ptr<std::vector<uint8_t>>           modelFile_;
        //std::string                                     fileName_;
    };

}
