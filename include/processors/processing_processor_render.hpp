#pragma once
#include <vulkan/vulkan.h>
#include "processing_processor.hpp"

namespace sgns::sgprocessing
{
    class RenderProcessor : public ProcessingProcessor
    {
    public:
        RenderProcessor() {}
        ~RenderProcessor() override = default;

        ProcessingResult StartProcessing( std::vector<std::vector<uint8_t>> &chunkhashes,
                           const sgns::IoDeclaration         &proc,
                           std::vector<char>                 &imageData,
                           std::vector<char>                 &modelFile,
                           const std::vector<sgns::Parameter> *parameters ) override;

    private:
        bool InitializeContext();

        static bool IsAcceptable( VkPhysicalDeviceType type );

        static VkDeviceSize LargestDeviceLocalHeap( VkPhysicalDevice device );

        VkInstance m_instance{VK_NULL_HANDLE};
        VkPhysicalDevice m_physicalDevice{VK_NULL_HANDLE};
        VkDevice m_device{VK_NULL_HANDLE};
        VkQueue m_queue{VK_NULL_HANDLE};
        bool m_contextInitialized{false};
    };
}
