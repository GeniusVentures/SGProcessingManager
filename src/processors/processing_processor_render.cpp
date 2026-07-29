#include "processors/processing_processor_render.hpp"
#include "processingbase/vulkan_init_guard.hpp"
#include <VkBootstrap.h>
#include <algorithm>
#include <mutex>

namespace sgns::sgprocessing
{

    bool RenderProcessor::IsAcceptable( VkPhysicalDeviceType type )
    {
        return type == VK_PHYSICAL_DEVICE_TYPE_DISCRETE_GPU
            || type == VK_PHYSICAL_DEVICE_TYPE_INTEGRATED_GPU;
    }

    VkDeviceSize RenderProcessor::LargestDeviceLocalHeap( VkPhysicalDevice device )
    {
        VkPhysicalDeviceMemoryProperties memProps;
        vkGetPhysicalDeviceMemoryProperties( device, &memProps );
        VkDeviceSize largest = 0;
        for ( uint32_t i = 0; i < memProps.memoryHeapCount; ++i )
        {
            if ( memProps.memoryHeaps[i].flags & VK_MEMORY_HEAP_DEVICE_LOCAL_BIT )
                largest = (std::max)( largest, memProps.memoryHeaps[i].size );
        }
        return largest;
    }

    bool RenderProcessor::InitializeContext()
    {
        if ( m_contextInitialized )
            return true;

        std::lock_guard<std::mutex> lock( sgns::sgprocessing::VulkanInitMutex() );

        if ( m_contextInitialized )
            return true;

        vkb::InstanceBuilder instance_builder;
        auto inst_ret = instance_builder
                            .set_app_name( "SGProcessingManager RenderProcessor" )
                            .set_app_version( 1, 0, 0 )
                            .request_validation_layers( false )
                            .build();
        if ( !inst_ret )
        {
            m_logger->error( "RenderProcessor: failed to create Vulkan instance: {}",
                             inst_ret.error().message() );
            return false;
        }
        auto vkb_instance = inst_ret.value();

        vkb::PhysicalDeviceSelector selector( vkb_instance );
        auto devices_ret = selector.select_devices();
        if ( !devices_ret )
        {
            m_logger->error( "RenderProcessor: failed to enumerate physical devices: {}",
                             devices_ret.error().message() );
            vkb::destroy_instance( vkb_instance );
            return false;
        }

        auto devices = devices_ret.value();

        devices.erase(
            std::remove_if( devices.begin(), devices.end(),
                []( const vkb::PhysicalDevice &d ) {
                    return !IsAcceptable( d.properties.deviceType );
                } ),
            devices.end() );

        if ( devices.empty() )
        {
            m_logger->error( "RenderProcessor: no acceptable physical device found "
                             "(none with device type DISCRETE_GPU or INTEGRATED_GPU)" );
            vkb::destroy_instance( vkb_instance );
            return false;
        }

        std::sort( devices.begin(), devices.end(),
            []( const vkb::PhysicalDevice &a, const vkb::PhysicalDevice &b ) {
                int rank_a = ( a.properties.deviceType == VK_PHYSICAL_DEVICE_TYPE_DISCRETE_GPU ) ? 2 : 1;
                int rank_b = ( b.properties.deviceType == VK_PHYSICAL_DEVICE_TYPE_DISCRETE_GPU ) ? 2 : 1;
                if ( rank_a != rank_b )
                    return rank_a > rank_b;
                return LargestDeviceLocalHeap( a.physical_device )
                     > LargestDeviceLocalHeap( b.physical_device );
            } );

        vkb::DeviceBuilder device_builder( devices[0] );
        auto dev_ret = device_builder.build();
        if ( !dev_ret )
        {
            m_logger->error( "RenderProcessor: failed to create Vulkan device: {}",
                             dev_ret.error().message() );
            vkb::destroy_instance( vkb_instance );
            return false;
        }

        auto vkb_device = dev_ret.value();
        auto queue_ret = vkb_device.get_queue( vkb::QueueType::graphics );
        if ( !queue_ret )
        {
            m_logger->error( "RenderProcessor: failed to get graphics queue: {}",
                             queue_ret.error().message() );
            vkb::destroy_device( vkb_device );
            vkb::destroy_instance( vkb_instance );
            return false;
        }

        m_instance = vkb_instance.instance;
        m_physicalDevice = vkb_device.physical_device;
        m_device = vkb_device.device;
        m_queue = queue_ret.value();
        m_contextInitialized = true;

        return true;
    }

    ProcessingResult RenderProcessor::StartProcessing(
        std::vector<std::vector<uint8_t>> &chunkhashes,
        const sgns::IoDeclaration         &proc,
        std::vector<char>                 &imageData,
        std::vector<char>                 &modelFile,
        const std::vector<sgns::Parameter> *parameters )
    {
        (void)proc;
        (void)imageData;
        (void)modelFile;
        (void)parameters;

        if ( !InitializeContext() )
        {
            ProcessingResult result;
            result.hash = std::vector<uint8_t>( 32, 0 );
            return result;
        }

        ProcessingResult result;
        result.hash = std::vector<uint8_t>( 32, 0 );
        m_progress = 100.0f;
        return result;
    }

}
