#include "processors/vulkan_gpu_probe.hpp"
#include "processors/processing_processor_render.hpp"
#include "processingbase/vulkan_init_guard.hpp"
#include <VkBootstrap.h>
#include <algorithm>
#include <mutex>

namespace sgns::sgprocessing
{
    bool HasUsableVulkanDevice()
    {
        try
        {
            std::lock_guard<std::mutex> lock( sgns::sgprocessing::VulkanInitMutex() );

            vkb::InstanceBuilder instance_builder;
            auto                 inst_ret = instance_builder.set_app_name( "SGProcessingManager GPU Probe" )
                                  .set_app_version( 1, 0, 0 )
                                  .request_validation_layers( false )
                                  .build();
            if ( !inst_ret )
            {
                // No instance was created -- nothing to destroy.
                return false;
            }
            auto vkb_instance = inst_ret.value();

            vkb::PhysicalDeviceSelector selector( vkb_instance );
            // Headless/offscreen probe -- no VkSurfaceKHR ever exists, same rationale
            // as RenderProcessor::InitializeContext()'s own require_present(false).
            selector.require_present( false );
            auto devices_ret = selector.select_devices();
            if ( !devices_ret )
            {
                vkb::destroy_instance( vkb_instance );
                return false;
            }

            auto devices = devices_ret.value();

            devices.erase( std::remove_if( devices.begin(),
                                            devices.end(),
                                            []( const vkb::PhysicalDevice &d )
                                            { return !RenderProcessor::IsAcceptable( d.properties.deviceType ); } ),
                           devices.end() );

            bool usable = !devices.empty();

            vkb::destroy_instance( vkb_instance );

            return usable;
        }
        catch ( ... )
        {
            // Never throw -- a probe failure of any kind means "no usable device".
            return false;
        }
    }
}
