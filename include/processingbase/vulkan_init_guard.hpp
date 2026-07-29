#pragma once
#include <mutex>

namespace sgns::sgprocessing
{
    // Process-wide, header-declared synchronization primitive guarding every
    // Vulkan instance/device-creation call site in this process (MNN's 3
    // existing createSession(MNN_FORWARD_VULKAN) sites plus RenderProcessor's
    // lazy-init path). Acquire via std::lock_guard at each call site, once per
    // Vulkan-init call, for the lifetime of the process -- this must be
    // acquired repeatedly (a run-once primitive would be the wrong tool here).
    inline std::mutex &VulkanInitMutex()
    {
        static std::mutex vulkan_init_mutex; // magic static -- thread-safe init, C++11+
        return vulkan_init_mutex;
    }
}
