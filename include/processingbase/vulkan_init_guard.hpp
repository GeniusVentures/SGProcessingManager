#pragma once
#include <mutex>

namespace sgns::sgprocessing
{
    // Process-wide, header-declared synchronization primitive guarding every
    // Vulkan instance/device-creation call site in this process: every MNN
    // processor's createSession() call that requests MNN_FORWARD_VULKAN,
    // plus RenderProcessor's lazy-init path. This is a pattern, not a fixed
    // count -- grep `MNN_FORWARD_VULKAN` across
    // SGProcessingManager/src/processors/*.cpp for the current, authoritative
    // call-site count; re-verify it whenever a processor migrates backends,
    // since a stale hardcoded number here has already undercounted the real
    // total once (see 01.1-03-SUMMARY.md). Acquire via std::lock_guard at
    // each call site, once per Vulkan-init call, for the lifetime of the
    // process -- this must be acquired repeatedly (a run-once primitive
    // would be the wrong tool here).
    inline std::mutex &VulkanInitMutex()
    {
        static std::mutex vulkan_init_mutex; // magic static -- thread-safe init, C++11+
        return vulkan_init_mutex;
    }
}
