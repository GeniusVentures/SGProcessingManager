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

    // Two-lock discipline for LLM processors (elmbridge Phase 3, D-01/D-03):
    //
    //   VulkanInitMutex() = the createLLM()/GPU context-creation window ONLY.
    //   LlmLoadMutex()    = serializes MNN::Transformer::Llm::load() against
    //                       other LLM loads, but NOT against the rest of the
    //                       grid.
    //
    // Render/MNN processors never wait on an LLM weight load: they only ever
    // contend on VulkanInitMutex() (the short createLLM window), so a
    // multi-second weight load cannot stall non-LLM processing. Two
    // concurrent ELM loads serialize on LlmLoadMutex() -- conservative, no
    // claim of concurrent-weight-load safety -- while each load itself runs
    // in the per-subtask worker thread (asio handlers only coordinate).
    //
    // Call-site convention: acquire via std::lock_guard in two SEPARATE
    // scoped blocks -- { VulkanInitMutex() around createLLM() } first, then
    // (with set_config() in between, outside any lock) { LlmLoadMutex()
    // around load() }. Re-verify the call-site inventory (grep
    // `LlmLoadMutex` across SGProcessingManager/src/) whenever an LLM
    // processor is added or migrated, same discipline as VulkanInitMutex
    // above.
    inline std::mutex &LlmLoadMutex()
    {
        static std::mutex llm_load_mutex; // magic static -- thread-safe init, C++11+
        return llm_load_mutex;
    }
}
