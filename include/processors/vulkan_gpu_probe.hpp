#pragma once

namespace sgns::sgprocessing
{
    /// Runtime probe answering "does this host have at least one usable Vulkan
    /// device?", mirroring RenderProcessor::IsAcceptable's DISCRETE_GPU/
    /// INTEGRATED_GPU device-type filter (D-32). Builds and immediately tears
    /// down its own throwaway VkInstance -- it never creates a VkDevice and
    /// never touches RenderProcessor's own Vulkan state.
    ///
    /// Callers MUST treat a `false` return as "skip GPU-dependent work" (e.g.
    /// via GTEST_SKIP()), never as a hard error -- a GPU-less host is an
    /// expected, valid environment (D-34), not a failure condition.
    ///
    /// Never throws.
    bool HasUsableVulkanDevice();

    /// Process-lifetime cached variant of HasUsableVulkanDevice().
    ///
    /// The uncached probe builds and destroys a whole VkInstance plus a full
    /// physical-device enumeration per call -- far too heavy for per-session
    /// or per-chunk MNN backend selection (MNN_Volume creates a session per
    /// chunk). The device set does not change over a process's lifetime in
    /// any environment we care about, so the first caller's result is cached
    /// in a function-local static (thread-safe initialization guaranteed by
    /// C++11) and every later caller gets a plain bool read.
    ///
    /// Intended use: MNN processors select MNN_FORWARD_VULKAN only when this
    /// returns true; on software-Vulkan-only hosts (llvmpipe/lavapipe in
    /// GPU-less CI containers) they select MNN_FORWARD_CPU instead -- the
    /// native CPU backend is dramatically faster than Vulkan-on-lavapipe and
    /// restores the pre-WHOLEARCHIVE behavior these hosts always had (MNN's
    /// empty creator map silently CPU-fell-back then). Render passes keep
    /// their existing GTEST_SKIP policy instead.
    ///
    /// Never throws.
    bool HasUsableVulkanDeviceCached();
}
