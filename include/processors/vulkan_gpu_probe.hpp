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
}
