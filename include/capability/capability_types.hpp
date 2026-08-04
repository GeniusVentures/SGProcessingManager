/**
 * Capability validation type system for Phase 06.
 *
 * Defines the data contracts (UnmetRequirement, CanExecuteResult, CapabilitySnapshot)
 * that all capability validation checks build against. No protobuf — plain C++ structs
 * per D-05.
 *
 * @brief Capability validation data types
 */
#ifndef SGPROCMGR_CAPABILITY_TYPES_HPP
#define SGPROCMGR_CAPABILITY_TYPES_HPP

#include <PassType.hpp>
#include <cstdint>
#include <string>
#include <vector>
#include <vulkan/vulkan.h>

namespace sgns::sgprocessing
{

    /// Category of unmet requirement for structured capability rejection (D-06).
    /// Follows same enum-prefix convention as ProcessingErrorStage in processing_processor.hpp.
    enum class UnmetRequirementCategory
    {
        VULKAN   = 0,  ///< Vulkan device/feature/limit insufficiency
        MNN      = 1,  ///< MNN model format or quantization not supported
        PASS_TYPE = 2, ///< No executor registered for the requested PassType
        RESOURCE = 3   ///< GPU memory or disk space insufficient
    };

    /// A single unmet capability requirement with category tag and human-readable detail.
    /// Pattern follows ProcessingError in processing_processor.hpp (D-06).
    struct UnmetRequirement
    {
        UnmetRequirementCategory category = UnmetRequirementCategory::RESOURCE;
        std::string              detail;  ///< Human-readable reason, e.g. "maxImageDimension2D: need 16384, have 8192"
    };

    /// Declared capability of a registered executor (D-11).
    struct ExecutorCapability
    {
        PassType                   passType;              ///< PassType this executor handles
        std::vector<std::string>   supportedModelFormats; ///< e.g. ".mnn", ".caffemodel"
        std::vector<std::string>   supportedQuantizations;///< e.g. "FP32", "FP16", "INT8"
        std::string                backend;               ///< "VULKAN" (per D-13, all MNN on Vulkan after Phase 01.1)
    };

    /// Full capability snapshot built once at startup (D-08, D-09).
    /// Cached for all subsequent CanExecute calls (D-12).
    struct CapabilitySnapshot
    {
        VkPhysicalDeviceProperties      vulkanProps;       ///< From vkGetPhysicalDeviceProperties() (D-14)
        VkPhysicalDeviceMemoryProperties memProps;         ///< From vkGetPhysicalDeviceMemoryProperties() (D-15)
        std::vector<ExecutorCapability>  executorCaps;     ///< From registry query (D-11)
        uint64_t                         availableDiskBytes = 0; ///< From platform syscall (D-16); 0 = query failed (degraded)
        std::vector<uint8_t>             identityHash;     ///< SHA-256 of serialized snapshot (D-08)
    };

    /// Result of a CanExecute check (D-05, D-07).
    struct CanExecuteResult
    {
        bool                          executable  = false; ///< True if all capability checks pass
        std::string                   executorId;          ///< Populated only when executable==true; hex prefix of identityHash (D-08)
        std::vector<UnmetRequirement> unmet;               ///< Populated only when executable==false; one entry per failing check
    };

} // namespace sgns::sgprocessing

#endif // SGPROCMGR_CAPABILITY_TYPES_HPP
