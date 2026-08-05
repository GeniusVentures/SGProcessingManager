/**
 * CapabilityValidator — pre-execution capability gate for SGProcessingManager.
 *
 * Constructed internally by ProcessingManager (D-01, D-04). Builds a capability
 * snapshot once at startup (D-09, D-12), then provides CanExecute() to validate
 * jobs against the cached snapshot before claiming work from the network (D-02).
 *
 * Uses PIMPL pattern to keep Vulkan headers out of transitive includes.
 *
 * @brief Pre-execution capability validation gate
 */
#ifndef SGPROCMGR_CAPABILITY_VALIDATOR_HPP
#define SGPROCMGR_CAPABILITY_VALIDATOR_HPP

#include <capability/capability_types.hpp>
#include <Pass.hpp>
#include <functional>
#include <memory>
#include <unordered_map>

namespace sgns::sgprocessing
{

    // Forward declaration — ProcessingManager provides the factory map.
    class ProcessingProcessor;

    /// Callback type for async CanExecute (D-03).
    using CanExecuteCallback = std::function<void( CanExecuteResult )>;

    /// Pre-execution capability validation gate.
    ///
    /// Built once at startup by ProcessingManager::Init(), then used by the scheduler
    /// to validate jobs before claiming them from the network. All checks run against
    /// the cached CapabilitySnapshot — no I/O needed at check time.
    class CapabilityValidator
    {
    public:
        CapabilityValidator();
        ~CapabilityValidator();

        // Non-copyable, non-movable (owns PIMPL)
        CapabilityValidator( const CapabilityValidator & )            = delete;
        CapabilityValidator &operator=( const CapabilityValidator & ) = delete;
        CapabilityValidator( CapabilityValidator && )                 = delete;
        CapabilityValidator &operator=( CapabilityValidator && )      = delete;

        /// Build the capability snapshot once at startup (D-09, D-12).
        /// Called by ProcessingManager::Init() after all Register* calls.
        /// Acquires VulkanInitMutex internally (D-10).
        ///
        /// @param passFactories — reference to ProcessingManager's m_passFactories,
        ///        used to enumerate registered PassType→executor mappings (D-11)
        /// @param mnnProcessorCount — number of registered MNN processors (DataType-keyed)
        /// @param ensureVulkanDevice — callable that ensures Vulkan device exists and
        ///        returns the VkPhysicalDevice handle
        void BuildSnapshot(
            const std::unordered_map<PassType,
                std::function<std::unique_ptr<ProcessingProcessor>()>,
                PassTypeHash> &passFactories,
            size_t             mnnProcessorCount,
            std::function<VkPhysicalDevice()> ensureVulkanDevice );

        /// Access the cached snapshot. Returns nullptr before BuildSnapshot().
        const CapabilitySnapshot *GetSnapshot() const;

        /// Validate whether a job can be executed on this node (CAP-02..05).
        /// Checks PassType registration, Vulkan limits, MNN model compatibility,
        /// GPU memory, and disk space against the cached snapshot.
        /// @param pass     — the job definition to validate
        /// @param callback — invoked with CanExecuteResult (D-03 async pattern)
        void CanExecute( const sgns::Pass &pass, CanExecuteCallback callback );

#ifdef SGPROCMGR_TEST_FRIEND
        /// Test-only: replace the cached snapshot with a mock.
        /// Only available when SGPROCMGR_TEST_FRIEND is defined.
        void SetSnapshotForTest( CapabilitySnapshot snap );
#endif

    private:
        struct Impl;
        std::unique_ptr<Impl> m_impl;
    };

} // namespace sgns::sgprocessing

#endif // SGPROCMGR_CAPABILITY_VALIDATOR_HPP
