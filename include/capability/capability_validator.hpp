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

        /// Local ELM resource preflight: host RAM + disk legs against the cached
        /// snapshot (plan 02-02, Pitfall 13 / ELM-13). A node refuses an ELM
        /// acquire whose model cannot fit BEFORE downloading hundreds of MB.
        ///
        /// Deliberately pass-free and manifest-free: ELM work items carry no
        /// sgns::Pass, and this library stays independent of the elmruntime
        /// generated manifest types (the root generated/ set and the
        /// elmruntime-manifest/ set cannot coexist in one translation unit --
        /// both define sgns::ClassMemberConstraints/ElmType). Callers extract
        /// plain values via sgns::elmruntime::ExtractElmResourceRequirements
        /// (include/elmruntime/ElmResourcePreflight.hpp) and hand them here;
        /// the call site arrives with the Phase 3 processor. This check is
        /// local-only: nothing is advertised on the network (binding Out of
        /// Scope).
        ///
        /// Degraded axes: a snapshot field of 0 (platform query failed) skips
        /// that leg -- the same convention as CanExecute's disk check. A
        /// requiredMemoryBytes of 0 (requirement absent) never fires the
        /// memory leg.
        ///
        /// @param requiredMemoryBytes — manifest runtime.required_memory_bytes (0 = no requirement)
        /// @param totalArtifactBytes  — sum of manifest artifact size_bytes
        /// @param callback            — invoked with CanExecuteResult; unmet
        ///        requirements carry UnmetRequirementCategory::RESOURCE with
        ///        details naming "required_memory_bytes" or the disk shortfall
        void CheckElmResources( uint64_t             requiredMemoryBytes,
                                uint64_t             totalArtifactBytes,
                                const CanExecuteCallback &callback );

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
