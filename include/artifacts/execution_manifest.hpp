/**
 * Execution manifest for Phase 08: Structured Artifacts & Execution Manifests.
 *
 * Defines the ExecutionManifest struct — a self-contained, fixed-field record
 * that captures everything needed for hashing, signing, caching, and verification
 * of an execution run (ARTF-04, D-13). All identity fields are inline; inapplicable
 * identities use the sentinel zero-hash convention (D-14). ARTF-09 adds a trailing
 * errorMessage field, appended without disturbing any existing field's offset.
 *
 * @brief Execution manifest data contract
 */
#ifndef SGPROCMGR_EXECUTION_MANIFEST_HPP
#define SGPROCMGR_EXECUTION_MANIFEST_HPP

#include <cstdint>
#include "artifacts/artifact_types.hpp"

namespace sgns::sgprocessing
{

    // Manifest-specific size constants (D-06).
    static constexpr size_t MAX_ARTIFACT_REFS = 64;   ///< Max input/output artifact hash references
    static constexpr size_t MAX_IDENTIFIER    = 256;  ///< Max bytes for execution/attempt/task/subtask/pass ID strings

    /// Self-contained execution manifest (ARTF-04, D-13).
    ///
    /// Captures: execution identifiers, executor & compatibility identities,
    /// input/output artifact hash references, timing, terminal state,
    /// resource-use summary, and a manifest self-hash (computed by
    /// artifact_serializer.hpp — see ComputeManifestHash).
    ///
    /// Sentinel zero-hash convention (D-14): modelIdentity, tokenizerIdentity,
    /// adapterIdentity, shaderIdentity, and quantizationIdentity are all-zero
    /// when the corresponding resource was not used in this execution.
    struct ExecutionManifest
    {
        // ── Execution identifiers (ARTF-04 first group) ────────────────

        char executionId[MAX_IDENTIFIER];   ///< Execution-scoped ID
        char attemptId[MAX_IDENTIFIER];     ///< Retry-attempt ID
        char taskId[MAX_IDENTIFIER];        ///< Task ID from job definition
        char subtaskId[MAX_IDENTIFIER];     ///< Subtask ID from processing pipeline
        char passId[MAX_RESOURCE_NAME];     ///< Producing pass identity

        // ── Executor & compatibility identities (D-13, D-14) ───────────

        uint8_t executorIdentity[SHA256_HASH_SIZE];      ///< From CapabilitySnapshot::identityHash (Phase 06 D-08); never zero
        uint8_t modelIdentity[SHA256_HASH_SIZE];         ///< SHA-256 of model bytes; all zeros if no model (D-14)
        uint8_t tokenizerIdentity[SHA256_HASH_SIZE];     ///< All zeros if no tokenizer (D-14)
        uint8_t adapterIdentity[SHA256_HASH_SIZE];       ///< All zeros if no adapter (D-14)
        uint8_t shaderIdentity[SHA256_HASH_SIZE];        ///< SHA-256 of compiled SPIR-V bytes; all zeros if no shader (D-14)
        uint8_t quantizationIdentity[SHA256_HASH_SIZE];  ///< All zeros if no quantization (D-14)

        // ── Input / output artifact hash references (D-13) ─────────────

        uint32_t inputArtifactCount = 0;
        uint8_t  inputArtifactHashes[MAX_ARTIFACT_REFS][SHA256_HASH_SIZE];

        uint32_t outputArtifactCount = 0;
        uint8_t  outputArtifactHashes[MAX_ARTIFACT_REFS][SHA256_HASH_SIZE];

        // ── Timing (D-13) ──────────────────────────────────────────────

        int64_t startTimeUsec = 0;  ///< Microseconds since Unix epoch, captured before StartProcessing()
        int64_t endTimeUsec   = 0;  ///< Microseconds since Unix epoch, captured after StartProcessing() returns

        // ── Terminal state (D-13, D-15) ────────────────────────────────

        TerminalState terminalState = TerminalState::Success;

        // ── Resource-use summary (D-13) ────────────────────────────────

        uint64_t gpuMemoryUsedBytes  = 0;  ///< Peak GPU memory during execution; 0 if not tracked
        uint64_t outputBytesProduced = 0;  ///< Sum of all output artifact byte sizes
        uint64_t wallClockUsec       = 0;  ///< endTimeUsec - startTimeUsec; computed at manifest assembly time

        // ── Manifest self-hash ─────────────────────────────────────────
        //
        // Computed by SerializeManifest() in artifact_serializer.hpp as
        // SHA-256 of the serialized manifest bytes with this field zeroed
        // (to avoid self-referential hashing). Write all zeros here before
        // calling SerializeManifest / ComputeManifestHash.
        uint8_t manifestHash[SHA256_HASH_SIZE];

        // ── Human-readable diagnostic (ARTF-09) ────────────────────────
        //
        // Carries ProcessingError::message. Defaults empty when
        // terminalState == TerminalState::Success. Silently truncated at
        // MAX_IDENTIFIER - 1 bytes with no truncation marker (D-10). Last
        // member of the struct — appended, never inserted, so no existing
        // field's offset changes (ARTF-10 append-only trailer convention).
        char errorMessage[MAX_IDENTIFIER];
    };

}  // namespace sgns::sgprocessing

#endif  // SGPROCMGR_EXECUTION_MANIFEST_HPP
