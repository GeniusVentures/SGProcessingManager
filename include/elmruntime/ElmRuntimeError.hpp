#ifndef SGPROCMGR_ELMRUNTIME_ERROR_HPP
#define SGPROCMGR_ELMRUNTIME_ERROR_HPP

#include <outcome/sgprocmgr-outcome.hpp>

namespace sgns::elmruntime
{
    /// @brief Structured error codes for the ELM runtime manifest/cache layer (Plan 02-01).
    ///
    /// Deliberately a standalone category (A26): extending ProcessingManager::Error would
    /// recompile every consumer of the processing base for a subsystem that Phase 3 maps to
    /// ProcessingErrorStage::RESOURCE_RESOLUTION at the processor boundary -- these codes never
    /// leak upward as-is. Every value is fail-closed: there is no debug flag, environment
    /// variable, or code path that bypasses hash verification or converts a mismatch into a
    /// warning (SC-1).
    enum class ElmRuntimeError : int
    {
        FETCH_FAILED = 0,            ///< A generic fetch through the FetchFn seam failed (including a FileManager range_error on an unregistered URI prefix).
        MANIFEST_FETCH_FAILED,       ///< Fetching the manifest bytes themselves failed before any verification could run.
        MANIFEST_HASH_MISMATCH,      ///< sha256(manifest bytes) does not match the work item's declared model_manifest_hash -- the bytes are never parsed (SC-1).
        MANIFEST_INVALID,            ///< The manifest failed the size ceiling, declared-hash normalization, JSON parse, or a semantic gate (roles/format/duplicates/bounds).
        ARTIFACT_FETCH_FAILED,       ///< Fetching one of the manifest's artifact URIs failed.
        ARTIFACT_HASH_MISMATCH,      ///< An artifact's fetched bytes do not hash to its declared sha256.
        ARTIFACT_SIZE_MISMATCH,      ///< An artifact's byte count does not match its declared size_bytes.
        CACHE_DIR_UNSET,             ///< The cache root directory is unset/empty -- fail closed, never guess a default (D-03).
        CACHE_ENTRY_QUARANTINED,     ///< The cache entry was moved to .bad-<hash> after a reuse-time verification failure (D-02).
        SMOKE_CHECK_FAILED,          ///< createLLM/load/1-token generation did not succeed on a materialized bundle (SC-2).
        SMOKE_CHECK_UNAVAILABLE,     ///< The smoke check could not run (e.g. MNN LLM support absent in this build) -- the entry is not marked usable.
    };
} // namespace sgns::elmruntime

// Category registration (header half of the ProcessingManager.hpp:290 pattern):
// declares make_error_code + the is_error_code_enum specialization. The
// message mapping lives in src/elmruntime/ElmRuntimeError.cpp via
// OUTCOME_CPP_DEFINE_CATEGORY_3 -- that macro emits non-inline external
// definitions (make_error_code body + the Category<Enum>::toString
// specialization) and MUST NOT appear in a header included by more than one
// TU, or the link fails with duplicate symbols.
OUTCOME_HPP_DECLARE_ERROR_2( sgns::elmruntime, ElmRuntimeError );

#endif // SGPROCMGR_ELMRUNTIME_ERROR_HPP
