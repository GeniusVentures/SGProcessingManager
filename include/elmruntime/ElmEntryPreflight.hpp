#ifndef SGPROCMGR_ELMRUNTIME_ELM_ENTRY_PREFLIGHT_HPP
#define SGPROCMGR_ELMRUNTIME_ELM_ENTRY_PREFLIGHT_HPP

// Pinned-entry preflight bridge (elmbridge Phase 3, plan 03-03).
//
// The processor TU includes the ROOT generated set (SGNSProcMain.hpp ->
// sgns::Elm) plus capability headers (also root set), while the manifest
// types live in the FALLBACK generated set (elmruntime-manifest/) -- the two
// sets redefine sgns::ClassMemberConstraints and can never share a TU (see
// ElmResourcePreflight.hpp's header comment). This header declares the
// plain-value bridge: the definition TU (src/elmruntime/ElmEntryPreflight.cpp)
// includes the fallback set ONLY and returns two uint64 values, so the
// processor passes them straight into
// CapabilityValidator::CheckElmResources(uint64_t, uint64_t, callback).
//
// ZERO generated includes here -- safe from any TU.

#include <outcome/sgprocmgr-outcome.hpp>

#include <cstdint>
#include <string>

namespace sgns::elmruntime
{
    /// @brief Plain-value preflight outcome for a pinned cache entry.
    struct ElmEntryPreflightValues
    {
        uint64_t requiredMemoryBytes = 0;  ///< runtime.required_memory_bytes (0 = absent)
        uint64_t totalArtifactBytes  = 0;  ///< sum of artifact size_bytes
    };

    /// @brief Read + re-verify the pinned entry's elm_manifest.json and extract
    ///        the preflight values.
    ///
    /// Reads <entryDir>/elm_manifest.json, runs ParseAndVerifyManifest against
    /// the work item's declared model_manifest hash (fail-closed), then
    /// ExtractElmResourceRequirements. The processor calls this BEFORE session
    /// creation and feeds the values to CheckElmResources.
    /// @param entryDir - pin.GetDir() (trailing-slash entry directory)
    /// @param declaredManifestHash - the work item's model_manifest_hash
    /// @return the two plain values, or an ElmRuntimeError failure
    outcome::result<ElmEntryPreflightValues> PreflightPinnedEntry(
        const std::string &entryDir, const std::string &declaredManifestHash );
} // namespace sgns::elmruntime

#endif // SGPROCMGR_ELMRUNTIME_ELM_ENTRY_PREFLIGHT_HPP
