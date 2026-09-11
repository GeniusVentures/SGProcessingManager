#ifndef SGPROCMGR_ELMRUNTIME_RESOURCE_PREFLIGHT_HPP
#define SGPROCMGR_ELMRUNTIME_RESOURCE_PREFLIGHT_HPP

// ELM resource-preflight value extraction (plan 02-02, elmbridge).
//
// CapabilityValidator::CheckElmResources takes PLAIN VALUES (required memory
// bytes + total artifact bytes) so the SGCapability library stays independent
// of the elmruntime generated manifest types: the root generated/ quicktype
// set (PassType.hpp et al., included by capability headers) and the
// generated/elmruntime-manifest/ fallback set BOTH define
// sgns::ClassMemberConstraints and sgns::ElmType in separate files with
// per-file #pragma once guards -- including both sets in one translation unit
// is a class redefinition error. The validator must never see a manifest.
//
// This header bridges the two worlds at the CALL SITE (Phase 3 processor):
// it lives in the elmruntime include set (fallback generated types only) and
// extracts the two plain values the validator needs.

#include "elmruntime-manifest/ElmModelManifest.hpp"

#include <cstdint>

namespace sgns::elmruntime
{
    /// Plain-value resource requirements extracted from a manifest, shaped for
    /// CapabilityValidator::CheckElmResources(uint64_t, uint64_t, callback).
    struct ElmResourceRequirements
    {
        /// runtime.required_memory_bytes; 0 when the runtime block is absent
        /// or the field is unset (no requirement -- the memory leg never fires).
        uint64_t requiredMemoryBytes = 0;

        /// Sum of artifact size_bytes (the bytes the acquire will store).
        uint64_t totalArtifactBytes = 0;
    };

    /// @brief Extract the local resource-preflight values from a manifest.
    ///
    /// Mirrors the runtime-block conventions of ElmManifest.cpp gates:
    ///  - get_runtime()/get_required_memory_bytes() return boost::optional BY
    ///    VALUE -- both are materialized into named locals before use (the
    ///    Phase 1 UB lesson; dereferencing the call result directly is UB).
    ///  - required_memory_bytes < 0 clamps to 0 (the schema minimum does not
    ///    survive codegen; negative values would wrap to ~2^64 as uint64_t and
    ///    become an accidental impossible requirement).
    ///  - Artifact bytes sum inline with the same arithmetic as
    ///    TotalArtifactBytes() -- a 3-line duplication kept deliberately so
    ///    this header stays independent of the ElmManifest.cpp TU (and its
    ///    sgprocmanagertypes linkage) for callers that only need the values.
    ///    Prefer TotalArtifactBytes(manifest) when already linking
    ///    sgprocmanagerelmruntime.
    /// @param manifest — a gate-passed manifest (ParseAndVerifyManifest output)
    /// @return the two plain values for the validator's preflight
    inline ElmResourceRequirements ExtractElmResourceRequirements(
        const sgns::ElmModelManifest &manifest )
    {
        ElmResourceRequirements reqs;

        // Materialize the by-value optionals into named locals (UB rule).
        const auto runtimeOpt          = manifest.get_runtime();
        const auto requiredMemOpt      = runtimeOpt
                                            ? runtimeOpt->get_required_memory_bytes()
                                            : boost::optional<int64_t>{};
        const int64_t requiredMemSigned = requiredMemOpt.value_or( 0 );

        if ( requiredMemSigned > 0 )
        {
            reqs.requiredMemoryBytes = static_cast<uint64_t>( requiredMemSigned );
        }

        for ( const auto &artifact : manifest.get_artifacts() )
        {
            const int64_t size = artifact.get_size_bytes();
            if ( size > 0 )
            {
                reqs.totalArtifactBytes += static_cast<uint64_t>( size );
            }
        }

        return reqs;
    }
} // namespace sgns::elmruntime

#endif // SGPROCMGR_ELMRUNTIME_RESOURCE_PREFLIGHT_HPP
