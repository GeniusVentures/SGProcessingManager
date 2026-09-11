#ifndef SGPROCMGR_ELMRUNTIME_MANIFEST_HPP
#define SGPROCMGR_ELMRUNTIME_MANIFEST_HPP

#include <elmruntime/ElmRuntimeError.hpp>
#include <util/sha256.hpp>

#include <outcome/sgprocmgr-outcome.hpp>

// The manifest types come from the A1-fallback generator set
// (generated/elmruntime-manifest/) -- SELF-CONTAINED: its headers include
// their own "helper.hpp"/"ElmType.hpp" siblings via quoted includes, so this
// must be a quoted include resolved against that directory ONLY (a
// global-include <ElmModelManifest.hpp> would work, but the quoted include
// keeps the subdir set from mixing with the root generated/ set in any TU
// that includes both -- their helper.hpp files redefine the same classes).
#include "elmruntime-manifest/ElmModelManifest.hpp"

#include <cstdint>
#include <string>
#include <vector>

namespace sgns::elmruntime
{
    /// @brief Manifest byte ceiling (DoS bound): manifests above 1 MiB reject with
    ///        MANIFEST_INVALID before any parsing (RESEARCH security table).
    inline constexpr size_t kMaxManifestBytes = 1024 * 1024;

    /// @brief Artifact count ceiling (DoS bound): manifests declaring more than 32
    ///        artifacts reject with MANIFEST_INVALID at the gate (RESEARCH security table).
    inline constexpr size_t kMaxArtifacts = 32;

    /// @brief Compute the 64-lowercase-hex sha256 digest of manifest bytes.
    ///
    /// This digest IS the cache-entry directory name (P2-4). The declared
    /// model_manifest_hash string is only ever COMPARED (after normalization) --
    /// it never becomes a path component.
    /// @param bytes - raw manifest bytes
    /// @return 64 lowercase hexadecimal characters
    std::string ComputeManifestHexDigest( const std::vector<uint8_t> &bytes );

    /// @brief Hash-verify then parse then semantically gate manifest bytes (SC-1 front door).
    ///
    /// Gate order (fail-closed at every step, no bypass):
    ///   a. bytes.size() <= kMaxManifestBytes
    ///   b. declaredHash normalization: optional "sha256:" prefix stripped, then exactly
    ///      64 hex chars (case-insensitive)
    ///   c. sha256(bytes) vs declaredHash -- MISMATCH REJECTS BEFORE PARSE: untrusted
    ///      bytes never reach the JSON parser until pinned by the hash
    ///   d. JSON parse via the generated from_json
    ///   e. semantic gates: model_format == "mnn"; non-empty artifacts; <= kMaxArtifacts;
    ///      unique roles; every role in the closed set; the four required roles
    ///      {llm_config, llm_model, llm_weight, tokenizer_file} present (context_file
    ///      optional); non-empty uri per artifact; size_bytes >= 0
    /// @param bytes - raw manifest bytes (fetched through the FetchFn seam)
    /// @param declaredHash - the work item's model_manifest_hash ("sha256:<hex64>" or bare hex64)
    /// @return the typed manifest, or an ElmRuntimeError failure
    outcome::result<sgns::ElmModelManifest> ParseAndVerifyManifest( const std::vector<uint8_t> &bytes,
                                                                   const std::string         &declaredHash );

    /// @brief Map a manifest artifact role onto MNN's default bundle filename.
    ///
    /// Roles materialize at these fixed names so a cache entry directory doubles as
    /// LlmConfig.base_dir (the manifest name is a ROLE, never a filesystem path).
    /// Unknown roles return nullptr (the parse gate makes this unreachable for parsed
    /// manifests; callers treating nullptr as a bug is correct).
    /// @param role - one of llm_config, llm_model, llm_weight, tokenizer_file, context_file
    /// @return the fixed filename, or nullptr for an unknown role
    const char *RoleFileName( const std::string &role );

    /// @brief Sum of all artifact size_bytes (drives D-04 LRU accounting and 02-02's disk preflight).
    /// @param manifest - a gate-passed manifest
    /// @return total declared bytes
    uint64_t TotalArtifactBytes( const sgns::ElmModelManifest &manifest );
} // namespace sgns::elmruntime

#endif // SGPROCMGR_ELMRUNTIME_MANIFEST_HPP
