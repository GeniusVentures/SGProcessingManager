/**
 * Artifact type system for Phase 08: Structured Artifacts & Execution Manifests.
 *
 * Defines the TerminalState enum and Artifact struct — the data contracts
 * for typed output records with content-hash-based identity, resource metadata,
 * and per-chunk SHA-256 hashes. No protobuf — plain C++ structs per D-04/D-05.
 *
 * @brief Artifact and terminal state data types
 */
#ifndef SGPROCMGR_ARTIFACT_TYPES_HPP
#define SGPROCMGR_ARTIFACT_TYPES_HPP

#include <cstdint>
#include <cstring>
#include "util/sha256.hpp"

namespace sgns::sgprocessing
{

    // Fixed-size constants shared between Artifact and ExecutionManifest.
    static constexpr size_t SHA256_HASH_SIZE  = 32;   ///< SHA-256 digest size in bytes
    static constexpr size_t MAX_RESOURCE_NAME = 256;  ///< Max bytes for resource/pass/binding name strings (D-06)
    static constexpr size_t MAX_MEDIA_TYPE    = 128;  ///< Max bytes for media type string (D-06)

    /// Terminal execution outcome for the manifest (D-15).
    /// Explicit uint8_t underlying type — maps directly to a single byte
    /// in the binary serialized manifest layout.
    enum class TerminalState : uint8_t
    {
        Success        = 0,  ///< Normal completion — all outputs valid
        Cancelled      = 1,  ///< Cancellation token triggered (07 D-01/D-05)
        Timeout        = 2,  ///< Deadline expired (07 D-02/D-09)
        BudgetExceeded = 3,  ///< Output byte budget exceeded (07 D-08/EXEC-03)
        Error          = 4   ///< All other failures (D-15: no error string in manifest)
    };

    /// Typed output artifact record (ARTF-01, ARTF-02, ARTF-03).
    ///
    /// Every artifact carries: identity (resource name + content-hash-based ID),
    /// format metadata (data type, format, dimensions, byte size, media type),
    /// and per-chunk SHA-256 hashes from the producing processor.
    ///
    /// Content-hash-based identity (D-01): artifactId = SHA-256 of raw artifact bytes.
    /// Same bytes → same ID across any execution — enables deduplication and caching.
    struct Artifact
    {
        // ── Artifact identity (ARTF-01) ────────────────────────────────

        char    resourceName[MAX_RESOURCE_NAME];    ///< Human-readable output name, null-terminated
        uint8_t artifactId[SHA256_HASH_SIZE];       ///< SHA-256 of raw artifact bytes (D-01: content-addressable ID)
        char    passId[MAX_RESOURCE_NAME];          ///< Producing pass identity from schema
        char    outputBinding[MAX_RESOURCE_NAME];   ///< Output binding reference (e.g. "output:render_target")

        // ── Artifact format metadata (ARTF-02) ─────────────────────────

        char     dataType[64];   ///< DataType string (e.g. "TEXTURE2_D", "TENSOR", "STRING")
        char     format[64];     ///< InputFormat string (e.g. "FLOAT32", "RGBA8", "INT8")
        uint32_t width  = 0;     ///< 0 if not applicable (e.g. string output)
        uint32_t height = 0;     ///< 0 if not applicable
        uint32_t depth  = 0;     ///< 0 if not applicable (1 for 2D textures by convention)
        uint64_t byteSize = 0;   ///< Size of raw artifact bytes in bytes
        char     mediaType[MAX_MEDIA_TYPE];  ///< e.g. "application/octet-stream", "image/png"

        // ── Artifact hashes (ARTF-03, D-07, D-08, D-09) ───────────────

        uint8_t  contentHash[SHA256_HASH_SIZE];              ///< SHA-256 of raw artifact bytes only (D-03/D-07)
        uint32_t chunkHashCount = 0;                          ///< How many chunk hashes the job produced (D-09); 0 if no chunking
        uint8_t  chunkHashes[1024][SHA256_HASH_SIZE] = {};   ///< Per-chunk SHA-256 hashes from processor output (D-08); max 1024 (D-06)
    };

    // ── Free-standing helpers (namespace scope, inline — avoid header bloat) ──

    /// Compute artifact identity from raw bytes (D-01, D-03).
    /// Delegates to sgprocmanagersha::sha256 for the actual hash.
    /// Fills both contentHash and artifactId in-place.
    inline void ComputeArtifactIdentity( Artifact &artifact, const uint8_t *rawBytes, size_t byteCount )
    {
        auto hash = sgns::sgprocmanagersha::sha256( rawBytes, byteCount );
        std::memcpy( artifact.contentHash, hash.data(), SHA256_HASH_SIZE );
        std::memcpy( artifact.artifactId, hash.data(), SHA256_HASH_SIZE );
    }

    /// Add a chunk hash to the artifact's chunk hash list (D-08).
    /// @return true on success, false if chunkHashCount >= 1024 (overflow guard).
    inline bool AddChunkHash( Artifact &artifact, const uint8_t hash[SHA256_HASH_SIZE] )
    {
        if ( artifact.chunkHashCount >= 1024 )
        {
            return false;
        }
        std::memcpy( artifact.chunkHashes[artifact.chunkHashCount], hash, SHA256_HASH_SIZE );
        ++artifact.chunkHashCount;
        return true;
    }

}  // namespace sgns::sgprocessing

#endif  // SGPROCMGR_ARTIFACT_TYPES_HPP
