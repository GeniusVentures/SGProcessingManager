/**
 * Deterministic binary serialization for Artifact and ExecutionManifest structs.
 *
 * Fixed-field, little-endian, fixed-offset binary layout (D-04, D-05, D-06).
 * Every multi-byte value at a known, fixed offset; all integers in native
 * little-endian byte order; variable-length data (strings, arrays) capped at
 * maximum sizes with inline storage.
 *
 * Byte-identical output across two runs with identical inputs (ARTF-05).
 *
 * @brief Artifact and manifest binary serialization
 */
#ifndef SGPROCMGR_ARTIFACT_SERIALIZER_HPP
#define SGPROCMGR_ARTIFACT_SERIALIZER_HPP

#include <cstdint>
#include <vector>
#include "artifacts/artifact_types.hpp"
#include "artifacts/execution_manifest.hpp"

namespace sgns::sgprocessing
{

    /// Fixed total size of a serialized Artifact in bytes.
    ///   resourceName[256] + artifactId[32] + passId[256] + outputBinding[256]
    /// + dataType[64] + format[64] + width[4] + height[4] + depth[4] + byteSize[8]
    /// + mediaType[128] + contentHash[32] + chunkHashCount[4] + chunkHashes[1024*32]
    static constexpr size_t ARTIFACT_SERIALIZED_SIZE = 33880;

    /// Fixed total size of a serialized ExecutionManifest in bytes.
    ///   5 * MAX_IDENTIFIER[256] + 6 * SHA256_HASH_SIZE[32] + inputArtifactCount[4]
    /// + inputArtifactHashes[64*32] + outputArtifactCount[4] + outputArtifactHashes[64*32]
    /// + startTimeUsec[8] + endTimeUsec[8] + terminalState[1] + gpuMemoryUsedBytes[8]
    /// + outputBytesProduced[8] + wallClockUsec[8] + manifestHash[32]
    static constexpr size_t MANIFEST_SERIALIZED_SIZE = 5649;

    /// Trailer appended after the unchanged MANIFEST_SERIALIZED_SIZE base region
    /// (ARTF-10, schema evolution): schemaVersion[4] + errorMessage[256].
    /// Expressed as an arithmetic expression (not a hardcoded literal) so it
    /// stays correct if MANIFEST_SERIALIZED_SIZE or MAX_IDENTIFIER ever change.
    static constexpr size_t MANIFEST_V2_SERIALIZED_SIZE = MANIFEST_SERIALIZED_SIZE + sizeof( uint32_t ) + MAX_IDENTIFIER;

    /// Serialize an Artifact to a fixed-size binary blob (ARTF-05).
    /// @return Vector of exactly ARTIFACT_SERIALIZED_SIZE bytes.
    std::vector<uint8_t> SerializeArtifact( const Artifact &artifact );

    /// Deserialize a binary blob back into an Artifact struct.
    /// @return true on success; false if input size != ARTIFACT_SERIALIZED_SIZE.
    bool DeserializeArtifact( const std::vector<uint8_t> &bytes, Artifact &out );

    /// Serialize an ExecutionManifest to a fixed-size binary blob (ARTF-05).
    ///
    /// CRITICAL (D-04): The manifestHash field is zeroed before serialization
    /// and restored afterward so it does NOT participate in its own hash computation.
    ///
    /// @return Vector of exactly MANIFEST_V2_SERIALIZED_SIZE bytes (unchanged base
    /// region + schemaVersion+errorMessage trailer, ARTF-10).
    std::vector<uint8_t> SerializeManifest( const ExecutionManifest &manifest );

    /// Deserialize a binary blob back into an ExecutionManifest struct.
    /// @return false if input size < MANIFEST_SERIALIZED_SIZE (base region);
    /// trailer fields (schemaVersion, errorMessage) are read only when present
    /// and within bounds, defaulting to absent/empty otherwise.
    bool DeserializeManifest( const std::vector<uint8_t> &bytes, ExecutionManifest &out );

    /// Compute the manifest self-hash: SHA-256 of serialized manifest bytes
    /// with the manifestHash field zeroed (handled internally by SerializeManifest).
    /// @return 32-byte SHA-256 hash.
    inline std::vector<uint8_t> ComputeManifestHash( const ExecutionManifest &manifest )
    {
        auto bytes = SerializeManifest( manifest );
        return sgns::sgprocmanagersha::sha256( bytes.data(), bytes.size() );
    }

}  // namespace sgns::sgprocessing

#endif  // SGPROCMGR_ARTIFACT_SERIALIZER_HPP
