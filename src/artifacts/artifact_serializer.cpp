/**
 * Implementation of deterministic binary serialization for Artifact and
 * ExecutionManifest structs. Fixed-field, little-endian layout per D-05.
 *
 * Follows the same memcpy-based approach as SerializeRenderPassConfig() in
 * ProcessingManager.cpp: pre-allocate zero-filled vector, then copy each
 * field at its documented fixed offset.
 */

#include "artifacts/artifact_serializer.hpp"
#include <algorithm>
#include <cstring>

namespace sgns::sgprocessing
{

    // ────────────────────────────────────────────────────────────────────
    //  Artifact Serialization
    // ────────────────────────────────────────────────────────────────────

    std::vector<uint8_t> SerializeArtifact( const Artifact &artifact )
    {
        // Fixed-field layout offsets for Artifact (D-05).
        // All fields at known offsets; multi-byte values in native little-endian.
        static constexpr size_t OFF_resourceName   = 0;
        static constexpr size_t OFF_artifactId      = 256;
        static constexpr size_t OFF_passId          = 288;
        static constexpr size_t OFF_outputBinding   = 544;
        static constexpr size_t OFF_dataType        = 800;
        static constexpr size_t OFF_format          = 864;
        static constexpr size_t OFF_width           = 928;
        static constexpr size_t OFF_height          = 932;
        static constexpr size_t OFF_depth           = 936;
        static constexpr size_t OFF_byteSize        = 940;
        static constexpr size_t OFF_mediaType       = 948;
        static constexpr size_t OFF_contentHash     = 1076;
        static constexpr size_t OFF_chunkHashCount  = 1108;
        static constexpr size_t OFF_chunkHashes     = 1112;

        // Pre-allocate zero-filled buffer at fixed total size
        std::vector<uint8_t> out( ARTIFACT_SERIALIZED_SIZE, 0 );

        // --- String fields: null-padded copy up to MAX-1 ---
        auto copyStr = [&]( size_t offset, const char *src, size_t maxLen )
        {
            size_t len = std::min( std::strlen( src ), maxLen - 1 );
            std::memcpy( out.data() + offset, src, len );
            // Rest is already zero from pre-allocation
        };

        copyStr( OFF_resourceName, artifact.resourceName, MAX_RESOURCE_NAME );
        copyStr( OFF_passId, artifact.passId, MAX_RESOURCE_NAME );
        copyStr( OFF_outputBinding, artifact.outputBinding, MAX_RESOURCE_NAME );
        copyStr( OFF_dataType, artifact.dataType, 64 );
        copyStr( OFF_format, artifact.format, 64 );
        copyStr( OFF_mediaType, artifact.mediaType, MAX_MEDIA_TYPE );

        // --- Fixed-size byte arrays ---
        std::memcpy( out.data() + OFF_artifactId, artifact.artifactId, SHA256_HASH_SIZE );
        std::memcpy( out.data() + OFF_contentHash, artifact.contentHash, SHA256_HASH_SIZE );

        // --- Integer fields (native little-endian, no byte swap needed on x86/ARM) ---
        std::memcpy( out.data() + OFF_width, &artifact.width, sizeof( uint32_t ) );
        std::memcpy( out.data() + OFF_height, &artifact.height, sizeof( uint32_t ) );
        std::memcpy( out.data() + OFF_depth, &artifact.depth, sizeof( uint32_t ) );
        std::memcpy( out.data() + OFF_byteSize, &artifact.byteSize, sizeof( uint64_t ) );
        std::memcpy( out.data() + OFF_chunkHashCount, &artifact.chunkHashCount, sizeof( uint32_t ) );

        // --- Chunk hashes: only chunkHashCount * 32 meaningful bytes ---
        std::memcpy( out.data() + OFF_chunkHashes, artifact.chunkHashes,
                     artifact.chunkHashCount * SHA256_HASH_SIZE );
        // Rest stays zero from pre-allocation

        return out;
    }

    bool DeserializeArtifact( const std::vector<uint8_t> &bytes, Artifact &out )
    {
        if ( bytes.size() != ARTIFACT_SERIALIZED_SIZE )
        {
            return false;
        }

        static constexpr size_t OFF_resourceName   = 0;
        static constexpr size_t OFF_artifactId      = 256;
        static constexpr size_t OFF_passId          = 288;
        static constexpr size_t OFF_outputBinding   = 544;
        static constexpr size_t OFF_dataType        = 800;
        static constexpr size_t OFF_format          = 864;
        static constexpr size_t OFF_width           = 928;
        static constexpr size_t OFF_height          = 932;
        static constexpr size_t OFF_depth           = 936;
        static constexpr size_t OFF_byteSize        = 940;
        static constexpr size_t OFF_mediaType       = 948;
        static constexpr size_t OFF_contentHash     = 1076;
        static constexpr size_t OFF_chunkHashCount  = 1108;
        static constexpr size_t OFF_chunkHashes     = 1112;

        // Zero the output struct
        out = Artifact{};

        // Copy string fields
        std::memcpy( out.resourceName, bytes.data() + OFF_resourceName, MAX_RESOURCE_NAME );
        out.resourceName[MAX_RESOURCE_NAME - 1] = '\0';  // force null terminator
        std::memcpy( out.passId, bytes.data() + OFF_passId, MAX_RESOURCE_NAME );
        out.passId[MAX_RESOURCE_NAME - 1] = '\0';
        std::memcpy( out.outputBinding, bytes.data() + OFF_outputBinding, MAX_RESOURCE_NAME );
        out.outputBinding[MAX_RESOURCE_NAME - 1] = '\0';
        std::memcpy( out.dataType, bytes.data() + OFF_dataType, 64 );
        out.dataType[63] = '\0';
        std::memcpy( out.format, bytes.data() + OFF_format, 64 );
        out.format[63] = '\0';
        std::memcpy( out.mediaType, bytes.data() + OFF_mediaType, MAX_MEDIA_TYPE );
        out.mediaType[MAX_MEDIA_TYPE - 1] = '\0';

        // Copy fixed-size byte arrays
        std::memcpy( out.artifactId, bytes.data() + OFF_artifactId, SHA256_HASH_SIZE );
        std::memcpy( out.contentHash, bytes.data() + OFF_contentHash, SHA256_HASH_SIZE );

        // Copy integers
        std::memcpy( &out.width, bytes.data() + OFF_width, sizeof( uint32_t ) );
        std::memcpy( &out.height, bytes.data() + OFF_height, sizeof( uint32_t ) );
        std::memcpy( &out.depth, bytes.data() + OFF_depth, sizeof( uint32_t ) );
        std::memcpy( &out.byteSize, bytes.data() + OFF_byteSize, sizeof( uint64_t ) );
        std::memcpy( &out.chunkHashCount, bytes.data() + OFF_chunkHashCount, sizeof( uint32_t ) );

        // Clamp chunk hash count
        if ( out.chunkHashCount > 1024 )
        {
            out.chunkHashCount = 1024;
        }

        // Copy chunk hashes
        std::memcpy( out.chunkHashes, bytes.data() + OFF_chunkHashes,
                     out.chunkHashCount * SHA256_HASH_SIZE );

        return true;
    }

    // ────────────────────────────────────────────────────────────────────
    //  ExecutionManifest Serialization
    // ────────────────────────────────────────────────────────────────────

    std::vector<uint8_t> SerializeManifest( const ExecutionManifest &manifest )
    {
        // Fixed-field layout offsets for ExecutionManifest (D-05).
        static constexpr size_t OFF_executionId          = 0;
        static constexpr size_t OFF_attemptId            = 256;
        static constexpr size_t OFF_taskId               = 512;
        static constexpr size_t OFF_subtaskId            = 768;
        static constexpr size_t OFF_passId               = 1024;
        static constexpr size_t OFF_executorIdentity     = 1280;
        static constexpr size_t OFF_modelIdentity        = 1312;
        static constexpr size_t OFF_tokenizerIdentity    = 1344;
        static constexpr size_t OFF_adapterIdentity      = 1376;
        static constexpr size_t OFF_shaderIdentity       = 1408;
        static constexpr size_t OFF_quantizationIdentity = 1440;
        static constexpr size_t OFF_inputArtifactCount   = 1472;
        static constexpr size_t OFF_inputArtifactHashes  = 1476;
        static constexpr size_t OFF_outputArtifactCount  = 3524;
        static constexpr size_t OFF_outputArtifactHashes = 3528;
        static constexpr size_t OFF_startTimeUsec        = 5576;
        static constexpr size_t OFF_endTimeUsec          = 5584;
        static constexpr size_t OFF_terminalState        = 5592;
        static constexpr size_t OFF_gpuMemoryUsedBytes   = 5593;
        static constexpr size_t OFF_outputBytesProduced  = 5601;
        static constexpr size_t OFF_wallClockUsec        = 5609;
        static constexpr size_t OFF_manifestHash         = 5617;

        // Pre-allocate zero-filled buffer
        std::vector<uint8_t> out( MANIFEST_SERIALIZED_SIZE, 0 );

        // --- CRITICAL: Save and zero manifestHash before serialization (D-04) ---
        uint8_t savedManifestHash[SHA256_HASH_SIZE];
        std::memcpy( savedManifestHash, manifest.manifestHash, SHA256_HASH_SIZE );
        // We need to zero the manifestHash in a mutable copy — cast away const
        // because SerializeManifest logically must mutate the hash field for correctness.
        std::memset( const_cast<uint8_t( & )[SHA256_HASH_SIZE]>( manifest.manifestHash ), 0, SHA256_HASH_SIZE );

        auto copyStr = [&]( size_t offset, const char *src, size_t maxLen )
        {
            size_t len = std::min( std::strlen( src ), maxLen - 1 );
            std::memcpy( out.data() + offset, src, len );
        };

        // --- Identifier strings ---
        copyStr( OFF_executionId, manifest.executionId, MAX_IDENTIFIER );
        copyStr( OFF_attemptId, manifest.attemptId, MAX_IDENTIFIER );
        copyStr( OFF_taskId, manifest.taskId, MAX_IDENTIFIER );
        copyStr( OFF_subtaskId, manifest.subtaskId, MAX_IDENTIFIER );
        copyStr( OFF_passId, manifest.passId, MAX_RESOURCE_NAME );

        // --- Identity hashes ---
        std::memcpy( out.data() + OFF_executorIdentity, manifest.executorIdentity, SHA256_HASH_SIZE );
        std::memcpy( out.data() + OFF_modelIdentity, manifest.modelIdentity, SHA256_HASH_SIZE );
        std::memcpy( out.data() + OFF_tokenizerIdentity, manifest.tokenizerIdentity, SHA256_HASH_SIZE );
        std::memcpy( out.data() + OFF_adapterIdentity, manifest.adapterIdentity, SHA256_HASH_SIZE );
        std::memcpy( out.data() + OFF_shaderIdentity, manifest.shaderIdentity, SHA256_HASH_SIZE );
        std::memcpy( out.data() + OFF_quantizationIdentity, manifest.quantizationIdentity, SHA256_HASH_SIZE );

        // --- Input artifact refs ---
        std::memcpy( out.data() + OFF_inputArtifactCount, &manifest.inputArtifactCount, sizeof( uint32_t ) );
        uint32_t inCount = std::min( manifest.inputArtifactCount, static_cast<uint32_t>( MAX_ARTIFACT_REFS ) );
        std::memcpy( out.data() + OFF_inputArtifactHashes, manifest.inputArtifactHashes,
                     inCount * SHA256_HASH_SIZE );

        // --- Output artifact refs ---
        std::memcpy( out.data() + OFF_outputArtifactCount, &manifest.outputArtifactCount, sizeof( uint32_t ) );
        uint32_t outCount = std::min( manifest.outputArtifactCount, static_cast<uint32_t>( MAX_ARTIFACT_REFS ) );
        std::memcpy( out.data() + OFF_outputArtifactHashes, manifest.outputArtifactHashes,
                     outCount * SHA256_HASH_SIZE );

        // --- Timing ---
        std::memcpy( out.data() + OFF_startTimeUsec, &manifest.startTimeUsec, sizeof( int64_t ) );
        std::memcpy( out.data() + OFF_endTimeUsec, &manifest.endTimeUsec, sizeof( int64_t ) );

        // --- Terminal state ---
        out[OFF_terminalState] = static_cast<uint8_t>( manifest.terminalState );

        // --- Resource summary ---
        std::memcpy( out.data() + OFF_gpuMemoryUsedBytes, &manifest.gpuMemoryUsedBytes, sizeof( uint64_t ) );
        std::memcpy( out.data() + OFF_outputBytesProduced, &manifest.outputBytesProduced, sizeof( uint64_t ) );
        std::memcpy( out.data() + OFF_wallClockUsec, &manifest.wallClockUsec, sizeof( uint64_t ) );

        // manifestHash field stays zeroed (we're serializing with hash excluded)

        // --- Restore manifestHash ---
        std::memcpy( const_cast<uint8_t( & )[SHA256_HASH_SIZE]>( manifest.manifestHash ), savedManifestHash, SHA256_HASH_SIZE );

        return out;
    }

    bool DeserializeManifest( const std::vector<uint8_t> &bytes, ExecutionManifest &out )
    {
        if ( bytes.size() != MANIFEST_SERIALIZED_SIZE )
        {
            return false;
        }

        static constexpr size_t OFF_executionId          = 0;
        static constexpr size_t OFF_attemptId            = 256;
        static constexpr size_t OFF_taskId               = 512;
        static constexpr size_t OFF_subtaskId            = 768;
        static constexpr size_t OFF_passId               = 1024;
        static constexpr size_t OFF_executorIdentity     = 1280;
        static constexpr size_t OFF_modelIdentity        = 1312;
        static constexpr size_t OFF_tokenizerIdentity    = 1344;
        static constexpr size_t OFF_adapterIdentity      = 1376;
        static constexpr size_t OFF_shaderIdentity       = 1408;
        static constexpr size_t OFF_quantizationIdentity = 1440;
        static constexpr size_t OFF_inputArtifactCount   = 1472;
        static constexpr size_t OFF_inputArtifactHashes  = 1476;
        static constexpr size_t OFF_outputArtifactCount  = 3524;
        static constexpr size_t OFF_outputArtifactHashes = 3528;
        static constexpr size_t OFF_startTimeUsec        = 5576;
        static constexpr size_t OFF_endTimeUsec          = 5584;
        static constexpr size_t OFF_terminalState        = 5592;
        static constexpr size_t OFF_gpuMemoryUsedBytes   = 5593;
        static constexpr size_t OFF_outputBytesProduced  = 5601;
        static constexpr size_t OFF_wallClockUsec        = 5609;
        static constexpr size_t OFF_manifestHash         = 5617;

        out = ExecutionManifest{};

        // Identifier strings
        std::memcpy( out.executionId, bytes.data() + OFF_executionId, MAX_IDENTIFIER );
        out.executionId[MAX_IDENTIFIER - 1] = '\0';
        std::memcpy( out.attemptId, bytes.data() + OFF_attemptId, MAX_IDENTIFIER );
        out.attemptId[MAX_IDENTIFIER - 1] = '\0';
        std::memcpy( out.taskId, bytes.data() + OFF_taskId, MAX_IDENTIFIER );
        out.taskId[MAX_IDENTIFIER - 1] = '\0';
        std::memcpy( out.subtaskId, bytes.data() + OFF_subtaskId, MAX_IDENTIFIER );
        out.subtaskId[MAX_IDENTIFIER - 1] = '\0';
        std::memcpy( out.passId, bytes.data() + OFF_passId, MAX_RESOURCE_NAME );
        out.passId[MAX_RESOURCE_NAME - 1] = '\0';

        // Identity hashes
        std::memcpy( out.executorIdentity, bytes.data() + OFF_executorIdentity, SHA256_HASH_SIZE );
        std::memcpy( out.modelIdentity, bytes.data() + OFF_modelIdentity, SHA256_HASH_SIZE );
        std::memcpy( out.tokenizerIdentity, bytes.data() + OFF_tokenizerIdentity, SHA256_HASH_SIZE );
        std::memcpy( out.adapterIdentity, bytes.data() + OFF_adapterIdentity, SHA256_HASH_SIZE );
        std::memcpy( out.shaderIdentity, bytes.data() + OFF_shaderIdentity, SHA256_HASH_SIZE );
        std::memcpy( out.quantizationIdentity, bytes.data() + OFF_quantizationIdentity, SHA256_HASH_SIZE );

        // Input artifact refs
        std::memcpy( &out.inputArtifactCount, bytes.data() + OFF_inputArtifactCount, sizeof( uint32_t ) );
        if ( out.inputArtifactCount > MAX_ARTIFACT_REFS )
            out.inputArtifactCount = static_cast<uint32_t>( MAX_ARTIFACT_REFS );
        std::memcpy( out.inputArtifactHashes, bytes.data() + OFF_inputArtifactHashes,
                     out.inputArtifactCount * SHA256_HASH_SIZE );

        // Output artifact refs
        std::memcpy( &out.outputArtifactCount, bytes.data() + OFF_outputArtifactCount, sizeof( uint32_t ) );
        if ( out.outputArtifactCount > MAX_ARTIFACT_REFS )
            out.outputArtifactCount = static_cast<uint32_t>( MAX_ARTIFACT_REFS );
        std::memcpy( out.outputArtifactHashes, bytes.data() + OFF_outputArtifactHashes,
                     out.outputArtifactCount * SHA256_HASH_SIZE );

        // Timing
        std::memcpy( &out.startTimeUsec, bytes.data() + OFF_startTimeUsec, sizeof( int64_t ) );
        std::memcpy( &out.endTimeUsec, bytes.data() + OFF_endTimeUsec, sizeof( int64_t ) );

        // Terminal state
        out.terminalState = static_cast<TerminalState>( bytes[OFF_terminalState] );

        // Resource summary
        std::memcpy( &out.gpuMemoryUsedBytes, bytes.data() + OFF_gpuMemoryUsedBytes, sizeof( uint64_t ) );
        std::memcpy( &out.outputBytesProduced, bytes.data() + OFF_outputBytesProduced, sizeof( uint64_t ) );
        std::memcpy( &out.wallClockUsec, bytes.data() + OFF_wallClockUsec, sizeof( uint64_t ) );

        // Manifest hash
        std::memcpy( out.manifestHash, bytes.data() + OFF_manifestHash, SHA256_HASH_SIZE );

        return true;
    }

}  // namespace sgns::sgprocessing
