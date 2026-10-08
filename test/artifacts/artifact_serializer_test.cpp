/**
 * Unit tests for Artifact and ExecutionManifest deterministic binary serialization.
 *
 * Tests ARTF-05 (byte-identical output across runs), fixed-field layout,
 * little-endian encoding, sentinel zero hashes (D-14), and manifest
 * self-hash computation (D-04).
 */

#include <artifacts/artifact_types.hpp>
#include <artifacts/artifact_serializer.hpp>
#include <artifacts/execution_manifest.hpp>
#include <gtest/gtest.h>
#include <cstring>

namespace sgns::sgprocessing
{
    namespace
    {

        // ────────────────────────────────────────────────────────────────
        //  Artifact Serialization Tests
        // ────────────────────────────────────────────────────────────────

        /// Fill an Artifact with known test values.
        Artifact MakeTestArtifact()
        {
            Artifact art{};

            // Identity
            std::strncpy( art.resourceName, "output_render", MAX_RESOURCE_NAME - 1 );
            std::strncpy( art.passId, "pass_0", MAX_RESOURCE_NAME - 1 );
            std::strncpy( art.outputBinding, "output:render_target", MAX_RESOURCE_NAME - 1 );

            // Format metadata
            std::strncpy( art.dataType, "TEXTURE2_D", 63 );
            std::strncpy( art.format, "RGBA8", 63 );
            art.width    = 1920;
            art.height   = 1080;
            art.depth    = 1;
            art.byteSize = 8294400;
            std::strncpy( art.mediaType, "image/png", MAX_MEDIA_TYPE - 1 );

            // Compute content hash from a known byte pattern
            const uint8_t knownBytes[] = { 0xDE, 0xAD, 0xBE, 0xEF, 0x00, 0x01, 0x02, 0x03 };
            ComputeArtifactIdentity( art, knownBytes, sizeof( knownBytes ) );

            // Add 3 chunk hashes
            uint8_t chunk1[SHA256_HASH_SIZE] = {};
            chunk1[0] = 0xAA;
            uint8_t chunk2[SHA256_HASH_SIZE] = {};
            chunk2[0] = 0xBB;
            uint8_t chunk3[SHA256_HASH_SIZE] = {};
            chunk3[0] = 0xCC;
            AddChunkHash( art, chunk1 );
            AddChunkHash( art, chunk2 );
            AddChunkHash( art, chunk3 );

            return art;
        }

        TEST( ArtifactSerializeRoundTrip, AllFieldsMatch )
        {
            auto art = MakeTestArtifact();

            auto bytes = SerializeArtifact( art );
            ASSERT_EQ( bytes.size(), ARTIFACT_SERIALIZED_SIZE );

            Artifact restored{};
            ASSERT_TRUE( DeserializeArtifact( bytes, restored ) );

            EXPECT_STREQ( restored.resourceName, "output_render" );
            EXPECT_STREQ( restored.passId, "pass_0" );
            EXPECT_STREQ( restored.outputBinding, "output:render_target" );
            EXPECT_STREQ( restored.dataType, "TEXTURE2_D" );
            EXPECT_STREQ( restored.format, "RGBA8" );
            EXPECT_EQ( restored.width, 1920u );
            EXPECT_EQ( restored.height, 1080u );
            EXPECT_EQ( restored.depth, 1u );
            EXPECT_EQ( restored.byteSize, 8294400ull );
            EXPECT_STREQ( restored.mediaType, "image/png" );

            // Content hash and artifact ID should match
            EXPECT_EQ( std::memcmp( restored.contentHash, art.contentHash, SHA256_HASH_SIZE ), 0 );
            EXPECT_EQ( std::memcmp( restored.artifactId, art.artifactId, SHA256_HASH_SIZE ), 0 );

            // Chunk hashes
            EXPECT_EQ( restored.chunkHashCount, 3u );
            EXPECT_EQ( restored.chunkHashes[0][0], 0xAA );
            EXPECT_EQ( restored.chunkHashes[1][0], 0xBB );
            EXPECT_EQ( restored.chunkHashes[2][0], 0xCC );
        }

        TEST( ArtifactDeterminism, ByteIdenticalAcrossTwoRuns )
        {
            auto art = MakeTestArtifact();

            auto bytes1 = SerializeArtifact( art );
            auto bytes2 = SerializeArtifact( art );

            ASSERT_EQ( bytes1.size(), bytes2.size() );
            EXPECT_EQ( std::memcmp( bytes1.data(), bytes2.data(), bytes1.size() ), 0 );
        }

        TEST( ArtifactMaxChunks, SerializeWith1024Chunks )
        {
            Artifact art{};
            uint8_t   hash[SHA256_HASH_SIZE] = {};
            for ( int i = 0; i < 1024; ++i )
            {
                hash[0] = static_cast<uint8_t>( i & 0xFF );
                ASSERT_TRUE( AddChunkHash( art, hash ) );
            }
            // 1025th should overflow
            EXPECT_FALSE( AddChunkHash( art, hash ) );

            auto bytes = SerializeArtifact( art );
            ASSERT_EQ( bytes.size(), ARTIFACT_SERIALIZED_SIZE );

            Artifact restored{};
            ASSERT_TRUE( DeserializeArtifact( bytes, restored ) );
            EXPECT_EQ( restored.chunkHashCount, 1024u );
            EXPECT_EQ( restored.chunkHashes[0][0], 0x00 );
            EXPECT_EQ( restored.chunkHashes[1023][0], 0xFF );
        }

        TEST( ArtifactEmptyStrings, NullPaddedAtOffsets )
        {
            Artifact art{};
            // All strings default to empty (zero-initialized)
            auto bytes = SerializeArtifact( art );

            // Bytes at resourceName offset (0) should be zero
            EXPECT_EQ( bytes[0], 0 );
            // Byte at passId offset should also be zero
            EXPECT_EQ( bytes[256 + 32], 0 ); // after artifactId

            Artifact restored{};
            ASSERT_TRUE( DeserializeArtifact( bytes, restored ) );
            EXPECT_STREQ( restored.resourceName, "" );
        }

        TEST( ArtifactZeroChunkCount, ChunkRegionAllZeros )
        {
            Artifact art{};
            art.chunkHashCount = 0;

            auto bytes = SerializeArtifact( art );

            // Count field should be 0
            uint32_t count;
            std::memcpy( &count, bytes.data() + 1108, sizeof( uint32_t ) );
            EXPECT_EQ( count, 0u );

            // First byte of chunk region should be 0
            EXPECT_EQ( bytes[1112], 0 );
        }

        TEST( ArtifactLittleEndian, Uint32Encoding )
        {
            Artifact art{};
            art.width = 0x01020304;

            auto bytes = SerializeArtifact( art );
            // Offset 928 = width: byte 0 should be 0x04 (LE)
            EXPECT_EQ( bytes[928], 0x04 );
            EXPECT_EQ( bytes[929], 0x03 );
            EXPECT_EQ( bytes[930], 0x02 );
            EXPECT_EQ( bytes[931], 0x01 );
        }

        // ────────────────────────────────────────────────────────────────
        //  ExecutionManifest Serialization Tests
        // ────────────────────────────────────────────────────────────────

        ExecutionManifest MakeTestManifest()
        {
            ExecutionManifest m{};

            // Identifiers
            std::strncpy( m.executionId, "exec_001", MAX_IDENTIFIER - 1 );
            std::strncpy( m.attemptId, "attempt_1", MAX_IDENTIFIER - 1 );
            std::strncpy( m.taskId, "task_42", MAX_IDENTIFIER - 1 );
            std::strncpy( m.subtaskId, "subtask_7", MAX_IDENTIFIER - 1 );
            std::strncpy( m.passId, "pass_render_main", MAX_RESOURCE_NAME - 1 );

            // Executor identity — non-zero
            std::memset( m.executorIdentity, 0xAB, SHA256_HASH_SIZE );

            // Model identity — non-zero (model was used)
            std::memset( m.modelIdentity, 0xCD, SHA256_HASH_SIZE );

            // Shader identity — non-zero
            std::memset( m.shaderIdentity, 0xEF, SHA256_HASH_SIZE );

            // tokenizer/adapter/quantization — zero (not used, D-14 sentinel)

            // Output artifacts
            m.outputArtifactCount = 2;
            std::memset( m.outputArtifactHashes[0], 0x11, SHA256_HASH_SIZE );
            std::memset( m.outputArtifactHashes[1], 0x22, SHA256_HASH_SIZE );

            // Timing
            m.startTimeUsec = 1700000000000000LL;
            m.endTimeUsec   = 1700000000123456LL;
            m.wallClockUsec = m.endTimeUsec - m.startTimeUsec;

            // Terminal state
            m.terminalState = TerminalState::Success;

            // Resource summary
            m.outputBytesProduced = 16588800; // two 1920x1080 RGBA8 images

            return m;
        }

        TEST( ManifestSerializeRoundTrip, AllFieldsMatch )
        {
            auto m = MakeTestManifest();

            auto bytes = SerializeManifest( m );
            ASSERT_EQ( bytes.size(), MANIFEST_SERIALIZED_SIZE );

            ExecutionManifest restored{};
            ASSERT_TRUE( DeserializeManifest( bytes, restored ) );

            EXPECT_STREQ( restored.executionId, "exec_001" );
            EXPECT_STREQ( restored.attemptId, "attempt_1" );
            EXPECT_STREQ( restored.taskId, "task_42" );
            EXPECT_STREQ( restored.subtaskId, "subtask_7" );
            EXPECT_STREQ( restored.passId, "pass_render_main" );

            EXPECT_EQ( std::memcmp( restored.executorIdentity, m.executorIdentity, SHA256_HASH_SIZE ), 0 );
            EXPECT_EQ( std::memcmp( restored.modelIdentity, m.modelIdentity, SHA256_HASH_SIZE ), 0 );
            EXPECT_EQ( std::memcmp( restored.shaderIdentity, m.shaderIdentity, SHA256_HASH_SIZE ), 0 );

            EXPECT_EQ( restored.outputArtifactCount, 2u );
            EXPECT_EQ( restored.outputArtifactHashes[0][0], 0x11 );
            EXPECT_EQ( restored.outputArtifactHashes[1][0], 0x22 );

            EXPECT_EQ( restored.startTimeUsec, 1700000000000000LL );
            EXPECT_EQ( restored.endTimeUsec, 1700000000123456LL );
            EXPECT_EQ( restored.wallClockUsec, 123456LL );

            EXPECT_EQ( restored.terminalState, TerminalState::Success );
            EXPECT_EQ( restored.outputBytesProduced, 16588800ull );
        }

        TEST( ManifestDeterminism, ByteIdenticalAcrossTwoRuns )
        {
            auto m = MakeTestManifest();

            auto bytes1 = SerializeManifest( m );
            auto bytes2 = SerializeManifest( m );

            ASSERT_EQ( bytes1.size(), bytes2.size() );
            EXPECT_EQ( std::memcmp( bytes1.data(), bytes2.data(), bytes1.size() ), 0 );
        }

        TEST( ManifestZeroIdentityHashes, SentinelZerosForInapplicable )
        {
            ExecutionManifest m{};
            // All identity hashes default to zero

            auto bytes = SerializeManifest( m );
            ASSERT_EQ( bytes.size(), MANIFEST_SERIALIZED_SIZE );

            // tokenizerIdentity at offset 1344: first byte should be 0
            EXPECT_EQ( bytes[1344], 0 );
            // adapterIdentity at offset 1376: first byte should be 0
            EXPECT_EQ( bytes[1376], 0 );
            // quantizationIdentity at offset 1440: first byte should be 0
            EXPECT_EQ( bytes[1440], 0 );
        }

        TEST( ManifestHashDeterminism, SameInputSameHash )
        {
            auto m = MakeTestManifest();

            auto hash1 = ComputeManifestHash( m );
            auto hash2 = ComputeManifestHash( m );

            ASSERT_EQ( hash1.size(), SHA256_HASH_SIZE );
            ASSERT_EQ( hash2.size(), SHA256_HASH_SIZE );
            EXPECT_EQ( std::memcmp( hash1.data(), hash2.data(), SHA256_HASH_SIZE ), 0 );
        }

        TEST( ManifestHashSensitive, ChangedFieldProducesDifferentHash )
        {
            auto m1 = MakeTestManifest();
            auto m2 = MakeTestManifest();

            // Change one byte in passId
            m2.passId[0] = 'X';

            auto hash1 = ComputeManifestHash( m1 );
            auto hash2 = ComputeManifestHash( m2 );

            EXPECT_NE( std::memcmp( hash1.data(), hash2.data(), SHA256_HASH_SIZE ), 0 );
        }

        TEST( ManifestHashExcludedFromSerialization, ManifestHashFieldNotInHash )
        {
            auto m = MakeTestManifest();

            // Set manifestHash to a non-zero value before serialization
            std::memset( m.manifestHash, 0xFF, SHA256_HASH_SIZE );

            auto hash = ComputeManifestHash( m );

            // The hash should NOT be all 0xFF (which would mean manifestHash leaked in)
            bool allFF = true;
            for ( size_t i = 0; i < SHA256_HASH_SIZE; ++i )
            {
                if ( hash[i] != 0xFF )
                {
                    allFF = false;
                    break;
                }
            }
            EXPECT_FALSE( allFF );

            // Verify manifestHash was restored after serialization
            EXPECT_EQ( m.manifestHash[0], 0xFF );
        }

    }  // namespace
}  // namespace sgns::sgprocessing
