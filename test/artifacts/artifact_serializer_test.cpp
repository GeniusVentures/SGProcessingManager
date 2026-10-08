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
            ASSERT_EQ( bytes.size(), MANIFEST_V2_SERIALIZED_SIZE );

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
            ASSERT_EQ( bytes.size(), MANIFEST_V2_SERIALIZED_SIZE );

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

        // ────────────────────────────────────────────────────────────────
        //  ARTF-09 / ARTF-10: errorMessage round-trip + schema evolution
        // ────────────────────────────────────────────────────────────────

        TEST( ManifestErrorMessage, RoundTripsAndTruncates )
        {
            // Short message round-trips exactly.
            auto m = MakeTestManifest();
            std::strncpy( m.errorMessage, "GPU device lost: VK_ERROR_DEVICE_LOST", MAX_IDENTIFIER - 1 );
            m.errorMessage[MAX_IDENTIFIER - 1] = '\0';

            auto bytes = SerializeManifest( m );
            ExecutionManifest restored{};
            ASSERT_TRUE( DeserializeManifest( bytes, restored ) );
            EXPECT_STREQ( restored.errorMessage, "GPU device lost: VK_ERROR_DEVICE_LOST" );

            // A 300-character message is truncated (silently, no marker) at
            // MAX_IDENTIFIER - 1 bytes before serialization, matching D-10's
            // convention -- this mirrors exactly what Plan 16-03's
            // ProcessingManager.cpp change will do when copying an unbounded
            // std::string into this fixed array.
            std::string longMessage( 300, 'x' );
            auto        m2 = MakeTestManifest();
            std::strncpy( m2.errorMessage, longMessage.c_str(), MAX_IDENTIFIER - 1 );
            m2.errorMessage[MAX_IDENTIFIER - 1] = '\0';

            auto bytes2 = SerializeManifest( m2 );
            ExecutionManifest restored2{};
            ASSERT_TRUE( DeserializeManifest( bytes2, restored2 ) );
            EXPECT_EQ( std::strlen( restored2.errorMessage ), static_cast<size_t>( MAX_IDENTIFIER - 1 ) );
            // No truncation marker anywhere in the string -- every character is 'x'.
            for ( size_t i = 0; i < std::strlen( restored2.errorMessage ); ++i )
            {
                ASSERT_EQ( restored2.errorMessage[i], 'x' );
            }
        }

        TEST( ManifestSchemaEvolution, OldWriterBytesNewReader )
        {
            // Direction 2 (SC4): old-shape bytes (no trailer) read by the new
            // reader. Simulates a genuine pre-ARTF-10, 5649-byte blob by
            // truncating a real serialized manifest down to the base region.
            auto m     = MakeTestManifest();
            auto bytes = SerializeManifest( m );
            bytes.resize( MANIFEST_SERIALIZED_SIZE );

            ExecutionManifest restored{};
            ASSERT_TRUE( DeserializeManifest( bytes, restored ) );

            EXPECT_STREQ( restored.errorMessage, "" );
            // Spot-check base fields still match.
            EXPECT_STREQ( restored.executionId, "exec_001" );
            EXPECT_EQ( restored.terminalState, TerminalState::Success );
            EXPECT_EQ( restored.outputArtifactCount, 2u );
        }

        // Direction 1 (SC4) helper: duplicates ONLY the unchanged base-region
        // offset constants and memcpy/strncpy calls from DeserializeManifest,
        // using the relaxed `>=` size check instead of the pre-ARTF-10 `!=`
        // check, and never reads anything at or past offset
        // MANIFEST_SERIALIZED_SIZE (5649) -- no trailer access at all.
        //
        // What this proves: the relaxed-size-check-plus-never-read-past-what-
        // you-understand mechanism is sufficient for forward-tolerance -- a
        // reader that only knows about the base region can still correctly
        // read a new-writer's larger output.
        //
        // What this does NOT prove: that a binary literally compiled before
        // this phase existed would already contain this relaxation. No such
        // pre-Phase-16 persisted manifest binary exists anywhere in this
        // repository (RESEARCH.md Assumption A3, confirmed by a grep sweep of
        // all SerializeManifest/DeserializeManifest call sites) -- this is a
        // same-mechanism proxy, not a literal historical-binary test.
        static bool DeserializeManifestBaseFieldsOnly( const std::vector<uint8_t> &bytes, ExecutionManifest &out )
        {
            if ( bytes.size() < MANIFEST_SERIALIZED_SIZE )
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

            std::memcpy( out.executorIdentity, bytes.data() + OFF_executorIdentity, SHA256_HASH_SIZE );
            std::memcpy( out.modelIdentity, bytes.data() + OFF_modelIdentity, SHA256_HASH_SIZE );
            std::memcpy( out.tokenizerIdentity, bytes.data() + OFF_tokenizerIdentity, SHA256_HASH_SIZE );
            std::memcpy( out.adapterIdentity, bytes.data() + OFF_adapterIdentity, SHA256_HASH_SIZE );
            std::memcpy( out.shaderIdentity, bytes.data() + OFF_shaderIdentity, SHA256_HASH_SIZE );
            std::memcpy( out.quantizationIdentity, bytes.data() + OFF_quantizationIdentity, SHA256_HASH_SIZE );

            std::memcpy( &out.inputArtifactCount, bytes.data() + OFF_inputArtifactCount, sizeof( uint32_t ) );
            if ( out.inputArtifactCount > MAX_ARTIFACT_REFS )
                out.inputArtifactCount = static_cast<uint32_t>( MAX_ARTIFACT_REFS );
            std::memcpy( out.inputArtifactHashes, bytes.data() + OFF_inputArtifactHashes,
                         out.inputArtifactCount * SHA256_HASH_SIZE );

            std::memcpy( &out.outputArtifactCount, bytes.data() + OFF_outputArtifactCount, sizeof( uint32_t ) );
            if ( out.outputArtifactCount > MAX_ARTIFACT_REFS )
                out.outputArtifactCount = static_cast<uint32_t>( MAX_ARTIFACT_REFS );
            std::memcpy( out.outputArtifactHashes, bytes.data() + OFF_outputArtifactHashes,
                         out.outputArtifactCount * SHA256_HASH_SIZE );

            std::memcpy( &out.startTimeUsec, bytes.data() + OFF_startTimeUsec, sizeof( int64_t ) );
            std::memcpy( &out.endTimeUsec, bytes.data() + OFF_endTimeUsec, sizeof( int64_t ) );

            out.terminalState = static_cast<TerminalState>( bytes[OFF_terminalState] );

            std::memcpy( &out.gpuMemoryUsedBytes, bytes.data() + OFF_gpuMemoryUsedBytes, sizeof( uint64_t ) );
            std::memcpy( &out.outputBytesProduced, bytes.data() + OFF_outputBytesProduced, sizeof( uint64_t ) );
            std::memcpy( &out.wallClockUsec, bytes.data() + OFF_wallClockUsec, sizeof( uint64_t ) );

            std::memcpy( out.manifestHash, bytes.data() + OFF_manifestHash, SHA256_HASH_SIZE );

            // Deliberately no trailer access -- this helper never reads at or
            // past offset MANIFEST_SERIALIZED_SIZE (5649).
            return true;
        }

        TEST( ManifestSchemaEvolution, NewWriterBytesOldReaderProxy )
        {
            // Direction 1 (SC4): a genuine new-writer 5909-byte blob, still
            // correctly readable by a base-region-only proxy reader.
            auto m = MakeTestManifest();
            std::strncpy( m.errorMessage, "shouldn't be read by the proxy", MAX_IDENTIFIER - 1 );
            m.errorMessage[MAX_IDENTIFIER - 1] = '\0';

            auto bytes = SerializeManifest( m );
            ASSERT_EQ( bytes.size(), MANIFEST_V2_SERIALIZED_SIZE );

            ExecutionManifest restored{};
            ASSERT_TRUE( DeserializeManifestBaseFieldsOnly( bytes, restored ) );

            EXPECT_STREQ( restored.executionId, "exec_001" );
            EXPECT_STREQ( restored.taskId, "task_42" );
            EXPECT_EQ( restored.terminalState, TerminalState::Success );
            EXPECT_EQ( restored.outputArtifactCount, 2u );
        }

    }  // namespace
}  // namespace sgns::sgprocessing
