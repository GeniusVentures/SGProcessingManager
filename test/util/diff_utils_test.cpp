// Phase 15, Plan 15-01: unit tests for diff_utils.hpp/.cpp -- the extracted
// capture_diff diff primitives (ComputeFloat32Diff/ComputeUint8Diff) plus the
// new D-03/D-04 tolerance-derivation functions
// (ResolveChunkElementTypeHint/IsFloatChunkWithinTolerance/
// IsByteChunkWithinTolerance).
//
// Mirrors quantization_test.cpp's structure: pure in-memory unit tests, no
// file fixtures, a local Parameter-vector-building helper via
// set_name/set_type/set_parameter_default, and memcpy-based bit-pattern
// construction (never approximate float comparison for exact cases).

#include <gtest/gtest.h>

#include <cstdint>
#include <cstring>
#include <string>
#include <vector>

#include "util/diff_utils.hpp"

namespace sgns::sgprocmanagerdiff
{
    namespace
    {
        uint32_t BitsOf( float value )
        {
            uint32_t bits = 0;
            std::memcpy( &bits, &value, sizeof( bits ) );
            return bits;
        }

        float FloatFromBits( uint32_t bits )
        {
            float value = 0.0f;
            std::memcpy( &value, &bits, sizeof( value ) );
            return value;
        }

        // Packs a vector of floats into a raw little/native-endian byte
        // buffer, mirroring how ComputeFloat32Diff reads raw capture bytes.
        std::vector<uint8_t> BuildFloatBytes( const std::vector<float> &values )
        {
            std::vector<uint8_t> bytes( values.size() * sizeof( float ) );
            for ( size_t i = 0; i < values.size(); ++i )
            {
                std::memcpy( bytes.data() + i * sizeof( float ), &values[i], sizeof( float ) );
            }
            return bytes;
        }

        std::vector<uint8_t> BuildByteBytes( const std::vector<uint8_t> &values )
        {
            return values;
        }

        // Phase 14, Plan 14-01's parameters-building helper, mirrored exactly
        // (quantization_test.cpp).
        std::vector<sgns::Parameter> MakeParameters( const std::string    &name,
                                                      sgns::ParameterType   type,
                                                      const nlohmann::json &defaultValue )
        {
            sgns::Parameter param;
            param.set_name( name );
            param.set_type( type );
            param.set_parameter_default( defaultValue );
            return { param };
        }
    } // namespace

    class DiffUtilsTest : public ::testing::Test
    {
    };

    // --- ComputeFloat32Diff / ComputeUint8Diff extraction correctness ---

    TEST_F( DiffUtilsTest, ComputeFloat32DiffMatchesKnownDelta )
    {
        auto bytesA = BuildFloatBytes( { 1.0f, 2.0f } );
        auto bytesB = BuildFloatBytes( { 1.0f, 2.5f } );

        ElementDiffStats stats = ComputeFloat32Diff( bytesA, bytesB );

        ASSERT_FALSE( stats.sizeMismatch );
        ASSERT_EQ( stats.elementCount, 2u );
        ASSERT_DOUBLE_EQ( stats.maxAbsDelta, 0.5 );
    }

    TEST_F( DiffUtilsTest, ComputeFloat32DiffDetectsSizeMismatch )
    {
        auto bytesA = BuildFloatBytes( { 1.0f } );
        auto bytesB = BuildFloatBytes( { 1.0f, 2.0f } );

        ElementDiffStats stats = ComputeFloat32Diff( bytesA, bytesB );

        ASSERT_TRUE( stats.sizeMismatch );
    }

    TEST_F( DiffUtilsTest, ComputeUint8DiffMatchesKnownDelta )
    {
        auto bytesA = BuildByteBytes( { 10, 20 } );
        auto bytesB = BuildByteBytes( { 10, 25 } );

        ElementDiffStats stats = ComputeUint8Diff( bytesA, bytesB );

        ASSERT_FALSE( stats.sizeMismatch );
        ASSERT_EQ( stats.elementCount, 2u );
        ASSERT_DOUBLE_EQ( stats.maxAbsDelta, 5.0 );
    }

    TEST_F( DiffUtilsTest, ComputeUint8DiffDetectsSizeMismatch )
    {
        auto bytesA = BuildByteBytes( { 10 } );
        auto bytesB = BuildByteBytes( { 10, 20 } );

        ElementDiffStats stats = ComputeUint8Diff( bytesA, bytesB );

        ASSERT_TRUE( stats.sizeMismatch );
    }

    // --- ResolveChunkElementTypeHint ---

    TEST_F( DiffUtilsTest, ResolveChunkElementTypeHintDefaultsToFloatWhenNothingDeclared )
    {
        ASSERT_EQ( ResolveChunkElementTypeHint( nullptr ), ChunkElementType::FLOAT32 );
    }

    TEST_F( DiffUtilsTest, ResolveChunkElementTypeHintDefaultsToFloatWhenOnlyQuantScaleDeclared )
    {
        const auto parameters = MakeParameters( "quantScale", sgns::ParameterType::FLOAT, 32768.0 );
        ASSERT_EQ( ResolveChunkElementTypeHint( &parameters ), ChunkElementType::FLOAT32 );
    }

    TEST_F( DiffUtilsTest, ResolveChunkElementTypeHintReturnsUint8WhenByteQuantModeDeclared )
    {
        const auto parameters = MakeParameters( "byteQuantMode", sgns::ParameterType::INT, 3 );
        ASSERT_EQ( ResolveChunkElementTypeHint( &parameters ), ChunkElementType::UINT8 );
    }

    // --- IsFloatChunkWithinTolerance: D-03 grid-step bound (declared quantScale) ---

    TEST_F( DiffUtilsTest, IsFloatChunkWithinToleranceUsesGridStepBoundWhenDeclaredPasses )
    {
        const auto parameters = MakeParameters( "quantScale", sgns::ParameterType::FLOAT, 32768.0 );
        auto bytesA = BuildFloatBytes( { 0.0f } );
        auto bytesB = BuildFloatBytes( { 2.0f / 32768.0f } ); // exactly at the 2/S bound

        ElementDiffStats stats;
        ASSERT_TRUE( IsFloatChunkWithinTolerance( bytesA, bytesB, &parameters, stats ) );
        ASSERT_FALSE( stats.sizeMismatch );
    }

    TEST_F( DiffUtilsTest, IsFloatChunkWithinToleranceUsesGridStepBoundWhenDeclaredFails )
    {
        const auto parameters = MakeParameters( "quantScale", sgns::ParameterType::FLOAT, 32768.0 );
        auto bytesA = BuildFloatBytes( { 0.0f } );
        auto bytesB = BuildFloatBytes( { 3.0f / 32768.0f } ); // strictly more than the 2/S bound

        ElementDiffStats stats;
        ASSERT_FALSE( IsFloatChunkWithinTolerance( bytesA, bytesB, &parameters, stats ) );
    }

    // --- IsFloatChunkWithinTolerance: D-04 fallback (no valid quantScale declared) ---

    TEST_F( DiffUtilsTest, IsFloatChunkWithinToleranceFallsBackToRelativeThresholdWhenUndeclaredPasses )
    {
        // relDelta = 0.00005 / 1.00005 ~= 5e-5, within kDefaultFloatRelativeThreshold (1e-4).
        auto bytesA = BuildFloatBytes( { 1.0f } );
        auto bytesB = BuildFloatBytes( { 1.00005f } );

        ElementDiffStats stats;
        ASSERT_TRUE( IsFloatChunkWithinTolerance( bytesA, bytesB, nullptr, stats ) );
    }

    TEST_F( DiffUtilsTest, IsFloatChunkWithinToleranceFallsBackToRelativeThresholdWhenUndeclaredFails )
    {
        // relDelta ~= 0.0003/1.0003 ~= 3e-4, exceeds kDefaultFloatRelativeThreshold (1e-4).
        auto bytesA = BuildFloatBytes( { 1.0f } );
        auto bytesB = BuildFloatBytes( { 1.0003f } );

        ElementDiffStats stats;
        ASSERT_FALSE( IsFloatChunkWithinTolerance( bytesA, bytesB, nullptr, stats ) );
    }

    TEST_F( DiffUtilsTest, IsFloatChunkWithinToleranceDetectsSizeMismatch )
    {
        auto bytesA = BuildFloatBytes( { 1.0f } );
        auto bytesB = BuildFloatBytes( { 1.0f, 2.0f } );

        ElementDiffStats stats;
        ASSERT_FALSE( IsFloatChunkWithinTolerance( bytesA, bytesB, nullptr, stats ) );
        ASSERT_TRUE( stats.sizeMismatch );
    }

    // --- IsByteChunkWithinTolerance: D-03 mask-width bound (declared byteQuantMode) ---

    TEST_F( DiffUtilsTest, IsByteChunkWithinToleranceUsesMaskBoundWhenDeclaredPasses )
    {
        const auto parameters = MakeParameters( "byteQuantMode", sgns::ParameterType::INT, 3 );
        auto bytesA = BuildByteBytes( { 0 } );
        auto bytesB = BuildByteBytes( { 7 } ); // exactly (1<<3)-1

        ElementDiffStats stats;
        ASSERT_TRUE( IsByteChunkWithinTolerance( bytesA, bytesB, &parameters, stats ) );
    }

    TEST_F( DiffUtilsTest, IsByteChunkWithinToleranceUsesMaskBoundWhenDeclaredFails )
    {
        const auto parameters = MakeParameters( "byteQuantMode", sgns::ParameterType::INT, 3 );
        auto bytesA = BuildByteBytes( { 0 } );
        auto bytesB = BuildByteBytes( { 8 } ); // strictly more than (1<<3)-1

        ElementDiffStats stats;
        ASSERT_FALSE( IsByteChunkWithinTolerance( bytesA, bytesB, &parameters, stats ) );
    }

    // --- IsByteChunkWithinTolerance: D-04 fallback (no valid byteQuantMode declared) ---

    TEST_F( DiffUtilsTest, IsByteChunkWithinToleranceFallsBackToAbsoluteThresholdWhenUndeclaredPasses )
    {
        auto bytesA = BuildByteBytes( { 10 } );
        auto bytesB = BuildByteBytes( { 11 } ); // absDelta = 1 == kDefaultByteAbsoluteThreshold

        ElementDiffStats stats;
        ASSERT_TRUE( IsByteChunkWithinTolerance( bytesA, bytesB, nullptr, stats ) );
    }

    TEST_F( DiffUtilsTest, IsByteChunkWithinToleranceFallsBackToAbsoluteThresholdWhenUndeclaredFails )
    {
        auto bytesA = BuildByteBytes( { 10 } );
        auto bytesB = BuildByteBytes( { 12 } ); // absDelta = 2, exceeds kDefaultByteAbsoluteThreshold

        ElementDiffStats stats;
        ASSERT_FALSE( IsByteChunkWithinTolerance( bytesA, bytesB, nullptr, stats ) );
    }

    TEST_F( DiffUtilsTest, IsByteChunkWithinToleranceDetectsSizeMismatch )
    {
        auto bytesA = BuildByteBytes( { 10 } );
        auto bytesB = BuildByteBytes( { 10, 20 } );

        ElementDiffStats stats;
        ASSERT_FALSE( IsByteChunkWithinTolerance( bytesA, bytesB, nullptr, stats ) );
        ASSERT_TRUE( stats.sizeMismatch );
    }

    // --- IsByteChunkWithinToleranceForMode: wrapper delegates to
    // IsByteChunkWithinTolerance unmodified (Phase 17-09, RENDTOL-02 gap
    // closure, D-11) ---

    TEST_F( DiffUtilsTest, IsByteChunkWithinToleranceForModeUsesMaskBoundWhenDeclaredPasses )
    {
        auto bytesA = BuildByteBytes( { 0 } );
        auto bytesB = BuildByteBytes( { 63 } ); // exactly (1<<6)-1, the mask bound

        ElementDiffStats stats;
        ASSERT_TRUE( IsByteChunkWithinToleranceForMode( bytesA, bytesB, /*byteQuantMode=*/6, stats ) );
    }

    TEST_F( DiffUtilsTest, IsByteChunkWithinToleranceForModeUsesMaskBoundWhenDeclaredFails )
    {
        auto bytesA = BuildByteBytes( { 0 } );
        auto bytesB = BuildByteBytes( { 64 } ); // one past the mask bound

        ElementDiffStats stats;
        ASSERT_FALSE( IsByteChunkWithinToleranceForMode( bytesA, bytesB, /*byteQuantMode=*/6, stats ) );
    }

    TEST_F( DiffUtilsTest, IsByteChunkWithinToleranceForModeMatchesBlendingRealDelta )
    {
        // Blending's real measured raw maxAbsDelta from both Round 1's
        // diff-render-blending.json and Round 2's preserved preQuantizeBytes.
        auto bytesA = BuildByteBytes( { 100 } );
        auto bytesB = BuildByteBytes( { 101 } ); // absDelta = 1

        ElementDiffStats stats;
        ASSERT_TRUE( IsByteChunkWithinToleranceForMode( bytesA, bytesB, /*byteQuantMode=*/6, stats ) );
    }

    // Silence unused-function warnings for BitsOf/FloatFromBits (kept for
    // parity with quantization_test.cpp's helper set, available for future
    // exact-bit-pattern assertions in this suite).
    TEST_F( DiffUtilsTest, BitHelpersRoundTrip )
    {
        ASSERT_EQ( FloatFromBits( BitsOf( 1.5f ) ), 1.5f );
    }

} // namespace sgns::sgprocmanagerdiff
