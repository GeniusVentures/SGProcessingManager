// Phase 12, Plan 12-01: unit tests for the real QuantizeFloatBuffer/
// QuantizeByteBuffer implementations (D-03 through D-09).
//
// Pure in-memory unit tests -- no file fixtures, no ProcessorConformanceFixture
// base needed. Bit patterns are compared via memcpy-extracted uint32_t and
// ASSERT_EQ, never via approximate float comparison, since D-09/D-06/D-08
// require exact canonical output.

#include <gtest/gtest.h>

#include <cstdint>
#include <cstring>
#include <cmath>

#include "util/quantization.hpp"

namespace sgns::sgprocmanagerquant
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
    } // namespace

    class QuantizationTest : public ::testing::Test
    {
    };

    TEST_F( QuantizationTest, QuantizeFloatBufferCanonicalizesNaN )
    {
        // NaN with nonzero payload -> exact canonical quiet-NaN bit pattern.
        float data1[1] = { FloatFromBits( 0x7FC00123u ) };
        QuantizeFloatBuffer( data1, 1 );
        ASSERT_EQ( BitsOf( data1[0] ), 0x7FC00000u );

        // Negative NaN -> sign discarded, same hardcoded canonical pattern (D-09).
        float data2[1] = { FloatFromBits( 0xFFC00000u ) };
        QuantizeFloatBuffer( data2, 1 );
        ASSERT_EQ( BitsOf( data2[0] ), 0x7FC00000u );
    }

    TEST_F( QuantizationTest, QuantizeFloatBufferCanonicalizesPositiveInfinity )
    {
        float data[1] = { FloatFromBits( 0x7F800000u ) };
        QuantizeFloatBuffer( data, 1 );
        ASSERT_EQ( BitsOf( data[0] ), 0x7F800000u );
    }

    TEST_F( QuantizationTest, QuantizeFloatBufferCanonicalizesNegativeInfinity )
    {
        // -Inf stays distinct from +Inf (D-06), never collapsed.
        float data[1] = { FloatFromBits( 0xFF800000u ) };
        QuantizeFloatBuffer( data, 1 );
        ASSERT_EQ( BitsOf( data[0] ), 0xFF800000u );
    }

    TEST_F( QuantizationTest, QuantizeFloatBufferCanonicalizesDenormals )
    {
        // Smallest positive denormal, smallest negative denormal -> both flush
        // to canonical +0.0.
        float data[2] = { FloatFromBits( 0x00000001u ), FloatFromBits( 0x80000001u ) };
        QuantizeFloatBuffer( data, 2 );
        ASSERT_EQ( BitsOf( data[0] ), 0x00000000u );
        ASSERT_EQ( BitsOf( data[1] ), 0x00000000u );
    }

    TEST_F( QuantizationTest, QuantizeFloatBufferCanonicalizesSignedZero )
    {
        // -0.0 and +0.0 both collapse to the single canonical zero bit pattern.
        float data[2] = { FloatFromBits( 0x80000000u ), FloatFromBits( 0x00000000u ) };
        QuantizeFloatBuffer( data, 2 );
        ASSERT_EQ( BitsOf( data[0] ), 0x00000000u );
        ASSERT_EQ( BitsOf( data[1] ), 0x00000000u );
    }

    TEST_F( QuantizationTest, QuantizeFloatBufferRoundsToFixedGrid )
    {
        // Ordinary finite value, not on the 2^-15 grid.
        constexpr float kScale = 32768.0f; // 2^15, matches Phase 13 Plan 13-04 gap-closure widening
        float           data[1] = { 0.1f };
        QuantizeFloatBuffer( data, 1 );

        const float expected = std::round( 0.1f * kScale ) / kScale;
        ASSERT_EQ( BitsOf( data[0] ), BitsOf( expected ) );

        // Grid-alignment property, independent of the formula-echo check above:
        // (output * scale) must itself be an exact integer.
        const float scaled = data[0] * kScale;
        ASSERT_EQ( scaled, std::round( scaled ) );
    }

    TEST_F( QuantizationTest, QuantizeByteBufferIsIdentity )
    {
        uint8_t data[5] = { 0, 1, 127, 128, 255 };
        const uint8_t expected[5] = { 0, 1, 127, 128, 255 };
        QuantizeByteBuffer( data, 5 );
        ASSERT_EQ( std::memcmp( data, expected, sizeof( data ) ), 0 );
    }

} // namespace sgns::sgprocmanagerquant
