// Phase 12, Plan 12-01: unit tests for the real QuantizeFloatBuffer/
// QuantizeByteBuffer implementations (D-03 through D-09).
//
// Pure in-memory unit tests -- no file fixtures, no ProcessorConformanceFixture
// base needed. Bit patterns are compared via memcpy-extracted uint32_t and
// ASSERT_EQ, never via approximate float comparison, since D-09/D-06/D-08
// require exact canonical output.
//
// Phase 14, Plan 14-01: extended with ResolveQuantScale/ResolveByteQuantMode
// fallback/boundary coverage (D-04/D-05/D-07/D-08), and every pre-existing
// QuantizeFloatBuffer/QuantizeByteBuffer call updated to the new required
// 3-arg signature (scale/maskBits are no longer compile-time constants).

#include <gtest/gtest.h>

#include <cstdint>
#include <cstring>
#include <cmath>
#include <string>
#include <vector>

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
        QuantizeFloatBuffer( data1, 1, 32768.0f );
        ASSERT_EQ( BitsOf( data1[0] ), 0x7FC00000u );

        // Negative NaN -> sign discarded, same hardcoded canonical pattern (D-09).
        float data2[1] = { FloatFromBits( 0xFFC00000u ) };
        QuantizeFloatBuffer( data2, 1, 32768.0f );
        ASSERT_EQ( BitsOf( data2[0] ), 0x7FC00000u );
    }

    TEST_F( QuantizationTest, QuantizeFloatBufferCanonicalizesPositiveInfinity )
    {
        float data[1] = { FloatFromBits( 0x7F800000u ) };
        QuantizeFloatBuffer( data, 1, 32768.0f );
        ASSERT_EQ( BitsOf( data[0] ), 0x7F800000u );
    }

    TEST_F( QuantizationTest, QuantizeFloatBufferCanonicalizesNegativeInfinity )
    {
        // -Inf stays distinct from +Inf (D-06), never collapsed.
        float data[1] = { FloatFromBits( 0xFF800000u ) };
        QuantizeFloatBuffer( data, 1, 32768.0f );
        ASSERT_EQ( BitsOf( data[0] ), 0xFF800000u );
    }

    TEST_F( QuantizationTest, QuantizeFloatBufferCanonicalizesDenormals )
    {
        // Smallest positive denormal, smallest negative denormal -> both flush
        // to canonical +0.0.
        float data[2] = { FloatFromBits( 0x00000001u ), FloatFromBits( 0x80000001u ) };
        QuantizeFloatBuffer( data, 2, 32768.0f );
        ASSERT_EQ( BitsOf( data[0] ), 0x00000000u );
        ASSERT_EQ( BitsOf( data[1] ), 0x00000000u );
    }

    TEST_F( QuantizationTest, QuantizeFloatBufferCanonicalizesSignedZero )
    {
        // -0.0 and +0.0 both collapse to the single canonical zero bit pattern.
        float data[2] = { FloatFromBits( 0x80000000u ), FloatFromBits( 0x00000000u ) };
        QuantizeFloatBuffer( data, 2, 32768.0f );
        ASSERT_EQ( BitsOf( data[0] ), 0x00000000u );
        ASSERT_EQ( BitsOf( data[1] ), 0x00000000u );
    }

    TEST_F( QuantizationTest, QuantizeFloatBufferRoundsToFixedGrid )
    {
        // Ordinary finite value, not on the 2^-15 grid.
        constexpr float kScale = 32768.0f; // 2^15, matches Phase 13 Plan 13-04 gap-closure widening
        float           data[1] = { 0.1f };
        QuantizeFloatBuffer( data, 1, kScale );

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
        QuantizeByteBuffer( data, 5, 0 );
        ASSERT_EQ( std::memcmp( data, expected, sizeof( data ) ), 0 );
    }

    namespace
    {
        // Phase 14, Task 2: builds a one-element parameters vector for a
        // Resolve* test case, using Parameter's public setters.
        std::vector<sgns::Parameter> MakeParameters( const std::string       &name,
                                                      sgns::ParameterType      type,
                                                      const nlohmann::json    &defaultValue )
        {
            sgns::Parameter param;
            param.set_name( name );
            param.set_type( type );
            param.set_parameter_default( defaultValue );
            return { param };
        }
    } // namespace

    TEST_F( QuantizationTest, ResolveQuantScaleFallsBackOnNullParameters )
    {
        ASSERT_EQ( BitsOf( ResolveQuantScale( nullptr ) ), BitsOf( 32768.0f ) );
    }

    TEST_F( QuantizationTest, ResolveQuantScaleFallsBackOnMissingEntry )
    {
        const std::vector<sgns::Parameter> parameters;
        ASSERT_EQ( BitsOf( ResolveQuantScale( &parameters ) ), BitsOf( 32768.0f ) );
    }

    TEST_F( QuantizationTest, ResolveQuantScaleUsesValidPowerOfTwo )
    {
        const auto parameters = MakeParameters( "quantScale", sgns::ParameterType::FLOAT, 16384.0 );
        ASSERT_EQ( BitsOf( ResolveQuantScale( &parameters ) ), BitsOf( 16384.0f ) );
    }

    TEST_F( QuantizationTest, ResolveQuantScaleFallsBackOnNonPowerOfTwo )
    {
        const auto parameters = MakeParameters( "quantScale", sgns::ParameterType::FLOAT, 100.0 );
        ASSERT_EQ( BitsOf( ResolveQuantScale( &parameters ) ), BitsOf( 32768.0f ) );
    }

    TEST_F( QuantizationTest, ResolveQuantScaleFallsBackOnNonPositive )
    {
        const auto zeroParameters = MakeParameters( "quantScale", sgns::ParameterType::FLOAT, 0.0 );
        ASSERT_EQ( BitsOf( ResolveQuantScale( &zeroParameters ) ), BitsOf( 32768.0f ) );

        const auto negativeParameters = MakeParameters( "quantScale", sgns::ParameterType::FLOAT, -8.0 );
        ASSERT_EQ( BitsOf( ResolveQuantScale( &negativeParameters ) ), BitsOf( 32768.0f ) );
    }

    TEST_F( QuantizationTest, ResolveQuantScaleFallsBackOnNonNumeric )
    {
        const auto parameters = MakeParameters( "quantScale", sgns::ParameterType::FLOAT, std::string( "16384" ) );
        ASSERT_EQ( BitsOf( ResolveQuantScale( &parameters ) ), BitsOf( 32768.0f ) );
    }

    TEST_F( QuantizationTest, ResolveByteQuantModeFallsBackOnNullParameters )
    {
        ASSERT_EQ( ResolveByteQuantMode( nullptr ), 0 );
    }

    TEST_F( QuantizationTest, ResolveByteQuantModeUsesValidValue )
    {
        const auto parameters = MakeParameters( "byteQuantMode", sgns::ParameterType::INT, 3 );
        ASSERT_EQ( ResolveByteQuantMode( &parameters ), 3 );
    }

    TEST_F( QuantizationTest, ResolveByteQuantModeAcceptsBoundaryEight )
    {
        const auto parameters = MakeParameters( "byteQuantMode", sgns::ParameterType::INT, 8 );
        ASSERT_EQ( ResolveByteQuantMode( &parameters ), 8 );
    }

    TEST_F( QuantizationTest, ResolveByteQuantModeFallsBackJustAboveBoundary )
    {
        const auto parameters = MakeParameters( "byteQuantMode", sgns::ParameterType::INT, 9 );
        ASSERT_EQ( ResolveByteQuantMode( &parameters ), 0 );
    }

    TEST_F( QuantizationTest, ResolveByteQuantModeFallsBackOnNegative )
    {
        const auto parameters = MakeParameters( "byteQuantMode", sgns::ParameterType::INT, -1 );
        ASSERT_EQ( ResolveByteQuantMode( &parameters ), 0 );
    }

} // namespace sgns::sgprocmanagerquant
