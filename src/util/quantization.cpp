

#include "util/quantization.hpp"

#include <cmath>
#include <cstring>

namespace sgns::sgprocmanagerquant
{
    void QuantizeFloatBuffer( float *data, size_t count )
    {
        // D-05: fixed power-of-two scale factor, 2^20 -- ~10x margin over
        // Phase 11's measured maxAbsDelta ≈ 1.043081283569336e-07 (see header
        // doc comment for the full citation).
        constexpr float kScale = 1048576.0f; // 2^20

        for ( size_t i = 0; i < count; ++i )
        {
            float x = data[i];

            // Extract the bit pattern via memcpy (never a reinterpret_cast
            // type-pun), mirroring HalfToFloat's existing bit-punning style
            // (processing_processor_mnn_float.cpp).
            uint32_t bits = 0;
            std::memcpy( &bits, &x, sizeof( bits ) );

            const uint32_t exponentBits = bits & 0x7F800000u;
            const uint32_t mantissaBits = bits & 0x007FFFFFu;

            // Branch order is itself the D-07 requirement: every special-value
            // check below is evaluated before the rounding arithmetic in the
            // final else arm ever runs.

            // 1. Denormal (both signs, D-07): biased exponent field is zero but
            //    mantissa is nonzero. Flush to canonical +0.0 (D-08).
            if ( exponentBits == 0u && mantissaBits != 0u )
            {
                data[i] = 0.0f;
            }
            // 2. NaN: canonicalize to the hardcoded quiet-NaN bit pattern
            //    0x7FC00000 (D-09), regardless of payload/sign/signaling bit.
            else if ( std::isnan( x ) )
            {
                constexpr uint32_t kCanonicalNaN = 0x7FC00000u;
                std::memcpy( &data[i], &kCanonicalNaN, sizeof( kCanonicalNaN ) );
            }
            // 3. Infinity: two distinct fixed bit patterns (D-06), never
            //    collapsed to one value.
            else if ( std::isinf( x ) )
            {
                constexpr uint32_t kPositiveInfinity = 0x7F800000u;
                constexpr uint32_t kNegativeInfinity = 0xFF800000u;
                if ( std::signbit( x ) )
                {
                    std::memcpy( &data[i], &kNegativeInfinity, sizeof( kNegativeInfinity ) );
                }
                else
                {
                    std::memcpy( &data[i], &kPositiveInfinity, sizeof( kPositiveInfinity ) );
                }
            }
            // 4. Signed zero (D-08): +0.0 and -0.0 both compare equal to 0.0f
            //    under IEEE equality; collapse to the single canonical zero.
            else if ( x == 0.0f )
            {
                data[i] = 0.0f;
            }
            // 5. Ordinary finite value: fixed-point scale-round-cast (D-03).
            else
            {
                data[i] = std::round( x * kScale ) / kScale;
            }
        }
    }

    void QuantizeByteBuffer( uint8_t *data, size_t count )
    {
        // D-01/QUANT-04: deliberate byte-identity pass-through for the render
        // uint8 path -- see header doc comment for the Phase 11 empirical
        // justification (contentHashMatch: true, all deltas 0.0). This is a
        // considered decision for this phase, not an unmodified carry-over
        // from Phase 10's placeholder stub.
        (void)data;
        (void)count;
    }
} // namespace sgns::sgprocmanagerquant
