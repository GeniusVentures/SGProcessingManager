

#include "util/quantization.hpp"

#include <algorithm>
#include <cctype>
#include <cmath>
#include <cstring>

namespace sgns::sgprocmanagerquant
{
    namespace
    {
        // Phase 14 D-05/Pitfall 2: never use std::log2/std::pow here -- a
        // transcendental-function-based check's last-bit behavior is
        // platform-dependent, which would reintroduce exactly the
        // cross-hardware nondeterminism this milestone exists to eliminate.
        // The integer bit-trick below is deterministic on every platform.
        bool IsPositivePowerOfTwo( double value )
        {
            if ( !( value > 0.0 ) )
            {
                return false;
            }
            if ( std::floor( value ) != value )
            {
                return false;
            }
            const auto asInt = static_cast<uint64_t>( value );
            return asInt != 0u && ( asInt & ( asInt - 1u ) ) == 0u;
        }
    } // namespace

    float ResolveQuantScale( const std::vector<sgns::Parameter> *parameters )
    {
        constexpr float kFallbackScale = 32768.0f; // 2^15, exact v2.1 constant (D-04)

        if ( parameters )
        {
            for ( const auto &param : *parameters )
            {
                if ( param.get_name() == "quantScale" && param.get_type() == sgns::ParameterType::FLOAT )
                {
                    const auto &def = param.get_parameter_default();
                    if ( def.is_number() )
                    {
                        const double declared = def.get<double>();
                        if ( IsPositivePowerOfTwo( declared ) )
                        {
                            return static_cast<float>( declared );
                        }
                    }
                    break;
                }
            }
        }

        return kFallbackScale;
    }

    int ResolveByteQuantMode( const std::vector<sgns::Parameter> *parameters )
    {
        constexpr int kFallbackMaskBits = 0; // Identity no-op, exact v2.1 behavior (D-08)

        if ( parameters )
        {
            for ( const auto &param : *parameters )
            {
                if ( param.get_name() == "byteQuantMode" && param.get_type() == sgns::ParameterType::INT )
                {
                    const auto &def = param.get_parameter_default();
                    if ( def.is_number_integer() )
                    {
                        const int declared = def.get<int>();
                        if ( declared >= 0 && declared <= 8 )
                        {
                            return declared;
                        }
                    }
                    break;
                }
            }
        }

        return kFallbackMaskBits;
    }

    MNNForwardType ResolveMnnBackend( const std::vector<sgns::Parameter> *parameters )
    {
        // Phase 13 (D-04): MNN_FORWARD_VULKAN is the fallback so every
        // existing caller (no "backend" parameter declared) keeps today's
        // exact hardcoded-Vulkan behavior.
        constexpr MNNForwardType kFallbackBackend = MNN_FORWARD_VULKAN;

        if ( parameters )
        {
            for ( const auto &param : *parameters )
            {
                if ( param.get_name() == "backend" && param.get_type() == sgns::ParameterType::STRING )
                {
                    const auto &def = param.get_parameter_default();
                    if ( def.is_string() )
                    {
                        // Lowercase-normalize so "CPU"/"Cpu" behave as "cpu"
                        // (T-13-02 mitigation: normalization happens before
                        // the accept-list check, and anything outside the
                        // two accepted values still falls back to Vulkan).
                        std::string declared = def.get<std::string>();
                        std::transform( declared.begin(),
                                        declared.end(),
                                        declared.begin(),
                                        []( unsigned char c ) { return static_cast<char>( std::tolower( c ) ); } );
                        if ( declared == "cpu" )
                        {
                            return MNN_FORWARD_CPU;
                        }
                        if ( declared == "vulkan" )
                        {
                            return MNN_FORWARD_VULKAN;
                        }
                    }
                    break;
                }
            }
        }

        return kFallbackBackend;
    }

    void QuantizeFloatBuffer( float *data, size_t count, float scale )
    {
        // Phase 13 Plan 13-04 gap-closure widening (supersedes Phase 12 D-05's
        // 2^20 value): the original S=2^20 grid step (9.5367431640625e-07)
        // gave only a ~9.14x margin over Phase 11's measured cross-machine
        // maxAbsDelta (1.043081283569336e-07); Phase 13's own fresh
        // re-validation (13-SCOPE-BOUNDARY.md) measured a post-quantization
        // maxAbsDelta of exactly 9.5367431640625e-07 (one full old-grid step)
        // with 12 of 15 MNN chunk hashes still diverging cross-hardware --
        // direct evidence the ~9x margin was insufficient.
        //
        // A local binary search over power-of-two S values (13-04-PLAN.md
        // Task 1, revised approach) against processing_conformance_security_
        // test's Secv01CounterTest.MnnCorruptedModelStillDiverges found:
        //   S=2^20 (9.5367431640625e-07 grid step) -- SECV-01 passes (baseline)
        //   S=2^17 (7.62939453125e-06 grid step)   -- SECV-01 passes
        //   S=2^16 (1.52587890625e-05 grid step)   -- SECV-01 passes
        //   S=2^15 (3.0517578125e-05 grid step)    -- SECV-01 passes
        //   S=2^14 (6.103515625e-05 grid step)     -- SECV-01 FAILS (the
        //     deliberately corrupted MNN model's artifactId collides
        //     bit-for-bit with the correct model's, memcmp equal, 0 vs 0 --
        //     confirmed deterministic, not flaky, by re-running twice)
        // S=2^15 is chosen: the widest power-of-two grid step confirmed safe,
        // one full power-of-two step of margin above the confirmed S=2^14
        // failure boundary (not the exact edge), giving 32x the old S=2^20
        // grid step (~292x Phase 11's original maxAbsDelta) while still
        // leaving SECV-01's corrupted-model divergence fully intact.
        //
        // Phase 14 (QUANT-CFG-01/02): this constant is no longer hardcoded
        // here -- callers resolve it via ResolveQuantScale() (D-04/D-05
        // fallback to this exact 32768.0f value) and pass it as `scale`.

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
                data[i] = std::round( x * scale ) / scale;
            }
        }
    }

    void QuantizeByteBuffer( uint8_t *data, size_t count, int maskBits )
    {
        // D-01/QUANT-04: byte-identity no-op when nothing (valid) is
        // schema-declared -- see header doc comment for the Phase 11
        // empirical justification (contentHashMatch: true, all deltas 0.0).
        // Phase 14 D-07: maskBits<=0 (absent/N=0) is exactly this v2.1
        // identity behavior, unchanged.
        if ( maskBits <= 0 )
        {
            return;
        }

        // D-06: clear the low `maskBits` bits of every byte. maskBits is
        // resolver-validated to [0, 8] (ResolveByteQuantMode), so the shift
        // below never exceeds the width of an unsigned int.
        const uint8_t mask = static_cast<uint8_t>( ~( ( 1u << maskBits ) - 1u ) );
        for ( size_t i = 0; i < count; ++i )
        {
            data[i] &= mask;
        }
    }
} // namespace sgns::sgprocmanagerquant
