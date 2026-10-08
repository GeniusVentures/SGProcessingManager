#include "util/diff_utils.hpp"

#include <algorithm>
#include <cmath>
#include <cstring>

#include <boost/optional.hpp>

namespace sgns::sgprocmanagerdiff
{
    namespace
    {
        // Mirrors quantization.cpp's power-of-two check exactly (Phase 14
        // D-05/Pitfall 2): never use a transcendental logarithm/exponent
        // function here -- a transcendental-function-based check's last-bit
        // behavior is platform-dependent, which would reintroduce exactly the
        // cross-hardware nondeterminism this milestone exists to eliminate.
        // The integer bit-trick below is deterministic on every platform.
        // Deliberately NOT shared with quantization.cpp -- see this plan's
        // <read_first> rationale: an isolated duplicate, not a refactor of
        // existing Phase 14 code.
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

        // Isolated parameter lookup -- deliberately duplicates
        // quantization.cpp's quantScale resolver find-by-name-and-type loop,
        // but returns boost::none on any invalid/missing case instead of a
        // fallback constant, since this plan's D-03/D-04 branch needs to
        // distinguish "validly declared" from "fell back" (a distinction the
        // existing quantization.cpp resolver's return type cannot express).
        // Must NOT call into or modify quantization.cpp's own resolver.
        boost::optional<float> TryGetDeclaredQuantScale( const std::vector<sgns::Parameter> *parameters )
        {
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
            return boost::none;
        }

        // Isolated parameter lookup -- deliberately duplicates
        // quantization.cpp's byteQuantMode resolver find-by-name-and-type
        // loop, but returns boost::none on any invalid/missing case instead
        // of a fallback constant. Must NOT call into or modify
        // quantization.cpp's own resolver.
        boost::optional<int> TryGetDeclaredByteQuantMode( const std::vector<sgns::Parameter> *parameters )
        {
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
            return boost::none;
        }
    } // namespace

    int64_t OrderedFloatBits( float f )
    {
        int32_t bits;
        std::memcpy( &bits, &f, sizeof( bits ) );
        int64_t wide = static_cast<int64_t>( bits );
        if ( bits < 0 )
        {
            wide = static_cast<int64_t>( 0x80000000LL ) - wide;
        }
        return wide;
    }

    int64_t UlpDistanceFloat( float a, float b )
    {
        return std::llabs( OrderedFloatBits( a ) - OrderedFloatBits( b ) );
    }

    ElementDiffStats ComputeFloat32Diff( const std::vector<uint8_t> &a, const std::vector<uint8_t> &b )
    {
        ElementDiffStats stats;
        if ( a.size() != b.size() )
        {
            stats.sizeMismatch = true;
            return stats;
        }

        stats.elementCount = a.size() / sizeof( float );
        size_t exceedingCount = 0;

        for ( size_t idx = 0; idx < stats.elementCount; ++idx )
        {
            float valA;
            float valB;
            std::memcpy( &valA, a.data() + idx * sizeof( float ), sizeof( float ) );
            std::memcpy( &valB, b.data() + idx * sizeof( float ), sizeof( float ) );

            float absDelta = std::fabs( valA - valB );
            float denom    = std::max( { std::fabs( valA ), std::fabs( valB ), kRelativeDeltaEpsilonFloor } );
            float relDelta = absDelta / denom;
            int64_t ulp    = UlpDistanceFloat( valA, valB );

            if ( relDelta > kDefaultFloatRelativeThreshold )
            {
                ++exceedingCount;
            }

            stats.maxAbsDelta    = std::max( stats.maxAbsDelta, static_cast<double>( absDelta ) );
            stats.maxRelDelta    = std::max( stats.maxRelDelta, static_cast<double>( relDelta ) );
            stats.maxUlpDistance = std::max( stats.maxUlpDistance, ulp );
        }

        stats.percentExceedingThreshold =
            stats.elementCount == 0 ? 0.0 : 100.0 * static_cast<double>( exceedingCount ) / static_cast<double>( stats.elementCount );

        return stats;
    }

    ElementDiffStats ComputeUint8Diff( const std::vector<uint8_t> &a, const std::vector<uint8_t> &b )
    {
        ElementDiffStats stats;
        if ( a.size() != b.size() )
        {
            stats.sizeMismatch = true;
            return stats;
        }

        stats.elementCount = a.size();
        size_t exceedingCount = 0;

        for ( size_t idx = 0; idx < stats.elementCount; ++idx )
        {
            int valA = static_cast<int>( a[idx] );
            int valB = static_cast<int>( b[idx] );

            int    absDelta = std::abs( valA - valB );
            double denom    = static_cast<double>( std::max( { valA, valB, 1 } ) );
            double relDelta = static_cast<double>( absDelta ) / denom;
            int64_t ulp     = absDelta;

            if ( absDelta > kDefaultByteAbsoluteThreshold )
            {
                ++exceedingCount;
            }

            stats.maxAbsDelta    = std::max( stats.maxAbsDelta, static_cast<double>( absDelta ) );
            stats.maxRelDelta    = std::max( stats.maxRelDelta, relDelta );
            stats.maxUlpDistance = std::max( stats.maxUlpDistance, ulp );
        }

        stats.percentExceedingThreshold =
            stats.elementCount == 0 ? 0.0 : 100.0 * static_cast<double>( exceedingCount ) / static_cast<double>( stats.elementCount );

        return stats;
    }

    ChunkElementType ResolveChunkElementTypeHint( const std::vector<sgns::Parameter> *parameters )
    {
        const auto declaredMaskBits = TryGetDeclaredByteQuantMode( parameters );
        if ( declaredMaskBits && *declaredMaskBits > 0 )
        {
            return ChunkElementType::UINT8;
        }
        return ChunkElementType::FLOAT32;
    }

    bool IsFloatChunkWithinTolerance( const std::vector<uint8_t>         &a,
                                      const std::vector<uint8_t>         &b,
                                      const std::vector<sgns::Parameter> *parameters,
                                      ElementDiffStats                   &statsOut )
    {
        statsOut = ComputeFloat32Diff( a, b );
        if ( statsOut.sizeMismatch )
        {
            return false;
        }

        const auto declaredScale = TryGetDeclaredQuantScale( parameters );
        if ( declaredScale )
        {
            // D-03: grid-step-derived bound -- two grid steps of margin,
            // mirroring the project's own "one full step of margin above the
            // confirmed boundary" philosophy (quantization.cpp's S=2^15
            // derivation history).
            return statsOut.maxAbsDelta <= 2.0 / static_cast<double>( *declaredScale );
        }

        // D-04: capture_diff's existing relative-threshold check, already
        // computed inside ComputeFloat32Diff against
        // kDefaultFloatRelativeThreshold. Zero-elements-may-exceed policy,
        // not a percentage-based bar.
        return statsOut.percentExceedingThreshold == 0.0;
    }

    bool IsByteChunkWithinTolerance( const std::vector<uint8_t>         &a,
                                     const std::vector<uint8_t>         &b,
                                     const std::vector<sgns::Parameter> *parameters,
                                     ElementDiffStats                   &statsOut )
    {
        statsOut = ComputeUint8Diff( a, b );
        if ( statsOut.sizeMismatch )
        {
            return false;
        }

        const auto declaredMaskBits = TryGetDeclaredByteQuantMode( parameters );
        if ( declaredMaskBits )
        {
            // D-03: mask-width-derived bound -- two values masking to the
            // same quantized value can differ by up to (1<<N)-1 in raw form.
            const double bound = static_cast<double>( ( 1 << *declaredMaskBits ) - 1 );
            return statsOut.maxAbsDelta <= bound;
        }

        // D-04: capture_diff's existing absolute-threshold check, already
        // computed inside ComputeUint8Diff against
        // kDefaultByteAbsoluteThreshold. Zero-elements-may-exceed policy,
        // not a percentage-based bar.
        return statsOut.percentExceedingThreshold == 0.0;
    }

    bool IsByteChunkWithinToleranceForMode( const std::vector<uint8_t> &a,
                                            const std::vector<uint8_t> &b,
                                            int                          byteQuantMode,
                                            ElementDiffStats            &statsOut )
    {
        sgns::Parameter param;
        param.set_name( "byteQuantMode" );
        param.set_type( sgns::ParameterType::INT );
        param.set_parameter_default( byteQuantMode );

        const std::vector<sgns::Parameter> parameters{ param };
        return IsByteChunkWithinTolerance( a, b, &parameters, statsOut );
    }
} // namespace sgns::sgprocmanagerdiff
