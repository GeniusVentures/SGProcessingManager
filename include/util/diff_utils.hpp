#ifndef SGPROCMGR_DIFF_UTILS_HPP
#define SGPROCMGR_DIFF_UTILS_HPP

#include <cstddef>
#include <cstdint>
#include <vector>

#include "Parameter.hpp"
#include "ParameterType.hpp"

namespace sgns::sgprocmanagerdiff
{
    /// Relative-delta denominator floor -- avoids divide-by-zero near
    /// zero-valued float elements. Extracted verbatim from capture_diff.cpp's
    /// former unnamed-namespace constant (Phase 10, Plan 10-05) so both the
    /// CLI tool and this shared library read the identical value.
    constexpr float kRelativeDeltaEpsilonFloor = 1e-6f;

    /// D-04 fallback: fixed float relative-delta threshold used by
    /// ComputeFloat32Diff's exceeding-count check and by
    /// IsFloatChunkWithinTolerance when no valid "quantScale" is declared.
    /// Exported (not file-local) so capture_diff.cpp and any runtime
    /// validator consumer read the exact same symbol -- guarantees zero
    /// behavioral drift between the offline CLI tool and Plan 15-02's
    /// runtime comparison.
    constexpr double kDefaultFloatRelativeThreshold = 1e-4;

    /// D-04 fallback: fixed byte absolute-delta threshold used by
    /// ComputeUint8Diff's exceeding-count check and by
    /// IsByteChunkWithinTolerance when no valid "byteQuantMode" is declared.
    constexpr int kDefaultByteAbsoluteThreshold = 1;

    /// Whole-buffer per-element divergence summary (DIFF-01/DIFF-02).
    /// Extracted verbatim from capture_diff.cpp's former unnamed-namespace
    /// struct of the same name/shape.
    struct ElementDiffStats
    {
        size_t  elementCount             = 0;
        double  maxAbsDelta              = 0.0;
        double  maxRelDelta              = 0.0;
        int64_t maxUlpDistance           = 0;
        double  percentExceedingThreshold = 0.0;
        bool    sizeMismatch             = false;
    };

    /// Standard ordered-integer bit-reinterpretation technique for float ULP
    /// distance. Extracted verbatim from capture_diff.cpp.
    int64_t OrderedFloatBits( float f );

    /// Extracted verbatim from capture_diff.cpp.
    int64_t UlpDistanceFloat( float a, float b );

    /// Computes per-element float32 divergence stats between two raw byte
    /// buffers (each buffer's size must be a multiple of sizeof(float)).
    /// Extracted verbatim from capture_diff.cpp's former unnamed-namespace
    /// function of the same name -- behavior-neutral relocation, byte-for-byte
    /// identical output to the pre-extraction version on the same inputs.
    ElementDiffStats ComputeFloat32Diff( const std::vector<uint8_t> &a, const std::vector<uint8_t> &b );

    /// Computes per-element uint8 divergence stats between two raw byte
    /// buffers. Extracted verbatim from capture_diff.cpp's former
    /// unnamed-namespace function of the same name.
    ElementDiffStats ComputeUint8Diff( const std::vector<uint8_t> &a, const std::vector<uint8_t> &b );

    /// Element-type hint for a chunk's raw output data, used to decide which
    /// tolerance-derivation function (float vs. byte) applies.
    enum class ChunkElementType
    {
        FLOAT32,
        UINT8
    };

    /// Phase 15 (XNODE-02): resolves whether a chunk's raw output data should
    /// be treated as float32 or uint8 for tolerance-comparison purposes.
    ///
    /// Returns UINT8 only when a job schema-declares "byteQuantMode" (INT
    /// type, integer value in [0, 8]) with a value greater than 0 -- i.e. the
    /// byte-quantization path is actually active for this job. Returns
    /// FLOAT32 in every other case: "quantScale" declared instead, a
    /// "byteQuantMode" of exactly 0 declared (the byte-identity no-op case,
    /// per quantization.hpp's own doc comments), or neither declared. This is
    /// a documented, deliberate default -- float32 is the more general/common
    /// MNN numeric case, and the render byte path has historically been a
    /// no-op per Phase 12/14's own doc comments -- not an attempt to solve
    /// per-output element-type inference generally.
    ///
    /// @param parameters Job schema's generic parameters array, or nullptr.
    /// @return UINT8 only when byteQuantMode is validly declared with a value
    ///         > 0; FLOAT32 otherwise.
    ChunkElementType ResolveChunkElementTypeHint( const std::vector<sgns::Parameter> *parameters );

    /// Phase 15 (XNODE-02, D-03/D-04): determines whether two float32 chunk
    /// buffers are numerically "close enough" to be treated as a tolerant
    /// match rather than a genuine cross-node divergence.
    ///
    /// D-03: when the job validly declares "quantScale" = S, the bound is
    /// derived from the quantization grid step (2/S, two grid steps of
    /// margin -- the same "one full step of margin" philosophy
    /// quantization.cpp's own S=2^15-over-2^14 derivation history documents)
    /// and compared against ComputeFloat32Diff's maxAbsDelta.
    ///
    /// D-04: when no valid "quantScale" is declared, falls back to
    /// capture_diff's existing kDefaultFloatRelativeThreshold (1e-4,
    /// relative), via ComputeFloat32Diff's own percentExceedingThreshold
    /// stat (zero-elements-may-exceed policy, not a percentage-based bar).
    ///
    /// @param a          First chunk's raw float32 bytes.
    /// @param b          Second chunk's raw float32 bytes.
    /// @param parameters Job schema's generic parameters array, or nullptr.
    /// @param statsOut   Populated with ComputeFloat32Diff's full stats,
    ///                   regardless of the boolean result.
    /// @return false immediately (statsOut.sizeMismatch=true) if a/b differ
    ///         in length; otherwise true iff within the D-03/D-04 tolerance.
    bool IsFloatChunkWithinTolerance( const std::vector<uint8_t>          &a,
                                      const std::vector<uint8_t>          &b,
                                      const std::vector<sgns::Parameter>  *parameters,
                                      ElementDiffStats                    &statsOut );

    /// Phase 15 (XNODE-02, D-03/D-04): determines whether two uint8 chunk
    /// buffers are numerically "close enough" to be treated as a tolerant
    /// match rather than a genuine cross-node divergence.
    ///
    /// D-03: when the job validly declares "byteQuantMode" = N, the bound is
    /// derived from the quantization mask width ((1<<N)-1, the maximum raw
    /// delta two values masking to the same quantized value can have) and
    /// compared against ComputeUint8Diff's maxAbsDelta.
    ///
    /// D-04: when no valid "byteQuantMode" is declared, falls back to
    /// capture_diff's existing kDefaultByteAbsoluteThreshold (1, absolute),
    /// via ComputeUint8Diff's own percentExceedingThreshold stat
    /// (zero-elements-may-exceed policy, not a percentage-based bar).
    ///
    /// @param a          First chunk's raw uint8 bytes.
    /// @param b          Second chunk's raw uint8 bytes.
    /// @param parameters Job schema's generic parameters array, or nullptr.
    /// @param statsOut   Populated with ComputeUint8Diff's full stats,
    ///                   regardless of the boolean result.
    /// @return false immediately (statsOut.sizeMismatch=true) if a/b differ
    ///         in length; otherwise true iff within the D-03/D-04 tolerance.
    bool IsByteChunkWithinTolerance( const std::vector<uint8_t>          &a,
                                     const std::vector<uint8_t>          &b,
                                     const std::vector<sgns::Parameter>  *parameters,
                                     ElementDiffStats                    &statsOut );
} // namespace sgns::sgprocmanagerdiff

#endif
