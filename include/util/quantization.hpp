#ifndef SGPROCMGR_QUANTIZATION_HPP
#define SGPROCMGR_QUANTIZATION_HPP

#include <cstddef>
#include <cstdint>

namespace sgns::sgprocmanagerquant
{
    /// Phase 12 real implementation (D-03 through D-09): IEEE-754 special-value
    /// canonicalization followed by fixed-precision scale-round-cast quantization.
    ///
    /// Canonicalization (evaluated strictly before any rounding arithmetic, D-07):
    ///  - Denormals (both signs) flush to canonical +0.0 (0x00000000), D-07/D-08.
    ///  - NaN (any payload/sign/signaling bit) canonicalizes to the hardcoded
    ///    quiet-NaN bit pattern 0x7FC00000 (D-09) -- never
    ///    std::numeric_limits<float>::quiet_NaN(), since that is not guaranteed
    ///    to be bit-identical across compilers/platforms.
    ///  - +Inf / -Inf canonicalize to two *distinct* fixed bit patterns,
    ///    0x7F800000 / 0xFF800000 respectively (D-06) -- never collapsed to one
    ///    value, so a wrong-sign divergence stays visible to SECV-01's
    ///    counter-test.
    ///  - -0.0 and +0.0 both collapse to the single canonical zero bit pattern
    ///    0x00000000, sign discarded (D-08).
    ///
    /// Rounding (only reached once every canonicalization branch above has been
    /// evaluated and found not to apply): q = round(x * S) / S, with
    /// S = 2^20 (1048576.0f, D-05) -- a power-of-two scale factor for exact
    /// float round-tripping. This grid step (~1e-6) provides roughly 10x margin
    /// over Phase 11's measured cross-machine (Mac vs Windows) MNN float32
    /// divergence: maxAbsDelta ≈ 1.043081283569336e-07, maxRelDelta ≈
    /// 7.269731577252969e-05, maxUlpDistance = 768 (512-element float32 MNN
    /// fixture; see 11-CAPTURE-RESULTS.md). The tolerance is a single fixed
    /// absolute epsilon (D-04) -- not magnitude-adaptive, not relative/ULP-based,
    /// and not schema-configurable.
    ///
    /// @param data  Pointer to a float buffer to quantize in place.
    /// @param count Number of float elements in the buffer.
    void QuantizeFloatBuffer( float *data, size_t count );

    /// Phase 12 deliberate identity pass-through for the render uint8 path.
    ///
    /// This is a considered design decision for this phase, not an inherited
    /// Phase 10 placeholder: Phase 11's empirical render fixture data
    /// (256-element uint8 RGBA/RGB pixel output, Mac vs Windows) showed
    /// contentHashMatch: true with maxAbsDelta/maxRelDelta/maxUlpDistance all
    /// 0.0 -- no observed cross-hardware divergence in the uint8 render path
    /// this milestone's fixtures exercise (see 11-CAPTURE-RESULTS.md). Applying
    /// a lossy tolerance-band here with no empirical justification would only
    /// enlarge the space of results indistinguishable from a correct one, so
    /// this stays byte-identity until new fixture data shows otherwise.
    ///
    /// @param data  Pointer to a byte buffer to quantize in place.
    /// @param count Number of bytes in the buffer.
    void QuantizeByteBuffer( uint8_t *data, size_t count );
}

#endif
