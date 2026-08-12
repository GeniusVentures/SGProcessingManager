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
    /// S = 2^15 (32768.0f) -- a power-of-two scale factor for exact float
    /// round-tripping. The tolerance is a single fixed absolute epsilon (D-04)
    /// -- not magnitude-adaptive, not relative/ULP-based, and not
    /// schema-configurable.
    ///
    /// Original Phase 12 derivation (D-05): S = 2^20 (1048576.0f), grid step
    /// ~9.5367431640625e-07, chosen for a ~9.14x margin over Phase 11's
    /// measured cross-machine (Mac vs Windows) MNN float32 divergence:
    /// maxAbsDelta ≈ 1.043081283569336e-07, maxRelDelta ≈ 7.269731577252969e-05,
    /// maxUlpDistance = 768 (512-element float32 MNN fixture; see
    /// 11-CAPTURE-RESULTS.md).
    ///
    /// Phase 13 Plan 13-04 gap-closure revision (this constant's current
    /// value): Phase 13's own fresh re-validation (13-SCOPE-BOUNDARY.md)
    /// measured a post-quantization maxAbsDelta of exactly 9.5367431640625e-07
    /// -- one full old-grid step -- with 12 of 15 MNN chunk hashes still
    /// diverging cross-hardware at the old S=2^20 grid, direct evidence the
    /// original ~9x margin was insufficient against per-element
    /// grid-boundary tie-break divergence for this fixture's real data.
    ///
    /// A local binary search over power-of-two S values (13-04-PLAN.md Task 1,
    /// revised approach) against processing_conformance_security_test's
    /// Secv01CounterTest.MnnCorruptedModelStillDiverges bracketed a hard
    /// boundary: S=2^15 (grid step 3.0517578125e-05) passes -- the corrupted
    /// MNN model's artifactId still diverges from the correct model's, as
    /// SECV-01 requires -- while S=2^14 (grid step 6.103515625e-05) FAILS
    /// deterministically (the corrupted model's post-quantization artifactId
    /// collides bit-for-bit with the correct model's, confirmed by re-running
    /// twice, not flaky). S=2^15 was chosen over S=2^14 specifically to keep
    /// one full power-of-two step of margin above this confirmed failure
    /// boundary rather than sitting at the exact edge (floating-point
    /// behavior can vary subtly build-to-build). S=2^15's grid step is 32x
    /// the original S=2^20 grid step and ~292x Phase 11's originally-measured
    /// maxAbsDelta -- substantially reducing (not mathematically eliminating)
    /// the per-element grid-boundary tie-break collision probability for this
    /// fixture's real values, while every SECV-01 case (corrupted MNN model,
    /// wrong render shader constant) still passes.
    ///
    /// This is a probabilistic engineering mitigation, not a one-shot
    /// guaranteed solution: a fixed rounding grid cannot mathematically
    /// guarantee zero cross-hardware divergence for arbitrary per-element
    /// deltas that happen to land arbitrarily close to a rounding boundary --
    /// it only reduces the probability of that happening for this fixture's
    /// actual values. See 13-SCOPE-BOUNDARY.md's Refit section (Plan 13-05)
    /// for the fresh empirical cross-machine outcome this constant change is
    /// validated against.
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
