#ifndef SGPROCMGR_QUANTIZATION_HPP
#define SGPROCMGR_QUANTIZATION_HPP

#include <cstddef>
#include <cstdint>
#include <vector>

#include <MNN/MNNForwardType.h>

#include "Parameter.hpp"
#include "ParameterType.hpp"

namespace sgns::sgprocmanagerquant
{
    /// Phase 14 (QUANT-CFG-01/02, D-01/D-02/D-04/D-05): resolves a job
    /// schema-declared "quantScale" entry from the generic `parameters` array,
    /// mirroring the existing find-by-name-in-parameters convention
    /// (ParseLayout / ResolveUniforms).
    ///
    /// Falls back to the exact v2.1 constant 32768.0f (2^15) -- no warning
    /// logged, no job rejection -- when `parameters` is null, no entry named
    /// "quantScale" of type FLOAT exists, its declared default value is not a
    /// JSON number, or the numeric value is not a strictly positive power of
    /// two (D-05's mandatory validation, guaranteeing the exact float
    /// round-trip property D-03's round(x*S)/S formula relies on can never be
    /// silently violated by a bad schema value).
    ///
    /// @param parameters Job schema's generic parameters array, or nullptr.
    /// @return The validated, schema-declared scale, or 32768.0f on any
    ///         invalid/missing declaration.
    float ResolveQuantScale( const std::vector<sgns::Parameter> *parameters );

    /// Phase 14 (QUANT-CFG-01/02, D-02/D-07/D-08): resolves a job
    /// schema-declared "byteQuantMode" entry from the generic `parameters`
    /// array, same lookup convention as ResolveQuantScale.
    ///
    /// Falls back to 0 (the exact v2.1 byte-identity no-op) -- no warning, no
    /// job rejection -- when `parameters` is null, no entry named
    /// "byteQuantMode" of type INT exists, its declared default value is not
    /// a JSON integer, or the integer value falls outside the inclusive range
    /// [0, 8]. N=8 (masking all 8 bits) is a valid, non-fallback boundary
    /// value by design (D-07/D-08); N=9 and above fall back to 0.
    ///
    /// @param parameters Job schema's generic parameters array, or nullptr.
    /// @return The validated, schema-declared mask-bit count in [0, 8], or 0
    ///         on any invalid/missing declaration.
    int ResolveByteQuantMode( const std::vector<sgns::Parameter> *parameters );

    /// Phase 13 (SGF-01, D-04/D-05): resolves a job schema-declared
    /// "backend" entry from the generic `parameters` array, same lookup
    /// convention as ResolveQuantScale/ResolveByteQuantMode, selecting the
    /// MNN session backend for MNN-based processors.
    ///
    /// Falls back to MNN_FORWARD_VULKAN -- the exact behavior every MNN
    /// processor had when `config.type` was hardcoded -- when `parameters`
    /// is null, no entry named "backend" of type STRING exists, its declared
    /// default value is not a JSON string, or the lowercased string is
    /// neither "cpu" nor "vulkan" (T-13-02: an untrusted schema value can
    /// never select an unintended backend; it only ever falls back to the
    /// safe default).
    ///
    /// @param parameters Job schema's generic parameters array, or nullptr.
    /// @return MNN_FORWARD_CPU only for an explicit "cpu" declaration;
    ///         MNN_FORWARD_VULKAN for "vulkan" and every fallback case.
    MNNForwardType ResolveMnnBackend( const std::vector<sgns::Parameter> *parameters );


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
    /// Phase 14 (QUANT-CFG-01/02): `S` is now schema-configurable via the
    /// caller-resolved `scale` argument, produced by calling
    /// ResolveQuantScale() once per StartProcessing() invocation. 32768.0f
    /// remains the exact fallback when nothing valid is schema-declared, and
    /// the D-03 round(x*S)/S formula plus the D-06..D-09 canonicalization
    /// branch order above are entirely unchanged by this addition -- this
    /// paragraph documents schema-configurability, it does not revise or
    /// contradict the S=2^15 derivation history above it.
    ///
    /// @param data  Pointer to a float buffer to quantize in place.
    /// @param count Number of float elements in the buffer.
    /// @param scale The resolved scale S to use (see ResolveQuantScale()).
    void QuantizeFloatBuffer( float *data, size_t count, float scale );

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
    /// Phase 14 (QUANT-CFG-01/02, D-06/D-07): the mask is now schema-
    /// configurable via the caller-resolved `maskBits` argument, produced by
    /// calling ResolveByteQuantMode() once per StartProcessing() invocation.
    /// `maskBits <= 0` (D-07's N=0/absent case) remains the exact v2.1
    /// byte-identity no-op; otherwise the low `maskBits` bits of every byte
    /// are cleared (D-06's bit-masking technique, `value &= ~((1<<N)-1)`).
    ///
    /// @param data     Pointer to a byte buffer to quantize in place.
    /// @param count    Number of bytes in the buffer.
    /// @param maskBits Number of low bits to clear per byte, in [0, 8] (see
    ///                 ResolveByteQuantMode()); <= 0 is the identity no-op.
    void QuantizeByteBuffer( uint8_t *data, size_t count, int maskBits );
}

#endif
