/**
 * Capture file binary format for Phase 10: Capture Harness & Diff Tool.
 *
 * Wraps the existing artifact_serializer.hpp binary convention
 * (SerializeArtifact/SerializeManifest, called unmodified) for the
 * metadata+hash portion, and appends one new length-prefixed raw-bytes
 * section per rawOutputCapture invocation (D-01 -- capture files must not
 * invent a second serialization convention alongside the one that already
 * exists). Fixed-field regions (artifact, manifest) keep their existing
 * little-endian, fixed-offset layout unchanged; the new sections added by
 * this file are little-endian, length-prefixed, variable-length -- mirroring
 * artifact_serializer.hpp's own "fixed-field, little-endian" doc convention
 * where a fixed layout applies, and falling back to explicit length
 * prefixes only where the data itself is inherently variable-length.
 *
 * Binary layout (little-endian throughout):
 *   [4]  magic "SGC1"
 *   [4]  machineIdTag byte length (uint32) + that many UTF-8 bytes
 *   [4]  fixtureLabel byte length (uint32) + that many UTF-8 bytes
 *   [4]  artifactCount (uint32)
 *   per artifact:
 *     [ARTIFACT_SERIALIZED_SIZE]  SerializeArtifact(artifact) bytes, unmodified
 *     [4]  recordCount for this artifact (uint32)
 *     per record:
 *       [8]  preQuantizeBytes.size() (uint64) + that many raw bytes
 *       [8]  quantizedBytes.size() (uint64) + that many raw bytes
 *   [MANIFEST_V2_SERIALIZED_SIZE]  SerializeManifest(manifest) bytes, unmodified
 *   [4]  combinedHash.size() (uint32) + that many raw bytes
 *
 * DeserializeCaptureFile validates every declared length/count against the
 * bytes actually remaining in the input buffer, and rejects (returns false)
 * any single declared length exceeding kMaxSectionBytes (1 GiB), BEFORE
 * allocating or reading that many bytes (T-10-02). Never throws; returns
 * false rather than partially populating `out` on any malformed, truncated,
 * or oversized input -- mirroring DeserializeArtifact/DeserializeManifest's
 * existing bool-return-false-on-malformed-input convention (T-10-01a).
 *
 * @brief Capture file binary format (per-run raw output bytes + hashes + manifest)
 */
#ifndef SGPROCMGR_CAPTURE_FILE_FORMAT_HPP
#define SGPROCMGR_CAPTURE_FILE_FORMAT_HPP

#include <cstdint>
#include <string>
#include <vector>
#include "artifacts/artifact_types.hpp"
#include "artifacts/execution_manifest.hpp"

namespace sgns::sgproccapture
{

    /// One rawOutputCapture invocation's worth of bytes -- either a per-chunk
    /// capture (paired with one entry of Artifact::chunkHashes) or the trailing
    /// combined-hash capture (paired with Artifact::contentHash), in call order.
    struct CaptureRecord
    {
        std::vector<uint8_t> preQuantizeBytes;  ///< Bytes offered to rawOutputCapture before Quantize*Buffer ran
        std::vector<uint8_t> quantizedBytes;    ///< Bytes offered to rawOutputCapture after Quantize*Buffer ran (identity stub in Phase 10)
    };

    /// A single capture run: machine identity, fixture label, every output
    /// artifact + the execution manifest from the run (via the existing
    /// artifact_serializer.hpp convention, unmodified), plus the raw
    /// pre-/post-quantization bytes captured at every rawOutputCapture call site.
    struct CaptureFile
    {
        std::string machineIdTag;  ///< Hostname + OS (D-03), e.g. "MacBook-Pro-M2 / macOS 15.1"
        std::string fixtureLabel;  ///< e.g. "render-happy-path" or "mnn-float" (D-02)

        std::vector<sgns::sgprocessing::Artifact> artifacts;  ///< One per job output

        /// Index-aligned with `artifacts`: rawRecordsPerArtifact[i] is the ordered
        /// list of every rawOutputCapture call that fed hashes for artifacts[i].
        /// For a single-output job, records[0 .. artifacts[i].chunkHashCount - 1]
        /// correspond 1:1 to artifacts[i].chunkHashes[0 .. chunkHashCount - 1], and
        /// an optional trailing record (if present) corresponds to
        /// artifacts[i].contentHash -- this pairing is what Wave 3's
        /// capture_harness self-check (CAPT-02) verifies.
        std::vector<std::vector<CaptureRecord>> rawRecordsPerArtifact;

        sgns::sgprocessing::ExecutionManifest manifest;  ///< The job's execution manifest

        std::vector<uint8_t> combinedHash;  ///< The job's ProcessOutput.combinedHash
    };

    /// Serialize a CaptureFile to bytes: SerializeArtifact/SerializeManifest calls,
    /// unmodified, for the metadata+hash portion, plus one new length-prefixed
    /// raw-bytes section per rawOutputCapture invocation (D-01).
    /// @return Serialized bytes per the layout documented above.
    std::vector<uint8_t> SerializeCaptureFile( const CaptureFile &capture );

    /// Deserialize bytes back into a CaptureFile. Never throws; returns false
    /// (without partially populating `out`) on wrong/missing magic, any length
    /// or count field pointing past the buffer's end or exceeding the 1 GiB cap,
    /// or a truncated SerializeArtifact/SerializeManifest region (T-10-01a, T-10-02).
    /// @return true on success and a fully populated `out`; false otherwise.
    bool DeserializeCaptureFile( const std::vector<uint8_t> &bytes, CaptureFile &out );

}  // namespace sgns::sgproccapture

#endif  // SGPROCMGR_CAPTURE_FILE_FORMAT_HPP
