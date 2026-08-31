/**
 * capture_diff -- standalone CLI tool (Phase 10, Plan 10-05, DIFF-01/02/03).
 *
 * Reads two .cap files (produced by capture_harness, possibly on different
 * machines) and reports quantitative per-element divergence (absolute delta,
 * relative delta, ULP distance, whole-buffer summary stats) plus independent
 * hash-match booleans, to both console and a JSON report (D-06).
 *
 * Not CTest-gated (Pattern 5) -- a meaningful cross-machine pass/fail needs
 * Phase 11's physical machines.
 *
 * Update (Phase 13 gap-closure, Plan 13-06): the per-element numeric-diff pass
 * described above originally examined only the trailing combined-hash capture
 * record. It now ADDITIONALLY numeric-diffs each individual per-chunk raw
 * capture record (rawRecordsPerArtifact[0][j] for j < chunkHashCount) via the
 * new `chunkDiffs` JSON output array (index-aligned with `chunkHashesMatch`),
 * closing the blind spot where a `chunkHashesMatch[j]: false` result carried
 * no magnitude information. The original trailing-record pass is unchanged.
 *
 * Usage:
 *   capture_diff --a <path> --b <path> --element-type <float32|uint8>
 *                [--json-output <path>]
 *
 * @brief Capture diff CLI (compares two .cap files, reports divergence stats)
 */
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <iostream>
#include <string>
#include <vector>

#include <nlohmann/json.hpp>

#include "capture_file_format.hpp"
#include "util/diff_utils.hpp"

namespace
{
    struct CliArgs
    {
        std::string pathA;
        std::string pathB;
        std::string elementType; // "float32" or "uint8"
        std::string jsonOutput = "diff_report.json";
    };

    void PrintUsage()
    {
        std::cerr << "Usage: capture_diff --a <path> --b <path> --element-type <float32|uint8> "
                     "[--json-output <path>]\n";
    }

    /// Parses argv into CliArgs.
    /// @return true on success; false (with an error already printed) on any parse failure.
    bool ParseArgs( int argc, char **argv, CliArgs &out )
    {
        for ( int i = 1; i < argc; ++i )
        {
            std::string arg = argv[i];
            if ( arg == "--a" && i + 1 < argc )
            {
                out.pathA = argv[++i];
            }
            else if ( arg == "--b" && i + 1 < argc )
            {
                out.pathB = argv[++i];
            }
            else if ( arg == "--element-type" && i + 1 < argc )
            {
                out.elementType = argv[++i];
            }
            else if ( arg == "--json-output" && i + 1 < argc )
            {
                out.jsonOutput = argv[++i];
            }
            else
            {
                std::cerr << "capture_diff: unrecognized or incomplete argument: " << arg << "\n";
                return false;
            }
        }

        if ( out.pathA.empty() || out.pathB.empty() )
        {
            std::cerr << "capture_diff: --a and --b are required\n";
            return false;
        }
        if ( out.elementType != "float32" && out.elementType != "uint8" )
        {
            std::cerr << "capture_diff: --element-type must be exactly \"float32\" or \"uint8\", got \""
                       << out.elementType << "\"\n";
            return false;
        }
        return true;
    }

    /// Reads an entire file into a byte vector.
    /// @return true on success; false (with an error already printed) if the file cannot be opened.
    bool ReadFileBytes( const std::string &path, std::vector<uint8_t> &out )
    {
        std::ifstream stream( path, std::ios::binary );
        if ( !stream.is_open() )
        {
            std::cerr << "capture_diff: could not open file " << path << "\n";
            return false;
        }
        out.assign( std::istreambuf_iterator<char>( stream ), std::istreambuf_iterator<char>() );
        return true;
    }

} // namespace

int main( int argc, char **argv )
{
    CliArgs args;
    if ( !ParseArgs( argc, argv, args ) )
    {
        PrintUsage();
        return 1;
    }

    std::vector<uint8_t> bytesA;
    std::vector<uint8_t> bytesB;
    if ( !ReadFileBytes( args.pathA, bytesA ) || !ReadFileBytes( args.pathB, bytesB ) )
    {
        return 1;
    }

    sgns::sgproccapture::CaptureFile captureA;
    sgns::sgproccapture::CaptureFile captureB;
    if ( !sgns::sgproccapture::DeserializeCaptureFile( bytesA, captureA ) )
    {
        std::cerr << "capture_diff: failed to parse capture file " << args.pathA << " (malformed or truncated)\n";
        return 1;
    }
    if ( !sgns::sgproccapture::DeserializeCaptureFile( bytesB, captureB ) )
    {
        std::cerr << "capture_diff: failed to parse capture file " << args.pathB << " (malformed or truncated)\n";
        return 1;
    }

    if ( captureA.artifacts.size() != 1 || captureB.artifacts.size() != 1 )
    {
        std::cerr << "capture_diff: expected exactly one artifact per capture file (Phase 10 scope), got "
                   << captureA.artifacts.size() << " in " << args.pathA << " and " << captureB.artifacts.size()
                   << " in " << args.pathB << "\n";
        return 1;
    }

    const auto &artifactA = captureA.artifacts[0];
    const auto &artifactB = captureB.artifacts[0];

    if ( artifactA.chunkHashCount != artifactB.chunkHashCount )
    {
        std::cerr << "capture_diff: chunkHashCount mismatch (a=" << artifactA.chunkHashCount
                   << ", b=" << artifactB.chunkHashCount
                   << ") -- the two capture files are not comparable (likely different fixtures or a "
                      "structural divergence, not a numeric one); distinct failure mode, not reported as "
                      "0% divergence\n";
        return 1;
    }

    // DIFF-03: hash-match booleans, computed purely from artifact/manifest metadata.
    bool contentHashMatch = std::equal( artifactA.contentHash,
                                        artifactA.contentHash + sgns::sgprocessing::SHA256_HASH_SIZE,
                                        artifactB.contentHash );

    std::vector<bool> chunkHashesMatch;
    uint32_t          sharedChunkCount = std::min( artifactA.chunkHashCount, artifactB.chunkHashCount );
    chunkHashesMatch.reserve( sharedChunkCount );
    for ( uint32_t j = 0; j < sharedChunkCount; ++j )
    {
        chunkHashesMatch.push_back( std::equal( artifactA.chunkHashes[j],
                                                artifactA.chunkHashes[j] + sgns::sgprocessing::SHA256_HASH_SIZE,
                                                artifactB.chunkHashes[j] ) );
    }

    bool combinedHashMatch = captureA.combinedHash == captureB.combinedHash;

    // DIFF-01/02: per-element numeric divergence over the LAST CaptureRecord's
    // quantizedBytes -- the same bytes that fed each run's contentHash.
    // Update (Phase 13 gap-closure, Plan 13-06): capture_diff now ALSO
    // numeric-diffs each individual per-chunk raw record below (see
    // `chunkStats`/`chunkDiffs`) -- this trailing-record-only pass is
    // preserved unchanged as its own distinct stat.
    sgns::sgprocmanagerdiff::ElementDiffStats stats;
    bool             haveRecords = !captureA.rawRecordsPerArtifact.empty() && !captureB.rawRecordsPerArtifact.empty() &&
                        !captureA.rawRecordsPerArtifact[0].empty() && !captureB.rawRecordsPerArtifact[0].empty();

    if ( !haveRecords )
    {
        std::cerr << "capture_diff: one or both capture files have no raw capture records for artifact 0 -- "
                     "skipping per-element numeric pass\n";
        stats.sizeMismatch = true;
    }
    else
    {
        const auto &lastRecordA = captureA.rawRecordsPerArtifact[0].back();
        const auto &lastRecordB = captureB.rawRecordsPerArtifact[0].back();

        if ( args.elementType == "float32" )
        {
            stats = sgns::sgprocmanagerdiff::ComputeFloat32Diff( lastRecordA.quantizedBytes, lastRecordB.quantizedBytes );
        }
        else
        {
            stats = sgns::sgprocmanagerdiff::ComputeUint8Diff( lastRecordA.quantizedBytes, lastRecordB.quantizedBytes );
        }

        if ( stats.sizeMismatch )
        {
            std::cerr << "capture_diff: size mismatch between the two files' final capture record bytes ("
                       << lastRecordA.quantizedBytes.size() << " vs " << lastRecordB.quantizedBytes.size()
                       << ") -- skipping per-element numeric pass (hash-match booleans above are still valid)\n";
        }
    }

    // DIFF-01/02 extension (Phase 13 gap-closure, Plan 13-06): per-chunk numeric
    // divergence over each individual rawRecordsPerArtifact[0][j] record
    // (j < chunkHashCount), closing the blind spot where chunkHashesMatch[j]
    // could report a divergence without ever reporting its magnitude. Reuses
    // ComputeFloat32Diff/ComputeUint8Diff unmodified -- only the caller loop
    // and its per-chunk inputs are new.
    std::vector<sgns::sgprocmanagerdiff::ElementDiffStats> chunkStats;
    chunkStats.reserve( chunkHashesMatch.size() );
    bool haveArtifactZeroRecords = !captureA.rawRecordsPerArtifact.empty() && !captureB.rawRecordsPerArtifact.empty();
    for ( size_t j = 0; j < chunkHashesMatch.size(); ++j )
    {
        bool haveChunkRecords = haveArtifactZeroRecords && captureA.rawRecordsPerArtifact[0].size() > j &&
                                 captureB.rawRecordsPerArtifact[0].size() > j;
        if ( !haveChunkRecords )
        {
            std::cerr << "capture_diff: chunk " << j
                       << " has no raw capture record in one or both files -- skipping its per-chunk numeric pass\n";
            sgns::sgprocmanagerdiff::ElementDiffStats missing;
            missing.sizeMismatch = true;
            chunkStats.push_back( missing );
            continue;
        }

        const auto &chunkRecordA = captureA.rawRecordsPerArtifact[0][j];
        const auto &chunkRecordB = captureB.rawRecordsPerArtifact[0][j];

        if ( args.elementType == "float32" )
        {
            chunkStats.push_back( sgns::sgprocmanagerdiff::ComputeFloat32Diff( chunkRecordA.quantizedBytes, chunkRecordB.quantizedBytes ) );
        }
        else
        {
            chunkStats.push_back( sgns::sgprocmanagerdiff::ComputeUint8Diff( chunkRecordA.quantizedBytes, chunkRecordB.quantizedBytes ) );
        }
    }

    // Console output.
    std::cout << "capture_diff: comparing " << args.pathA << " vs " << args.pathB << " (element-type "
              << args.elementType << ")\n";
    std::cout << "  contentHashMatch:   " << ( contentHashMatch ? "true" : "false" ) << "\n";
    std::cout << "  combinedHashMatch:  " << ( combinedHashMatch ? "true" : "false" ) << "\n";
    std::cout << "  chunkHashesMatch:   [";
    for ( size_t j = 0; j < chunkHashesMatch.size(); ++j )
    {
        std::cout << ( chunkHashesMatch[j] ? "true" : "false" );
        if ( j + 1 < chunkHashesMatch.size() )
        {
            std::cout << ", ";
        }
    }
    std::cout << "]\n";
    for ( size_t j = 0; j < chunkStats.size(); ++j )
    {
        std::cout << "  chunk[" << j << "] match=" << ( chunkHashesMatch[j] ? "true" : "false" );
        if ( chunkStats[j].sizeMismatch )
        {
            std::cout << " sizeMismatch=true (per-chunk numeric pass skipped)\n";
        }
        else
        {
            std::cout << " maxAbsDelta=" << chunkStats[j].maxAbsDelta << " maxRelDelta=" << chunkStats[j].maxRelDelta
                       << " maxUlpDistance=" << chunkStats[j].maxUlpDistance << "\n";
        }
    }
    if ( stats.sizeMismatch )
    {
        std::cout << "  sizeMismatch:       true (per-element numeric pass skipped)\n";
    }
    else
    {
        std::cout << "  elementCount:               " << stats.elementCount << "\n";
        std::cout << "  maxAbsDelta:                " << stats.maxAbsDelta << "\n";
        std::cout << "  maxRelDelta:                " << stats.maxRelDelta << "\n";
        std::cout << "  maxUlpDistance:             " << stats.maxUlpDistance << "\n";
        std::cout << "  percentExceedingThreshold:  " << stats.percentExceedingThreshold << "%\n";
    }

    // JSON output (D-06 -- both console and JSON, not deferred).
    nlohmann::json report;
    report["elementType"]               = args.elementType;
    report["elementCount"]              = stats.elementCount;
    report["maxAbsDelta"]               = stats.maxAbsDelta;
    report["maxRelDelta"]               = stats.maxRelDelta;
    report["maxUlpDistance"]            = stats.maxUlpDistance;
    report["percentExceedingThreshold"] = stats.percentExceedingThreshold;
    report["sizeMismatch"]              = stats.sizeMismatch;
    report["contentHashMatch"]          = contentHashMatch;
    report["combinedHashMatch"]         = combinedHashMatch;
    report["chunkHashesMatch"]          = chunkHashesMatch;

    report["chunkDiffs"] = nlohmann::json::array();
    for ( size_t j = 0; j < chunkStats.size(); ++j )
    {
        nlohmann::json chunkEntry;
        chunkEntry["chunkIndex"]                = j;
        chunkEntry["elementCount"]              = chunkStats[j].elementCount;
        chunkEntry["maxAbsDelta"]               = chunkStats[j].maxAbsDelta;
        chunkEntry["maxRelDelta"]               = chunkStats[j].maxRelDelta;
        chunkEntry["maxUlpDistance"]            = chunkStats[j].maxUlpDistance;
        chunkEntry["percentExceedingThreshold"] = chunkStats[j].percentExceedingThreshold;
        chunkEntry["sizeMismatch"]              = chunkStats[j].sizeMismatch;
        report["chunkDiffs"].push_back( chunkEntry );
    }

    std::ofstream jsonStream( args.jsonOutput );
    if ( !jsonStream.is_open() )
    {
        std::cerr << "capture_diff: failed to open " << args.jsonOutput << " for writing\n";
        return 1;
    }
    jsonStream << report.dump( 2 );
    jsonStream.close();

    std::cout << "capture_diff: wrote " << args.jsonOutput << "\n";

    return 0;
}
