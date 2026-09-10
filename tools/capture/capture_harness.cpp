/**
 * capture_harness -- standalone CLI tool (Phase 10, Plan 10-05, CAPT-01/02/03).
 *
 * Runs a Phase 09 fixture --repeat N times via ProcessingManager::Process()'s
 * 5-argument ExecutionContext overload, captures per-run raw output bytes via
 * ExecutionContext::rawOutputCapture, independently re-hashes every captured
 * buffer to prove it is the literal pre-hash bytes production hashing saw
 * (CAPT-02 self-check), and verifies same-node stability across all N runs
 * (CAPT-03/D-04) before writing a single .cap file. Writes NO file if either
 * check fails (D-05).
 *
 * Not CTest-gated (Pattern 5) -- a meaningful cross-machine pass/fail needs
 * Phase 11's physical machines.
 *
 * Usage:
 *   capture_harness --fixture-root <dir> --fixture <relative-path> --label <name>
 *                    [--repeat N] [--output-dir <dir>] [--model-input-source <src>]
 *                    [--write-render-vertex-fixture]
 *
 * @brief Capture harness CLI (runs a fixture N times, self-checks, writes a .cap file)
 */
#include <algorithm>
#include <chrono>
#include <cstdint>
#include <cstdlib>
#include <ctime>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <regex>
#include <sstream>
#include <string>
#include <vector>

#include <boost/asio/io_context.hpp>
#include <boost/asio/ip/host_name.hpp>
#include <boost/system/error_code.hpp>

#if defined( __APPLE__ )
#include <TargetConditionals.h>
#endif

#include <artifacts/artifact_types.hpp>
#include <execution/execution_context.hpp>
#include <processingbase/ProcessingManager.hpp>
#include <util/sha256.hpp>

#include "capture_file_format.hpp"

namespace
{
    /// Hand-rolled CLI argument bundle -- no new parsing dependency, per
    /// Plan 10-CONTEXT.md's discretion note.
    struct CliArgs
    {
        std::string fixtureRoot;
        std::string fixture;
        std::string label;
        int         repeat                  = 3;
        std::string outputDir               = ".";
        std::string modelInputSource;
        bool        writeRenderVertexFixture = false;
    };

    void PrintUsage()
    {
        std::cerr << "Usage: capture_harness --fixture-root <dir> --fixture <relative-path> "
                     "--label <name> [--repeat N] [--output-dir <dir>] "
                     "[--model-input-source <source>] [--write-render-vertex-fixture]\n";
    }

    /// Parses argv into CliArgs.
    /// @return true on success; false (with an error already printed) on any parse failure.
    bool ParseArgs( int argc, char **argv, CliArgs &out )
    {
        for ( int i = 1; i < argc; ++i )
        {
            std::string arg = argv[i];
            if ( arg == "--fixture-root" && i + 1 < argc )
            {
                out.fixtureRoot = argv[++i];
            }
            else if ( arg == "--fixture" && i + 1 < argc )
            {
                out.fixture = argv[++i];
            }
            else if ( arg == "--label" && i + 1 < argc )
            {
                out.label = argv[++i];
            }
            else if ( arg == "--repeat" && i + 1 < argc )
            {
                out.repeat = std::atoi( argv[++i] );
            }
            else if ( arg == "--output-dir" && i + 1 < argc )
            {
                out.outputDir = argv[++i];
            }
            else if ( arg == "--model-input-source" && i + 1 < argc )
            {
                out.modelInputSource = argv[++i];
            }
            else if ( arg == "--write-render-vertex-fixture" )
            {
                out.writeRenderVertexFixture = true;
            }
            else
            {
                std::cerr << "capture_harness: unrecognized or incomplete argument: " << arg << "\n";
                return false;
            }
        }

        if ( out.fixtureRoot.empty() || out.fixture.empty() || out.label.empty() )
        {
            std::cerr << "capture_harness: --fixture-root, --fixture, and --label are required\n";
            return false;
        }
        if ( out.repeat < 2 )
        {
            std::cerr << "capture_harness: --repeat must be >= 2 (CAPT-03/D-04 requires at least "
                          "two runs to self-check stability), got "
                       << out.repeat << "\n";
            return false;
        }
        return true;
    }

    /// Standalone (no <gtest/gtest.h> dependency -- this is CLI tooling, not a GTest
    /// binary) local copy of processing_conformance_fixture.hpp's
    /// PatchJsonUrisToAbsolute, using --fixture-root in place of that function's
    /// bin_path parameter.
    std::string PatchJsonUrisToAbsolute( const std::string &jsonStr, const std::string &fixtureRoot )
    {
        std::string normalizedRoot = fixtureRoot;
        for ( auto &c : normalizedRoot )
        {
            if ( c == '\\' )
            {
                c = '/';
            }
        }
        if ( !normalizedRoot.empty() && normalizedRoot.back() != '/' )
        {
            normalizedRoot += '/';
        }

        std::string          result;
        std::regex            relativeFileUriPattern( R"delim("(file://(?!/)(?![A-Za-z]:)[^"]+)")delim" );
        size_t                lastPos = 0;
        std::sregex_iterator  iter( jsonStr.begin(), jsonStr.end(), relativeFileUriPattern );
        std::sregex_iterator  end;

        while ( iter != end )
        {
            result += jsonStr.substr( lastPos, iter->position() - lastPos );

            std::string originalUri  = ( *iter )[1].str();
            std::string relativePath = originalUri.substr( 7 ); // skip "file://"
            result += "\"file://" + normalizedRoot + relativePath + "\"";

            lastPos = iter->position() + iter->length();
            ++iter;
        }
        result += jsonStr.substr( lastPos );

        return result;
    }

    /// Writes the render-pass-happy-path fixture's vertex data (3 scalar floats),
    /// mirroring processing_dispatch_test.cpp's WriteHappyPathVertexData() exactly --
    /// this raw binary is never checked into source control.
    /// @return true on success; false (with an error already printed) on failure.
    bool WriteHappyPathVertexData( const std::string &fixtureRoot )
    {
        std::error_code        ec;
        std::filesystem::path dir = std::filesystem::path( fixtureRoot ) / "processing_dispatch";
        std::filesystem::create_directories( dir, ec );
        if ( ec )
        {
            std::cerr << "capture_harness: failed to create directory " << dir.string() << ": " << ec.message()
                       << "\n";
            return false;
        }

        std::filesystem::path file = dir / "happy-path-vertex-data.raw";
        std::ofstream          stream( file, std::ios::binary );
        if ( !stream.is_open() )
        {
            std::cerr << "capture_harness: failed to open " << file.string() << " for writing\n";
            return false;
        }

        float values[3] = { -0.5f, 0.0f, 0.5f };
        stream.write( reinterpret_cast<const char *>( values ), sizeof( values ) );
        return true;
    }

    /// Compile-time platform name (D-03 -- hostname + OS only, no GPU vendor/driver detail).
    const char *PlatformName()
    {
#if defined( _WIN32 )
        return "Windows";
#elif defined( __ANDROID__ )
        return "Android";
#elif defined( __APPLE__ )
#if defined( TARGET_OS_IPHONE ) && TARGET_OS_IPHONE
        return "iOS";
#else
        return "macOS";
#endif
#elif defined( __linux__ )
        return "Linux";
#else
        return "Unknown";
#endif
    }

    /// Machine-identity tag (D-03): "<hostname> / <OS>".
    std::string MachineIdTag()
    {
        boost::system::error_code ec;
        std::string               hostname = boost::asio::ip::host_name( ec );
        if ( ec || hostname.empty() )
        {
            hostname = "unknown-host";
        }
        return hostname + " / " + PlatformName();
    }

    /// Sanitizes a machine-identity tag for filesystem safety (D-02): spaces and '/'
    /// become '-'.
    std::string SanitizeForFilename( const std::string &tag )
    {
        std::string result = tag;
        for ( auto &c : result )
        {
            if ( c == ' ' || c == '/' )
            {
                c = '-';
            }
        }
        return result;
    }

    /// UTC timestamp formatted yyyymmddThhmmss (D-02).
    std::string UtcTimestampNow()
    {
        auto        now = std::chrono::system_clock::now();
        std::time_t t   = std::chrono::system_clock::to_time_t( now );
        std::tm     tmUtc{};
#if defined( _WIN32 )
        gmtime_s( &tmUtc, &t );
#else
        gmtime_r( &t, &tmUtc );
#endif
        std::ostringstream oss;
        oss << std::put_time( &tmUtc, "%Y%m%dT%H%M%S" );
        return oss.str();
    }

    /// One run's captured data: the structured ProcessOutput plus every
    /// rawOutputCapture record collected during that run, in call order.
    struct IterationResult
    {
        sgns::sgprocessing::ProcessOutput               output;
        std::vector<sgns::sgproccapture::CaptureRecord> records;
    };

    /// CAPT-02 self-check: independently re-hashes every captured buffer and confirms
    /// it equals the paired chunk/combined hash from the same run -- this is the
    /// literal proof that captured bytes are the same bytes production hashing
    /// actually saw, not a downstream copy (Pitfall 6).
    /// @return true if every check passes; false (with an error already printed) otherwise.
    bool SelfCheckCapturedBytes( const sgns::sgprocessing::Artifact                    &artifact,
                                 const std::vector<sgns::sgproccapture::CaptureRecord> &records,
                                 int                                                     iterationIndex )
    {
        if ( records.size() < static_cast<size_t>( artifact.chunkHashCount ) )
        {
            std::cerr << "capture_harness: iteration " << iterationIndex << " self-check failed -- captured "
                       << records.size() << " records but artifact declares " << artifact.chunkHashCount
                       << " chunk hashes\n";
            return false;
        }

        for ( uint32_t j = 0; j < artifact.chunkHashCount; ++j )
        {
            auto hash = sgns::sgprocmanagersha::sha256( records[j].quantizedBytes.data(),
                                                         records[j].quantizedBytes.size() );
            if ( !std::equal( hash.begin(), hash.end(), artifact.chunkHashes[j] ) )
            {
                std::cerr << "capture_harness: iteration " << iterationIndex
                           << " self-check failed -- re-hashed captured chunk " << j
                           << " does not match artifact.chunkHashes[" << j << "]\n";
                return false;
            }
        }

        if ( records.size() == static_cast<size_t>( artifact.chunkHashCount ) + 1 )
        {
            auto hash = sgns::sgprocmanagersha::sha256( records.back().quantizedBytes.data(),
                                                         records.back().quantizedBytes.size() );
            if ( !std::equal( hash.begin(), hash.end(), artifact.contentHash ) )
            {
                std::cerr << "capture_harness: iteration " << iterationIndex
                           << " self-check failed -- re-hashed trailing combined-level capture does not "
                              "match artifact.contentHash\n";
                return false;
            }
        }

        return true;
    }

    /// CAPT-03/D-04/D-05 stability check: compares iteration 0's contentHash,
    /// chunkHashes, and combinedHash against every other iteration's same fields.
    /// @return true if every iteration matches iteration 0; false (with an error
    /// already printed) on any divergence.
    bool CheckStability( const std::vector<IterationResult> &iterations )
    {
        const auto &baseArtifact = iterations[0].output.artifacts[0];
        const auto &baseCombined = iterations[0].output.combinedHash;

        for ( size_t i = 1; i < iterations.size(); ++i )
        {
            const auto &curArtifact = iterations[i].output.artifacts[0];

            if ( !std::equal( baseArtifact.contentHash,
                              baseArtifact.contentHash + sgns::sgprocessing::SHA256_HASH_SIZE,
                              curArtifact.contentHash ) )
            {
                std::cerr << "capture_harness: instability detected -- iteration " << i
                           << "'s contentHash diverged from iteration 0's; aborting, no capture file written\n";
                return false;
            }

            if ( baseArtifact.chunkHashCount != curArtifact.chunkHashCount )
            {
                std::cerr << "capture_harness: instability detected -- iteration " << i << "'s chunkHashCount ("
                           << curArtifact.chunkHashCount << ") diverged from iteration 0's ("
                           << baseArtifact.chunkHashCount << "); aborting, no capture file written\n";
                return false;
            }

            for ( uint32_t j = 0; j < baseArtifact.chunkHashCount; ++j )
            {
                if ( !std::equal( baseArtifact.chunkHashes[j],
                                  baseArtifact.chunkHashes[j] + sgns::sgprocessing::SHA256_HASH_SIZE,
                                  curArtifact.chunkHashes[j] ) )
                {
                    std::cerr << "capture_harness: instability detected -- iteration " << i << "'s chunkHashes["
                               << j << "] diverged from iteration 0's; aborting, no capture file written\n";
                    return false;
                }
            }

            if ( iterations[i].output.combinedHash != baseCombined )
            {
                std::cerr << "capture_harness: instability detected -- iteration " << i
                           << "'s combinedHash diverged from iteration 0's; aborting, no capture file written\n";
                return false;
            }
        }

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

    if ( args.writeRenderVertexFixture )
    {
        if ( !WriteHappyPathVertexData( args.fixtureRoot ) )
        {
            return 1;
        }
    }

    std::filesystem::path fixturePath = std::filesystem::path( args.fixtureRoot ) / args.fixture;
    std::ifstream          fixtureStream( fixturePath );
    if ( !fixtureStream.is_open() )
    {
        std::cerr << "capture_harness: could not open fixture file " << fixturePath.string() << "\n";
        return 1;
    }
    std::string rawJson( ( std::istreambuf_iterator<char>( fixtureStream ) ), std::istreambuf_iterator<char>() );
    if ( rawJson.empty() )
    {
        std::cerr << "capture_harness: fixture file " << fixturePath.string() << " is empty\n";
        return 1;
    }

    std::string patchedJson = PatchJsonUrisToAbsolute( rawJson, args.fixtureRoot );

    std::vector<IterationResult> iterations;
    iterations.reserve( static_cast<size_t>( args.repeat ) );

    for ( int i = 0; i < args.repeat; ++i )
    {
        auto mgrResult = sgns::sgprocessing::ProcessingManager::Create( patchedJson );
        if ( !mgrResult.has_value() )
        {
            std::cerr << "capture_harness: iteration " << i << ": ProcessingManager::Create failed\n";
            return 1;
        }
        auto manager = mgrResult.value();

        auto        processingData = manager->GetProcessingData();
        // Phase 01-01 (D-04): passes is schema-optional now (root required
        // relaxed so minimal ELM jobs parse); this harness always runs legacy
        // fixtures that passed the non-ELM parity gate, so the value_or shim
        // only satisfies the compiler. Materialized into a local because the
        // quicktype getter returns boost::optional<T> by value.
        const auto  passesOpt      = processingData.get_passes();
        const auto  passes         = passesOpt.value_or( std::vector<sgns::Pass>{} );
        if ( passes.empty() )
        {
            std::cerr << "capture_harness: iteration " << i << ": fixture has no passes\n";
            return 1;
        }

        sgns::ModelNode modelNode;
        auto            modelOpt = passes[0].get_model();
        if ( modelOpt.has_value() )
        {
            auto        model      = modelOpt.value();
            const auto &inputNodes = model.get_input_nodes();
            if ( inputNodes.empty() )
            {
                std::cerr << "capture_harness: iteration " << i << ": model has no input nodes\n";
                return 1;
            }
            modelNode = inputNodes[0];
        }
        else
        {
            if ( args.modelInputSource.empty() )
            {
                std::cerr << "capture_harness: fixture's pass has no model -- pass "
                              "--model-input-source (e.g. input:renderInput)\n";
                return 1;
            }
            modelNode.set_source( args.modelInputSource );
        }

        sgns::sgprocessing::ExecutionContext execCtx;
        execCtx.cancelToken.SetCallback( []() {} );

        std::vector<sgns::sgproccapture::CaptureRecord> captured;
        execCtx.rawOutputCapture = [&captured]( const std::vector<uint8_t> &quantizedBytes,
                                                const std::vector<uint8_t> &preQuantizeBytes )
        {
            sgns::sgproccapture::CaptureRecord record;
            record.quantizedBytes   = quantizedBytes;
            record.preQuantizeBytes = preQuantizeBytes;
            captured.push_back( std::move( record ) );
        };

        auto                               ioc = std::make_shared<boost::asio::io_context>();
        std::vector<std::vector<uint8_t>> chunkhashes;
        std::vector<std::string>          outputLocations;

        auto processResult = manager->Process( ioc, chunkhashes, modelNode, outputLocations, execCtx );
        if ( !processResult.has_value() )
        {
            std::cerr << "capture_harness: iteration " << i
                       << ": Process() failed: " << processResult.error().message() << "\n";
            return 1;
        }

        auto &output = processResult.value();
        if ( output.artifacts.empty() )
        {
            std::cerr << "capture_harness: iteration " << i << ": Process() produced no artifacts\n";
            return 1;
        }

        if ( !SelfCheckCapturedBytes( output.artifacts[0], captured, i ) )
        {
            return 1;
        }

        iterations.push_back( IterationResult{ std::move( output ), std::move( captured ) } );
    }

    if ( !CheckStability( iterations ) )
    {
        return 1;
    }

    sgns::sgproccapture::CaptureFile captureFile;
    captureFile.machineIdTag          = MachineIdTag();
    captureFile.fixtureLabel          = args.label;
    captureFile.artifacts             = { iterations[0].output.artifacts[0] };
    captureFile.rawRecordsPerArtifact = { iterations[0].records };
    captureFile.manifest              = iterations[0].output.manifest;
    captureFile.combinedHash          = iterations[0].output.combinedHash;

    auto serialized = sgns::sgproccapture::SerializeCaptureFile( captureFile );

    std::error_code ec;
    std::filesystem::create_directories( args.outputDir, ec ); // no-op if it already exists or is "."

    std::string filename = args.label + "_" + SanitizeForFilename( captureFile.machineIdTag ) + "_" +
                           UtcTimestampNow() + ".cap";
    std::filesystem::path outputPath = std::filesystem::path( args.outputDir ) / filename;

    std::ofstream outFile( outputPath, std::ios::binary );
    if ( !outFile.is_open() )
    {
        std::cerr << "capture_harness: failed to open output file " << outputPath.string() << " for writing\n";
        return 1;
    }
    outFile.write( reinterpret_cast<const char *>( serialized.data() ),
                   static_cast<std::streamsize>( serialized.size() ) );
    outFile.close();

    std::cout << "capture_harness: wrote " << outputPath.string() << " (" << serialized.size() << " bytes, "
              << args.repeat << "/" << args.repeat << " stable runs, "
              << iterations[0].output.artifacts[0].chunkHashCount << " chunk hashes self-checked)\n";

    return 0;
}
