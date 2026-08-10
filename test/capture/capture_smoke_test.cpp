/**
 * capture_smoke_test -- CTest-registered smoke test (Phase 10, Plan 10-06, CAPT-01).
 *
 * Proves capture_harness actually builds, runs, and produces a well-formed,
 * round-trippable .cap file in ordinary CI -- it deliberately does NOT assert
 * cross-machine hash equality (Pattern 5, ARCHITECTURE.md); that empirical,
 * multi-machine comparison is Phase 11's manual job.
 *
 * @brief Smoke test: runs capture_harness as a subprocess and round-trips its output
 */
#include <gtest/gtest.h>

#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <string>
#include <vector>

#include <processors/vulkan_gpu_probe.hpp>

#include "tools/capture/capture_file_format.hpp"

namespace
{
    /// Reads an entire file's bytes into memory.
    std::vector<uint8_t> ReadAllBytes( const std::filesystem::path &path )
    {
        std::ifstream stream( path, std::ios::binary );
        return std::vector<uint8_t>( ( std::istreambuf_iterator<char>( stream ) ),
                                      std::istreambuf_iterator<char>() );
    }
} // namespace

TEST( CaptureSmokeTest, HarnessProducesWellFormedFile )
{
    if ( !sgns::sgprocessing::HasUsableVulkanDevice() )
    {
        GTEST_SKIP() << "No usable Vulkan device found on this host; skipping this GPU-dependent "
                        "smoke test, not failing it.";
    }

    const std::string kLabel = "smoke-mnn-float";

    std::filesystem::path outputDir( OUTPUT_DIR_PATH );
    std::string           prefix = kLabel + "_";

    // Capture filenames are timestamped (D-02) so capture_harness never overwrites a
    // prior run's file -- but that means repeated local/CI runs of this test against
    // the same OUTPUT_DIR accumulate stale *.cap files from earlier runs, and the
    // "exactly one" assertion below would then match all of them, not just this run's.
    // Remove any pre-existing matches first so this test is idempotent across reruns.
    for ( const auto &entry : std::filesystem::directory_iterator( outputDir ) )
    {
        std::string name = entry.path().filename().string();
        if ( name.rfind( prefix, 0 ) == 0 && name.size() >= 4 && name.substr( name.size() - 4 ) == ".cap" )
        {
            std::filesystem::remove( entry.path() );
        }
    }

    std::string command = std::string( CAPTURE_HARNESS_PATH ) + " --fixture-root \"" + FIXTURE_ROOT_PATH +
                          "\" --fixture processing_datatypes/float-processing-definition.json --label " +
                          kLabel + " --repeat 2 --output-dir \"" + OUTPUT_DIR_PATH + "\"";

    int rc = std::system( command.c_str() );
    ASSERT_EQ( rc, 0 ) << "capture_harness exited non-zero";

    std::vector<std::filesystem::path> matches;
    for ( const auto &entry : std::filesystem::directory_iterator( outputDir ) )
    {
        std::string name = entry.path().filename().string();
        if ( name.rfind( prefix, 0 ) == 0 && name.size() >= 4 && name.substr( name.size() - 4 ) == ".cap" )
        {
            matches.push_back( entry.path() );
        }
    }

    ASSERT_EQ( matches.size(), 1u ) << "Expected exactly one " << prefix << "*.cap file in " << outputDir.string();

    std::error_code ec;
    auto             fileSize = std::filesystem::file_size( matches[0], ec );
    ASSERT_FALSE( ec ) << "Failed to stat " << matches[0].string() << ": " << ec.message();
    ASSERT_GT( fileSize, 0u ) << matches[0].string() << " is empty";

    std::vector<uint8_t> bytes = ReadAllBytes( matches[0] );

    sgns::sgproccapture::CaptureFile out;
    ASSERT_TRUE( sgns::sgproccapture::DeserializeCaptureFile( bytes, out ) )
        << "DeserializeCaptureFile failed to round-trip " << matches[0].string();

    EXPECT_EQ( out.artifacts.size(), 1u );
    EXPECT_EQ( out.combinedHash.size(), 32u );
}
