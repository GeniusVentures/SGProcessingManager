// ELM resource-preflight manifest-extraction tests (Plan 02-02, Task 3).
//
// Pins the composition the Phase 3 processor will wire:
// sgns::elmruntime::ExtractElmResourceRequirements(manifest) →
// CapabilityValidator::CheckElmResources(values...). The extraction cases
// build manifests in-memory via generated setters ONLY (no JSON parsing) and
// assert the extracted values; the validator legs (mock snapshots) live in
// elm_resources_test.cpp.
//
// This TU includes ONLY the fallback generated/elmruntime-manifest/ quicktype
// set. The root generated/ set (PassType.hpp et al., required by capability
// headers) and this set both define sgns::ClassMemberConstraints/ElmType —
// they can never meet in one translation unit. The validator's manifest
// overload was dropped for exactly this reason (see plan fallback note).

#include <gtest/gtest.h>

// Quoted include resolved against generated/elmruntime-manifest/ ONLY —
// never a global <ElmModelManifest.hpp> (mixing the sets is a redefinition
// error; same discipline as ElmManifest.cpp). ElmType.hpp is included
// explicitly: ElmModelManifest.hpp only forward-declares the enum, and this
// TU sets it (ElmManifest.cpp gets the definition via Generators.hpp).
#include "elmruntime-manifest/ElmModelManifest.hpp"
#include "elmruntime-manifest/ElmType.hpp"

#include <elmruntime/ElmResourcePreflight.hpp>

#include <cstdint>
#include <string>
#include <vector>

namespace
{

    constexpr uint64_t kKiB = 1024ULL;
    constexpr uint64_t kMiB = 1024ULL * kKiB;
    constexpr uint64_t kGiB = 1024ULL * kMiB;

    /// 64-hex placeholder — the extraction under test never hashes, so any
    /// schema-valid string works (generated setters only, no JSON parsing).
    std::string AnyHash()
    {
        return std::string( 64, 'a' );
    }

    /// One artifact via generated setters (role, uri, size; fixed hash).
    sgns::ElmModelArtifact MakeArtifact( const std::string &role, int64_t sizeBytes )
    {
        sgns::ElmModelArtifact artifact;
        artifact.set_name( role );
        artifact.set_uri( "ipfs://artifact-" + role );
        artifact.set_sha256( AnyHash() );
        artifact.set_size_bytes( sizeBytes );
        return artifact;
    }

    /// Manifest via generated setters: fixed top fields + artifacts.
    sgns::ElmModelManifest MakeBaseManifest(
        const std::vector<std::pair<std::string, int64_t>> &artifacts )
    {
        sgns::ElmModelManifest manifest;
        manifest.set_schema_version( 1 );
        manifest.set_elm_type( sgns::ElmType::CAUSAL_LM );
        manifest.set_model_format( "mnn" );

        std::vector<sgns::ElmModelArtifact> entries;
        entries.reserve( artifacts.size() );
        for ( const auto &[role, size] : artifacts )
        {
            entries.push_back( MakeArtifact( role, size ) );
        }
        manifest.set_artifacts( entries );
        return manifest;
    }

} // anonymous namespace

TEST( ElmResourceExtractionTest, ExtractsRuntimeBlockAndArtifactSum )
{
    auto manifest = MakeBaseManifest(
        { { "llm_config", 2 * kKiB },
          { "llm_model", 600 * kMiB },
          { "llm_weight", 100 * kMiB },
          { "tokenizer_file", 5 * kMiB } } );

    sgns::ElmModelRuntime runtime;
    runtime.set_required_memory_bytes(
        boost::optional<int64_t>( static_cast<int64_t>( 2 * kGiB ) ) );
    manifest.set_runtime( boost::optional<sgns::ElmModelRuntime>( runtime ) );

    const auto reqs = sgns::elmruntime::ExtractElmResourceRequirements( manifest );

    EXPECT_EQ( reqs.requiredMemoryBytes, 2 * kGiB );
    EXPECT_EQ( reqs.totalArtifactBytes,
               2 * kKiB + 600 * kMiB + 100 * kMiB + 5 * kMiB );
}

TEST( ElmResourceExtractionTest, AbsentRuntimeBlockYieldsZeroRequirement )
{
    // No runtime block at all → memory leg never fires (value_or(0)).
    auto manifest = MakeBaseManifest(
        { { "llm_config", 1 },
          { "llm_model", 300 * kMiB },
          { "llm_weight", 100 * kMiB },
          { "tokenizer_file", 1 * kMiB } } );

    const auto reqs = sgns::elmruntime::ExtractElmResourceRequirements( manifest );

    EXPECT_EQ( reqs.requiredMemoryBytes, 0u );
    EXPECT_EQ( reqs.totalArtifactBytes,
               1u + 300 * kMiB + 100 * kMiB + 1 * kMiB );
}

TEST( ElmResourceExtractionTest, UnsetRequiredMemoryFieldYieldsZero )
{
    // Runtime block present but required_memory_bytes unset → 0.
    auto manifest = MakeBaseManifest(
        { { "llm_config", 1 },
          { "llm_model", 1 },
          { "llm_weight", 1 },
          { "tokenizer_file", 1 } } );

    sgns::ElmModelRuntime runtime; // field left unset
    manifest.set_runtime( boost::optional<sgns::ElmModelRuntime>( runtime ) );

    const auto reqs = sgns::elmruntime::ExtractElmResourceRequirements( manifest );

    EXPECT_EQ( reqs.requiredMemoryBytes, 0u );
}

TEST( ElmResourceExtractionTest, ZeroRequiredMemoryFieldYieldsZero )
{
    // Explicit 0 → treated as absent (never fires the memory leg).
    auto manifest = MakeBaseManifest(
        { { "llm_config", 1 },
          { "llm_model", 1 },
          { "llm_weight", 1 },
          { "tokenizer_file", 1 } } );

    sgns::ElmModelRuntime runtime;
    runtime.set_required_memory_bytes( boost::optional<int64_t>( 0 ) );
    manifest.set_runtime( boost::optional<sgns::ElmModelRuntime>( runtime ) );

    const auto reqs = sgns::elmruntime::ExtractElmResourceRequirements( manifest );

    EXPECT_EQ( reqs.requiredMemoryBytes, 0u );
}

TEST( ElmResourceExtractionTest, NegativeRequiredMemoryClampsToZero )
{
    // The schema minimum does not survive codegen; negative values would wrap
    // to ~2^64 as uint64_t — extraction clamps them to 0 instead.
    auto manifest = MakeBaseManifest(
        { { "llm_config", 1 },
          { "llm_model", 1 },
          { "llm_weight", 1 },
          { "tokenizer_file", 1 } } );

    sgns::ElmModelRuntime runtime;
    runtime.set_required_memory_bytes( boost::optional<int64_t>( -4096 ) );
    manifest.set_runtime( boost::optional<sgns::ElmModelRuntime>( runtime ) );

    const auto reqs = sgns::elmruntime::ExtractElmResourceRequirements( manifest );

    EXPECT_EQ( reqs.requiredMemoryBytes, 0u );
}

TEST( ElmResourceExtractionTest, NegativeArtifactSizesExcludedFromSum )
{
    // Defensive: negative size_bytes never subtracts from the total.
    auto manifest = MakeBaseManifest(
        { { "llm_config", 2 * kKiB },
          { "llm_model", 300 * kMiB },
          { "llm_weight", 100 * kMiB },
          { "tokenizer_file", 5 * kMiB } } );

    auto mutableArts = manifest.get_mutable_artifacts();
    mutableArts[0].set_size_bytes( -1024 );
    manifest.set_artifacts( mutableArts );

    const auto reqs = sgns::elmruntime::ExtractElmResourceRequirements( manifest );

    EXPECT_EQ( reqs.totalArtifactBytes,
               300 * kMiB + 100 * kMiB + 5 * kMiB );
    EXPECT_EQ( reqs.requiredMemoryBytes, 0u );
}

TEST( ElmResourceExtractionTest, ExtractedValuesFeedValidatorRedPath )
{
    // The Phase 3 composition, pinned at the value seam (the validator call
    // itself lives in elm_resources_test.cpp — the two generated sets cannot
    // meet in one TU): a 2 GiB requirement extracted from a manifest must be
    // large enough to trip CheckElmResources' memory leg against a 1 GiB
    // host, i.e. the exact comparison the validator performs.
    auto manifest = MakeBaseManifest(
        { { "llm_config", 2 * kKiB },
          { "llm_model", 600 * kMiB },
          { "llm_weight", 100 * kMiB },
          { "tokenizer_file", 5 * kMiB } } );

    sgns::ElmModelRuntime runtime;
    runtime.set_required_memory_bytes(
        boost::optional<int64_t>( static_cast<int64_t>( 2 * kGiB ) ) );
    manifest.set_runtime( boost::optional<sgns::ElmModelRuntime>( runtime ) );

    const auto reqs = sgns::elmruntime::ExtractElmResourceRequirements( manifest );

    // The validator's memory-leg predicate (capability_validator.cpp):
    // requiredMemoryBytes > 0 && availableMemoryBytes > 0 &&
    // requiredMemoryBytes > availableMemoryBytes — true for 2 GiB vs 1 GiB.
    const uint64_t oneGiBHost = 1 * kGiB;
    ASSERT_GT( reqs.requiredMemoryBytes, 0u );
    ASSERT_GT( oneGiBHost, 0u );
    EXPECT_TRUE( reqs.requiredMemoryBytes > oneGiBHost );

    // ...and the disk-leg predicate against an ample 100 GiB disk stays green.
    const uint64_t hundredGiBDisk = 100 * kGiB;
    EXPECT_FALSE( reqs.totalArtifactBytes > hundredGiBDisk );
}
