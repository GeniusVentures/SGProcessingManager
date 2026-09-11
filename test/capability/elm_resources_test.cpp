// ELM resource-preflight validator tests (Plan 02-02, Task 3).
//
// Locks the Pitfall 13 / ELM-13 preflight: CapabilityValidator::CheckElmResources
// refuses an ELM acquire whose model cannot fit BEFORE any download —
// memory leg (manifest runtime.required_memory_bytes vs host RAM) and disk
// leg (total artifact bytes vs available disk) — across green / red /
// degraded-0 / absent-requirement axes.
//
// This TU includes ONLY the root generated/ quicktype set (transitively via
// capability_validator.hpp -> PassType.hpp). The manifest-extraction cases
// live in elm_resource_extraction_test.cpp (fallback generated set) — the two
// generated sets both define sgns::ClassMemberConstraints/ElmType and can
// never meet in one TU.
//
// Snapshots are injected exclusively via SetSnapshotForTest (SGPROCMGR_TEST_FRIEND,
// the capability_validator_test.cpp:99/163/194 precedent): no BuildSnapshot,
// no Vulkan/GPU query, no network.

// SGPROCMGR_TEST_FRIEND comes from target_compile_definitions in
// test/capability/CMakeLists.txt (same as the sibling target) — it unlocks
// SetSnapshotForTest below.
#include <capability/capability_validator.hpp>
#include <gtest/gtest.h>

#include <cstdint>
#include <string>

namespace sgns::sgprocessing
{
    namespace
    {

        constexpr uint64_t kKiB = 1024ULL;
        constexpr uint64_t kMiB = 1024ULL * kKiB;
        constexpr uint64_t kGiB = 1024ULL * kMiB;
        constexpr uint64_t kTiB = 1024ULL * kGiB;

        /// Ample-resources mock snapshot: 16 GiB host RAM, 100 GiB disk.
        /// No Vulkan/GPU reliance — CheckElmResources reads only the two
        /// byte axes (and identityHash for the executor id).
        CapabilitySnapshot MakeElmSnapshot( uint64_t memBytes, uint64_t diskBytes )
        {
            CapabilitySnapshot snap;
            snap.availableMemoryBytes = memBytes;
            snap.availableDiskBytes   = diskBytes;
            snap.identityHash         = { 0xab, 0xcd, 0xef, 0x01, 0x23, 0x45,
                                   0x67, 0x89, 0x00, 0x11, 0x22, 0x33,
                                   0x44, 0x55, 0x66, 0x77 };
            return snap;
        }

        /// Run the preflight and capture the single result.
        CanExecuteResult RunCheck( CapabilityValidator &validator,
                                   uint64_t            requiredMemoryBytes,
                                   uint64_t            totalArtifactBytes )
        {
            CanExecuteResult result;
            validator.CheckElmResources(
                requiredMemoryBytes, totalArtifactBytes,
                [&]( CanExecuteResult r ) { result = std::move( r ); } );
            return result;
        }

    } // anonymous namespace

    // =========================================================================
    // Red paths — one leg exceeded each
    // =========================================================================

    TEST( ElmResourcesTest, MemoryExceeded )
    {
        // 1 GiB host RAM vs a 2 GiB requirement → exactly one RESOURCE unmet
        CapabilityValidator validator;
        validator.SetSnapshotForTest( MakeElmSnapshot( 1 * kGiB, 100 * kGiB ) );

        const auto result = RunCheck( validator, 2 * kGiB, 500 * kMiB );

        EXPECT_FALSE( result.executable );
        ASSERT_EQ( result.unmet.size(), 1u );
        EXPECT_EQ( result.unmet[0].category, UnmetRequirementCategory::RESOURCE );
        EXPECT_NE( result.unmet[0].detail.find( "required_memory_bytes" ),
                   std::string::npos );
        EXPECT_TRUE( result.executorId.empty() );
    }

    TEST( ElmResourcesTest, DiskExceeded )
    {
        // 100 MB free disk vs a 500 MB artifact set → exactly one RESOURCE unmet
        CapabilityValidator validator;
        validator.SetSnapshotForTest( MakeElmSnapshot( 16 * kGiB, 100 * kMiB ) );

        const auto result = RunCheck( validator, 600 * kMiB, 500 * kMiB );

        EXPECT_FALSE( result.executable );
        ASSERT_EQ( result.unmet.size(), 1u );
        EXPECT_EQ( result.unmet[0].category, UnmetRequirementCategory::RESOURCE );
        EXPECT_NE( result.unmet[0].detail.find( "disk" ), std::string::npos );
        EXPECT_TRUE( result.executorId.empty() );
    }

    TEST( ElmResourcesTest, BothLegsExceededYieldTwoUnmet )
    {
        CapabilityValidator validator;
        validator.SetSnapshotForTest( MakeElmSnapshot( 1 * kGiB, 100 * kMiB ) );

        const auto result = RunCheck( validator, 2 * kGiB, 500 * kMiB );

        EXPECT_FALSE( result.executable );
        ASSERT_EQ( result.unmet.size(), 2u );
        EXPECT_EQ( result.unmet[0].category, UnmetRequirementCategory::RESOURCE );
        EXPECT_EQ( result.unmet[1].category, UnmetRequirementCategory::RESOURCE );
    }

    // =========================================================================
    // Green path
    // =========================================================================

    TEST( ElmResourcesTest, GreenPathAmpleResources )
    {
        // Realistic values: 600 MB model vs 16 GiB RAM; 500 MB artifacts vs
        // 100 GiB disk → executable, zero unmet, executor id populated.
        CapabilityValidator validator;
        validator.SetSnapshotForTest( MakeElmSnapshot( 16 * kGiB, 100 * kGiB ) );

        const auto result = RunCheck( validator, 600 * kMiB, 500 * kMiB );

        EXPECT_TRUE( result.executable );
        EXPECT_TRUE( result.unmet.empty() );
        EXPECT_FALSE( result.executorId.empty() );
        EXPECT_EQ( result.executorId.find( "sgproc-" ), 0u );
    }

    // =========================================================================
    // Degraded axes — query failure (field == 0) skips the leg, never fails
    // =========================================================================

    TEST( ElmResourcesTest, DegradedMemorySkipsMemoryLeg )
    {
        // Host-RAM query failed (0) + an impossible requirement → NO memory
        // unmet; disk fine → fully green despite the huge requirement.
        CapabilityValidator validator;
        validator.SetSnapshotForTest( MakeElmSnapshot( 0, 100 * kGiB ) );

        const auto result = RunCheck( validator, 512 * kGiB, 500 * kMiB );

        EXPECT_TRUE( result.executable );
        EXPECT_TRUE( result.unmet.empty() );
    }

    TEST( ElmResourcesTest, DegradedDiskSkipsDiskLeg )
    {
        // Disk query failed (0) + an impossible artifact set → NO disk unmet.
        CapabilityValidator validator;
        validator.SetSnapshotForTest( MakeElmSnapshot( 16 * kGiB, 0 ) );

        const auto result = RunCheck( validator, 600 * kMiB, 100 * kTiB );

        EXPECT_TRUE( result.executable );
        EXPECT_TRUE( result.unmet.empty() );
    }

    TEST( ElmResourcesTest, BothAxesDegradedStillGreen )
    {
        CapabilityValidator validator;
        validator.SetSnapshotForTest( MakeElmSnapshot( 0, 0 ) );

        const auto result = RunCheck( validator, 8 * kGiB, 8 * kGiB );

        EXPECT_TRUE( result.executable );
        EXPECT_TRUE( result.unmet.empty() );
    }

    // =========================================================================
    // Absent requirement — requiredMemoryBytes == 0 never fires the memory leg
    // =========================================================================

    TEST( ElmResourcesTest, ZeroRequirementSkipsMemoryLeg )
    {
        // Even against a tiny 1 GiB host, a 0 (absent) requirement is green.
        CapabilityValidator validator;
        validator.SetSnapshotForTest( MakeElmSnapshot( 1 * kGiB, 100 * kGiB ) );

        const auto result = RunCheck( validator, 0, 500 * kMiB );

        EXPECT_TRUE( result.executable );
        EXPECT_TRUE( result.unmet.empty() );
    }

    TEST( ElmResourcesTest, ZeroArtifactsSkipsDiskLeg )
    {
        // Zero-byte artifact set against a tiny disk stays green.
        CapabilityValidator validator;
        validator.SetSnapshotForTest( MakeElmSnapshot( 16 * kGiB, 1 * kMiB ) );

        const auto result = RunCheck( validator, 600 * kMiB, 0 );

        EXPECT_TRUE( result.executable );
        EXPECT_TRUE( result.unmet.empty() );
    }

    // =========================================================================
    // Uninitialized validator — same fail-closed shape as CanExecute
    // =========================================================================

    TEST( ElmResourcesTest, RejectBeforeBuildSnapshot )
    {
        CapabilityValidator validator; // no SetSnapshotForTest

        const auto result = RunCheck( validator, 1, 1 );

        EXPECT_FALSE( result.executable );
        ASSERT_GE( result.unmet.size(), 1u );
        EXPECT_EQ( result.unmet[0].category, UnmetRequirementCategory::RESOURCE );
        EXPECT_NE( result.unmet[0].detail.find( "not initialized" ), std::string::npos );
    }

    // =========================================================================
    // Boundary — requirement exactly equal to availability is green (>)
    // =========================================================================

    TEST( ElmResourcesTest, ExactFitIsGreen )
    {
        CapabilityValidator validator;
        validator.SetSnapshotForTest( MakeElmSnapshot( 2 * kGiB, 500 * kMiB ) );

        const auto result = RunCheck( validator, 2 * kGiB, 500 * kMiB );

        EXPECT_TRUE( result.executable );
        EXPECT_TRUE( result.unmet.empty() );
    }

} // namespace sgns::sgprocessing
