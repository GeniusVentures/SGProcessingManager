/**
 * Unit tests for CapabilityValidator — all rejection categories + acceptance path.
 *
 * Uses mock CapabilitySnapshots via SetSnapshotForTest (no real GPU needed).
 * Tests CAP-01 through CAP-06 per Plan 06-03.
 */

#define SGPROCMGR_TEST_FRIEND
#include <capability/capability_validator.hpp>
#include <ColorFormat.hpp>
#include <DepthFormat.hpp>
#include <ModelFormat.hpp>
#include <gtest/gtest.h>

namespace sgns::sgprocessing
{
    namespace
    {

        /// Build a minimal mock snapshot with a single RENDER executor.
        CapabilitySnapshot MakeMockSnapshot()
        {
            CapabilitySnapshot snap;

            // Vulkan device props — a "mock" DISCRETE_GPU with generous limits
            snap.vulkanProps.deviceType = VK_PHYSICAL_DEVICE_TYPE_DISCRETE_GPU;
            snap.vulkanProps.limits.maxImageDimension2D      = 16384;
            snap.vulkanProps.limits.maxColorAttachments       = 8;
            snap.vulkanProps.limits.maxMemoryAllocationCount  = 4096;
            snap.vulkanProps.deviceID   = 0x1234;
            snap.vulkanProps.vendorID   = 0x10DE;
            snap.vulkanProps.driverVersion = 0x80000001;
            std::strncpy( snap.vulkanProps.deviceName, "Mock GPU", VK_MAX_PHYSICAL_DEVICE_NAME_SIZE );

            // Memory: 1 GB device-local heap
            snap.memProps.memoryHeapCount = 2;
            snap.memProps.memoryHeaps[0].size  = 1024ULL * 1024 * 1024; // 1 GB device-local
            snap.memProps.memoryHeaps[0].flags = VK_MEMORY_HEAP_DEVICE_LOCAL_BIT;
            snap.memProps.memoryHeaps[1].size  = 8ULL * 1024 * 1024 * 1024; // 8 GB host
            snap.memProps.memoryHeaps[1].flags = 0;

            // Executor: RENDER only
            ExecutorCapability cap;
            cap.passType = PassType::RENDER;
            cap.backend  = "VULKAN";
            snap.executorCaps.push_back( cap );

            // Plenty of disk
            snap.availableDiskBytes = 100ULL * 1024 * 1024 * 1024; // 100 GB

            // Dummy identity hash
            snap.identityHash = { 0xab, 0xcd, 0xef, 0x01, 0x23, 0x45, 0x67, 0x89,
                                   0x00, 0x11, 0x22, 0x33, 0x44, 0x55, 0x66, 0x77 };

            return snap;
        }

        /// Helper: create a minimal mock RENDER Pass.
        sgns::Pass MakeMockRenderPass( int64_t width = 256, int64_t height = 256 )
        {
            sgns::Pass pass;
            pass.set_type( PassType::RENDER );

            sgns::RenderTarget rt;
            rt.set_width( width );
            rt.set_height( height );
            rt.set_color_format( sgns::ColorFormat::RGBA8 );
            rt.set_depth_format( sgns::DepthFormat::D32_SFLOAT );
            pass.set_render_target( rt );

            return pass;
        }

        /// Helper: create a minimal mock INFERENCE Pass.
        sgns::Pass MakeMockInferencePass( sgns::ModelFormat fmt = sgns::ModelFormat::MNN )
        {
            sgns::Pass pass;
            pass.set_type( PassType::INFERENCE );

            sgns::ModelConfig model;
            model.set_format( fmt );
            model.set_source_uri_param( "model.mnn" );
            pass.set_model( model );

            return pass;
        }

    } // anonymous namespace

    // =========================================================================
    // Test fixture
    // =========================================================================

    class CapabilityValidatorTest : public ::testing::Test
    {
    protected:
        void SetUp() override
        {
            validator.SetSnapshotForTest( MakeMockSnapshot() );
        }

        CapabilityValidator validator;
    };

    // =========================================================================
    // CAP-04: PassType registration
    // =========================================================================

    TEST_F( CapabilityValidatorTest, RejectUnregisteredPassType )
    {
        sgns::Pass pass;
        pass.set_type( PassType::INFERENCE ); // not in mock snapshot

        bool     called = false;
        CanExecuteResult result;
        validator.CanExecute( pass, [&]( CanExecuteResult r )
        {
            called = true;
            result = std::move( r );
        } );

        ASSERT_TRUE( called );
        EXPECT_FALSE( result.executable );
        ASSERT_EQ( result.unmet.size(), 1u );
        EXPECT_EQ( result.unmet[0].category, UnmetRequirementCategory::PASS_TYPE );
        EXPECT_TRUE( result.unmet[0].detail.find( "INFERENCE" ) != std::string::npos
                     || result.unmet[0].detail.find( "1" ) != std::string::npos );
        EXPECT_TRUE( result.unmet[0].detail.find( "Available" ) != std::string::npos );
        EXPECT_TRUE( result.executorId.empty() );
    }

    // =========================================================================
    // CAP-02: Vulkan limit checks
    // =========================================================================

    TEST_F( CapabilityValidatorTest, RejectVulkanImageDimensionExceeded )
    {
        // Mock has maxImageDimension2D=16384, pass has 99999 → rejected
        auto pass = MakeMockRenderPass( /*width=*/99999, /*height=*/256 );

        CanExecuteResult result;
        validator.CanExecute( pass, [&]( CanExecuteResult r ) { result = std::move( r ); } );

        EXPECT_FALSE( result.executable );
        ASSERT_GE( result.unmet.size(), 1u );
        bool found = false;
        for ( const auto &u : result.unmet )
        {
            if ( u.category == UnmetRequirementCategory::VULKAN
                 && u.detail.find( "maxImageDimension2D" ) != std::string::npos )
            {
                found = true;
                EXPECT_TRUE( u.detail.find( "99999" ) != std::string::npos );
                EXPECT_TRUE( u.detail.find( "16384" ) != std::string::npos );
            }
        }
        EXPECT_TRUE( found ) << "Expected maxImageDimension2D unmet requirement";
    }

    TEST_F( CapabilityValidatorTest, RejectDeviceTypeNotAcceptable )
    {
        auto snap = MakeMockSnapshot();
        snap.vulkanProps.deviceType = VK_PHYSICAL_DEVICE_TYPE_CPU;
        validator.SetSnapshotForTest( snap );

        auto pass = MakeMockRenderPass();

        CanExecuteResult result;
        validator.CanExecute( pass, [&]( CanExecuteResult r ) { result = std::move( r ); } );

        EXPECT_FALSE( result.executable );
        bool found = false;
        for ( const auto &u : result.unmet )
        {
            if ( u.detail.find( "Device type not acceptable" ) != std::string::npos )
                found = true;
        }
        EXPECT_TRUE( found );
    }

    // =========================================================================
    // CAP-03: MNN model format checks
    // =========================================================================

    TEST_F( CapabilityValidatorTest, RejectUnsupportedModelFormat )
    {
        // Add INFERENCE executor to mock snapshot
        auto snap = MakeMockSnapshot();
        ExecutorCapability cap;
        cap.passType               = PassType::INFERENCE;
        cap.backend                = "VULKAN";
        cap.supportedModelFormats  = { "MNN" };
        cap.supportedQuantizations = { "FP32", "FP16", "INT8" };
        snap.executorCaps.push_back( cap );
        validator.SetSnapshotForTest( snap );

        // ONNX model — not in supported list
        auto pass = MakeMockInferencePass( sgns::ModelFormat::ONNX );

        CanExecuteResult result;
        validator.CanExecute( pass, [&]( CanExecuteResult r ) { result = std::move( r ); } );

        EXPECT_FALSE( result.executable );
        ASSERT_GE( result.unmet.size(), 1u );
        EXPECT_EQ( result.unmet[0].category, UnmetRequirementCategory::MNN );
        EXPECT_TRUE( result.unmet[0].detail.find( "ONNX" ) != std::string::npos );
        EXPECT_TRUE( result.unmet[0].detail.find( "MNN" ) != std::string::npos );
    }

    // =========================================================================
    // CAP-05: GPU memory + disk space checks
    // =========================================================================

    TEST_F( CapabilityValidatorTest, RejectGpuMemoryExceeded )
    {
        // 16384×16384 RGBA8 + D32 ≈ 16384*16384*(4+4) ≈ 2GB, but heap is 1GB
        auto pass = MakeMockRenderPass( /*width=*/16384, /*height=*/16384 );

        CanExecuteResult result;
        validator.CanExecute( pass, [&]( CanExecuteResult r ) { result = std::move( r ); } );

        EXPECT_FALSE( result.executable );
        bool found = false;
        for ( const auto &u : result.unmet )
        {
            if ( u.category == UnmetRequirementCategory::RESOURCE
                 && u.detail.find( "GPU memory" ) != std::string::npos )
            {
                found = true;
            }
        }
        EXPECT_TRUE( found ) << "Expected GPU memory exceeded unmet requirement";
    }

    TEST_F( CapabilityValidatorTest, AcceptGpuMemoryOk )
    {
        // 256×256 RGBA8 + D32 ≈ 256*256*8 ≈ 512KB, heap is 1GB → OK
        auto pass = MakeMockRenderPass( /*width=*/256, /*height=*/256 );

        CanExecuteResult result;
        validator.CanExecute( pass, [&]( CanExecuteResult r ) { result = std::move( r ); } );

        EXPECT_TRUE( result.executable );
        EXPECT_FALSE( result.executorId.empty() );
        EXPECT_TRUE( result.unmet.empty() );
    }

    TEST_F( CapabilityValidatorTest, RejectDiskSpaceExceeded )
    {
        auto snap = MakeMockSnapshot();
        snap.availableDiskBytes = 100; // only 100 bytes
        validator.SetSnapshotForTest( snap );

        // 256×256 RGBA8 = 256KB output → exceeds 100 bytes
        auto pass = MakeMockRenderPass( /*width=*/256, /*height=*/256 );

        CanExecuteResult result;
        validator.CanExecute( pass, [&]( CanExecuteResult r ) { result = std::move( r ); } );

        EXPECT_FALSE( result.executable );
        bool found = false;
        for ( const auto &u : result.unmet )
        {
            if ( u.detail.find( "disk space" ) != std::string::npos )
                found = true;
        }
        EXPECT_TRUE( found );
    }

    TEST_F( CapabilityValidatorTest, DiskSpaceCheckSkippedWhenZero )
    {
        auto snap = MakeMockSnapshot();
        snap.availableDiskBytes = 0; // degraded mode
        validator.SetSnapshotForTest( snap );

        auto pass = MakeMockRenderPass();

        CanExecuteResult result;
        validator.CanExecute( pass, [&]( CanExecuteResult r ) { result = std::move( r ); } );

        // Should still be executable — disk check skipped in degraded mode
        EXPECT_TRUE( result.executable );
    }

    // =========================================================================
    // CAP-06: Executor identity stability
    // =========================================================================

    TEST_F( CapabilityValidatorTest, ExecutorIdStableAcrossCalls )
    {
        auto pass = MakeMockRenderPass();

        std::string firstId;
        validator.CanExecute( pass, [&]( CanExecuteResult r ) { firstId = r.executorId; } );

        for ( int i = 0; i < 10; ++i )
        {
            std::string id;
            validator.CanExecute( pass, [&]( CanExecuteResult r ) { id = r.executorId; } );
            EXPECT_EQ( id, firstId ) << "Executor ID changed on iteration " << i;
        }
    }

    // =========================================================================
    // CAP-01: Acceptance path (valid pass)
    // =========================================================================

    TEST_F( CapabilityValidatorTest, AcceptValidRenderPass )
    {
        auto pass = MakeMockRenderPass();

        CanExecuteResult result;
        validator.CanExecute( pass, [&]( CanExecuteResult r ) { result = std::move( r ); } );

        EXPECT_TRUE( result.executable );
        EXPECT_FALSE( result.executorId.empty() );
        EXPECT_TRUE( result.executorId.find( "sgproc-" ) == 0 );
        EXPECT_TRUE( result.unmet.empty() );
    }

    TEST_F( CapabilityValidatorTest, AcceptValidInferencePass )
    {
        auto snap = MakeMockSnapshot();
        ExecutorCapability cap;
        cap.passType               = PassType::INFERENCE;
        cap.backend                = "VULKAN";
        cap.supportedModelFormats  = { "MNN" };
        cap.supportedQuantizations = { "FP32", "FP16", "INT8" };
        snap.executorCaps.push_back( cap );
        validator.SetSnapshotForTest( snap );

        auto pass = MakeMockInferencePass( sgns::ModelFormat::MNN );

        CanExecuteResult result;
        validator.CanExecute( pass, [&]( CanExecuteResult r ) { result = std::move( r ); } );

        EXPECT_TRUE( result.executable );
        EXPECT_FALSE( result.executorId.empty() );
        EXPECT_TRUE( result.unmet.empty() );
    }

    // =========================================================================
    // Edge cases
    // =========================================================================

    TEST_F( CapabilityValidatorTest, RejectBeforeBuildSnapshot )
    {
        CapabilityValidator v;
        sgns::Pass          pass;
        pass.set_type( PassType::RENDER );

        CanExecuteResult result;
        v.CanExecute( pass, [&]( CanExecuteResult r ) { result = std::move( r ); } );

        EXPECT_FALSE( result.executable );
        ASSERT_GE( result.unmet.size(), 1u );
        EXPECT_EQ( result.unmet[0].category, UnmetRequirementCategory::RESOURCE );
        EXPECT_TRUE( result.unmet[0].detail.find( "not initialized" ) != std::string::npos );
    }

} // namespace sgns::sgprocessing
