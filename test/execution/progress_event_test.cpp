/**
 * Progress event tests — EXEC-04.
 *
 * Tests ProgressEvent struct, stage enums, and factory methods.
 */
#include <gtest/gtest.h>
#include <execution/execution_context.hpp>

namespace sgns::sgprocessing
{
namespace test
{

    class ProgressEventTest : public ::testing::Test
    {
    };

    /// Verify ProgressEvent default values.
    TEST_F( ProgressEventTest, DefaultValues )
    {
        ProgressEvent ev;
        EXPECT_TRUE( ev.pass_id.empty() );
        EXPECT_EQ( ev.render_stage, RenderStage::COMPILE );
        EXPECT_EQ( ev.mnn_stage, MNNStage::LOAD_MODEL );
        EXPECT_FLOAT_EQ( ev.percent, 0.0f );
    }

    /// Verify ForRender factory populates render_stage.
    TEST_F( ProgressEventTest, ForRenderFactory )
    {
        auto ev = ProgressEvent::ForRender( "render_pass_1", RenderStage::BUILD_PIPELINE, 50.0f );
        EXPECT_EQ( ev.pass_id, "render_pass_1" );
        EXPECT_EQ( ev.render_stage, RenderStage::BUILD_PIPELINE );
        EXPECT_EQ( ev.mnn_stage, MNNStage::LOAD_MODEL ); // default
        EXPECT_FLOAT_EQ( ev.percent, 50.0f );
    }

    /// Verify ForMNN factory populates mnn_stage.
    TEST_F( ProgressEventTest, ForMNNFactory )
    {
        auto ev = ProgressEvent::ForMNN( "mnn_pass_1", MNNStage::RUN, 75.0f );
        EXPECT_EQ( ev.pass_id, "mnn_pass_1" );
        EXPECT_EQ( ev.render_stage, RenderStage::COMPILE ); // default
        EXPECT_EQ( ev.mnn_stage, MNNStage::RUN );
        EXPECT_FLOAT_EQ( ev.percent, 75.0f );
    }

    /// Verify stage enums have correct numeric values.
    TEST_F( ProgressEventTest, RenderStageEnumValues )
    {
        EXPECT_EQ( static_cast<int>( RenderStage::COMPILE ), 0 );
        EXPECT_EQ( static_cast<int>( RenderStage::BUILD_PIPELINE ), 1 );
        EXPECT_EQ( static_cast<int>( RenderStage::DRAW ), 2 );
        EXPECT_EQ( static_cast<int>( RenderStage::READBACK ), 3 );
    }

    /// Verify MNN stage enums have correct numeric values.
    TEST_F( ProgressEventTest, MNNStageEnumValues )
    {
        EXPECT_EQ( static_cast<int>( MNNStage::LOAD_MODEL ), 0 );
        EXPECT_EQ( static_cast<int>( MNNStage::CREATE_SESSION ), 1 );
        EXPECT_EQ( static_cast<int>( MNNStage::RUN ), 2 );
        EXPECT_EQ( static_cast<int>( MNNStage::READ_OUTPUT ), 3 );
    }

    // TODO: Integration tests requiring Vulkan:
    // - RenderProcessorProgressEvents: capture 4 events with correct progression
    // - MNNProcessorProgressEvents: capture 4 MNN stages

} // namespace test
} // namespace sgns::sgprocessing
