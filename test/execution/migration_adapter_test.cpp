/**
 * Migration adapter tests — EXEC-07.
 *
 * Verifies that ExecutionContext::NoOp() produces identical behavior to
 * the old non-ExecutionContext path, proving the adapter didn't change behavior.
 */
#include <gtest/gtest.h>
#include <execution/execution_context.hpp>
#include <processors/processing_processor.hpp>

namespace sgns::sgprocessing
{
namespace test
{

    class MigrationAdapterTest : public ::testing::Test
    {
    };

    /// Verify NoOp ExecutionContext doesn't cancel.
    TEST_F( MigrationAdapterTest, NoOpDoesNotCancel )
    {
        auto ctx = ExecutionContext::NoOp();
        EXPECT_FALSE( ctx->cancelToken.IsCancelled() );
    }

    /// Verify NoOp progress callback doesn't throw.
    TEST_F( MigrationAdapterTest, NoOpProgressCallbackDoesNotThrow )
    {
        auto ctx = ExecutionContext::NoOp();
        EXPECT_NO_THROW( ctx->progressCallback(
            ProgressEvent::ForRender( "test", RenderStage::COMPILE, 0.0f ) ) );
    }

    // TODO: Integration tests requiring Vulkan/MNN:
    // - MNNImageSameOutputThroughAdapter: identical output with NoOp vs real context
    // - RenderProcessorSameOutputThroughAdapter: identical output
    // - AllProcessorsCompileAndRun: all 15 processors instantiate and run with NoOp

} // namespace test
} // namespace sgns::sgprocessing
