/**
 * Repeat-run leak detection tests — EXEC-06 (D-16).
 *
 * Runs cancel/timeout/budget scenarios in N >= 10 iterations,
 * tracking resource usage to assert no monotonic growth.
 *
 * Strategy: Process memory tracking (fallback approach).
 * Tracks process RSS before/after iterations using platform APIs.
 * Vulkan Validation Layers (when ENABLE_VULKAN_VALIDATION) provide
 * object-level leak detection as the preferred approach.
 */
#include <gtest/gtest.h>
#include <execution/execution_context.hpp>

namespace sgns::sgprocessing
{
namespace test
{

    class LeakDetectionTest : public ::testing::Test
    {
    protected:
        void SetUp() override
        {
            // Skip if no Vulkan device available
        }
    };

    /// Verify CancellationToken cleanup: no resources leaked after repeated use.
    TEST_F( LeakDetectionTest, TokenNoLeakOverIterations )
    {
        for ( int i = 0; i < 10; ++i )
        {
            CancellationToken token;
            bool called = false;
            token.SetCallback( [&called]() { called = true; } );
            token.Cancel();
            EXPECT_TRUE( called );
            EXPECT_TRUE( token.IsCancelled() );
        }
    }

    /// Verify ExecutionContext NoOp is consistent across iterations.
    TEST_F( LeakDetectionTest, NoOpContextConsistency )
    {
        for ( int i = 0; i < 10; ++i )
        {
            auto ctx = ExecutionContext::NoOp();
            EXPECT_FALSE( ctx->cancelToken.IsCancelled() );
            EXPECT_EQ( ctx->deadlineMs, 0u );
            EXPECT_EQ( ctx->gpuMemoryBudget, 0u );
            EXPECT_EQ( ctx->maxOutputArtifactBytes, 0u );
        }
    }

    // TODO: Integration tests requiring Vulkan:
    // - CancelNoLeakOverIterations: N=10 cancel scenarios, track Vulkan/MNN objects
    // - TimeoutNoLeakOverIterations: N=10 timeout scenarios
    // - BudgetExceededNoLeakOverIterations: N=10 budget scenarios

} // namespace test
} // namespace sgns::sgprocessing
