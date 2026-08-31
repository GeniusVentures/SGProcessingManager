/**
 * Timeout tests for ExecutionContext — EXEC-02.
 *
 * Tests deadline expiry produces TIMED_OUT distinct from CANCELLED.
 */
#include <gtest/gtest.h>
#include <execution/execution_context.hpp>
#include <processors/processing_processor.hpp>

namespace sgns::sgprocessing
{
namespace test
{

    class TimeoutTest : public ::testing::Test
    {
    };

    /// Verify TIMED_OUT error stage is distinct from CANCELLED and BUDGET_EXCEEDED.
    TEST_F( TimeoutTest, ErrorStagesAreDistinct )
    {
        EXPECT_NE( static_cast<int>( ProcessingErrorStage::TIMED_OUT ),
                   static_cast<int>( ProcessingErrorStage::CANCELLED ) );
        EXPECT_NE( static_cast<int>( ProcessingErrorStage::TIMED_OUT ),
                   static_cast<int>( ProcessingErrorStage::BUDGET_EXCEEDED ) );
        EXPECT_NE( static_cast<int>( ProcessingErrorStage::CANCELLED ),
                   static_cast<int>( ProcessingErrorStage::BUDGET_EXCEEDED ) );
    }

    /// Verify CancellationToken cancel callback is invoked at most once.
    TEST_F( TimeoutTest, CancelCallbackInvokedOnce )
    {
        CancellationToken token;
        int callCount = 0;
        token.SetCallback( [&callCount]() { ++callCount; } );

        token.Cancel();
        EXPECT_EQ( callCount, 1 );

        // Second Cancel() should not invoke again
        token.Cancel();
        EXPECT_EQ( callCount, 1 );
    }

    /// Verify IsCancelled() returns true after Cancel().
    TEST_F( TimeoutTest, IsCancelledAfterCancel )
    {
        CancellationToken token;
        EXPECT_FALSE( token.IsCancelled() );
        token.Cancel();
        EXPECT_TRUE( token.IsCancelled() );
    }

    // TODO: Integration tests requiring Vulkan:
    // - DeadlineExpiryReturnsTimedOut: job with per_pass_deadline_ms=100, sleep >100ms
    // - NoDeadlineRunsNormally: job with deadline=0 completes normally
    // - DeadlineDistinctFromCancel: TIMED_OUT != CANCELLED error codes

} // namespace test
} // namespace sgns::sgprocessing
