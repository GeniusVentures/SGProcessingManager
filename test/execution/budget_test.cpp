/**
 * Budget tests for ExecutionContext — EXEC-03.
 *
 * Tests BUDGET_EXCEEDED error on output size exceeding max_output_artifact_bytes.
 */
#include <gtest/gtest.h>
#include <execution/execution_context.hpp>
#include <processors/processing_processor.hpp>

namespace sgns::sgprocessing
{
namespace test
{

    class BudgetTest : public ::testing::Test
    {
    };

    /// Verify BUDGET_EXCEEDED stage exists and is distinct.
    TEST_F( BudgetTest, BudgetExceededStageExists )
    {
        EXPECT_EQ( static_cast<int>( ProcessingErrorStage::BUDGET_EXCEEDED ), 14 );
    }

    /// Verify ExecutionContext budget fields default to 0 (no budget).
    TEST_F( BudgetTest, BudgetFieldsDefaultToZero )
    {
        ExecutionContext ctx;
        EXPECT_EQ( ctx.gpuMemoryBudget, 0u );
        EXPECT_EQ( ctx.maxOutputArtifactBytes, 0u );
        EXPECT_EQ( ctx.deadlineMs, 0u );
    }

    /// Verify NoOp ExecutionContext has all budgets at 0.
    TEST_F( BudgetTest, NoOpContextHasZeroBudgets )
    {
        auto ctx = ExecutionContext::NoOp();
        EXPECT_EQ( ctx->gpuMemoryBudget, 0u );
        EXPECT_EQ( ctx->maxOutputArtifactBytes, 0u );
        EXPECT_EQ( ctx->deadlineMs, 0u );
        EXPECT_FALSE( ctx->cancelToken.IsCancelled() );
    }

    // TODO: Integration tests requiring Vulkan:
    // - OutputSizeExceedsBudget: max_output_artifact_bytes=1, produces >1 byte → BUDGET_EXCEEDED
    // - OutputWithinBudget: max_output_artifact_bytes=0 runs normally
    // - BudgetCheckForRender: render pass with budget exceeded

} // namespace test
} // namespace sgns::sgprocessing
