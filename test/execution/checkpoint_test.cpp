/**
 * Checkpoint support tests — EXEC-05.
 *
 * Tests that supports_checkpointing is false for all registered executors.
 */
#include <gtest/gtest.h>
#include <capability/capability_types.hpp>
#include <execution/execution_context.hpp>

namespace sgns::sgprocessing
{
namespace test
{

    class CheckpointTest : public ::testing::Test
    {
    };

    /// Verify PassTypeHash works correctly for map lookups.
    TEST_F( CheckpointTest, PassTypeHashWorks )
    {
        PassTypeHash hash;
        EXPECT_EQ( hash( PassType::RENDER ), static_cast<size_t>( PassType::RENDER ) );
    }

    /// Verify CapabilitySnapshot::checkpointSupport exists and is empty by default.
    TEST_F( CheckpointTest, CheckpointSupportDefaultEmpty )
    {
        CapabilitySnapshot snap;
        EXPECT_TRUE( snap.checkpointSupport.empty() );
    }

    // TODO: Integration tests requiring CapabilityValidator:
    // - CheckpointNotSupportedForRender: query supports_checkpointing for RENDER → false
    // - CheckpointFlagInRegistry: ExecutorRegistryEntry.supports_checkpointing is false
    // - AllExecutorsCheckpointFalse: all registered executors have false

} // namespace test
} // namespace sgns::sgprocessing
