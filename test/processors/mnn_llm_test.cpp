// MNN_Llm unit tests (Phase 04-sgprocessing-integration, Plan 04-03, PROC-01)
//
// Exercises MNN_Llm::StartProcessing() directly -- never via
// ProcessingManager::Create() -- so these tests are fully deterministic and
// do not hit the known, out-of-scope VulkanInitMutex re-entrancy deadlock
// (sgproc-render Phase 18) or require a real Vulkan device/MNN LLM model
// fixture (none exists in this repo's test data).
//
// This entire translation unit is only compiled in when SGPROC_HAS_MNN_LLM is
// defined (see test/processors/CMakeLists.txt) -- i.e. when the vendored MNN
// static library was actually built with MNN_BUILD_LLM=ON. In checkouts
// without LLM support (like the one these tests were authored against),
// MNN_Llm::StartProcessing() has no compiled definition anywhere (see
// src/processors/CMakeLists.txt), so this file is excluded from the test
// build entirely rather than failing to link.

#include <chrono>
#include <vector>

#include <gtest/gtest.h>

#include "processors/processing_processor_mnn_llm.hpp"

namespace
{
    using sgns::sgprocessing::ExecutionContext;
    using sgns::sgprocessing::MNN_Llm;
    using sgns::sgprocessing::ProcessingErrorStage;
    using sgns::sgprocessing::ProcessingResult;

    // Builds a minimal LLM-typed IoDeclaration. No JSON parsing needed --
    // generated setters are plain C++ accessors.
    sgns::IoDeclaration MakeLlmDeclaration()
    {
        sgns::IoDeclaration decl;
        decl.set_type( sgns::DataType::LLM );
        return decl;
    }

    struct CallResult
    {
        ProcessingResult result;
        double           elapsedMs = 0.0;
    };

    CallResult CallStartProcessing( std::vector<char> promptData, std::vector<char> modelFile, bool preCancel )
    {
        MNN_Llm                           processor;
        std::vector<std::vector<uint8_t>> chunkhashes;
        sgns::IoDeclaration               decl = MakeLlmDeclaration();
        auto                               execCtx = ExecutionContext::NoOp();
        if ( preCancel )
        {
            execCtx->cancelToken.Cancel();
        }

        const auto t0 = std::chrono::steady_clock::now();
        auto       result =
            processor.StartProcessing( chunkhashes, decl, promptData, modelFile, nullptr, *execCtx );
        const auto t1 = std::chrono::steady_clock::now();

        CallResult callResult;
        callResult.result    = result;
        callResult.elapsedMs = std::chrono::duration<double, std::milli>( t1 - t0 ).count();
        return callResult;
    }
} // namespace

// An empty modelFile buffer must fail closed with a structured
// RESOURCE_RESOLUTION error before any generation is attempted -- never crash,
// never return a bare default-constructed result with no diagnostic.
TEST( MnnLlmTest, EmptyModelFileFailsClosedWithResourceResolution )
{
    std::vector<char> promptData( { 'h', 'i' } );
    std::vector<char> modelFile; // intentionally empty

    auto callResult = CallStartProcessing( std::move( promptData ), std::move( modelFile ), /*preCancel=*/false );

    ASSERT_TRUE( callResult.result.error.has_value() );
    EXPECT_EQ( callResult.result.error->stage, ProcessingErrorStage::RESOURCE_RESOLUTION );
    // Proves this call never took VulkanInitMutex()/attempted a real MNN LLM
    // load -- a real load attempt would take orders of magnitude longer or
    // deadlock outright (sgproc-render Phase 18).
    EXPECT_LT( callResult.elapsedMs, 1000.0 );
}

// A pre-cancelled cancellation token must short-circuit before any work is
// attempted -- including before the model-load attempt -- returning a
// structured CANCELLED error. StartProcessing() checks execCtx.cancelToken
// as the very first thing, ahead of the empty-model check, so a cancelled
// job never pays the cost of a (potentially expensive) load attempt; this
// also makes the CANCELLED path deterministically testable without any real
// MNN LLM model fixture.
TEST( MnnLlmTest, PreCancelledTokenFailsClosedWithCancelled )
{
    std::vector<char> promptData( { 'h', 'i' } );
    std::vector<char> modelFile; // irrelevant -- cancellation is checked first

    auto callResult = CallStartProcessing( std::move( promptData ), std::move( modelFile ), /*preCancel=*/true );

    ASSERT_TRUE( callResult.result.error.has_value() );
    EXPECT_EQ( callResult.result.error->stage, ProcessingErrorStage::CANCELLED );
    EXPECT_LT( callResult.elapsedMs, 1000.0 );
}

// Task 1's DataType::LLM json round-trip test lives in mnn_tensor_fp4_test.cpp
// instead of here, since that target builds unconditionally in every checkout
// (this file only builds when SGPROC_HAS_MNN_LLM is set -- Task 1's enum/json
// mapping has no dependency on MNN's LLM engine being compiled in at all, so
// pinning its test coverage to this conditional target would leave it
// untested in checkouts without MNN_BUILD_LLM=ON, like this one).
