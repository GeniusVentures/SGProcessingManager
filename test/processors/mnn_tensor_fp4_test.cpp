// MNN_Tensor FP4_ULTRA unit tests (Phase 04-sgprocessing-integration, Plan 04-02, PROC-02)
//
// Exercises MNN_Tensor::StartProcessing() directly -- never via
// ProcessingManager::Create() -- so these tests are fully deterministic and
// do not hit the known, out-of-scope VulkanInitMutex re-entrancy deadlock
// (sgproc-render Phase 18) or require a real Vulkan device/model file.

#include <chrono>
#include <vector>

#include <gtest/gtest.h>

#include "processors/processing_processor_mnn_tensor.hpp"

namespace
{
    using sgns::sgprocessing::MNN_Tensor;
    using sgns::sgprocessing::ProcessingErrorStage;
    using sgns::sgprocessing::ProcessingResult;
    using sgns::sgprocessing::ExecutionContext;

    // Builds a minimal TENSOR-typed IoDeclaration with the given format and
    // declared width. No JSON parsing needed -- generated setters are plain
    // C++ accessors.
    sgns::IoDeclaration MakeTensorDeclaration( sgns::InputFormat format, int64_t width )
    {
        sgns::Dimensions dims;
        dims.set_width( width );

        sgns::IoDeclaration decl;
        decl.set_type( sgns::DataType::TENSOR );
        decl.set_format( format );
        decl.set_dimensions( dims );
        return decl;
    }

    struct CallResult
    {
        ProcessingResult result;
        double           elapsedMs = 0.0;
    };

    CallResult CallStartProcessing( sgns::InputFormat format, int64_t width, std::vector<char> tensorData )
    {
        MNN_Tensor                              processor;
        std::vector<std::vector<uint8_t>>       chunkhashes;
        sgns::IoDeclaration                     decl = MakeTensorDeclaration( format, width );
        std::vector<char>                       modelFile; // intentionally empty -- FP4_ULTRA's
                                                             // structured-failure path must return
                                                             // before any model/session work begins.
        auto                                     execCtx = ExecutionContext::NoOp();

        const auto t0 = std::chrono::steady_clock::now();
        auto       result = processor.StartProcessing( chunkhashes, decl, tensorData, modelFile, nullptr, *execCtx );
        const auto t1 = std::chrono::steady_clock::now();

        CallResult callResult;
        callResult.result    = result;
        callResult.elapsedMs = std::chrono::duration<double, std::milli>( t1 - t0 ).count();
        return callResult;
    }
}

// FP4_ULTRA with a buffer large enough to hold the declared elements: recognized
// as a valid TENSOR format, but decode is unavailable in this build (D-04/D-09) --
// returns a structured FORMAT_UNSUPPORTED error describing the pending MNN_Ultra
// decode kernel, and completes near-instantly (never touches VulkanInitMutex/MNN
// session creation).
TEST( MnnTensorFp4Test, Fp4UltraRecognizedButDecodeUnavailable )
{
    constexpr int64_t kWidth = 64;
    // FP4 packs two 4-bit elements per byte -- ceil(64 / 2) = 32 bytes is
    // exactly sufficient, so this buffer size should pass the size check and
    // reach the "decode unavailable" return path.
    std::vector<char> tensorData( 32, 0 );

    auto callResult = CallStartProcessing( sgns::InputFormat::FP4_ULTRA, kWidth, std::move( tensorData ) );

    ASSERT_TRUE( callResult.result.error.has_value() );
    EXPECT_EQ( callResult.result.error->stage, ProcessingErrorStage::FORMAT_UNSUPPORTED );
    EXPECT_NE( callResult.result.error->message.find( "MNN_Ultra" ), std::string::npos )
        << "message was: " << callResult.result.error->message;

    // Proves this call never took VulkanInitMutex()/created an MNN session --
    // a real session-creation path would take orders of magnitude longer
    // (model load + Vulkan device init) or deadlock outright (Pitfall 4).
    EXPECT_LT( callResult.elapsedMs, 1000.0 );
}

// FP4_ULTRA with a buffer smaller than its declared width can hold: fails the
// buffer-size-vs-dimensions check (T-04-03) before ever reaching the
// decode-unavailable return -- a distinct, honest, structured error, not an
// out-of-bounds read.
TEST( MnnTensorFp4Test, Fp4UltraUndersizedBufferFailsSizeCheck )
{
    constexpr int64_t kWidth = 64;
    std::vector<char> tensorData; // empty -- far smaller than the required 32 bytes

    auto callResult = CallStartProcessing( sgns::InputFormat::FP4_ULTRA, kWidth, std::move( tensorData ) );

    ASSERT_TRUE( callResult.result.error.has_value() );
    EXPECT_EQ( callResult.result.error->stage, ProcessingErrorStage::FORMAT_UNSUPPORTED );
    EXPECT_NE( callResult.result.error->message.find( "buffer size" ), std::string::npos )
        << "message was: " << callResult.result.error->message;
    EXPECT_LT( callResult.elapsedMs, 1000.0 );
}

// Regression: an unrecognized format (RGB8, not a valid TENSOR format at all)
// must still take the pre-existing, unmodified rejection path -- a bare
// default-constructed ProcessingResult with no structured error and no
// output -- proving Task 2 did not widen format acceptance beyond FP4_ULTRA.
TEST( MnnTensorFp4Test, UnrecognizedFormatStillUsesPreExistingRejection )
{
    constexpr int64_t kWidth = 64;
    std::vector<char> tensorData; // irrelevant -- rejected before size is checked

    auto callResult = CallStartProcessing( sgns::InputFormat::RGB8, kWidth, std::move( tensorData ) );

    EXPECT_FALSE( callResult.result.error.has_value() );
    EXPECT_TRUE( callResult.result.hash.empty() );
    EXPECT_EQ( callResult.result.output_buffers, nullptr );
    EXPECT_LT( callResult.elapsedMs, 1000.0 );
}
