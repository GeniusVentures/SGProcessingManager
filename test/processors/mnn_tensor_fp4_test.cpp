// MNN_Tensor FP4_ULTRA unit tests (Phase 04-sgprocessing-integration, Plan 04-02, PROC-02;
// rewritten Phase 13, Plan 13-01, SGF-02/SGF-04a)
//
// Exercises MNN_Tensor::StartProcessing() directly -- never via
// ProcessingManager::Create() -- so these tests are fully deterministic and
// do not require a real Vulkan device or model file for their negative
// paths: malformed model bytes make Process() return nullptr before any
// session is ever created (its null-return path 1, createFromBuffer).

#include <chrono>
#include <vector>

#include <gtest/gtest.h>

#include "processors/processing_processor_mnn_tensor.hpp"
#include "Generators.hpp" // from_json/to_json(DataType) -- for DataTypeLlmJsonRoundTrip below

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

    CallResult CallStartProcessing( sgns::InputFormat                       format,
                                    int64_t                                 width,
                                    std::vector<char>                        tensorData,
                                    const std::vector<sgns::Parameter>      *parameters = nullptr,
                                    std::vector<char>                        modelFile = {} )
    {
        MNN_Tensor                              processor;
        std::vector<std::vector<uint8_t>>       chunkhashes;
        sgns::IoDeclaration                     decl = MakeTensorDeclaration( format, width );
        auto                                    execCtx = ExecutionContext::NoOp();

        const auto t0 = std::chrono::steady_clock::now();
        auto       result
            = processor.StartProcessing( chunkhashes, decl, tensorData, modelFile, parameters, *execCtx );
        const auto t1 = std::chrono::steady_clock::now();

        CallResult callResult;
        callResult.result    = result;
        callResult.elapsedMs = std::chrono::duration<double, std::milli>( t1 - t0 ).count();
        return callResult;
    }
}

// FP4_ULTRA with a buffer large enough to hold the declared elements: the
// format passes the input-format gate and FP4_ULTRA input decode is LIVE
// (MNN::dequant_fp4_packed_cpu, SGProcessingManager e1f28d7) -- so a valid
// buffer proceeds all the way into model/session work. With no usable model
// bytes, Process() returns nullptr and StartProcessing() must surface that
// as a structured FORMAT_UNSUPPORTED error (Phase 13, SGF-02/D-11/SGF-04a)
// -- never a crash on the null procresults pointer. This replaces the stale
// Fp4UltraRecognizedButDecodeUnavailable assertion (decode-unavailable is
// no longer true since e1f28d7 wired the real E2M1 decode).
TEST( MnnTensorFp4Test, Fp4UltraRecognizedAndMalformedModelReturnsCleanError )
{
    constexpr int64_t kWidth = 64;
    // FP4 packs two 4-bit elements per byte -- ceil(64 / 2) = 32 bytes is
    // exactly sufficient to pass the size check and reach the decode +
    // model/session path.
    std::vector<char> tensorData( 32, 0 );
    // Garbage model bytes: not a valid MNN flatbuffer, so
    // Interpreter::createFromBuffer returns null and Process() must return
    // nullptr (its documented null-return path 1).
    std::vector<char> modelFile( 16, static_cast<char>( 0xAB ) );

    auto callResult = CallStartProcessing( sgns::InputFormat::FP4_ULTRA, kWidth, std::move( tensorData ), nullptr, std::move( modelFile ) );

    ASSERT_TRUE( callResult.result.error.has_value() );
    EXPECT_EQ( callResult.result.error->stage, ProcessingErrorStage::FORMAT_UNSUPPORTED );
    EXPECT_NE( callResult.result.error->message.find( "malformed or incompatible model" ), std::string::npos )
        << "message was: " << callResult.result.error->message;
}

// SGF-02/D-11 negative regression: a malformed model buffer fed to a plain
// FLOAT32 tensor input must produce the structured clean error, not a null
// dereference crash. Before the Phase 13 null-check, this exact call crashed
// on procresults->host<float>() when Process() returned nullptr.
TEST( MnnTensorFp4Test, MalformedModelBufferReturnsCleanErrorNoCrash )
{
    constexpr int64_t kWidth = 8;
    std::vector<char> tensorData( kWidth * sizeof( float ), 0 );
    std::vector<char> modelFile( 16, static_cast<char>( 0xCD ) ); // garbage -- not an MNN flatbuffer

    auto callResult = CallStartProcessing( sgns::InputFormat::FLOAT32, kWidth, std::move( tensorData ), nullptr, std::move( modelFile ) );

    ASSERT_TRUE( callResult.result.error.has_value() );
    EXPECT_EQ( callResult.result.error->stage, ProcessingErrorStage::FORMAT_UNSUPPORTED );
    EXPECT_EQ( callResult.result.error->message, "MNN_Tensor::Process returned null (malformed or incompatible model)" );
}

// FP4_ULTRA with a buffer smaller than its declared width can hold: fails the
// buffer-size-vs-dimensions check (T-04-03) before ever reaching the
// model/session path -- a distinct, honest, structured error, not an
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

// Task 1 (Phase 04-sgprocessing-integration, Plan 04-03, PROC-01): DataType::LLM's
// json round-trip. Lives here (builds unconditionally in every checkout) rather than
// in mnn_llm_test.cpp (which only builds when the vendored MNN was built with
// MNN_BUILD_LLM=ON) since the enum/json mapping itself has no dependency on MNN's LLM
// engine being compiled in at all.
TEST( MnnTensorFp4Test, DataTypeLlmJsonRoundTrip )
{
    nlohmann::json j = "llm";
    sgns::DataType x;
    sgns::from_json( j, x );
    EXPECT_EQ( x, sgns::DataType::LLM );

    nlohmann::json j2;
    sgns::to_json( j2, x );
    EXPECT_EQ( j2, "llm" );
}
