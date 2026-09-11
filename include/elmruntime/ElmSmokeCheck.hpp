#ifndef SGPROCMGR_ELMRUNTIME_SMOKE_CHECK_HPP
#define SGPROCMGR_ELMRUNTIME_SMOKE_CHECK_HPP

#include <outcome/sgprocmgr-outcome.hpp>

#include <functional>
#include <string>

namespace sgns::elmruntime
{
    /// @brief Injectable loadability probe over a materialized MNN bundle (SC-2).
    ///
    /// The cache publish path calls this against the STAGING directory BEFORE the
    /// atomic rename (A5): a failed check aborts the publish, so a final-path entry
    /// is usable-by-construction. The probe is one createLLM + load + 1-token greedy
    /// generation -- wrong-tokenizer/garbage bundles die inside load() and surface
    /// as SMOKE_CHECK_FAILED (structured, never an exception).
    ///
    /// The header contains ZERO MNN includes: the concrete MNN::Transformer::Llm
    /// usage is confined to ElmSmokeCheck.cpp behind SGPROC_HAS_MNN_LLM (the
    /// include-isolation pattern of include/processors/processing_processor_mnn_llm.hpp
    /// :16-29 -- consumers of this header never need <llm/llm.hpp>).
    /// @param bundleDirWithTrailingSlash - staging dir path WITH a trailing separator
    ///        (P2-5: LlmConfig uses the string verbatim as a path prefix; the cache
    ///        guarantees the trailing slash)
    /// @return success, or ElmRuntimeError::SMOKE_CHECK_FAILED / SMOKE_CHECK_UNAVAILABLE
    using SmokeCheckFn = std::function<outcome::result<void>( const std::string &bundleDirWithTrailingSlash )>;

    /// @brief Production smoke-check factory (SGPROC_HAS_MNN_LLM-gated TU).
    ///
    /// When the vendored MNN has LLM support (gate TRUE -- detected at configure time
    /// in src/elmruntime/CMakeLists.txt), returns a SmokeCheckFn performing the full
    /// probe: VulkanInitMutex lock (one load per entry bounds Pitfall 5's damage) ->
    /// createLLM(dir) -> set_config(greedy, max_new_tokens=1) -> load() ->
    /// response("Hello", ..., 1) -> destroy on EVERY path.
    ///
    /// When the gate is FALSE, returns a fail-closed stub that always fails with
    /// SMOKE_CHECK_UNAVAILABLE: without the engine, entries can never be marked
    /// usable (SC-2's letter -- no bypass).
    /// @return the smoke-check function
    SmokeCheckFn MakeMnnLlmSmokeCheck();
} // namespace sgns::elmruntime

#endif // SGPROCMGR_ELMRUNTIME_SMOKE_CHECK_HPP
