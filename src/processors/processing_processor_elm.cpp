#include "processors/processing_processor_elm.hpp"

#include <elmruntime/ElmEnvelope.hpp>
#include <elmruntime/ElmEntryPreflight.hpp>
#include <elmruntime/ElmModelCache.hpp>
#include <elmruntime/ElmRuntimeError.hpp>
#include <elmruntime/ElmStopStringStreamBuf.hpp>

#include <capability/capability_validator.hpp>

// SGNSProcMain.hpp pulls the full generated type set (including Elm.hpp via
// its own quoted include resolved against generated/ on the include path).
#include <SGNSProcMain.hpp>

#include "util/sha256.hpp"

#include <util/sgprocmgr-logger.hpp>

#include <atomic>
#include <cstdio>
#include <memory>
#include <mutex>
#include <sstream>
#include <string>
#include <vector>

#if defined( SGPROC_HAS_MNN_LLM )

// The ENTIRE MNN surface of this processor lives in this TU, inside the gate
// (the ElmSmokeCheck.cpp include-isolation pattern). The header above
// forward-declares MNN::Transformer::Llm only; no header reachable from
// processing_processor_elm.hpp names an MNN type.
#include "processingbase/vulkan_init_guard.hpp"

#include <llm/llm.hpp>

#include <nlohmann/json.hpp>

namespace sgns::sgprocessing
{
    namespace
    {
        /// Builds the error-envelope ProcessingResult (D-11: published terminal
        /// result, never a throw). Called on every pre-generation failure path.
        ProcessingResult MakeErrorResult( std::vector<std::vector<uint8_t>> &chunkhashes,
                                          const std::string                 &workItemId,
                                          const std::string                 &manifestHash,
                                          const std::string                 &code,
                                          const std::string                 &message,
                                          int64_t                            promptTokens = 0,
                                          int64_t                            completionTokens = 0 )
        {
            sgns::elmruntime::ElmEnvelope envelope;
            envelope.work_item_id        = workItemId;
            envelope.text                = {};
            envelope.prompt_tokens      = promptTokens;
            envelope.completion_tokens  = completionTokens;
            envelope.finish_reason      = sgns::elmruntime::ElmFinishReason::Error;
            envelope.model_manifest_hash = manifestHash;
            envelope.error = sgns::elmruntime::ElmEnvelopeError{ code, message };

            const std::string json     = sgns::elmruntime::ElmEnvelopeToJson( envelope );
            const auto        resultHash = sgprocmanagersha::sha256( json.c_str(), json.size() );
            chunkhashes.push_back( resultHash );

            ProcessingResult result;
            result.hash           = resultHash;
            result.output_buffers = std::make_shared<
                std::pair<std::vector<std::string>, std::vector<std::vector<char>>>>();
            result.output_buffers->first.push_back( "" );
            result.output_buffers->second.push_back( std::vector<char>( json.begin(), json.end() ) );
            result.error = ProcessingError{ ProcessingErrorStage::RESOURCE_RESOLUTION, code + ": " + message };
            return result;
        }
    } // namespace

    ProcessingResult ElmProcessor::StartProcessing( std::vector<std::vector<uint8_t>> &chunkhashes,
                                                    const sgns::IoDeclaration         &proc,
                                                    std::vector<char>                 &promptData,
                                                    std::vector<char>                 &modelFile,
                                                    const std::vector<sgns::Parameter> *parameters,
                                                    const ExecutionContext            &execCtx )
    {
        (void) proc;
        (void) promptData;
        (void) modelFile;
        (void) parameters;

        if ( execCtx.cancelToken.IsCancelled() )
        {
            return ProcessingResult{ {}, nullptr, {},
                ProcessingError{ ProcessingErrorStage::CANCELLED, "ELM work item cancelled" } };
        }

        return ProcessingResult{ {}, nullptr, {},
            ProcessingError{ ProcessingErrorStage::RESOURCE_RESOLUTION,
                "ELM work items must enter via StartProcessingElm (Phase 4 wires grid routing)" } };
    }

    ProcessingResult ElmProcessor::StartProcessingElm( std::vector<std::vector<uint8_t>> &chunkhashes,
                                                        const std::string                 &promptText,
                                                        const std::vector<std::string>    &stopStrings,
                                                        const sgns::Elm                   &elm,
                                                        const ExecutionContext            &execCtx,
                                                        std::shared_ptr<sgns::elmruntime::ElmModelCache> cache,
                                                        CapabilityValidator               *capabilityValidator )
    {
        const std::string passId = elm.get_work_item_id();

        // (a) Pre-cancel check BEFORE any work (the mnn_llm.cpp:137-143 pattern).
        if ( execCtx.cancelToken.IsCancelled() )
        {
            return ProcessingResult{ {}, nullptr, {},
                ProcessingError{ ProcessingErrorStage::CANCELLED, "ELM work item cancelled" } };
        }

        // (b) Materialize the by-value quicktype optionals into named locals
        // ONCE (the Phase 1 UB rule), then default-fill settings per Phase 1
        // semantics. NEVER clamp -- bounds were enforced at parse (Phase 1 D-05).
        const auto generationOpt    = elm.get_generation(); // boost::optional<ElmGeneration> BY VALUE
        const auto maxOutputOpt     = generationOpt ? generationOpt->get_max_output_tokens() : boost::optional<int64_t>{};
        const auto seedOpt          = generationOpt ? generationOpt->get_seed() : boost::optional<int64_t>{};
        const auto temperatureOpt   = generationOpt ? generationOpt->get_temperature() : boost::optional<double>{};
        const auto topPOpt          = generationOpt ? generationOpt->get_top_p() : boost::optional<double>{};

        const bool     hasMaxOutputTokens = maxOutputOpt.has_value();
        const int64_t  maxOutputTokens    = maxOutputOpt.value_or( 0 );
        const bool     hasSeed            = seedOpt.has_value();
        const int64_t  seed               = seedOpt.value_or( 0 );
        // Phase 1 D-04 named defaults: temperature=1.0, top_p=1.0.
        const double   temperature        = temperatureOpt.value_or( 1.0 );
        const double   topP               = topPOpt.value_or( 1.0 );

        const std::string &manifestUri  = elm.get_model_manifest_uri();
        const std::string &manifestHash = elm.get_model_manifest_hash();
        const std::string &workItemId   = elm.get_work_item_id();

        // (c) Cache acquire: the ONLY model source (T-03-07). The pin is held
        // by value in this scope -- RAII releases on EVERY return (SC-3).
        if ( !cache )
        {
            return MakeErrorResult( chunkhashes, workItemId, manifestHash,
                "CACHE_DIR_UNSET", "ELM processor invoked without a model cache" );
        }
        auto acquireResult = cache->Acquire( manifestUri, manifestHash );
        if ( !acquireResult )
        {
            const auto err = static_cast<sgns::elmruntime::ElmRuntimeError>( acquireResult.error().value() );
            m_logger->error( "ELM[{}]: cache acquire failed: {}", workItemId, acquireResult.error().message() );
            return MakeErrorResult( chunkhashes, workItemId, manifestHash,
                "ELM_ACQUIRE_FAILED", acquireResult.error().message() );
        }
        const sgns::elmruntime::ElmCachePin pin = std::move( acquireResult.value() );

        // (d) Preflight (Phase 2's deferred wiring) BEFORE session creation.
        // The manifest read + extraction goes through the plain-value bridge
        // (ElmEntryPreflight): this TU includes the ROOT generated set for
        // sgns::Elm, which cannot coexist with the fallback manifest set
        // (ClassMemberConstraints redefinition) -- the bridge TU holds the
        // fallback includes and hands back two uint64 values.
        {
            auto preflight = sgns::elmruntime::PreflightPinnedEntry( pin.GetDir(), manifestHash );
            if ( !preflight )
            {
                const auto err = static_cast<sgns::elmruntime::ElmRuntimeError>( preflight.error().value() );
                (void) err;
                return MakeErrorResult( chunkhashes, workItemId, pin.GetHash(),
                    "MANIFEST_INVALID", "pinned entry preflight failed: " + preflight.error().message() );
            }
            if ( capabilityValidator != nullptr )
            {
                // Degraded-0 convention: 0-valued snapshot legs skip, never fail
                // spuriously (capability_validator.cpp's own convention).
                std::atomic<bool> executable{ false };
                std::string       unmetDetails;
                capabilityValidator->CheckElmResources( preflight.value().requiredMemoryBytes,
                    preflight.value().totalArtifactBytes,
                    [ &executable, &unmetDetails ]( const CanExecuteResult &result ) {
                        executable.store( result.executable );
                        for ( const auto &u : result.unmet )
                        {
                            unmetDetails += u.detail + "; ";
                        }
                    } );
                if ( !executable.load() )
                {
                    m_logger->error( "ELM[{}]: resource preflight unmet: {}", workItemId, unmetDetails );
                    return MakeErrorResult( chunkhashes, workItemId, pin.GetHash(),
                        "ELM_RESOURCE_PREFLIGHT", "resource preflight unmet: " + unmetDetails );
                }
            }
            else
            {
                m_logger->info( "ELM[{}]: resource preflight SKIPPED (no validator injected -- test seam)", workItemId );
            }
        }

        // (e) Progress: LOAD_MODEL.
        if ( execCtx.progressCallback )
        {
            execCtx.progressCallback( ProgressEvent::ForMNN( passId, MNNStage::LOAD_MODEL, 10.0f ) );
        }

        // (f) Session creation under the SPLIT locks (D-01/D-03).
        // Scoped block 1: createLLM under VulkanInitMutex ONLY (the GPU-init
        // window; render/MNN processors never wait on a weight load).
        MNN::Transformer::Llm *llm = nullptr;
        {
            std::lock_guard<std::mutex> gpuInit( sgns::sgprocessing::VulkanInitMutex() );
            llm = MNN::Transformer::Llm::createLLM( pin.GetDir() );
        }
        if ( llm == nullptr )
        {
            m_logger->error( "ELM[{}]: createLLM returned null for {}", workItemId, pin.GetDir() );
            return MakeErrorResult( chunkhashes, workItemId, pin.GetHash(),
                "SMOKE_CHECK_FAILED", "createLLM returned null for the pinned bundle" );
        }

        // set_config BEFORE load (Pitfall 3 -- hard ordering: the Sampler is
        // constructed inside load() and reads config_ in its ctor; a key
        // applied after load never lands). Build ONE JSON object with only the
        // present keys: max_new_tokens/seed ONLY when set (absent max tokens =
        // model's own llm_config.json default via the -1 sentinel; absent seed
        // = non-deterministic by request). Deliberately NO timeout_ms (planner
        // resolution: deadline semantics flow through the cancel token ->
        // cancelled per Phase 1 D-03; timeout_ms would mislabel overruns).
        {
            nlohmann::json config;
            config["temperature"] = temperature;
            config["top_p"]       = topP;
            if ( hasMaxOutputTokens )
            {
                config["max_new_tokens"] = maxOutputTokens;
            }
            if ( hasSeed )
            {
                config["seed"] = seed;
            }
            if ( !llm->set_config( config.dump() ) )
            {
                MNN::Transformer::Llm::destroy( llm );
                return MakeErrorResult( chunkhashes, workItemId, pin.GetHash(),
                    "ELM_CONFIG_FAILED", "set_config rejected the generation settings" );
            }

            // ASSERT application (Pitfall 2 -- set_config silently ignores
            // unknown keys): parse dump_config() back and verify EVERY key just
            // set round-trips with the exact value. A missing seed is the SC-2
            // no-silent-no-op trap.
            try
            {
                const nlohmann::json dumped = nlohmann::json::parse( llm->dump_config() );
                const auto checkKey = [ &dumped ]( const char *key, const nlohmann::json &expected ) -> bool {
                    const auto it = dumped.find( key );
                    return it != dumped.end() && *it == expected;
                };
                if ( !checkKey( "temperature", nlohmann::json( temperature ) )
                    || !checkKey( "top_p", nlohmann::json( topP ) )
                    || ( hasMaxOutputTokens && !checkKey( "max_new_tokens", nlohmann::json( maxOutputTokens ) ) )
                    || ( hasSeed && !checkKey( "seed", nlohmann::json( seed ) ) ) )
                {
                    MNN::Transformer::Llm::destroy( llm );
                    m_logger->error( "ELM[{}]: dump_config round-trip mismatch: {}", workItemId, llm->dump_config() );
                    return MakeErrorResult( chunkhashes, workItemId, pin.GetHash(),
                        "ELM_CONFIG_FAILED", "generation settings failed to apply (dump_config round-trip mismatch)" );
                }
            }
            catch ( const std::exception &parseError )
            {
                MNN::Transformer::Llm::destroy( llm );
                return MakeErrorResult( chunkhashes, workItemId, pin.GetHash(),
                    "ELM_CONFIG_FAILED", std::string( "dump_config unparseable: " ) + parseError.what() );
            }
        }

        // Scoped block 2: load under LlmLoadMutex (serializes against other
        // LLM loads only; NOT against the rest of the grid).
        {
            std::lock_guard<std::mutex> llmLoad( sgns::sgprocessing::LlmLoadMutex() );
            if ( !llm->load() )
            {
                MNN::Transformer::Llm::destroy( llm );
                m_logger->error( "ELM[{}]: Llm::load() failed for {}", workItemId, pin.GetDir() );
                return MakeErrorResult( chunkhashes, workItemId, pin.GetHash(),
                    "SMOKE_CHECK_FAILED", "Llm::load() failed for the pinned bundle" );
            }
        }

        // (g) Teardown registration immediately after a successful load
        // (the mnn_llm.cpp:172-175 pattern); post-load cancel re-check.
        PushTeardown( [llm]() { MNN::Transformer::Llm::destroy( llm ); } );

        if ( execCtx.cancelToken.IsCancelled() )
        {
            RunTeardown();
            sgns::elmruntime::ElmEnvelope envelope;
            envelope.work_item_id        = workItemId;
            envelope.prompt_tokens      = 0;
            envelope.completion_tokens  = 0;
            envelope.finish_reason      = sgns::elmruntime::ElmFinishReason::Cancelled;
            envelope.model_manifest_hash = pin.GetHash();
            const std::string json       = sgns::elmruntime::ElmEnvelopeToJson( envelope );
            const auto        resultHash = sgprocmanagersha::sha256( json.c_str(), json.size() );
            chunkhashes.push_back( resultHash );
            ProcessingResult result;
            result.hash           = resultHash;
            result.output_buffers = std::make_shared<
                std::pair<std::vector<std::string>, std::vector<std::vector<char>>>>();
            result.output_buffers->first.push_back( "" );
            result.output_buffers->second.push_back( std::vector<char>( json.begin(), json.end() ) );
            return result;
        }

        if ( execCtx.progressCallback )
        {
            execCtx.progressCallback( ProgressEvent::ForMNN( passId, MNNStage::RUN, 50.0f ) );
        }

        // (h) Generation. The streambuf is the single stop-string injection
        // seam: stop strings arrive ONLY via the stopStrings PARAMETER (the
        // Phase 1 schema has no stop field -- Phase 4's amendment; an empty
        // vector means no stop-string scanning). onCancelMatch fires the D-13
        // fork-patch cancel; SetExternalCancelPoll wires the job cancel token
        // (the Pitfall 4 streambuf-poll resolution: latency ~ one token, zero
        // ProcessManager callback changes; two intent latches per D-09).
        sgns::elmruntime::ElmStopStringStreamBuf stopBuf(
            stopStrings, [llm]() { llm->cancel(); } );
        stopBuf.SetExternalCancelPoll( [ &execCtx ]() { return execCtx.cancelToken.IsCancelled(); } );

        std::ostream os( &stopBuf );
        // end_with is the EXPLICIT empty string (Pitfall 5: nullptr defaults
        // to "\n" and the stop-token path writes end_with into the stream).
        // max_new_tokens: the -1 sentinel defers capping to the model's own
        // llm_config.json default when the work item omitted the key (Phase 1
        // "unset = model default"; NO 512 hard default).
        llm->response( promptText, &os, "",
            hasMaxOutputTokens ? static_cast<int>( maxOutputTokens ) : -1 );

        // (i) Reconcile (Pitfall 1/10): LlmContext is the authority.
        const MNN::Transformer::LlmContext *ctx = llm->getContext();
        const int64_t promptTokens     = ctx ? static_cast<int64_t>( ctx->prompt_len ) : 0;
        const int64_t completionTokens = ctx ? static_cast<int64_t>( ctx->output_tokens.size() ) : 0;
        if ( ctx && ctx->gen_seq_len != static_cast<int>( ctx->output_tokens.size() ) )
        {
            // Early-stop paths diverge by design (the stop token is counted
            // before the loop breaks); warn, never average or pick max.
            m_logger->warn( "ELM[{}]: gen_seq_len ({}) != output_tokens.size() ({})",
                workItemId, ctx->gen_seq_len, ctx->output_tokens.size() );
        }

        // Finish-reason mapping (the D-08..D-11 table).
        sgns::elmruntime::ElmFinishReason finishReason;
        std::string                       envelopeText;
        if ( stopBuf.Matched() ) // stop-string match -> stop with truncated text (D-07/D-09)
        {
            finishReason  = sgns::elmruntime::ElmFinishReason::Stop;
            envelopeText  = std::string( stopBuf.VisibleText() );
        }
        else if ( stopBuf.CancelRequested() || execCtx.cancelToken.IsCancelled() )
        {
            finishReason  = sgns::elmruntime::ElmFinishReason::Cancelled;
            envelopeText  = stopBuf.AccumulatedText();
        }
        else if ( ctx && ctx->status == MNN::Transformer::LlmStatus::MAX_TOKENS_FINISHED )
        {
            finishReason  = sgns::elmruntime::ElmFinishReason::MaxTokens;
            envelopeText  = stopBuf.AccumulatedText();
        }
        else if ( ctx && ctx->status == MNN::Transformer::LlmStatus::TIMEOUT )
        {
            // D-08: TIMEOUT is an execution fault (deadline firing flows
            // through the cancel token -> Cancelled, not here).
            finishReason = sgns::elmruntime::ElmFinishReason::Error;
            envelopeText = stopBuf.AccumulatedText();
        }
        else if ( ctx && ctx->status == MNN::Transformer::LlmStatus::NORMAL_FINISHED )
        {
            finishReason  = sgns::elmruntime::ElmFinishReason::Stop;
            envelopeText  = stopBuf.AccumulatedText();
        }
        else
        {
            // INTERNAL_ERROR, or the CHECK_LLM_RUNNING early-return-with-empty-
            // output case (Pitfall 10): never "stop with empty text".
            finishReason = sgns::elmruntime::ElmFinishReason::Error;
            envelopeText = stopBuf.AccumulatedText();
        }

        // (j) Envelope + output construction.
        sgns::elmruntime::ElmEnvelope envelope;
        envelope.work_item_id        = workItemId;
        envelope.text                = envelopeText;
        envelope.prompt_tokens      = promptTokens;
        envelope.completion_tokens  = completionTokens;
        envelope.finish_reason      = finishReason;
        envelope.model_manifest_hash = pin.GetHash();
        if ( finishReason == sgns::elmruntime::ElmFinishReason::Error )
        {
            std::string statusName = "INTERNAL_ERROR";
            if ( ctx && ctx->status == MNN::Transformer::LlmStatus::TIMEOUT )
            {
                statusName = "TIMEOUT";
            }
            else if ( ctx && ctx->status == MNN::Transformer::LlmStatus::NOT_LOADED )
            {
                statusName = "NOT_LOADED";
            }
            envelope.error = sgns::elmruntime::ElmEnvelopeError{
                statusName, "generation ended in LlmStatus " + statusName };
        }

        const std::string json     = sgns::elmruntime::ElmEnvelopeToJson( envelope );
        const auto        resultHash = sgprocmanagersha::sha256( json.c_str(), json.size() );

        // Budget check on the ENVELOPE size (never a silent truncation).
        if ( execCtx.maxOutputArtifactBytes > 0 && json.size() > execCtx.maxOutputArtifactBytes )
        {
            RunTeardown();
            return ProcessingResult{ {}, nullptr, {},
                ProcessingError{ ProcessingErrorStage::BUDGET_EXCEEDED,
                    "Envelope size " + std::to_string( json.size() ) + " exceeds budget "
                        + std::to_string( execCtx.maxOutputArtifactBytes ) } };
        }

        chunkhashes.push_back( resultHash );

        ProcessingResult result;
        result.hash           = resultHash;
        result.output_buffers = std::make_shared<
            std::pair<std::vector<std::string>, std::vector<std::vector<char>>>>();
        result.output_buffers->first.push_back( "" );
        result.output_buffers->second.push_back( std::vector<char>( json.begin(), json.end() ) );

        m_progress = 100.0f;
        if ( execCtx.progressCallback )
        {
            execCtx.progressCallback( ProgressEvent::ForMNN( passId, MNNStage::READ_OUTPUT, 100.0f ) );
        }

        m_logger->info( "ELM[{}]: generation complete ({} byte(s), finish_reason={})",
            workItemId, json.size(), sgns::elmruntime::ToString( finishReason ) );

        RunTeardown();
        return result;
    }

} // namespace sgns::sgprocessing

#else // !SGPROC_HAS_MNN_LLM

// Fail-closed fallback (ElmSmokeCheck.cpp's exact idiom): without the engine
// there is no way to execute an ELM work item; every entry point returns a
// structured RESOURCE_RESOLUTION error instead of linking away.
namespace sgns::sgprocessing
{
    ProcessingResult ElmProcessor::StartProcessing( std::vector<std::vector<uint8_t>> &chunkhashes,
                                                    const sgns::IoDeclaration         &proc,
                                                    std::vector<char>                 &promptData,
                                                    std::vector<char>                 &modelFile,
                                                    const std::vector<sgns::Parameter> *parameters,
                                                    const ExecutionContext            &execCtx )
    {
        (void) chunkhashes;
        (void) proc;
        (void) promptData;
        (void) modelFile;
        (void) parameters;
        (void) execCtx;
        return ProcessingResult{ {}, nullptr, {},
            ProcessingError{ ProcessingErrorStage::RESOURCE_RESOLUTION,
                "ElmProcessor built without MNN LLM support" } };
    }

    ProcessingResult ElmProcessor::StartProcessingElm( std::vector<std::vector<uint8_t>> &chunkhashes,
                                                        const std::string                 &promptText,
                                                        const std::vector<std::string>    &stopStrings,
                                                        const sgns::Elm                   &elm,
                                                        const ExecutionContext            &execCtx,
                                                        std::shared_ptr<sgns::elmruntime::ElmModelCache> cache,
                                                        CapabilityValidator               *capabilityValidator )
    {
        (void) chunkhashes;
        (void) promptText;
        (void) stopStrings;
        (void) elm;
        (void) execCtx;
        (void) cache;
        (void) capabilityValidator;
        return ProcessingResult{ {}, nullptr, {},
            ProcessingError{ ProcessingErrorStage::RESOURCE_RESOLUTION,
                "ElmProcessor built without MNN LLM support" } };
    }
} // namespace sgns::sgprocessing

#endif // SGPROC_HAS_MNN_LLM
