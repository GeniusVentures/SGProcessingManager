#include <elmruntime/ElmSmokeCheck.hpp>

#include <elmruntime/ElmRuntimeError.hpp>

#include <util/sgprocmgr-logger.hpp>

#if defined( SGPROC_HAS_MNN_LLM )

// The ENTIRE MNN surface of the elmruntime layer lives in this TU, inside the
// gate. No header reachable from ElmSmokeCheck.hpp names an MNN type, so every
// consumer compiles without <llm/llm.hpp> (the processing_processor_mnn_llm
// include-isolation pattern).
#include "processingbase/vulkan_init_guard.hpp"

#include <llm/llm.hpp>

#include <mutex>
#include <sstream>
#include <string>

namespace sgns::elmruntime
{
    namespace
    {
        sgns::sgprocmanager::Logger SmokeCheckLogger()
        {
            return sgns::sgprocmanager::createLogger( "ElmSmokeCheck" );
        }

        /// Defensive trailing-slash normalization (P2-5): the cache guarantees a
        /// trailing separator, but LlmConfig uses the string VERBATIM as a path
        /// prefix on Windows too -- a missing separator would resolve files as
        /// "<dir>llm_config.json". Normalize the materializer's way if violated.
        std::string EnsureTrailingSlash( const std::string &dir )
        {
            std::string out = dir;
            if ( !out.empty() && out.back() != '/' && out.back() != '\\' )
            {
                out += '/';
            }
            return out;
        }
    } // namespace

    SmokeCheckFn MakeMnnLlmSmokeCheck()
    {
        return []( const std::string &bundleDirWithTrailingSlash ) -> outcome::result<void> {
            const auto logger = SmokeCheckLogger();
            const std::string dir = EnsureTrailingSlash( bundleDirWithTrailingSlash );

            // One load per entry under the process-wide init mutex: adapting
            // MNN_Llm::LoadModel's exact sequence (processing_processor_mnn_llm.cpp:86-93).
            // Scope the lock to the probe: createLLM + load + the 1-token response
            // are one indivisible usability check.
            std::lock_guard<std::mutex> lock( sgns::sgprocessing::VulkanInitMutex() );

            MNN::Transformer::Llm *llm = MNN::Transformer::Llm::createLLM( dir );
            if ( llm == nullptr )
            {
                logger->error( "ElmSmokeCheck: createLLM returned null for {}", dir );
                return outcome::failure( ElmRuntimeError::SMOKE_CHECK_FAILED );
            }

            // Greedy = deterministic probe (no RNG); 1 new token = minimal work.
            // set_config before load so the sampler config applies to the probe.
            if ( !llm->set_config( R"({"sampler_type":"greedy","max_new_tokens":1})" ) )
            {
                logger->error( "ElmSmokeCheck: set_config failed for {}", dir );
                MNN::Transformer::Llm::destroy( llm );
                return outcome::failure( ElmRuntimeError::SMOKE_CHECK_FAILED );
            }

            if ( !llm->load() )
            {
                // Wrong-tokenizer / garbage bundles die here (Llm::load() checks
                // llm_config.json, llm.mnn, llm.mnn.weight, tokenizer.txt
                // unconditionally -- llm.cpp:265-283): SC-2's structured error.
                logger->error( "ElmSmokeCheck: Llm::load() failed for {}", dir );
                MNN::Transformer::Llm::destroy( llm );
                return outcome::failure( ElmRuntimeError::SMOKE_CHECK_FAILED );
            }

            std::ostringstream oss;
            llm->response( "Hello", &oss, nullptr, 1 );
            const std::string text = oss.str();
            if ( text.empty() )
            {
                // A loaded bundle that still produces no token is not usable.
                logger->error( "ElmSmokeCheck: 1-token response produced empty output for {}", dir );
                MNN::Transformer::Llm::destroy( llm );
                return outcome::failure( ElmRuntimeError::SMOKE_CHECK_FAILED );
            }

            MNN::Transformer::Llm::destroy( llm );
            return outcome::success();
        };
    }
} // namespace sgns::elmruntime

#else // !SGPROC_HAS_MNN_LLM

// Fail-closed fallback (SC-2): without the engine there is NO way to prove a
// bundle loadable, so the probe always fails and entries can never be marked
// usable. There is no bypass for this.
namespace sgns::elmruntime
{
    namespace
    {
        sgns::sgprocmanager::Logger SmokeCheckLogger()
        {
            return sgns::sgprocmanager::createLogger( "ElmSmokeCheck" );
        }
    } // namespace

    SmokeCheckFn MakeMnnLlmSmokeCheck()
    {
        return []( const std::string & ) -> outcome::result<void> {
            SmokeCheckLogger()->error( "ElmSmokeCheck: MNN LLM support absent in this build -- failing closed" );
            return outcome::failure( ElmRuntimeError::SMOKE_CHECK_UNAVAILABLE );
        };
    }
} // namespace sgns::elmruntime

#endif // SGPROC_HAS_MNN_LLM
