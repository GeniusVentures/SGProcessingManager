#include "processors/processing_processor_mnn_llm.hpp"
#include "processingbase/vulkan_init_guard.hpp"

#include <chrono>
#include <cstdio>
#include <filesystem>
#include <fstream>
#include <mutex>
#include <sstream>
#include <string>

#include "util/sha256.hpp"

#include <llm/llm.hpp>

namespace sgns::sgprocessing
{
    namespace
    {
        // Materializes MNN LLM model bytes to a fresh temp directory. No existing
        // materialize-to-disk mechanism was found elsewhere in SGProcessingManager --
        // every other MNN processor loads models via MNN::Interpreter::createFromBuffer's
        // in-memory API, which needs no filesystem path at all. MNN::Transformer::Llm's
        // createLLM(), by contrast, structurally requires a directory path (it expects
        // to find llm_config.json and weight files alongside each other on disk), so
        // this is a new, minimal helper scoped to this processor only.
        bool MaterializeModelToTempDir( const std::vector<uint8_t> &modelBytes, std::string &outDir )
        {
            namespace fs = std::filesystem;

            std::error_code ec;
            fs::path        base = fs::temp_directory_path( ec );
            if ( ec )
            {
                return false;
            }

            const auto stamp = std::chrono::high_resolution_clock::now().time_since_epoch().count();
            fs::path   dir   = base / ( "sgproc_mnn_llm_" + std::to_string( stamp ) );
            fs::create_directories( dir, ec );
            if ( ec )
            {
                return false;
            }

            fs::path      modelPath = dir / "model.mnn";
            std::ofstream out( modelPath, std::ios::binary );
            if ( !out )
            {
                return false;
            }
            out.write( reinterpret_cast<const char *>( modelBytes.data() ),
                       static_cast<std::streamsize>( modelBytes.size() ) );
            out.close();
            if ( !out )
            {
                return false;
            }

            std::string dirStr = dir.string();
            if ( !dirStr.empty() && dirStr.back() != '/' && dirStr.back() != '\\' )
            {
                dirStr += '/';
            }
            outDir = dirStr;
            return true;
        }
    } // namespace

    MNN::Transformer::Llm *MNN_Llm::LoadModel( const std::vector<uint8_t> &modelFileBytes )
    {
        if ( modelFileBytes.empty() )
        {
            return nullptr;
        }

        std::string tempDir;
        if ( !MaterializeModelToTempDir( modelFileBytes, tempDir ) )
        {
            m_logger->error( "MNN LLM: failed to materialize model buffer to a temp directory" );
            return nullptr;
        }

        MNN::Transformer::Llm *llm = nullptr;
        {
            std::lock_guard<std::mutex> lock( sgns::sgprocessing::VulkanInitMutex() );
            llm = MNN::Transformer::Llm::createLLM( tempDir );
            if ( llm && !llm->load() )
            {
                MNN::Transformer::Llm::destroy( llm );
                llm = nullptr;
            }
        }

        return llm;
    }

    namespace
    {
        // Reads the schema-declared "maxNewTokens" INT parameter, mirroring MNN_String's
        // "maxLength" find-by-name idiom (processing_processor_mnn_string.cpp). Defaults
        // to 512 (matching MNNInferenceEngine::Config::kDefaultMaxTokens) when absent or
        // invalid -- T-04-05 (DoS via unbounded generation length) mitigation: this bound
        // is always applied, never an unbounded loop.
        int ResolveMaxNewTokens( const std::vector<sgns::Parameter> *parameters )
        {
            constexpr int kDefaultMaxNewTokens = 512;
            int           maxNewTokens         = kDefaultMaxNewTokens;
            if ( parameters )
            {
                for ( const auto &param : *parameters )
                {
                    if ( param.get_name() == "maxNewTokens" && param.get_type() == sgns::ParameterType::INT )
                    {
                        const auto &def = param.get_parameter_default();
                        if ( def.is_number_integer() && def.get<int>() > 0 )
                        {
                            maxNewTokens = def.get<int>();
                        }
                        break;
                    }
                }
            }
            return maxNewTokens;
        }
    } // namespace

    ProcessingResult MNN_Llm::StartProcessing( std::vector<std::vector<uint8_t>> &chunkhashes,
                                                const sgns::IoDeclaration         &proc,
                                                std::vector<char>                 &promptData,
                                                std::vector<char>                 &modelFile,
                                                const std::vector<sgns::Parameter> *parameters,
                                                const ExecutionContext            &execCtx )
    {
        const std::string    passId = proc.get_name();
        std::vector<uint8_t> modelFileBytes( modelFile.begin(), modelFile.end() );

        // Check cancellation before doing any work at all -- including before the
        // (potentially expensive) model materialization/load attempt below -- so a
        // job cancelled prior to dispatch never pays that cost. Re-checked again
        // after a successful load (T-04-07) since PushTeardown() only has anything
        // to unwind from that point on.
        if ( execCtx.cancelToken.IsCancelled() )
        {
            return ProcessingResult{ {}, nullptr, {},
                ProcessingError{ ProcessingErrorStage::CANCELLED, "LLM pass cancelled" } };
        }

        if ( modelFileBytes.empty() )
        {
            m_logger->error( "MNN LLM: no model file provided" );
            return ProcessingResult{ {}, nullptr, {},
                ProcessingError{ ProcessingErrorStage::RESOURCE_RESOLUTION,
                    "MNN LLM model failed to load: empty model buffer" } };
        }

        if ( execCtx.progressCallback )
        {
            execCtx.progressCallback( ProgressEvent::ForMNN( passId, MNNStage::LOAD_MODEL, 10.0f ) );
        }

        MNN::Transformer::Llm *llm = LoadModel( modelFileBytes );
        if ( !llm )
        {
            m_logger->error( "MNN LLM: model failed to load" );
            return ProcessingResult{ {}, nullptr, {},
                ProcessingError{ ProcessingErrorStage::RESOURCE_RESOLUTION, "MNN LLM model failed to load" } };
        }

        // D-14/T-04-07: register teardown immediately after a successful load so a
        // cancelled/timed-out job cannot leak this long-lived MNN LLM session --
        // every return path below this point goes through RunTeardown().
        PushTeardown( [llm]() { MNN::Transformer::Llm::destroy( llm ); } );

        if ( execCtx.cancelToken.IsCancelled() )
        {
            RunTeardown();
            return ProcessingResult{ {}, nullptr, {},
                ProcessingError{ ProcessingErrorStage::CANCELLED, "LLM pass cancelled" } };
        }

        if ( execCtx.progressCallback )
        {
            execCtx.progressCallback( ProgressEvent::ForMNN( passId, MNNStage::CREATE_SESSION, 25.0f ) );
        }

        const int   maxNewTokens = ResolveMaxNewTokens( parameters );
        std::string promptText( promptData.begin(), promptData.end() );

        if ( execCtx.progressCallback )
        {
            execCtx.progressCallback( ProgressEvent::ForMNN( passId, MNNStage::RUN, 50.0f ) );
        }

        // Port MNN's own native autoregressive API -- the exact call NEO-SWARM's
        // InferViaMnnLlm() already makes correctly -- NOT a hand-rolled sampling loop.
        // MNN::Transformer::Llm::response() handles tokenization, KV-cache, sampling,
        // and stopping criteria internally.
        std::ostringstream oss;
        llm->response( promptText, &oss, nullptr, maxNewTokens );

        // MNN::Transformer::Llm::response() is a single blocking call with no
        // cancellation hook exposed by its public API, so true mid-generation
        // cancellation cannot be implemented without deeper MNN API support (see
        // SUMMARY.md deviations). Re-checking here at minimum ensures a cancellation
        // that raced with generation is still surfaced as a structured CANCELLED
        // result rather than a false-success return.
        if ( execCtx.cancelToken.IsCancelled() )
        {
            RunTeardown();
            return ProcessingResult{ {}, nullptr, {},
                ProcessingError{ ProcessingErrorStage::CANCELLED, "LLM pass cancelled" } };
        }

        if ( execCtx.progressCallback )
        {
            execCtx.progressCallback( ProgressEvent::ForMNN( passId, MNNStage::READ_OUTPUT, 90.0f ) );
        }

        const std::string outputText = oss.str();
        const auto        subTaskResultHash =
            sgprocmanagersha::sha256( outputText.c_str(), outputText.size() );
        chunkhashes.push_back( subTaskResultHash );

        // Output budget check (EXEC-03), mirroring MNN_Tensor's existing pattern.
        if ( execCtx.maxOutputArtifactBytes > 0 && outputText.size() > execCtx.maxOutputArtifactBytes )
        {
            RunTeardown();
            return ProcessingResult{ {}, nullptr, {},
                ProcessingError{ ProcessingErrorStage::BUDGET_EXCEEDED,
                    "Output artifact size " + std::to_string( outputText.size() ) + " exceeds budget " +
                        std::to_string( execCtx.maxOutputArtifactBytes ) } };
        }

        ProcessingResult result;
        result.hash            = subTaskResultHash;
        result.output_buffers  = std::make_shared<std::pair<std::vector<std::string>, std::vector<std::vector<char>>>>();
        result.output_buffers->first.push_back( "" );
        result.output_buffers->second.push_back( std::vector<char>( outputText.begin(), outputText.end() ) );

        m_progress = 100.0f;
        if ( execCtx.progressCallback )
        {
            execCtx.progressCallback( ProgressEvent::ForMNN( passId, MNNStage::READ_OUTPUT, 100.0f ) );
        }

        m_logger->info( "MNN LLM generation complete: {} output byte(s)", outputText.size() );

        RunTeardown();
        return result;
    }
}
