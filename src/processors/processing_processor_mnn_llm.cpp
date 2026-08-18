#include "processors/processing_processor_mnn_llm.hpp"
#include "processingbase/vulkan_init_guard.hpp"

#include <chrono>
#include <cstdio>
#include <filesystem>
#include <fstream>
#include <mutex>
#include <sstream>
#include <string>

#include <MNN/llm/llm.hpp>

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

    ProcessingResult MNN_Llm::StartProcessing( std::vector<std::vector<uint8_t>> &chunkhashes,
                                                const sgns::IoDeclaration         &proc,
                                                std::vector<char>                 &promptData,
                                                std::vector<char>                 &modelFile,
                                                const std::vector<sgns::Parameter> *parameters,
                                                const ExecutionContext            &execCtx )
    {
        std::vector<uint8_t> modelFileBytes( modelFile.begin(), modelFile.end() );

        if ( modelFileBytes.empty() )
        {
            m_logger->error( "MNN LLM: no model file provided" );
            return ProcessingResult{ {}, nullptr, {},
                ProcessingError{ ProcessingErrorStage::RESOURCE_RESOLUTION,
                    "MNN LLM model failed to load: empty model buffer" } };
        }

        MNN::Transformer::Llm *llm = LoadModel( modelFileBytes );
        if ( !llm )
        {
            m_logger->error( "MNN LLM: model failed to load" );
            return ProcessingResult{ {}, nullptr, {},
                ProcessingError{ ProcessingErrorStage::RESOURCE_RESOLUTION, "MNN LLM model failed to load" } };
        }

        m_logger->info( "MNN LLM: model loaded successfully from materialized directory" );

        // Task 3 (this plan) extends this function past a successful load with the
        // full generation path: PushTeardown() registration, cancellation checks,
        // maxNewTokens-bounded response(), progress events, and hashing/output
        // population. Until then, intentionally return immediately after load,
        // per this task's own scope (skeleton + fail-closed load path only).
        (void) proc;
        (void) promptData;
        (void) parameters;
        (void) execCtx;
        (void) chunkhashes;
        MNN::Transformer::Llm::destroy( llm );

        return ProcessingResult{};
    }
}
