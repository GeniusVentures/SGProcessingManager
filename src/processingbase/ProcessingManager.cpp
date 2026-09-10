#include <processingbase/ProcessingManager.hpp>

#include <datasplitter/ImageSplitter.hpp>
#include "FileManager.hpp"
#include "URLStringUtil.h"
#include "shaders/shader_compiler.hpp"

#include <boost/asio/deadline_timer.hpp>
#include <boost/date_time/posix_time/posix_time.hpp>
#include <chrono>
#include <cstring>
#include <map>
#include <set>
#include "artifacts/artifact_serializer.hpp"

OUTCOME_CPP_DEFINE_CATEGORY_3( sgns::sgprocessing, ProcessingManager::Error, e )
{
    switch ( e )
    {
        case sgns::sgprocessing::ProcessingManager::Error::PROCESS_INFO_MISSING:
            return "Processing information missing on JSON file";
        case sgns::sgprocessing::ProcessingManager::Error::INVALID_JSON:
            return "Json cannot be parsed";
        case sgns::sgprocessing::ProcessingManager::Error::INVALID_BLOCK_PARAMETERS:
            return "Json missing block params";
        case sgns::sgprocessing::ProcessingManager::Error::NO_PROCESSOR:
            return "Json missing processor";
        case sgns::sgprocessing::ProcessingManager::Error::MISSING_INPUT:
            return "Input missing";
        case sgns::sgprocessing::ProcessingManager::Error::INPUT_UNAVAIL:
            return "Could not get input from source";
        case sgns::sgprocessing::ProcessingManager::Error::SHADER_COMPILE_FAILED:
            return "Shader source failed to compile";
        case sgns::sgprocessing::ProcessingManager::Error::SPIRV_VALIDATION_FAILED:
            return "SPIR-V failed validation";
        case sgns::sgprocessing::ProcessingManager::Error::PROCESSING_FAILED:
            return "Processor failed to produce a valid result";
        case sgns::sgprocessing::ProcessingManager::Error::MODEL_MISSING:
            return "Inference or retrain pass is missing required model configuration";
        case sgns::sgprocessing::ProcessingManager::Error::MODEL_FORMAT_UNSUPPORTED:
            return "Model format is not supported for execution (only MNN format is executable)";
        case sgns::sgprocessing::ProcessingManager::Error::RENDER_SHADER_MISSING:
            return "Render pass is missing required shader configuration";
        case sgns::sgprocessing::ProcessingManager::Error::UNKNOWN_PASS_TYPE:
            return "Job definition references an unrecognized or unregistered pass type";
    }
    return "Unknown error";
}

namespace sgns::sgprocessing
{
    namespace
    {
        bool IsUrl( const std::string &value )
        {
            return value.find( "://" ) != std::string::npos;
        }

        bool EndsWithSlash( const std::string &value )
        {
            if ( value.empty() )
            {
                return false;
            }
            const char last = value.back();
            return last == '/' || last == '\\';
        }

        bool UrlHasExtension( const std::string &value )
        {
            std::string prefix;
            std::string base;
            std::string extension;
            if ( !getURLComponents( value, prefix, base, extension ) )
            {
                return false;
            }
            return !extension.empty();
        }

        /**
         * Packs validated per-stage SPIR-V into a single byte buffer.
         *
         * PROVISIONAL WIRE FORMAT -- this is this plan's own choice, not a
         * negotiated Phase-3 contract. Phase 3's RenderProcessor has not been
         * designed yet and does not currently consume mainbuffers->first for
         * render passes at all; Phase 3's planning may revise this format
         * once RenderProcessor's actual pipeline-construction needs are
         * known.
         *
         * Layout (all integers little-endian, native uint32_t width):
         *   uint32_t stage_count
         *   per stage:
         *     uint32_t stage_tag       (static_cast<uint32_t>(sgns::Stage))
         *     uint32_t entry_point_len (number of following raw UTF-8 bytes)
         *     entry_point_len raw UTF-8 bytes (no null terminator)
         *     uint32_t word_count      (number of following uint32_t SPIR-V words)
         *     word_count * uint32_t spirv_words
         */
        std::vector<char> SerializeCompiledStages(
            const std::vector<sgns::sgprocessing::CompiledShaderStage> &stages,
            const std::vector<std::string>                             &entryPoints )
        {
            std::vector<char> out;

            if ( entryPoints.size() != stages.size() )
            {
                // Invariant of this plan's own call site -- a mismatch indicates a
                // caller bug, not malformed job-supplied input.
                return out;
            }

            auto appendU32 = [&out]( uint32_t value )
            {
                size_t offset = out.size();
                out.resize( offset + sizeof( uint32_t ) );
                std::memcpy( out.data() + offset, &value, sizeof( uint32_t ) );
            };

            appendU32( static_cast<uint32_t>( stages.size() ) );
            for ( size_t i = 0; i < stages.size(); ++i )
            {
                const auto &compiled = stages[i];
                appendU32( static_cast<uint32_t>( compiled.stage ) );

                const std::string &entryPoint = entryPoints[i];
                appendU32( static_cast<uint32_t>( entryPoint.size() ) );
                if ( !entryPoint.empty() )
                {
                    size_t offset = out.size();
                    out.resize( offset + entryPoint.size() );
                    std::memcpy( out.data() + offset, entryPoint.data(), entryPoint.size() );
                }

                appendU32( static_cast<uint32_t>( compiled.spirv.size() ) );
                if ( !compiled.spirv.empty() )
                {
                    size_t offset = out.size();
                    out.resize( offset + compiled.spirv.size() * sizeof( uint32_t ) );
                    std::memcpy( out.data() + offset,
                                 compiled.spirv.data(),
                                 compiled.spirv.size() * sizeof( uint32_t ) );
                }
            }

            return out;
        }

        /**
         * Packs render_target/pipeline_state/vertex_layout/uniforms alongside the
         * independently-resolved vertex/index buffer bytes into the single wire-format
         * buffer GetCidForProc() places into mainbuffers->second for a render pass.
         *
         * This is the ONLY channel any of this Pass-level data has to reach
         * RenderProcessor -- StartProcessing()'s fixed signature never carries the
         * Pass or RenderShaderConfig object itself (D-25's no-signature-change
         * constraint). Plan 03-03's RenderProcessor parser must be the exact inverse
         * of this function.
         *
         * Layout (all integers little-endian, native uint32_t width; clear_color/
         * clear_depth narrowed from the schema's double to float on write):
         *   uint32_t width
         *   uint32_t height
         *   uint32_t color_format_tag   (static_cast<uint32_t>(ColorFormat))
         *   uint32_t depth_format_tag   (static_cast<uint32_t>(DepthFormat))
         *   float    clear_color[4]
         *   float    clear_depth
         *   uint8_t  has_pipeline_state
         *   if has_pipeline_state:
         *     uint8_t has_topology         + [uint32_t topology_tag]
         *     uint8_t has_cull_mode        + [uint32_t cull_mode_tag]
         *     uint8_t has_front_face       + [uint32_t front_face_tag]
         *     uint8_t has_depth_test       + [uint32_t depth_test_tag]
         *     uint8_t has_blend_enable     + [uint8_t  blend_enable_value]      (Phase 17, D-05)
         *     uint8_t has_blend_src_factor + [uint32_t blend_src_factor_tag]    (Phase 17, D-05)
         *     uint8_t has_blend_dst_factor + [uint32_t blend_dst_factor_tag]    (Phase 17, D-05)
         *   uint32_t vertex_layout_count
         *   per entry:
         *     uint32_t name_len + name bytes (raw UTF-8, no null terminator)
         *     uint32_t format_tag   (static_cast<uint32_t>(VertexLayoutFormat))
         *     uint32_t offset
         *   uint8_t has_uniforms
         *   if has_uniforms:
         *     uint32_t uniform_count
         *     per entry (std::map's natural key-sorted iteration order):
         *       uint32_t name_len + name bytes
         *       uint8_t has_source + [uint32_t source_len + source bytes]
         *       uint8_t has_type   + [uint32_t type_tag (static_cast<uint32_t>(DataType))]
         *       uint32_t value_json_len + value bytes (nlohmann::json::dump() UTF-8;
         *                                               empty string if get_value().is_null())
         *   uint32_t vertex_len + vertex bytes
         *   uint8_t has_index
         *   if has_index:
         *     uint32_t index_type_tag (static_cast<uint32_t>(IndexType))
         *     uint32_t index_len + index bytes
         *   uint32_t data_transform_count
         *   uint8_t has_texture_buffer                                          (Phase 17, D-05)
         *   if has_texture_buffer:
         *     uint32_t texture_width
         *     uint32_t texture_height
         *     uint32_t texture_len + texture bytes (raw RGBA8)
         */
        std::vector<char> SerializeRenderPassConfig(
            const sgns::RenderTarget                                                &target,
            const boost::optional<sgns::PipelineState>                              &pipelineState,
            const std::vector<sgns::VertexLayoutEntry>                              &vertexLayout,
            const boost::optional<std::map<std::string, sgns::RenderShaderUniform>> &uniforms,
            const std::vector<char>                                                 &vertexBytes,
            bool                                                                     hasIndexBuffer,
            sgns::IndexType                                                          indexType,
            const std::vector<char>                                                 &indexBytes,
            uint32_t                                                                 dataTransformCount,
            bool                                                                     hasTextureBuffer,
            uint32_t                                                                 textureWidth,
            uint32_t                                                                 textureHeight,
            const std::vector<char>                                                 &textureBytes )
        {
            std::vector<char> out;

            auto appendBytes = [&out]( const char *data, size_t size )
            {
                if ( size > 0 )
                {
                    size_t offset = out.size();
                    out.resize( offset + size );
                    std::memcpy( out.data() + offset, data, size );
                }
            };
            auto appendU32 = [&out]( uint32_t value )
            {
                size_t offset = out.size();
                out.resize( offset + sizeof( uint32_t ) );
                std::memcpy( out.data() + offset, &value, sizeof( uint32_t ) );
            };
            auto appendU8 = [&out]( uint8_t value ) { out.push_back( static_cast<char>( value ) ); };
            auto appendF32 = [&out]( float value )
            {
                size_t offset = out.size();
                out.resize( offset + sizeof( float ) );
                std::memcpy( out.data() + offset, &value, sizeof( float ) );
            };
            auto appendString = [&]( const std::string &value )
            {
                appendU32( static_cast<uint32_t>( value.size() ) );
                appendBytes( value.data(), value.size() );
            };

            appendU32( static_cast<uint32_t>( target.get_width() ) );
            appendU32( static_cast<uint32_t>( target.get_height() ) );
            appendU32( static_cast<uint32_t>( target.get_color_format() ) );
            appendU32( static_cast<uint32_t>( target.get_depth_format() ) );

            const auto &clearColor = target.get_clear_color();
            for ( size_t i = 0; i < 4; ++i )
            {
                appendF32( i < clearColor.size() ? static_cast<float>( clearColor[i] ) : 0.0f );
            }
            appendF32( static_cast<float>( target.get_clear_depth() ) );

            if ( pipelineState )
            {
                appendU8( 1 );
                const auto &ps = pipelineState.value();

                if ( ps.get_topology() )
                {
                    appendU8( 1 );
                    appendU32( static_cast<uint32_t>( ps.get_topology().value() ) );
                }
                else
                {
                    appendU8( 0 );
                }

                if ( ps.get_cull_mode() )
                {
                    appendU8( 1 );
                    appendU32( static_cast<uint32_t>( ps.get_cull_mode().value() ) );
                }
                else
                {
                    appendU8( 0 );
                }

                if ( ps.get_front_face() )
                {
                    appendU8( 1 );
                    appendU32( static_cast<uint32_t>( ps.get_front_face().value() ) );
                }
                else
                {
                    appendU8( 0 );
                }

                if ( ps.get_depth_test() )
                {
                    appendU8( 1 );
                    appendU32( static_cast<uint32_t>( ps.get_depth_test().value() ) );
                }
                else
                {
                    appendU8( 0 );
                }

                if ( ps.get_blend_enable() )
                {
                    appendU8( 1 );
                    appendU8( ps.get_blend_enable().value() ? 1 : 0 );
                }
                else
                {
                    appendU8( 0 );
                }

                if ( ps.get_blend_src_factor() )
                {
                    appendU8( 1 );
                    appendU32( static_cast<uint32_t>( ps.get_blend_src_factor().value() ) );
                }
                else
                {
                    appendU8( 0 );
                }

                if ( ps.get_blend_dst_factor() )
                {
                    appendU8( 1 );
                    appendU32( static_cast<uint32_t>( ps.get_blend_dst_factor().value() ) );
                }
                else
                {
                    appendU8( 0 );
                }
            }
            else
            {
                appendU8( 0 );
            }

            appendU32( static_cast<uint32_t>( vertexLayout.size() ) );
            for ( const auto &entry : vertexLayout )
            {
                appendString( entry.get_name() );
                appendU32( static_cast<uint32_t>( entry.get_format() ) );
                appendU32( static_cast<uint32_t>( entry.get_offset() ) );
            }

            if ( uniforms )
            {
                appendU8( 1 );
                const auto &uniformMap = uniforms.value();
                appendU32( static_cast<uint32_t>( uniformMap.size() ) );
                // std::map iterates in key-sorted order already -- matches plan
                // 03-03's ResolveUniforms iteration-order decision.
                for ( const auto &uniformEntry : uniformMap )
                {
                    appendString( uniformEntry.first );
                    const auto &uniform = uniformEntry.second;

                    if ( uniform.get_source() )
                    {
                        appendU8( 1 );
                        appendString( uniform.get_source().value() );
                    }
                    else
                    {
                        appendU8( 0 );
                    }

                    if ( uniform.get_type() )
                    {
                        appendU8( 1 );
                        appendU32( static_cast<uint32_t>( uniform.get_type().value() ) );
                    }
                    else
                    {
                        appendU8( 0 );
                    }

                    std::string valueJson =
                        uniform.get_value().is_null() ? std::string() : uniform.get_value().dump();
                    appendString( valueJson );
                }
            }
            else
            {
                appendU8( 0 );
            }

            appendU32( static_cast<uint32_t>( vertexBytes.size() ) );
            appendBytes( vertexBytes.data(), vertexBytes.size() );

            if ( hasIndexBuffer )
            {
                appendU8( 1 );
                appendU32( static_cast<uint32_t>( indexType ) );
                appendU32( static_cast<uint32_t>( indexBytes.size() ) );
                appendBytes( indexBytes.data(), indexBytes.size() );
            }
            else
            {
                appendU8( 0 );
            }

            appendU32( dataTransformCount );

            if ( hasTextureBuffer )
            {
                appendU8( 1 );
                appendU32( textureWidth );
                appendU32( textureHeight );
                appendU32( static_cast<uint32_t>( textureBytes.size() ) );
                appendBytes( textureBytes.data(), textureBytes.size() );
            }
            else
            {
                appendU8( 0 );
            }

            return out;
        }
    }

    ProcessingManager::~ProcessingManager() {}

    outcome::result<std::shared_ptr<ProcessingManager>> ProcessingManager::Create( const std::string &jsondata )
    {
        auto instance = std::shared_ptr<ProcessingManager>( new ProcessingManager() );
        BOOST_OUTCOME_TRY( instance->Init( jsondata ) );
        return instance;
    }

    outcome::result<void> ProcessingManager::Init( const std::string &jsondata )
    {
        m_processor = nullptr;
        //Register Processors
        RegisterProcessorFactory( static_cast<int>( DataType::TEXTURE2_D ),
                                  [] { return std::make_unique<sgprocessing::MNN_Image>(); } );
        RegisterProcessorFactory( static_cast<int>( DataType::STRING ),
                                  [] { return std::make_unique<sgprocessing::MNN_String>(); } );
        RegisterProcessorFactory( static_cast<int>( DataType::BOOL ),
                                  [] { return std::make_unique<sgprocessing::MNN_Bool>(); } );
        RegisterProcessorFactory( static_cast<int>( DataType::BUFFER ),
                                  [] { return std::make_unique<sgprocessing::MNN_Buffer>(); } );
        RegisterProcessorFactory( static_cast<int>( DataType::FLOAT ),
                                  [] { return std::make_unique<sgprocessing::MNN_Float>(); } );
        RegisterProcessorFactory( static_cast<int>( DataType::INT ),
                                  [] { return std::make_unique<sgprocessing::MNN_Int>(); } );
#ifdef SGPROC_HAS_MNN_LLM
        // PROC-01: only registered when the vendored MNN was built with MNN_BUILD_LLM=ON
        // (see ProcessingManager.hpp's include guard and src/processors/CMakeLists.txt's
        // configure-time detection). In checkouts without LLM support (like this one),
        // DataType::LLM has no registered factory and SetProcessorByName() returns false,
        // so ProcessInternal() fails closed with the existing Error::NO_PROCESSOR path --
        // the same behavior any other unregistered DataType already has.
        RegisterProcessorFactory( static_cast<int>( DataType::LLM ),
                                  [] { return std::make_unique<sgprocessing::MNN_Llm>(); } );
#endif
        RegisterProcessorFactory( static_cast<int>( DataType::MAT2 ),
                                  [] { return std::make_unique<sgprocessing::MNN_Mat2>(); } );
        RegisterProcessorFactory( static_cast<int>( DataType::MAT3 ),
                                  [] { return std::make_unique<sgprocessing::MNN_Mat3>(); } );
        RegisterProcessorFactory( static_cast<int>( DataType::MAT4 ),
                                  [] { return std::make_unique<sgprocessing::MNN_Mat4>(); } );
        RegisterProcessorFactory( static_cast<int>( DataType::VEC2 ),
                                  [] { return std::make_unique<sgprocessing::MNN_Vec2>(); } );
        RegisterProcessorFactory( static_cast<int>( DataType::VEC3 ),
                                  [] { return std::make_unique<sgprocessing::MNN_Vec3>(); } );
        RegisterProcessorFactory( static_cast<int>( DataType::VEC4 ),
                                  [] { return std::make_unique<sgprocessing::MNN_Vec4>(); } );
        RegisterProcessorFactory( static_cast<int>( DataType::TENSOR ),
                                  [] { return std::make_unique<sgprocessing::MNN_Tensor>(); } );
        RegisterProcessorFactory( static_cast<int>( DataType::TEXTURE1_D ),
                                  [] { return std::make_unique<sgprocessing::MNN_Texture1D>(); } );
        RegisterProcessorFactory( static_cast<int>( DataType::TEXTURE3_D ),
                                  [] { return std::make_unique<sgprocessing::MNN_Volume>(); } );
        RegisterProcessorFactory( static_cast<int>( DataType::TEXTURE_CUBE ),
                                  [] { return std::make_unique<sgprocessing::MNN_TextureCube>(); } );
        RegisterPassProcessorFactory( PassType::RENDER,
                                      [] { return std::make_unique<sgprocessing::RenderProcessor>(); },
                                      false /* supports_checkpointing */ );

        // Build capability snapshot after all executors are registered (D-01, D-09)
        m_capabilityValidator = std::make_unique<CapabilityValidator>();
        {
            // Extract factory functions from ExecutorRegistryEntry for BuildSnapshot
            std::unordered_map<PassType, std::function<std::unique_ptr<ProcessingProcessor>()>, PassTypeHash> factoriesOnly;
            std::unordered_map<PassType, bool, PassTypeHash> checkpointFlags;
            for ( auto &entry : m_passFactories )
            {
                factoriesOnly[entry.first]       = entry.second.factory;
                checkpointFlags[entry.first]     = entry.second.supports_checkpointing;
            }
            m_capabilityValidator->BuildSnapshot(
                factoriesOnly,
                m_processorFactories.size(),
                []() -> VkPhysicalDevice
                {
                    // Ensure Vulkan device exists via a temporary RenderProcessor
                    // that lazy-initializes the shared Vulkan context under VulkanInitMutex.
                    static auto s_renderProc = std::make_unique<sgprocessing::RenderProcessor>();
                    if ( !s_renderProc->InitializeContext() )
                        return VK_NULL_HANDLE;
                    return s_renderProc->GetPhysicalDevice();
                } );
            // Populate checkpoint support flags onto the snapshot (D-20)
            if ( auto *snap = m_capabilityValidator->GetSnapshot() )
            {
                // const_cast: GetSnapshot returns const*, but we own the snapshot
                // and this is the only place it's populated during Init().
                const_cast<CapabilitySnapshot *>( snap )->checkpointSupport = std::move( checkpointFlags );
            }
        }

        //Parse Json
        //This will check required fields inherently.
        try
        {
            auto data = nlohmann::json::parse( jsondata );

            // Pre-parse validation: intercept unrecognized passes[].type and
            // passes[].model.format raw strings *before* sgns::from_json() runs.
            // The quicktype-generated from_json(PassType&)/from_json(ModelFormat&)
            // throw a plain std::runtime_error with no field context when a job
            // submits an unrecognized enum string, which the generic catch below
            // would otherwise collapse into a context-free Error::INVALID_JSON.
            // Recognized-but-unsupported formats (e.g. ONNX) are intentionally left
            // alone here -- they parse successfully and are rejected afterward by
            // CheckProcessValidity()'s explicit MNN-executability check instead.
            if ( data.contains( "passes" ) && data[ "passes" ].is_array() )
            {
                static const std::set<std::string> kRecognizedPassTypes    = { "compute", "data_transform",
                                                                                "inference", "render", "retrain" };
                static const std::set<std::string> kRecognizedModelFormats = { "MNN", "ONNX", "PyTorch",
                                                                                "TensorFlow" };
                for ( const auto &passEntry : data[ "passes" ] )
                {
                    if ( !passEntry.is_object() || !passEntry.contains( "type" ) || !passEntry[ "type" ].is_string() )
                    {
                        continue;
                    }
                    const std::string passType = passEntry[ "type" ].get<std::string>();
                    if ( kRecognizedPassTypes.find( passType ) == kRecognizedPassTypes.end() )
                    {
                        m_logger->error( "Job definition references an unrecognized pass type: " + passType );
                        return outcome::failure( Error::UNKNOWN_PASS_TYPE );
                    }
                    if ( ( passType == "inference" || passType == "retrain" ) && passEntry.contains( "model" ) &&
                         passEntry[ "model" ].is_object() )
                    {
                        const auto &modelEntry = passEntry[ "model" ];
                        if ( modelEntry.contains( "format" ) && modelEntry[ "format" ].is_string() )
                        {
                            const std::string modelFormat = modelEntry[ "format" ].get<std::string>();
                            if ( kRecognizedModelFormats.find( modelFormat ) == kRecognizedModelFormats.end() )
                            {
                                m_logger->error( "Job definition references an unsupported model format: " +
                                                 modelFormat );
                                return outcome::failure( Error::MODEL_FORMAT_UNSUPPORTED );
                            }
                        }
                    }
                }
            }

            sgns::from_json( data, processing_ );
        }
        catch ( const nlohmann::json::exception &e )
        {
            return outcome::failure( Error::INVALID_JSON );
        }
        catch ( const std::exception &e )
        {
            // quicktype-generated enum from_json functions (e.g. the narrowed
            // ShaderSourceType) throw a plain std::runtime_error -- not a
            // nlohmann::json::exception subclass -- when a job submits a
            // schema-invalid enum value (e.g. legacy "hlsl"/"metal"). Must be
            // caught here as well or it propagates uncaught out of Init().
            return outcome::failure( Error::INVALID_JSON );
        }
        auto isvalid = CheckProcessValidity();
        if ( !isvalid )
        {
            return isvalid.error();
        }
        // Phase 01-01 (D-04): inputs is now optional at the schema level (root
        // required relaxed to name/version/gnus_spec_version so minimal ELM jobs
        // parse). For non-ELM jobs the Init parity gate below guarantees the
        // optional is engaged before we reach this point, so value_or(empty) is
        // a compile-shim, never a behavioral path.
        const auto inputs = processing_.get_inputs().value_or( std::vector<sgns::IoDeclaration>{} );
        for ( size_t i = 0; i < inputs.size(); ++i )
        {
            std::string sourceKey = "input:" + inputs[i].get_name();
            m_inputMap[sourceKey] = i;
        }
        // Successful parse
        return outcome::success();
    }

    outcome::result<void> ProcessingManager::CheckProcessValidity()
    {
        // Phase 01-01 (D-04): passes is now optional at the schema level. The
        // non-ELM parity gate in Init() rejects non-ELM jobs without a
        // non-empty passes array before CheckProcessValidity runs; for ELM jobs
        // there are no passes to iterate. value_or(empty) keeps this loop a
        // no-op for both cases instead of a compile error.
        const auto passes = processing_.get_passes().value_or( std::vector<sgns::Pass>{} );
        for ( auto &pass : passes )
        {
            //Check optional params if needed
            switch ( pass.get_type() )
            {
                case PassType::INFERENCE:
                {
                    if ( !pass.get_model() )
                    {
                        m_logger->error( "Inference json has no model" );
                        return outcome::failure( Error::MODEL_MISSING );
                    }
                    if ( pass.get_model().value().get_format() != ModelFormat::MNN )
                    {
                        m_logger->error( "Inference pass model format is not executable (only MNN is supported), pass: " +
                                         pass.get_name() );
                        return outcome::failure( Error::MODEL_FORMAT_UNSUPPORTED );
                    }
                    break;
                }
                case PassType::COMPUTE:
                    break;
                case PassType::DATA_TRANSFORM:
                    break;
                case PassType::RENDER:
                {
                    if ( !pass.get_render_shader() )
                    {
                        m_logger->error( "Render pass has no render_shader config" );
                        return outcome::failure( Error::RENDER_SHADER_MISSING );
                    }
                    if ( !pass.get_render_target() )
                    {
                        m_logger->error( "Render pass has no render_target config" );
                        return outcome::failure( Error::PROCESS_INFO_MISSING );
                    }
                    if ( !pass.get_vertex_buffer() )
                    {
                        m_logger->error( "Render pass has no vertex_buffer binding" );
                        return outcome::failure( Error::PROCESS_INFO_MISSING );
                    }
                    if ( !pass.get_vertex_layout() || pass.get_vertex_layout()->empty() )
                    {
                        m_logger->error( "Render pass has no vertex_layout entries" );
                        return outcome::failure( Error::PROCESS_INFO_MISSING );
                    }

                    // Task 2: defensively reject vertex_buffer/index_buffer/uniform
                    // sources this phase has no real resolution path for
                    // (output:/internal:/parameter: for buffers; anything but
                    // parameter: for uniforms) -- fail closed at Create() time
                    // rather than reaching RenderProcessor unchecked (T-03-02-01,
                    // T-03-02-02).
                    auto rejectUnsupportedBufferSourcePrefix =
                        [this]( const char *fieldName, const std::string &source ) -> outcome::result<void>
                    {
                        m_logger->error(
                            "Render pass {}.source '{}' uses an unsupported prefix -- "
                            "only input: is resolvable (no cross-pass output:/internal: dependency "
                            "graph exists; parameter:-sourced raw buffers are not supported)",
                            fieldName,
                            source );
                        return outcome::failure( Error::PROCESS_INFO_MISSING );
                    };

                    {
                        const auto        vertexBufferCfg = pass.get_vertex_buffer().value();
                        const std::string vertexSource    = vertexBufferCfg.get_source();
                        if ( vertexSource.rfind( "input:", 0 ) != 0 )
                        {
                            return rejectUnsupportedBufferSourcePrefix( "vertex_buffer", vertexSource );
                        }
                    }

                    if ( pass.get_index_buffer() && pass.get_index_buffer().value().get_source() )
                    {
                        const auto  indexBufferCfg = pass.get_index_buffer().value();
                        std::string indexSource    = indexBufferCfg.get_source().value();
                        if ( indexSource.rfind( "input:", 0 ) != 0 )
                        {
                            return rejectUnsupportedBufferSourcePrefix( "index_buffer", indexSource );
                        }
                    }

                    // texture_buffer is optional (Phase 17 D-05, texturing scope); if present,
                    // it must be input:-resolvable, mirroring vertex_buffer/index_buffer's gate.
                    if ( pass.get_texture_buffer() )
                    {
                        const auto        textureBufferCfg = pass.get_texture_buffer().value();
                        const std::string textureSource    = textureBufferCfg.get_source();
                        if ( textureSource.rfind( "input:", 0 ) != 0 )
                        {
                            return rejectUnsupportedBufferSourcePrefix( "texture_buffer", textureSource );
                        }
                    }

                    {
                        const auto renderShaderCfg = pass.get_render_shader().value();
                        if ( renderShaderCfg.get_uniforms() )
                        {
                            const auto uniformsCfg = renderShaderCfg.get_uniforms().value();
                            for ( const auto &uniformEntry : uniformsCfg )
                            {
                                const std::string &uniformName = uniformEntry.first;
                                const auto         &uniform     = uniformEntry.second;

                                if ( uniform.get_source() )
                                {
                                    const std::string &uniformSource = uniform.get_source().value();
                                    if ( uniformSource.rfind( "parameter:", 0 ) != 0 )
                                    {
                                        m_logger->error(
                                            "Render pass uniform '{}' has source '{}' with an "
                                            "unsupported prefix -- only parameter: is resolvable for "
                                            "uniform values in this phase",
                                            uniformName,
                                            uniformSource );
                                        return outcome::failure( Error::PROCESS_INFO_MISSING );
                                    }
                                }
                                else if ( uniform.get_value().is_null() )
                                {
                                    m_logger->error(
                                        "Render pass uniform '{}' has neither a source nor a usable "
                                        "value",
                                        uniformName );
                                    return outcome::failure( Error::PROCESS_INFO_MISSING );
                                }
                            }
                        }
                    }
                    break;
                }
                case PassType::RETRAIN:
                    break;
                default:
                    m_logger->error( "Somehow pass has no type" );
                    return outcome::failure( Error::PROCESS_INFO_MISSING );
            }
        }
        //Check Input optionals
        // Phase 01-01 (D-04): inputs is schema-optional now; see the parity
        // gate note above -- value_or(empty) is the compile shim.
        const auto inputsToCheck = processing_.get_inputs().value_or( std::vector<sgns::IoDeclaration>{} );
        for ( auto &input : inputsToCheck )
        {
            switch ( input.get_type() )
            {
                case DataType::BOOL:
                {
                    if ( !input.get_dimensions() || !input.get_dimensions()->get_width() )
                    {
                        m_logger->error( "Bool type missing width" );
                        return outcome::failure( Error::PROCESS_INFO_MISSING );
                    }

                    if ( input.get_format() )
                    {
                        const auto format = input.get_format().value();
                        if ( format != sgns::InputFormat::FLOAT32 && format != sgns::InputFormat::FLOAT16 &&
                             format != sgns::InputFormat::INT8 )
                        {
                            m_logger->error( "Bool type supports FLOAT32/FLOAT16/INT8 formats only" );
                            return outcome::failure( Error::PROCESS_INFO_MISSING );
                        }
                    }
                    else
                    {
                        m_logger->warn( "Bool input missing format; defaulting to FLOAT32" );
                    }
                    break;
                }
                case DataType::BUFFER:
                {
                    if ( !input.get_dimensions() || !input.get_dimensions()->get_width() )
                    {
                        m_logger->error( "Buffer type missing width" );
                        return outcome::failure( Error::PROCESS_INFO_MISSING );
                    }

                    if ( input.get_format() )
                    {
                        const auto format = input.get_format().value();
                        if ( format != sgns::InputFormat::INT8 )
                        {
                            m_logger->error( "Buffer type supports INT8 format only" );
                            return outcome::failure( Error::PROCESS_INFO_MISSING );
                        }
                    }
                    else
                    {
                        m_logger->warn( "Buffer input missing format; defaulting to INT8" );
                    }
                    break;
                }
                case DataType::FLOAT:
                {
                    if ( !input.get_dimensions() || !input.get_dimensions()->get_width() )
                    {
                        m_logger->error( "Float type missing width" );
                        return outcome::failure( Error::PROCESS_INFO_MISSING );
                    }

                    if ( input.get_format() )
                    {
                        const auto format = input.get_format().value();
                        if ( format != sgns::InputFormat::FLOAT32 && format != sgns::InputFormat::FLOAT16 )
                        {
                            m_logger->error( "Float type supports FLOAT32/FLOAT16 formats only" );
                            return outcome::failure( Error::PROCESS_INFO_MISSING );
                        }
                    }
                    else
                    {
                        m_logger->warn( "Float input missing format; defaulting to FLOAT32" );
                    }
                    break;
                }
                case DataType::INT:
                {
                    if ( !input.get_dimensions() || !input.get_dimensions()->get_width() )
                    {
                        m_logger->error( "Int type missing width" );
                        return outcome::failure( Error::PROCESS_INFO_MISSING );
                    }

                    if ( input.get_format() )
                    {
                        const auto format = input.get_format().value();
                        if ( format != sgns::InputFormat::INT32 && format != sgns::InputFormat::INT16 &&
                             format != sgns::InputFormat::INT8 )
                        {
                            m_logger->error( "Int type supports INT32/INT16/INT8 formats only" );
                            return outcome::failure( Error::PROCESS_INFO_MISSING );
                        }
                    }
                    else
                    {
                        m_logger->warn( "Int input missing format; defaulting to INT32" );
                    }
                    break;
                }
                case DataType::LLM:
                {
                    // LLM inputs (PROC-01, plan 04-03) carry a text prompt via the input source,
                    // not a fixed-width buffer -- MNN::Transformer::Llm::response() takes plain
                    // text, so unlike the tensor-shaped types above there is no dimensions/format
                    // requirement here. The only recognized parameter is an optional "maxNewTokens"
                    // INT (processing_processor_mnn_llm.cpp's ResolveMaxNewTokens() already defaults
                    // it when absent) -- validate its type only when the schema author supplied one.
                    if ( processing_.get_parameters() )
                    {
                        for ( const auto &param : processing_.get_parameters().value() )
                        {
                            if ( param.get_name() == "maxNewTokens" && param.get_type() != sgns::ParameterType::INT )
                            {
                                m_logger->error( "LLM maxNewTokens parameter must be INT type" );
                                return outcome::failure( Error::PROCESS_INFO_MISSING );
                            }
                        }
                    }
                    break;
                }
                case DataType::MAT2:
                {
                    if ( !input.get_dimensions() || !input.get_dimensions()->get_width() )
                    {
                        m_logger->error( "Mat2 type missing width" );
                        return outcome::failure( Error::PROCESS_INFO_MISSING );
                    }

                    if ( input.get_format() )
                    {
                        const auto format = input.get_format().value();
                        if ( format != sgns::InputFormat::FLOAT32 && format != sgns::InputFormat::FLOAT16 )
                        {
                            m_logger->error( "Mat2 type supports FLOAT32/FLOAT16 formats only" );
                            return outcome::failure( Error::PROCESS_INFO_MISSING );
                        }
                    }
                    else
                    {
                        m_logger->warn( "Mat2 input missing format; defaulting to FLOAT32" );
                    }
                    break;
                }
                case DataType::MAT3:
                {
                    if ( !input.get_dimensions() || !input.get_dimensions()->get_width() )
                    {
                        m_logger->error( "Mat3 type missing width" );
                        return outcome::failure( Error::PROCESS_INFO_MISSING );
                    }

                    if ( input.get_format() )
                    {
                        const auto format = input.get_format().value();
                        if ( format != sgns::InputFormat::FLOAT32 && format != sgns::InputFormat::FLOAT16 )
                        {
                            m_logger->error( "Mat3 type supports FLOAT32/FLOAT16 formats only" );
                            return outcome::failure( Error::PROCESS_INFO_MISSING );
                        }
                    }
                    else
                    {
                        m_logger->warn( "Mat3 input missing format; defaulting to FLOAT32" );
                    }
                    break;
                }
                case DataType::MAT4:
                {
                    if ( !input.get_dimensions() || !input.get_dimensions()->get_width() )
                    {
                        m_logger->error( "Mat4 type missing width" );
                        return outcome::failure( Error::PROCESS_INFO_MISSING );
                    }

                    if ( input.get_format() )
                    {
                        const auto format = input.get_format().value();
                        if ( format != sgns::InputFormat::FLOAT32 && format != sgns::InputFormat::FLOAT16 )
                        {
                            m_logger->error( "Mat4 type supports FLOAT32/FLOAT16 formats only" );
                            return outcome::failure( Error::PROCESS_INFO_MISSING );
                        }
                    }
                    else
                    {
                        m_logger->warn( "Mat4 input missing format; defaulting to FLOAT32" );
                    }
                    break;
                }
                case DataType::STRING:
                {
                    if ( !processing_.get_parameters() )
                    {
                        m_logger->error( "String input missing parameters" );
                        return outcome::failure( Error::PROCESS_INFO_MISSING );
                    }

                    const auto params     = processing_.get_parameters().value();
                    auto       find_param = [&params]( const std::string &name ) -> const sgns::Parameter *
                    {
                        for ( const auto &param : params )
                        {
                            if ( param.get_name() == name )
                            {
                                return &param;
                            }
                        }
                        return nullptr;
                    };

                    const auto *tokenizer_mode = find_param( "tokenizerMode" );
                    if ( !tokenizer_mode || tokenizer_mode->get_type() != sgns::ParameterType::STRING )
                    {
                        m_logger->error( "String input missing tokenizerMode parameter" );
                        return outcome::failure( Error::PROCESS_INFO_MISSING );
                    }

                    std::string mode;
                    const auto &mode_default = tokenizer_mode->get_parameter_default();
                    if ( mode_default.is_string() )
                    {
                        mode = mode_default.get<std::string>();
                    }
                    else
                    {
                        m_logger->error( "tokenizerMode default must be a string" );
                        return outcome::failure( Error::PROCESS_INFO_MISSING );
                    }

                    if ( mode == "raw_text" )
                    {
                        const auto *vocab_uri = find_param( "vocabUri" );
                        if ( !vocab_uri || vocab_uri->get_type() != sgns::ParameterType::URI )
                        {
                            m_logger->error( "raw_text tokenizer mode requires vocabUri parameter" );
                            return outcome::failure( Error::PROCESS_INFO_MISSING );
                        }
                    }
                    break;
                }
                case DataType::TENSOR:
                {
                    if ( !input.get_dimensions() || !input.get_dimensions()->get_width() )
                    {
                        m_logger->error( "Tensor type missing width" );
                        return outcome::failure( Error::PROCESS_INFO_MISSING );
                    }

                    if ( input.get_format() )
                    {
                        const auto format = input.get_format().value();
                        if ( format != sgns::InputFormat::FLOAT32 && format != sgns::InputFormat::FLOAT16 &&
                             format != sgns::InputFormat::INT32 && format != sgns::InputFormat::INT16 &&
                             format != sgns::InputFormat::INT8
                             && format != sgns::InputFormat::FP4_ULTRA )
                        {
                            m_logger->error( "Tensor type supports FLOAT32/FLOAT16/INT32/INT16/INT8 only" );
                            return outcome::failure( Error::PROCESS_INFO_MISSING );
                        }
                    }
                    else
                    {
                        m_logger->warn( "Tensor input missing format; defaulting to FLOAT32" );
                    }
                    break;
                }
                case DataType::TEXTURE1_D:
                {
                    if ( !input.get_dimensions() )
                    {
                        m_logger->error( "Texture1d type has no dimensions" );
                        return outcome::failure( Error::PROCESS_INFO_MISSING );
                    }

                    auto dimensions = input.get_dimensions().value();
                    if ( !dimensions.get_width() )
                    {
                        m_logger->error( "Texture1d type missing width" );
                        return outcome::failure( Error::PROCESS_INFO_MISSING );
                    }

                    if ( input.get_format() )
                    {
                        const auto format = input.get_format().value();
                        if ( format != sgns::InputFormat::FLOAT32 && format != sgns::InputFormat::FLOAT16 )
                        {
                            m_logger->error( "Texture1d type supports FLOAT32/FLOAT16 formats only" );
                            return outcome::failure( Error::PROCESS_INFO_MISSING );
                        }
                    }
                    else
                    {
                        m_logger->warn( "Texture1d input missing format; defaulting to FLOAT32" );
                    }
                    break;
                }
                case DataType::TEXTURE2_D:
                {
                    if ( !input.get_dimensions() )
                    {
                        m_logger->error( "Texture2d type has no dimensions" );
                        return outcome::failure( Error::PROCESS_INFO_MISSING );
                    }
                    else
                    {
                        auto dimensions = input.get_dimensions().value();
                        //We need these dimensions
                        if ( !dimensions.get_block_len() || !dimensions.get_block_line_stride() ||
                             !dimensions.get_width() || !dimensions.get_height() || !dimensions.get_block_stride() ||
                             !dimensions.get_chunk_line_stride() || !dimensions.get_chunk_offset() ||
                             !dimensions.get_chunk_stride() || !dimensions.get_chunk_subchunk_height() ||
                             !dimensions.get_chunk_subchunk_width() )
                        {
                            m_logger->error( "Texture2d type missing dimension values" );
                            return outcome::failure( Error::PROCESS_INFO_MISSING );
                        }
                        uint64_t block_len         = dimensions.get_block_len().value();
                        uint64_t block_line_stride = dimensions.get_block_line_stride().value();

                        // Ensure block_len is evenly divisible by block_line_stride
                        if ( block_line_stride == 0 || ( block_len % block_line_stride ) != 0 )
                        {
                            m_logger->error( "Texture2d type has dimensions not divisible" );
                            return outcome::failure( Error::INVALID_BLOCK_PARAMETERS );
                        }

                        if ( !dimensions.get_chunk_count() )
                        {
                            m_logger->error( "Texture2d type has no chunk count" );
                            return outcome::failure( Error::PROCESS_INFO_MISSING );
                        }

                        break;
                    }
                }
                case DataType::TEXTURE3_D:
                {
                    if ( !input.get_dimensions() )
                    {
                        m_logger->error( "Texture3d type has no dimensions" );
                        return outcome::failure( Error::PROCESS_INFO_MISSING );
                    }

                    auto dimensions = input.get_dimensions().value();
                    if ( !dimensions.get_width() || !dimensions.get_height() || !dimensions.get_chunk_count() )
                    {
                        m_logger->error( "Texture3d type missing width/height/chunk_count" );
                        return outcome::failure( Error::PROCESS_INFO_MISSING );
                    }

                    if ( !dimensions.get_chunk_subchunk_width() || !dimensions.get_chunk_subchunk_height() ||
                         !dimensions.get_block_len() )
                    {
                        m_logger->error( "Texture3d type missing patch size parameters" );
                        return outcome::failure( Error::PROCESS_INFO_MISSING );
                    }

                    if ( input.get_format() )
                    {
                        const auto format = input.get_format().value();
                        if ( format != sgns::InputFormat::FLOAT32 && format != sgns::InputFormat::FLOAT16 )
                        {
                            m_logger->error( "Texture3d type supports FLOAT32/FLOAT16 formats only" );
                            return outcome::failure( Error::PROCESS_INFO_MISSING );
                        }
                    }
                    else
                    {
                        m_logger->warn( "Texture3d input missing format; defaulting to FLOAT32" );
                    }
                    break;
                }
                case DataType::TEXTURE_CUBE:
                {
                    if ( !input.get_dimensions() )
                    {
                        m_logger->error( "TextureCube type has no dimensions" );
                        return outcome::failure( Error::PROCESS_INFO_MISSING );
                    }

                    auto dimensions = input.get_dimensions().value();
                    if ( !dimensions.get_width() || !dimensions.get_height() )
                    {
                        m_logger->error( "TextureCube type missing width/height" );
                        return outcome::failure( Error::PROCESS_INFO_MISSING );
                    }

                    const bool hasAnyChunk = dimensions.get_block_len() || dimensions.get_block_line_stride() ||
                                             dimensions.get_block_stride() || dimensions.get_chunk_line_stride() ||
                                             dimensions.get_chunk_offset() || dimensions.get_chunk_stride() ||
                                             dimensions.get_chunk_subchunk_height() ||
                                             dimensions.get_chunk_subchunk_width() || dimensions.get_chunk_count();

                    if ( hasAnyChunk )
                    {
                        const bool hasAllChunk = dimensions.get_block_len() && dimensions.get_block_line_stride() &&
                                                 dimensions.get_block_stride() && dimensions.get_chunk_line_stride() &&
                                                 dimensions.get_chunk_offset() && dimensions.get_chunk_stride() &&
                                                 dimensions.get_chunk_subchunk_height() &&
                                                 dimensions.get_chunk_subchunk_width() && dimensions.get_chunk_count();
                        if ( !hasAllChunk )
                        {
                            m_logger->error( "TextureCube chunking requires all texture2D chunk fields" );
                            return outcome::failure( Error::PROCESS_INFO_MISSING );
                        }
                    }

                    if ( input.get_format() )
                    {
                        const auto format = input.get_format().value();
                        if ( format != sgns::InputFormat::RGB8 && format != sgns::InputFormat::RGBA8 &&
                             format != sgns::InputFormat::FLOAT32 && format != sgns::InputFormat::FLOAT16 )
                        {
                            m_logger->error( "TextureCube supports RGB8/RGBA8/FLOAT32/FLOAT16 formats only" );
                            return outcome::failure( Error::PROCESS_INFO_MISSING );
                        }
                    }
                    else
                    {
                        m_logger->warn( "TextureCube input missing format; defaulting to RGB8" );
                    }
                    break;
                }
                case DataType::VEC2:
                {
                    if ( !input.get_dimensions() || !input.get_dimensions()->get_width() )
                    {
                        m_logger->error( "Vec2 type missing width" );
                        return outcome::failure( Error::PROCESS_INFO_MISSING );
                    }

                    if ( input.get_format() )
                    {
                        const auto format = input.get_format().value();
                        if ( format != sgns::InputFormat::FLOAT32 && format != sgns::InputFormat::FLOAT16 )
                        {
                            m_logger->error( "Vec2 type supports FLOAT32/FLOAT16 formats only" );
                            return outcome::failure( Error::PROCESS_INFO_MISSING );
                        }
                    }
                    else
                    {
                        m_logger->warn( "Vec2 input missing format; defaulting to FLOAT32" );
                    }
                    break;
                }
                case DataType::VEC3:
                {
                    if ( !input.get_dimensions() || !input.get_dimensions()->get_width() )
                    {
                        m_logger->error( "Vec3 type missing width" );
                        return outcome::failure( Error::PROCESS_INFO_MISSING );
                    }

                    if ( input.get_format() )
                    {
                        const auto format = input.get_format().value();
                        if ( format != sgns::InputFormat::FLOAT32 && format != sgns::InputFormat::FLOAT16 )
                        {
                            m_logger->error( "Vec3 type supports FLOAT32/FLOAT16 formats only" );
                            return outcome::failure( Error::PROCESS_INFO_MISSING );
                        }
                    }
                    else
                    {
                        m_logger->warn( "Vec3 input missing format; defaulting to FLOAT32" );
                    }
                    break;
                }
                case DataType::VEC4:
                {
                    if ( !input.get_dimensions() || !input.get_dimensions()->get_width() )
                    {
                        m_logger->error( "Vec4 type missing width" );
                        return outcome::failure( Error::PROCESS_INFO_MISSING );
                    }

                    if ( input.get_format() )
                    {
                        const auto format = input.get_format().value();
                        if ( format != sgns::InputFormat::FLOAT32 && format != sgns::InputFormat::FLOAT16 )
                        {
                            m_logger->error( "Vec4 type supports FLOAT32/FLOAT16 formats only" );
                            return outcome::failure( Error::PROCESS_INFO_MISSING );
                        }
                    }
                    else
                    {
                        m_logger->warn( "Vec4 input missing format; defaulting to FLOAT32" );
                    }
                    break;
                }
                default:
                    return outcome::failure( Error::PROCESS_INFO_MISSING );
            }
        }
        //Check Output optionals. Anything to do here?
        // Phase 01-01 (D-04): outputs is schema-optional now; see the parity
        // gate note above -- value_or(empty) is the compile shim.
        const auto outputsToCheck = processing_.get_outputs().value_or( std::vector<sgns::IoDeclaration>{} );
        for ( auto &output : outputsToCheck )
        {
        }

        return outcome::success();
    }

    outcome::result<uint64_t> ProcessingManager::ParseBlockSize() const
    {
        uint64_t block_total_len = 0;
        // Phase 01-01 (D-04): passes/inputs are schema-optional now; ELM jobs
        // carry neither. For non-ELM jobs the Init parity gate guarantees both
        // are present before any Process/ParseBlockSize call can succeed, so the
        // value_or shims only satisfy the compiler.
        auto     passes          = processing_.get_passes().value_or( std::vector<sgns::Pass>{} );
        for ( const auto &pass : passes )
        {
            if ( !pass.get_model() )
            {
                continue;
            }
            auto input_nodes = pass.get_model().value().get_input_nodes();
            for ( auto &model : input_nodes )
            {
                auto index = GetInputIndex( model.get_source().value() );
                if ( !index )
                {
                    return index.error();
                }
                block_total_len += processing_.get_inputs()
                                       .value_or( std::vector<sgns::IoDeclaration>{} )[index.value()]
                                       .get_dimensions()
                                       .value()
                                       .get_block_len()
                                       .value();
            }
        }
        return block_total_len;
    }

    outcome::result<ProcessOutput> ProcessingManager::Process( std::shared_ptr<boost::asio::io_context> ioc,
                                                                      std::vector<std::vector<uint8_t>> &chunkhashes,
                                                                      sgns::ModelNode                   &model,
                                                                      std::vector<std::string>          &output_locations )
    {
        // Legacy 4-arg overload: construct a fresh, internally-owned ExecutionContext
        // (unchanged behavior for every existing caller) and delegate to ProcessInternal.
        ExecutionContext execCtx;
        return ProcessInternal( ioc, chunkhashes, model, output_locations, execCtx );
    }

    outcome::result<ProcessOutput> ProcessingManager::Process( std::shared_ptr<boost::asio::io_context> ioc,
                                                                      std::vector<std::vector<uint8_t>> &chunkhashes,
                                                                      sgns::ModelNode                   &model,
                                                                      std::vector<std::string>          &output_locations,
                                                                      ExecutionContext                  &externalExecCtx )
    {
        // New 5-arg overload: caller owns the ExecutionContext, so cancellation,
        // deadline, and budget fields may be pre-set/cancelled from another thread.
        return ProcessInternal( ioc, chunkhashes, model, output_locations, externalExecCtx );
    }

    outcome::result<ProcessOutput> ProcessingManager::ProcessInternal( std::shared_ptr<boost::asio::io_context> ioc,
                                                                      std::vector<std::vector<uint8_t>> &chunkhashes,
                                                                      sgns::ModelNode                   &model,
                                                                      std::vector<std::string>          &output_locations,
                                                                      ExecutionContext                  &execCtx )
    {
        //Get input index
        auto modelname = model.get_source().value();
        auto index     = GetInputIndex( modelname );
        if ( !index )
        {
            return outcome::failure( Error::MISSING_INPUT );
        }
        auto maybe_buffers = GetCidForProc( ioc, model );
        if ( !maybe_buffers )
        {
            return maybe_buffers.error();
        }
        auto buffers = maybe_buffers.value();
        // Phase 01-01 (D-04): passes/inputs are schema-optional now (see the
        // Init parity gate). ProcessInternal is only reachable for non-ELM
        // jobs that already passed that gate, so these value_or shims are
        // compile-only, never behavioral.
        const auto passesVec = processing_.get_passes().value_or( std::vector<sgns::Pass>{} );
        const auto inputsVec = processing_.get_inputs().value_or( std::vector<sgns::IoDeclaration>{} );
        const auto &pass     = passesVec[index.value()];

        // Extract budget fields from pass schema (D-06, D-07, D-08)
        uint64_t gpuMemoryBudget    = pass.get_estimated_gpu_memory_bytes().value_or( 0 );
        uint64_t outputArtifactBudget = pass.get_max_output_artifact_bytes().value_or( 0 );
        uint64_t deadlineMs         = pass.get_per_pass_deadline_ms().value_or( 0 );

        if ( pass.get_type() == PassType::RENDER )
        {
            if ( !SetProcessorByPassType( PassType::RENDER ) )
            {
                return outcome::failure( Error::NO_PROCESSOR );
            }
        }
        else
        {
            if ( !SetProcessorByName( static_cast<int>( inputsVec[index.value()].get_type() ) ) )
            {
                return outcome::failure( Error::NO_PROCESSOR );
            }
        }
        const auto  maybeParameters = processing_.get_parameters();
        const auto *parameters      = maybeParameters ? &maybeParameters.value() : nullptr;

        try
        {
            // Apply schema-derived budgets (D-06, D-07, D-08) only when the incoming
            // execCtx still has the field at its "unset" sentinel (0). A caller of the
            // 5-arg Process() overload may have pre-set any of these fields explicitly;
            // that caller-supplied value is never overwritten. For the legacy 4-arg
            // overload's freshly-constructed ExecutionContext, every field starts at 0,
            // so this is behavior-neutral — the schema default always applies.
            if ( execCtx.gpuMemoryBudget == 0 )
            {
                execCtx.gpuMemoryBudget = gpuMemoryBudget;
            }
            if ( execCtx.maxOutputArtifactBytes == 0 )
            {
                execCtx.maxOutputArtifactBytes = outputArtifactBudget;
            }
            if ( execCtx.deadlineMs == 0 )
            {
                execCtx.deadlineMs = deadlineMs;
            }

            // Progress callback logs events at stage boundaries (D-10). Only install the
            // default logging callback when the caller did not already supply their own
            // via the 5-arg Process() overload — otherwise a caller-supplied callback
            // (e.g. one capturing ProgressEvents for a test) would be silently discarded.
            if ( !execCtx.progressCallback )
            {
                execCtx.progressCallback = [this]( const ProgressEvent &ev )
                {
                    m_logger->info( "Progress: pass={} percent={:.1f}", ev.pass_id, ev.percent );
                };
            }

            // Wire deadline timer (D-05, D-09)
            boost::asio::deadline_timer deadlineTimer( *ioc );
            if ( deadlineMs > 0 )
            {
                deadlineTimer.expires_from_now( boost::posix_time::milliseconds( deadlineMs ) );
                deadlineTimer.async_wait( [&execCtx]( const boost::system::error_code &ec )
                {
                    if ( !ec )
                    {
                        // D-09: deadline → unified cancel path
                        execCtx.cancelToken.Cancel();
                    }
                } );
            }

            // Register cancel callback: if explicit cancel happens first, cancel the timer
            execCtx.cancelToken.SetCallback( [&deadlineTimer]()
            {
                deadlineTimer.cancel();
            } );

            // Capture start time before StartProcessing (D-13)
            auto startTimeUsec = std::chrono::duration_cast<std::chrono::microseconds>(
                std::chrono::system_clock::now().time_since_epoch() ).count();

            // Extract executor identity from CapabilityValidator (Phase 06 D-08)
            uint8_t executorId[SHA256_HASH_SIZE] = {};
            if ( m_capabilityValidator )
            {
                auto *snap = m_capabilityValidator->GetSnapshot();
                if ( snap && snap->identityHash.size() >= SHA256_HASH_SIZE )
                {
                    std::memcpy( executorId, snap->identityHash.data(), SHA256_HASH_SIZE );
                }
            }

            // Call new 6-arg StartProcessing() overload (D-18)
            auto processResult = m_processor->StartProcessing( chunkhashes,
                                                               inputsVec[index.value()],
                                                               *buffers->second,
                                                               *buffers->first,
                                                               parameters,
                                                               execCtx );

            // Cancel deadline timer after StartProcessing returns (whether success or failure)
            deadlineTimer.cancel();

            // Capture end time after StartProcessing returns (D-13)
            auto endTimeUsec = std::chrono::duration_cast<std::chrono::microseconds>(
                std::chrono::system_clock::now().time_since_epoch() ).count();

            // Map terminal error to TerminalState for manifest (D-15)
            TerminalState terminalState = TerminalState::Success;
            if ( processResult.error )
            {
                switch ( processResult.error->stage )
                {
                    case ProcessingErrorStage::CANCELLED:       terminalState = TerminalState::Cancelled;      break;
                    case ProcessingErrorStage::TIMED_OUT:       terminalState = TerminalState::Timeout;        break;
                    case ProcessingErrorStage::BUDGET_EXCEEDED: terminalState = TerminalState::BudgetExceeded; break;
                    default:                                    terminalState = TerminalState::Error;          break;
                }
            }

            // Build a minimal ExecutionManifest on every terminal path (ARTF-09) so
            // GetLastManifest() is reachable even when Process() returns failure
            // before the full manifest-assembly block below ever runs. Mirrors the
            // success-path assembly's identity/timing/executor-identity population.
            auto buildFailureManifest = [&]()
            {
                ExecutionManifest fm{};
                std::strncpy( fm.executionId, processing_.get_name().c_str(), MAX_IDENTIFIER - 1 );
                std::strncpy( fm.passId, pass.get_name().c_str(), MAX_RESOURCE_NAME - 1 );
                std::memcpy( fm.executorIdentity, executorId, SHA256_HASH_SIZE );
                fm.startTimeUsec = startTimeUsec;
                fm.endTimeUsec   = endTimeUsec;
                fm.wallClockUsec = endTimeUsec - startTimeUsec;
                fm.terminalState = terminalState;
                std::strncpy( fm.errorMessage,
                              processResult.error
                                  ? processResult.error->message.c_str()
                                  : "processor returned an empty hash with no result (legacy failure sentinel)",
                              MAX_IDENTIFIER - 1 );
                m_lastManifest = fm;
            };

            // Check terminal conditions before saving (D-15)
            if ( processResult.error )
            {
                if ( processResult.error->stage == ProcessingErrorStage::CANCELLED )
                {
                    m_logger->error( "Processing cancelled" );
                    buildFailureManifest();
                    return outcome::failure( Error::PROCESSING_FAILED );
                }
                if ( processResult.error->stage == ProcessingErrorStage::TIMED_OUT )
                {
                    m_logger->error( "Processing deadline exceeded" );
                    buildFailureManifest();
                    return outcome::failure( Error::PROCESSING_FAILED );
                }
                if ( processResult.error->stage == ProcessingErrorStage::BUDGET_EXCEEDED )
                {
                    m_logger->error( "Processing output budget exceeded" );
                    buildFailureManifest();
                    return outcome::failure( Error::PROCESSING_FAILED );
                }
            }

            if ( processResult.error || processResult.hash.empty() )
            {
                m_logger->error( "Processing failed: {}",
                                 processResult.error
                                     ? processResult.error->message
                                     : std::string( "processor returned an empty hash with no result (legacy failure sentinel)" ) );
                buildFailureManifest();
                return outcome::failure( Error::PROCESSING_FAILED );
            }

            // ── Build ProcessOutput: artifact records + execution manifest (Phase 08) ──
            ProcessOutput output{};
            const auto   &procInput = inputsVec[index.value()];
            const auto   outputs    = processing_.get_outputs().value_or( std::vector<sgns::IoDeclaration>{} );

            if ( processResult.output_buffers && !outputs.empty() )
            {
                const auto &bufferNames = processResult.output_buffers->first;
                const auto &bufferData  = processResult.output_buffers->second;

                // Build one Artifact per output buffer
                for ( size_t outIdx = 0; outIdx < outputs.size() && outIdx < bufferData.size(); ++outIdx )
                {
                    Artifact art{};

                    // Identity (ARTF-01)
                    std::strncpy( art.resourceName, outputs[outIdx].get_name().c_str(), MAX_RESOURCE_NAME - 1 );
                    std::strncpy( art.passId, pass.get_name().c_str(), MAX_RESOURCE_NAME - 1 );
                    {
                        std::string binding = "output:" + outputs[outIdx].get_name();
                        std::strncpy( art.outputBinding, binding.c_str(), MAX_RESOURCE_NAME - 1 );
                    }

                    // Format metadata (ARTF-02)
                    {
                        // Map DataType enum to string
                        static const char *dataTypeNames[] = {
                            "BOOL", "BUFFER", "FLOAT", "INT", "MAT2", "MAT3", "MAT4",
                            "STRING", "TENSOR", "TEXTURE1_D", "TEXTURE2_D", "TEXTURE3_D",
                            "TEXTURE_CUBE", "VEC2", "VEC3", "VEC4"
                        };
                        int dtIdx = static_cast<int>( procInput.get_type() );
                        if ( dtIdx >= 0 && dtIdx < static_cast<int>( sizeof( dataTypeNames ) / sizeof( dataTypeNames[0] ) ) )
                        {
                            std::strncpy( art.dataType, dataTypeNames[dtIdx], 63 );
                        }
                    }
                    {
                        // Map InputFormat enum to string. procInput.get_format() is optional --
                        // e.g. BUFFER-type inputs (such as a render pass's vertex_buffer source)
                        // may omit "format" entirely, defaulting to INT8 per the same convention
                        // already applied above in CheckProcessValidity()'s BUFFER case (see the
                        // "Buffer input missing format; defaulting to INT8" warning).
                        static const char *formatNames[] = {
                            "FLOAT16", "FLOAT32", "FP4_ULTRA", "INT16", "INT32", "INT8", "RGB8", "RGBA8"
                        };
                        sgns::InputFormat fmt    = procInput.get_format().value_or( sgns::InputFormat::INT8 );
                        int               fmtIdx = static_cast<int>( fmt );
                        if ( fmtIdx >= 0 && fmtIdx < static_cast<int>( sizeof( formatNames ) / sizeof( formatNames[0] ) ) )
                        {
                            std::strncpy( art.format, formatNames[fmtIdx], 63 );
                        }
                    }
                    if ( procInput.get_dimensions() )
                    {
                        auto dims = procInput.get_dimensions().value();
                        if ( dims.get_block_len() )
                            art.width  = static_cast<uint32_t>( dims.get_block_len().value() );
                        if ( dims.get_block_line_stride() )
                            art.height = static_cast<uint32_t>( dims.get_block_line_stride().value() );
                        art.depth = 1; // 2D texture convention
                    }
                    art.byteSize = bufferData[outIdx].size();
                    std::strncpy( art.mediaType, "application/octet-stream", MAX_MEDIA_TYPE - 1 );

                    // Content hash (ARTF-03, D-01)
                    ComputeArtifactIdentity( art,
                                             reinterpret_cast<const uint8_t *>( bufferData[outIdx].data() ),
                                             bufferData[outIdx].size() );

                    // Chunk hashes from processor output (D-08)
                    for ( const auto &ch : chunkhashes )
                    {
                        if ( ch.size() >= SHA256_HASH_SIZE )
                        {
                            AddChunkHash( art, ch.data() );
                        }
                    }

                    output.artifacts.push_back( art );
                }

                // ── Assemble ExecutionManifest (ARTF-04, D-13) ──
                ExecutionManifest &manifest = output.manifest;

                // Identifiers — use schema name as executionId; attempt/task/subtask
                // IDs are not tracked in v2.0 schema (left as empty strings per D-14 sentinel convention)
                std::strncpy( manifest.executionId, processing_.get_name().c_str(), MAX_IDENTIFIER - 1 );
                // attemptId, taskId, subtaskId default to empty (zero-initialized)
                std::strncpy( manifest.passId, pass.get_name().c_str(), MAX_RESOURCE_NAME - 1 );

                // Executor identity
                std::memcpy( manifest.executorIdentity, executorId, SHA256_HASH_SIZE );

                // Model identity: SHA-256 of model bytes if model used (D-14)
                if ( pass.get_model() )
                {
                    const auto &modelBytes = *buffers->first; // model file bytes
                    if ( !modelBytes.empty() )
                    {
                        auto modelHash = sgns::sgprocmanagersha::sha256(
                            modelBytes.data(), modelBytes.size() );
                        std::memcpy( manifest.modelIdentity, modelHash.data(), SHA256_HASH_SIZE );
                    }
                }

                // Shader identity: SHA-256 of SPIR-V bytes if RENDER pass (D-14)
                if ( pass.get_type() == PassType::RENDER && pass.get_render_target() )
                {
                    // The SPIR-V was compiled earlier in GetCidForProc — we compute
                    // the model identity from the shader bytes stored in buffers->second
                    // (second is image data, first is model/shader data for render passes)
                    // For now: shaderIdentity stays zero — SPIR-V bytes not tracked separately.
                    // Future: populate from compiled SPIR-V cache.
                }

                // Output artifact hashes
                manifest.outputArtifactCount = static_cast<uint32_t>(
                    std::min( output.artifacts.size(), static_cast<size_t>( MAX_ARTIFACT_REFS ) ) );
                for ( size_t i = 0; i < manifest.outputArtifactCount; ++i )
                {
                    std::memcpy( manifest.outputArtifactHashes[i],
                                 output.artifacts[i].artifactId, SHA256_HASH_SIZE );
                }

                // Timing
                manifest.startTimeUsec = startTimeUsec;
                manifest.endTimeUsec   = endTimeUsec;
                manifest.wallClockUsec = endTimeUsec - startTimeUsec;

                // Terminal state
                manifest.terminalState = terminalState;

                // Resource summary
                manifest.outputBytesProduced = 0;
                for ( const auto &art : output.artifacts )
                {
                    manifest.outputBytesProduced += art.byteSize;
                }

                // Compute manifest self-hash (D-04)
                // Hash a timing-zeroed copy so combinedHash/manifestHash are deterministic
                // across separate Process() calls; the live manifest returned to the caller
                // keeps its real startTimeUsec/endTimeUsec/wallClockUsec for provenance (ARTF-04).
                ExecutionManifest hashInput  = manifest;
                hashInput.startTimeUsec      = 0;
                hashInput.endTimeUsec        = 0;
                hashInput.wallClockUsec      = 0;
                auto mHash                   = ComputeManifestHash( hashInput );
                std::memcpy( manifest.manifestHash, mHash.data(), SHA256_HASH_SIZE );
                output.combinedHash = mHash;
            }

            // ── Existing FileManager save loop (unchanged) ──
            if ( processResult.output_buffers && !outputs.empty() )
            {
                const auto &bufferNames = processResult.output_buffers->first;
                const auto &bufferData  = processResult.output_buffers->second;

                if ( !bufferData.empty() )
                {
                    FileManager::GetInstance().InitializeSingletons();
                    bool hasSaves = false;

                    // Pre-allocate location slots matching the number of outputs
                    output_locations.clear();
                    output_locations.resize( outputs.size() );

                    // Collect save location shared_ptrs for post-ioc collection
                    std::vector<std::shared_ptr<std::string>> locationPtrs;
                    locationPtrs.resize( outputs.size() );

                    for ( size_t outputIndex = 0; outputIndex < outputs.size(); ++outputIndex )
                    {
                        const auto &output    = outputs[outputIndex];
                        const auto &outputUrl = output.get_source_uri_param();
                        if ( outputUrl.empty() )
                        {
                            continue;
                        }
                        if ( !IsUrl( outputUrl ) )
                        {
                            m_logger->warn( "Output source_uri_param '{}' is not a URL; skipping save", outputUrl );
                            continue;
                        }

                        const size_t dataIndex = ( bufferData.size() == outputs.size() ) ? outputIndex : 0;
                        if ( dataIndex >= bufferData.size() )
                        {
                            continue;
                        }

                        const size_t nameIndex = ( bufferNames.size() == outputs.size() ) ? outputIndex : 0;
                        std::string  outputFileName;
                        if ( !UrlHasExtension( outputUrl ) )
                        {
                            std::string baseName;
                            if ( nameIndex < bufferNames.size() && !bufferNames[nameIndex].empty() )
                            {
                                baseName = bufferNames[nameIndex];
                            }
                            else
                            {
                                baseName = output.get_name() + ".raw";
                            }

                            if ( EndsWithSlash( outputUrl ) )
                            {
                                outputFileName = baseName;
                            }
                            else
                            {
                                outputFileName = "/" + baseName;
                            }
                        }

                        auto saveBuffers =
                            std::make_shared<std::pair<std::vector<std::string>, std::vector<std::vector<char>>>>();
                        saveBuffers->first.push_back( outputFileName );
                        saveBuffers->second.push_back( bufferData[dataIndex] );

                        // Create a shared_ptr to capture the save location from the saver
                        auto saveLocation = std::make_shared<std::string>();
                        locationPtrs[outputIndex] = saveLocation;

                        FileManager::GetInstance().SaveASync( outputUrl,
                                                              outcome::success( saveBuffers ),
                                                              ioc,
                                                              [this, outputUrl]( const FileManager::ResultType &result )
                                                              {
                                                                  if ( !result )
                                                                  {
                                                                      m_logger->error( "Failed to save output to {}: {}",
                                                                                       outputUrl,
                                                                                       result.error().message() );
                                                                  }
                                                              },
                                                              saveLocation );
                        hasSaves = true;

                        // Dual-save: persist a local copy when output is IPFS
                        // This ensures the producing node can re-serve data after restart.
                        std::string urlPrefix, urlPath, urlExt;
                        getURLComponents( outputUrl, urlPrefix, urlPath, urlExt );
                        if ( urlPrefix == "ipfs" )
                        {
                            auto cacheDir = FileManager::GetInstance().getCacheDir();
                            if ( !cacheDir.empty() )
                            {
                                auto localUrl = "file://" + cacheDir + "/results/" +
                                                output.get_name() + outputFileName;
                                FileManager::GetInstance().SaveASync(
                                    localUrl,
                                    outcome::success( saveBuffers ),
                                    ioc,
                                    nullptr,  // no callback needed for local save
                                    nullptr ); // no save_location needed
                            }
                        }
                    }

                    if ( hasSaves )
                    {
                        ioc->reset();
                        ioc->run();

                        // After async IO completes, collect the save locations
                        for ( size_t i = 0; i < locationPtrs.size(); ++i )
                        {
                            if ( locationPtrs[i] && !locationPtrs[i]->empty() )
                            {
                                output_locations[i] = *locationPtrs[i];
                            }
                        }
                    }
                }
            }

            m_lastManifest = output.manifest;
            return output;
        }
        catch ( const std::exception &e )
        {
            m_logger->error( "Process() exception: {}", e.what() );
            if ( m_processor )
            {
                m_processor->RunTeardown();
            }
            return outcome::failure( Error::PROCESSING_FAILED );
        }
    }

    // ── ProcessingResult Migration Adapter (D-10) ──────────────────────────
    // Temporary: maps new ProcessOutput back to legacy ProcessingResult shape.
    // Removed before Phase 08 ships per D-10/D-12.

    ProcessingResult ProcessingResult::FromProcessOutput( const ProcessOutput &output )
    {
        ProcessingResult result;
        result.hash = output.combinedHash;

        if ( !output.artifacts.empty() )
        {
            auto buffers = std::make_shared<std::pair<std::vector<std::string>, std::vector<std::vector<char>>>>();
            for ( const auto &art : output.artifacts )
            {
                buffers->first.push_back( std::string( art.resourceName ) );
                // Raw bytes not stored in Artifact struct (only hash).
                // Callers needing raw bytes must use ProcessOutput directly.
                buffers->second.push_back( {} );
            }
            result.output_buffers = buffers;
        }

        return result;
    }

    outcome::result<std::shared_ptr<std::pair<std::shared_ptr<std::vector<char>>, std::shared_ptr<std::vector<char>>>>>
    ProcessingManager::GetCidForProc( std::shared_ptr<boost::asio::io_context> ioc, sgns::ModelNode &model )
    {
        auto modelname = model.get_source().value();
        auto index     = GetInputIndex( modelname );
        if ( !index )
        {
            return outcome::failure( Error::MISSING_INPUT );
        }
        boost::asio::io_context::executor_type                                   executor = ioc->get_executor();
        boost::asio::executor_work_guard<boost::asio::io_context::executor_type> workGuard( executor );

        auto mainbuffers =
            std::make_shared<std::pair<std::shared_ptr<std::vector<char>>, std::shared_ptr<std::vector<char>>>>(
                std::make_shared<std::vector<char>>(),
                std::make_shared<std::vector<char>>() );

        // Phase 01-01 (D-04): passes/inputs are schema-optional now; see the
        // Init parity gate. GetCidForProc is only reachable for non-ELM jobs
        // that passed it, so these value_or shims are compile-only.
        const auto passesForCid = processing_.get_passes().value_or( std::vector<sgns::Pass>{} );
        const auto &p        = passesForCid[index.value()];
        const bool  isRender = ( p.get_type() == PassType::RENDER && p.get_render_shader() );

        //Init Loaders
        FileManager::GetInstance().InitializeSingletons();

        // Per-stage fetch buffers for the render path -- queued alongside the
        // existing image fetch below so the single existing ioc->run() call
        // still drains everything in one pass (no new synchronization
        // primitive needed).
        std::vector<std::pair<sgns::ShaderStage, std::shared_ptr<std::vector<char>>>> stageBuffers;

        // Independently-resolved vertex/index buffer fetch buffers (Task 1 --
        // resolved via the "input:name" prefix, NOT the coincidental single
        // model-index input `mainbuffers->second` used to carry today).
        std::shared_ptr<std::vector<char>> vertexBuffer;
        std::shared_ptr<std::vector<char>> indexBuffer;
        bool                               hasIndexBuffer = false;
        sgns::IndexType                    indexType      = sgns::IndexType::UINT16;

        // Independently-resolved texture buffer fetch (Phase 17, D-05 -- texturing
        // "define contracts" half). texture_buffer is optional; if present it is
        // resolved via the same "input:name" prefix convention as vertex_buffer.
        auto     textureBuffer     = std::make_shared<std::vector<char>>();
        uint32_t textureWidth      = 0;
        uint32_t textureHeight     = 0;
        bool     hasTextureBuffer = false;

        if ( isRender )
        {
            // NOTE: get_render_shader() returns boost::optional<RenderShaderConfig> BY VALUE
            // (quicktype's standard convention for optional accessors) -- binding `stages` as a
            // reference into a chained `.value().get_stages()` call would dangle the moment this
            // statement ends, since the temporary optional/RenderShaderConfig backing that
            // reference is destroyed at the semicolon. Copy the optional into a named local first
            // so its lifetime covers the loop below.
            const auto                        renderShader = p.get_render_shader().value();
            const std::vector<sgns::ShaderStage> &stages    = renderShader.get_stages();
            for ( const auto &stage : stages )
            {
                auto tempBuffer = std::make_shared<std::vector<char>>();
                GetSubCidForProc( ioc, stage.get_source(), tempBuffer );
                stageBuffers.emplace_back( stage, tempBuffer );
            }
            // For a render pass, mainbuffers->first is populated by
            // SerializeCompiledStages() below, not by a raw modelURL fetch --
            // skip the old single GetSubCidForProc(ioc, modelURL, ...) call
            // entirely for this pass type.

            // Resolve vertex_buffer.source as an independently-named "input:"
            // reference. CheckProcessValidity() already requires vertex_buffer to
            // be present and (Task 2) requires its source to start with "input:" --
            // this call-site check is defense-in-depth, not the primary rejection
            // point.
            const auto        vertexBufferCfg = p.get_vertex_buffer().value();
            const std::string vertexSource    = vertexBufferCfg.get_source();
            if ( vertexSource.rfind( "input:", 0 ) != 0 )
            {
                return outcome::failure( Error::MISSING_INPUT );
            }
            auto vertexInputIndex = GetInputIndex( vertexSource );
            if ( !vertexInputIndex )
            {
                return outcome::failure( Error::MISSING_INPUT );
            }
            const auto inputsForVertex = processing_.get_inputs().value_or( std::vector<sgns::IoDeclaration>{} );
            std::string vertexUrl = inputsForVertex[vertexInputIndex.value()].get_source_uri_param();
            vertexBuffer          = std::make_shared<std::vector<char>>();
            GetSubCidForProc( ioc, vertexUrl, vertexBuffer );

            // index_buffer is optional; if present but its source is absent, that's
            // a schema-permitted-but-unusable-here state -- treat as no index
            // buffer (skip, do not error).
            const auto indexBufferOpt = p.get_index_buffer();
            if ( indexBufferOpt && indexBufferOpt.value().get_source() )
            {
                const auto  indexBufferCfg = indexBufferOpt.value();
                std::string indexSource    = indexBufferCfg.get_source().value();
                if ( indexSource.rfind( "input:", 0 ) != 0 )
                {
                    return outcome::failure( Error::MISSING_INPUT );
                }
                auto indexInputIndex = GetInputIndex( indexSource );
                if ( !indexInputIndex )
                {
                    return outcome::failure( Error::MISSING_INPUT );
                }
                const auto inputsForIndex = processing_.get_inputs().value_or( std::vector<sgns::IoDeclaration>{} );
                std::string indexUrl = inputsForIndex[indexInputIndex.value()].get_source_uri_param();
                indexBuffer           = std::make_shared<std::vector<char>>();
                hasIndexBuffer        = true;
                indexType             = indexBufferCfg.get_index_type().value_or( sgns::IndexType::UINT16 );
                GetSubCidForProc( ioc, indexUrl, indexBuffer );
            }

            // texture_buffer is optional (Phase 17 D-05); if present, its source is
            // already required to be "input:"-prefixed by CheckProcessValidity() --
            // this call-site check is defense-in-depth, mirroring vertex_buffer's
            // own re-check comment above.
            if ( p.get_texture_buffer() )
            {
                const auto        textureBufferCfg = p.get_texture_buffer().value();
                const std::string textureSource    = textureBufferCfg.get_source();
                if ( textureSource.rfind( "input:", 0 ) != 0 )
                {
                    return outcome::failure( Error::MISSING_INPUT );
                }
                auto texInputIndex = GetInputIndex( textureSource );
                if ( !texInputIndex )
                {
                    return outcome::failure( Error::MISSING_INPUT );
                }
                const auto inputsForTex = processing_.get_inputs().value_or( std::vector<sgns::IoDeclaration>{} );
                std::string textureUrl = inputsForTex[texInputIndex.value()].get_source_uri_param();
                GetSubCidForProc( ioc, textureUrl, textureBuffer );
                hasTextureBuffer = true;
                textureWidth     = static_cast<uint32_t>( textureBufferCfg.get_width() );
                textureHeight    = static_cast<uint32_t>( textureBufferCfg.get_height() );
            }
        }
        else
        {
            std::string modelFile = p.get_model().value().get_source_uri_param();
            m_logger->info( "Model Input URL: {}", modelFile );

            string modelURL = modelFile;
            GetSubCidForProc( ioc, modelURL, mainbuffers->first );
        }

        if ( !isRender )
        {
            // For a render pass, mainbuffers->second is populated by
            // SerializeRenderPassConfig() below, not by this raw single fetch --
            // `index` here is the coincidental pass-index-as-input-index value,
            // not any render-specific buffer.
            const auto inputsForImage = processing_.get_inputs().value_or( std::vector<sgns::IoDeclaration>{} );
            std::string image = inputsForImage[index.value()].get_source_uri_param();
            m_logger->info( "Data Input URL: {}", image );

            string imageUrl = image;
            GetSubCidForProc( ioc, imageUrl, mainbuffers->second );
        }

        //Run IO
        ioc->reset();
        ioc->run();

        if ( isRender )
        {
            std::vector<sgns::sgprocessing::CompiledShaderStage> compiledStages;
            std::vector<std::string>                             entryPoints;
            compiledStages.reserve( stageBuffers.size() );
            entryPoints.reserve( stageBuffers.size() );
            for ( auto &entry : stageBuffers )
            {
                const auto &stage      = entry.first;
                auto       &tempBuffer = entry.second;

                std::string entryPoint = stage.get_entry_point().value_or( "main" );

                sgns::sgprocessing::ShaderCompiler compiler;
                auto compileResult = compiler.CompileAndValidate( *tempBuffer,
                                                                   stage.get_stage(),
                                                                   stage.get_type(),
                                                                   entryPoint );
                if ( !compileResult )
                {
                    if ( compileResult.error() == sgns::sgprocessing::ShaderCompiler::Error::VALIDATION_FAILED )
                    {
                        return outcome::failure( Error::SPIRV_VALIDATION_FAILED );
                    }
                    return outcome::failure( Error::SHADER_COMPILE_FAILED );
                }
                compiledStages.push_back( compileResult.value() );
                entryPoints.push_back( std::move( entryPoint ) );
            }

            *mainbuffers->first = SerializeCompiledStages( compiledStages, entryPoints );

            // Preserve the pre-existing INPUT_UNAVAIL failure semantics: previously
            // this pass type's mainbuffers->second WAS the raw vertex/model fetch
            // buffer, so an unresolvable source URI surfaced here via the
            // mainbuffers->second->size() <= 0 check below. Now mainbuffers->second
            // is always populated with a non-empty SerializeRenderPassConfig()
            // header regardless of fetch success, so that check alone would no
            // longer catch a failed vertex-buffer fetch -- check it explicitly.
            if ( vertexBuffer->empty() )
            {
                return outcome::failure( Error::INPUT_UNAVAIL );
            }

            static const std::vector<char> kEmptyIndexBytes;
            static const std::vector<char> kEmptyTextureBytes;
            *mainbuffers->second = SerializeRenderPassConfig( p.get_render_target().value(),
                                                               p.get_pipeline_state(),
                                                               p.get_vertex_layout().value(),
                                                               p.get_render_shader().value().get_uniforms(),
                                                               *vertexBuffer,
                                                               hasIndexBuffer,
                                                               indexType,
                                                               hasIndexBuffer ? *indexBuffer : kEmptyIndexBytes,
                                                               p.get_data_transforms()
                                                                   ? static_cast<uint32_t>(
                                                                         p.get_data_transforms()->size() )
                                                                   : 0u,
                                                               hasTextureBuffer,
                                                               textureWidth,
                                                               textureHeight,
                                                               hasTextureBuffer ? *textureBuffer
                                                                                 : kEmptyTextureBytes );
        }

        if ( mainbuffers == nullptr )
        {
            return outcome::failure( Error::INPUT_UNAVAIL );
        }
        if ( mainbuffers->first->size() <= 0 || mainbuffers->second->size() <= 0 )
        {
            return outcome::failure( Error::INPUT_UNAVAIL );
        }

        return mainbuffers;
    }

    sgns::SgnsProcessing ProcessingManager::GetProcessingData()
    {
        return processing_;
    }

    outcome::result<size_t> ProcessingManager::GetInputIndex( const std::string &input ) const
    {
        auto it = m_inputMap.find( input );
        if ( it != m_inputMap.end() )
        {
            return it->second;
        }
        return outcome::failure( Error::MISSING_INPUT );
    }

    void ProcessingManager::GetSubCidForProc( std::shared_ptr<boost::asio::io_context> ioc,
                                              std::string                              url,
                                              std::shared_ptr<std::vector<char>>       results )
    {
        auto modeldata = FileManager::GetInstance().LoadASync(
            url,
            false,
            false,
            ioc,
            [this, results](
                outcome::result<std::shared_ptr<std::pair<std::vector<std::string>, std::vector<std::vector<char>>>>>
                    buffers )
            {
                if ( buffers )
                {
                    if ( results )
                    {
                        results->insert( results->end(),
                                         buffers.value()->second[0].begin(),
                                         buffers.value()->second[0].end() );
                    }
                }
                else
                {
                    m_logger->error( "Failed to obtain processing source: {}", buffers.error().message() );
                }
            },
            "file" );
    }

    void ProcessingManager::CanExecute( const sgns::Pass                         &pass,
                                        sgns::sgprocessing::CanExecuteCallback callback )
    {
        if ( !m_capabilityValidator )
        {
            CanExecuteResult result;
            result.executable = false;
            result.unmet.push_back(
                { UnmetRequirementCategory::RESOURCE,
                  "CapabilityValidator not initialized" } );
            callback( result );
            return;
        }
        m_capabilityValidator->CanExecute( pass, std::move( callback ) );
    }

    bool ProcessingManager::IsProcessingValid( const std::string &jsondata )
    {
        auto result = Create( jsondata );
        return result.has_value();
    }

    bool ProcessingManager::IsProcessingModelValid( const std::string &jsondata )
    {
        auto result = GetModelNodeFromJson( jsondata );
        return result.has_value();
    }

    outcome::result<sgns::ModelNode> ProcessingManager::GetModelNodeFromJson( const std::string &jsondata )
    {
        sgns::ModelNode model;
        try
        {
            auto data = nlohmann::json::parse( jsondata );
            sgns::from_json( data, model );
        }
        catch ( const nlohmann::json::exception &e )
        {
            return outcome::failure( Error::INVALID_JSON );
        }
        return model;
    }
}
