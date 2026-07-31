#include <processingbase/ProcessingManager.hpp>

#include <datasplitter/ImageSplitter.hpp>
#include "FileManager.hpp"
#include "URLStringUtil.h"
#include "shaders/shader_compiler.hpp"

#include <cstring>

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
                                      [] { return std::make_unique<sgprocessing::RenderProcessor>(); } );

        //Parse Json
        //This will check required fields inherently.
        try
        {
            auto data = nlohmann::json::parse( jsondata );
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
        const auto &inputs = processing_.get_inputs();
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
        for ( auto &pass : processing_.get_passes() )
        {
            //Check optional params if needed
            switch ( pass.get_type() )
            {
                case PassType::INFERENCE:
                {
                    if ( !pass.get_model() )
                    {
                        m_logger->error( "Inference json has no model" );
                        return outcome::failure( Error::PROCESS_INFO_MISSING );
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
                        return outcome::failure( Error::PROCESS_INFO_MISSING );
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
        for ( auto &input : processing_.get_inputs() )
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
                             /*&& format != sgns::InputFormat::FP4_ULTRA*/ )
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
        for ( auto &output : processing_.get_outputs() )
        {
        }

        return outcome::success();
    }

    outcome::result<uint64_t> ProcessingManager::ParseBlockSize()
    {
        uint64_t block_total_len = 0;
        auto     passes          = processing_.get_passes();
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
                block_total_len +=
                    processing_.get_inputs()[index.value()].get_dimensions().value().get_block_len().value();
            }
        }
        return block_total_len;
    }

    outcome::result<std::vector<uint8_t>> ProcessingManager::Process( std::shared_ptr<boost::asio::io_context> ioc,
                                                                      std::vector<std::vector<uint8_t>> &chunkhashes,
                                                                      sgns::ModelNode                   &model,
                                                                      std::vector<std::string>          &output_locations )
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
        const auto &pass = processing_.get_passes()[index.value()];
        if ( pass.get_type() == PassType::RENDER )
        {
            if ( !SetProcessorByPassType( PassType::RENDER ) )
            {
                return outcome::failure( Error::NO_PROCESSOR );
            }
        }
        else
        {
            if ( !SetProcessorByName( static_cast<int>( processing_.get_inputs()[index.value()].get_type() ) ) )
            {
                return outcome::failure( Error::NO_PROCESSOR );
            }
        }
        const auto  maybeParameters = processing_.get_parameters();
        const auto *parameters      = maybeParameters ? &maybeParameters.value() : nullptr;

        auto processResult = m_processor->StartProcessing( chunkhashes,
                                                           processing_.get_inputs()[index.value()],
                                                           *buffers->second,
                                                           *buffers->first,
                                                           parameters );

        if ( processResult.error || processResult.hash.empty() )
        {
            m_logger->error( "Processing failed: {}",
                             processResult.error
                                 ? processResult.error->message
                                 : std::string( "processor returned an empty hash with no result (legacy failure sentinel)" ) );
            return outcome::failure( Error::PROCESSING_FAILED );
        }

        const auto &outputs = processing_.get_outputs();
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

        return processResult.hash;
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

        const auto &p        = processing_.get_passes()[index.value()];
        const bool  isRender = ( p.get_type() == PassType::RENDER && p.get_render_shader() );

        //Init Loaders
        FileManager::GetInstance().InitializeSingletons();

        // Per-stage fetch buffers for the render path -- queued alongside the
        // existing image fetch below so the single existing ioc->run() call
        // still drains everything in one pass (no new synchronization
        // primitive needed).
        std::vector<std::pair<sgns::ShaderStage, std::shared_ptr<std::vector<char>>>> stageBuffers;

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
        }
        else
        {
            std::string modelFile = p.get_model().value().get_source_uri_param();
            m_logger->info( "Model Input URL: {}", modelFile );

            string modelURL = modelFile;
            GetSubCidForProc( ioc, modelURL, mainbuffers->first );
        }

        std::string image = processing_.get_inputs()[index.value()].get_source_uri_param();
        m_logger->info( "Data Input URL: {}", image );

        string imageUrl = image;
        GetSubCidForProc( ioc, imageUrl, mainbuffers->second );

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

    outcome::result<size_t> ProcessingManager::GetInputIndex( const std::string &input )
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
