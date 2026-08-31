#include "shaders/shader_compiler.hpp"

#include <shaderc/shaderc.hpp>
#include <spirv-tools/libspirv.hpp>

#include <cstring>

OUTCOME_CPP_DEFINE_CATEGORY_3( sgns::sgprocessing, ShaderCompiler::Error, e )
{
    switch ( e )
    {
        case sgns::sgprocessing::ShaderCompiler::Error::COMPILE_FAILED:
            return "Shader source failed to compile (GLSL -> SPIR-V)";
        case sgns::sgprocessing::ShaderCompiler::Error::VALIDATION_FAILED:
            return "SPIR-V failed SPIRV-Tools validation";
    }
    return "Unknown error";
}

namespace sgns::sgprocessing
{
    outcome::result<CompiledShaderStage> ShaderCompiler::CompileAndValidate( const std::vector<char> &source_bytes,
                                                                              sgns::Stage              stage,
                                                                              sgns::ShaderSourceType   type,
                                                                              const std::string       &entry_point )
    {
        auto message_consumer = [ this ]( spv_message_level_t, const char *, const spv_position_t &,
                                           const char *message )
        {
            m_logger->error( "SPIRV-Tools validation message: {}", message ? message : "" );
        };

        if ( type == sgns::ShaderSourceType::GLSL )
        {
            // ---- GLSL -> SPIR-V compile path ----
            shaderc_shader_kind kind = ( stage == sgns::Stage::VERTEX ) ? shaderc_glsl_vertex_shader
                                                                         : shaderc_glsl_fragment_shader;

            shaderc::Compiler       compiler;
            shaderc::CompileOptions options;
            options.SetTargetEnvironment( shaderc_target_env_vulkan, shaderc_env_version_vulkan_1_3 );
            options.SetOptimizationLevel( shaderc_optimization_level_zero );

            shaderc::SpvCompilationResult result = compiler.CompileGlslToSpv(
                source_bytes.data(), source_bytes.size(), kind, "shader", entry_point.c_str(), options );

            if ( result.GetCompilationStatus() != shaderc_compilation_status_success )
            {
                m_logger->error( "ShaderCompiler: GLSL compile failed: {}", result.GetErrorMessage() );
                return outcome::failure( Error::COMPILE_FAILED );
            }

            std::vector<uint32_t> spirv_words( result.cbegin(), result.cend() );

            // Mandatory validation gate -- shaderc's CompileGlslToSpv() success does NOT mean
            // SPIRV-Tools has validated the module. Never skip this call.
            spvtools::SpirvTools tools( SPV_ENV_VULKAN_1_3 );
            tools.SetMessageConsumer( message_consumer );
            if ( !tools.Validate( spirv_words.data(), spirv_words.size() ) )
            {
                m_logger->error( "ShaderCompiler: compiled SPIR-V failed SPIRV-Tools validation" );
                return outcome::failure( Error::VALIDATION_FAILED );
            }

            return CompiledShaderStage{ std::move( spirv_words ), stage };
        }
        else // sgns::ShaderSourceType::SPIRV -- compilation skipped entirely
        {
            if ( source_bytes.size() % 4 != 0 )
            {
                m_logger->error( "ShaderCompiler: direct SPIR-V submission has a size ({}) that is "
                                  "not a multiple of 4 bytes",
                                  source_bytes.size() );
                return outcome::failure( Error::VALIDATION_FAILED );
            }

            std::vector<uint32_t> spirv_words( source_bytes.size() / 4 );
            std::memcpy( spirv_words.data(), source_bytes.data(), source_bytes.size() );

            // Mandatory validation gate -- a directly-submitted payload gets zero exemption from
            // this check, whether it is adversarial bytes or a mutated copy of previously-valid
            // SPIR-V. This is the exact gate Pitfall 2 warns must never be bypassed.
            spvtools::SpirvTools tools( SPV_ENV_VULKAN_1_3 );
            tools.SetMessageConsumer( message_consumer );
            if ( !tools.Validate( spirv_words.data(), spirv_words.size() ) )
            {
                m_logger->error( "ShaderCompiler: directly-submitted SPIR-V failed SPIRV-Tools validation" );
                return outcome::failure( Error::VALIDATION_FAILED );
            }

            return CompiledShaderStage{ std::move( spirv_words ), stage };
        }
    }
}
