#ifndef SGPROCESSINGMANAGER_SHADER_COMPILER_HPP
#define SGPROCESSINGMANAGER_SHADER_COMPILER_HPP

#include <outcome/sgprocmgr-outcome.hpp>
#include <util/sgprocmgr-logger.hpp>
#include <Stage.hpp>
#include <ShaderSourceType.hpp>

#include <cstdint>
#include <string>
#include <vector>

namespace sgns::sgprocessing
{
    /**
     * A single compiled-and-validated shader stage: the SPIR-V words that
     * survived the mandatory spirv-val gate, plus which pipeline stage they
     * target.
     */
    struct CompiledShaderStage
    {
        std::vector<uint32_t> spirv;
        sgns::Stage            stage;
    };

    /**
     * Standalone, Vulkan-device-free GLSL->SPIR-V compiler + mandatory
     * SPIRV-Tools validation gate. No VkInstance/VkDevice/VkPhysicalDevice
     * member anywhere -- pure CPU-side text/bytecode transformation and
     * validation, fully unit-testable in isolation from any Vulkan context.
     *
     * Every code path -- GLSL-compiled or directly-submitted SPIR-V -- passes
     * through spvtools::SpirvTools::Validate() before CompileAndValidate()
     * can ever return success. shaderc's CompileGlslToSpv() does NOT run
     * spirv-val internally; "shaderc compiled it" must never be conflated
     * with "SPIRV-Tools validated it."
     */
    class ShaderCompiler
    {
    public:
        enum class Error
        {
            COMPILE_FAILED    = 1,
            VALIDATION_FAILED = 2
        };

        /**
         * Compile (if GLSL) and unconditionally validate a single shader
         * stage's source bytes.
         *
         * @param source_bytes raw source bytes -- GLSL text if type == GLSL,
         *        raw SPIR-V bytes if type == SPIRV
         * @param stage which pipeline stage this shader targets
         * @param type whether source_bytes is GLSL text or raw SPIR-V bytes
         * @param entry_point the shader's entry point function name (only
         *        meaningful for the GLSL path)
         * @return validated SPIR-V words + stage on success, or a structured
         *         Error on failure -- never throws
         */
        outcome::result<CompiledShaderStage> CompileAndValidate( const std::vector<char> &source_bytes,
                                                                   sgns::Stage              stage,
                                                                   sgns::ShaderSourceType   type,
                                                                   const std::string       &entry_point );

    private:
        sgns::sgprocmanager::Logger m_logger = sgns::sgprocmanager::createLogger( "ShaderCompiler" );
    };
}

OUTCOME_HPP_DECLARE_ERROR_2( sgns::sgprocessing, ShaderCompiler::Error );

#endif // SGPROCESSINGMANAGER_SHADER_COMPILER_HPP
