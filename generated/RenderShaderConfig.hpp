//  To parse this JSON data, first install
//
//      Boost     http://www.boost.org
//      json.hpp  https://github.com/nlohmann/json
//
//  Then include this file, and then do
//
//     RenderShaderConfig.hpp data = nlohmann::json::parse(jsonString);

#pragma once

#include <boost/optional.hpp>
#include <nlohmann/json.hpp>
#include "helper.hpp"

#include "ShaderStage.hpp"
#include "RenderShaderUniform.hpp"

namespace sgns {
    /**
     * Multi-stage (vertex+fragment) shader configuration for render passes
     */

    using nlohmann::json;

    /**
     * Multi-stage (vertex+fragment) shader configuration for render passes
     */
    class RenderShaderConfig {
        public:
        RenderShaderConfig() = default;
        virtual ~RenderShaderConfig() = default;

        private:
        std::vector<ShaderStage> stages;
        boost::optional<std::map<std::string, RenderShaderUniform>> uniforms;

        public:
        /**
         * Ordered shader stages (vertex, fragment) making up this render pass's pipeline
         */
        const std::vector<ShaderStage> & get_stages() const { return stages; }
        std::vector<ShaderStage> & get_mutable_stages() { return stages; }
        void set_stages(const std::vector<ShaderStage> & value) { this->stages = value; }

        /**
         * Uniform variable declarations, shared across all stages
         */
        boost::optional<std::map<std::string, RenderShaderUniform>> get_uniforms() const { return uniforms; }
        void set_uniforms(boost::optional<std::map<std::string, RenderShaderUniform>> value) { this->uniforms = value; }
    };
}
