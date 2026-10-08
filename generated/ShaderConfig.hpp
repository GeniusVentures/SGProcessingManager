//  To parse this JSON data, first install
//
//      Boost     http://www.boost.org
//      json.hpp  https://github.com/nlohmann/json
//
//  Then include this file, and then do
//
//     ShaderConfig.hpp data = nlohmann::json::parse(jsonString);

#pragma once

#include <boost/optional.hpp>
#include <nlohmann/json.hpp>
#include "helper.hpp"

#include "ShaderUniform.hpp"

namespace sgns {
    enum class ShaderSourceType : int;
}

namespace sgns {
    /**
     * Shader configuration for compute passes
     */

    using nlohmann::json;

    /**
     * Shader configuration for compute passes
     */
    class ShaderConfig {
        public:
        ShaderConfig() = default;
        virtual ~ShaderConfig() = default;

        private:
        boost::optional<std::string> entry_point;
        std::string source;
        boost::optional<ShaderSourceType> type;
        boost::optional<std::map<std::string, ShaderUniform>> uniforms;

        public:
        const boost::optional<std::string> & get_entry_point() const { return entry_point; }
        boost::optional<std::string> & get_mutable_entry_point() { return entry_point; }
        void set_entry_point(const boost::optional<std::string> & value) { this->entry_point = value; }

        /**
         * Shader source path or URI parameter
         */
        const std::string & get_source() const { return source; }
        std::string & get_mutable_source() { return source; }
        void set_source(const std::string & value) { this->source = value; }

        const boost::optional<ShaderSourceType> & get_type() const { return type; }
        boost::optional<ShaderSourceType> & get_mutable_type() { return type; }
        void set_type(const boost::optional<ShaderSourceType> & value) { this->type = value; }

        /**
         * Uniform variable declarations
         */
        const boost::optional<std::map<std::string, ShaderUniform>> & get_uniforms() const { return uniforms; }
        boost::optional<std::map<std::string, ShaderUniform>> & get_mutable_uniforms() { return uniforms; }
        void set_uniforms(const boost::optional<std::map<std::string, ShaderUniform>> & value) { this->uniforms = value; }
    };
}
