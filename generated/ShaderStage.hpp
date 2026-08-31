//  To parse this JSON data, first install
//
//      Boost     http://www.boost.org
//      json.hpp  https://github.com/nlohmann/json
//
//  Then include this file, and then do
//
//     ShaderStage.hpp data = nlohmann::json::parse(jsonString);

#pragma once

#include <boost/optional.hpp>
#include <nlohmann/json.hpp>
#include "helper.hpp"

namespace sgns {
    enum class Stage : int;
    enum class ShaderSourceType : int;
}

namespace sgns {
    using nlohmann::json;

    class ShaderStage {
        public:
        ShaderStage() = default;
        virtual ~ShaderStage() = default;

        private:
        boost::optional<std::string> entry_point;
        std::string source;
        Stage stage;
        ShaderSourceType type;

        public:
        boost::optional<std::string> get_entry_point() const { return entry_point; }
        void set_entry_point(boost::optional<std::string> value) { this->entry_point = value; }

        /**
         * Shader source path or URI parameter for this stage
         */
        const std::string & get_source() const { return source; }
        std::string & get_mutable_source() { return source; }
        void set_source(const std::string & value) { this->source = value; }

        /**
         * Which pipeline stage this shader source targets
         */
        const Stage & get_stage() const { return stage; }
        Stage & get_mutable_stage() { return stage; }
        void set_stage(const Stage & value) { this->stage = value; }

        const ShaderSourceType & get_type() const { return type; }
        ShaderSourceType & get_mutable_type() { return type; }
        void set_type(const ShaderSourceType & value) { this->type = value; }
    };
}
