//  To parse this JSON data, first install
//
//      Boost     http://www.boost.org
//      json.hpp  https://github.com/nlohmann/json
//
//  Then include this file, and then do
//
//     RenderShaderUniform.hpp data = nlohmann::json::parse(jsonString);

#pragma once

#include <boost/optional.hpp>
#include <nlohmann/json.hpp>
#include "helper.hpp"

namespace sgns {
    enum class DataType : int;
}

namespace sgns {
    using nlohmann::json;

    class RenderShaderUniform {
        public:
        RenderShaderUniform() = default;
        virtual ~RenderShaderUniform() = default;

        private:
        boost::optional<std::string> source;
        boost::optional<DataType> type;
        nlohmann::json value;

        public:
        const boost::optional<std::string> & get_source() const { return source; }
        boost::optional<std::string> & get_mutable_source() { return source; }
        void set_source(const boost::optional<std::string> & value) { this->source = value; }

        const boost::optional<DataType> & get_type() const { return type; }
        boost::optional<DataType> & get_mutable_type() { return type; }
        void set_type(const boost::optional<DataType> & value) { this->type = value; }

        const nlohmann::json & get_value() const { return value; }
        nlohmann::json & get_mutable_value() { return value; }
        void set_value(const nlohmann::json & value) { this->value = value; }
    };
}
