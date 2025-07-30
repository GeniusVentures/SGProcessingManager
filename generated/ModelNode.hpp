//  To parse this JSON data, first install
//
//      Boost     http://www.boost.org
//      json.hpp  https://github.com/nlohmann/json
//
//  Then include this file, and then do
//
//     ModelNode.hpp data = nlohmann::json::parse(jsonString);

#pragma once

#include <boost/optional.hpp>
#include <nlohmann/json.hpp>
#include "helper.hpp"

namespace sgns {
    enum class DataType : int;
}

namespace sgns {
    using nlohmann::json;

    class ModelNode {
        public:
        ModelNode() :
            source_constraint(boost::none, boost::none, boost::none, boost::none, boost::none, boost::none, std::string("^(input|output|internal|parameter):[a-zA-Z][a-zA-Z0-9_]*$")),
            target_constraint(boost::none, boost::none, boost::none, boost::none, boost::none, boost::none, std::string("^(output|internal):[a-zA-Z][a-zA-Z0-9_]*$"))
        {}
        virtual ~ModelNode() = default;

        private:
        std::string name;
        boost::optional<std::vector<int64_t>> shape;
        std::string source;
        ClassMemberConstraints source_constraint;
        std::string target;
        ClassMemberConstraints target_constraint;
        DataType type;

        public:
        /**
         * Node name in the model graph
         */
        const std::string & get_name() const { return name; }
        std::string & get_mutable_name() { return name; }
        void set_name(const std::string & value) { this->name = value; }

        /**
         * Expected tensor shape
         */
        boost::optional<std::vector<int64_t>> get_shape() const { return shape; }
        void set_shape(boost::optional<std::vector<int64_t>> value) { this->shape = value; }

        /**
         * Data source using prefix notation (input:, output:, internal:, parameter:)
         */
        const std::string & get_source() const { return source; }
        std::string & get_mutable_source() { return source; }
        void set_source(const std::string & value) { CheckConstraint("source", source_constraint, value); this->source = value; }

        /**
         * Data target using prefix notation
         */
        const std::string & get_target() const { return target; }
        std::string & get_mutable_target() { return target; }
        void set_target(const std::string & value) { CheckConstraint("target", target_constraint, value); this->target = value; }

        const DataType & get_type() const { return type; }
        DataType & get_mutable_type() { return type; }
        void set_type(const DataType & value) { this->type = value; }
    };
}
