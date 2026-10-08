//  To parse this JSON data, first install
//
//      Boost     http://www.boost.org
//      json.hpp  https://github.com/nlohmann/json
//
//  Then include this file, and then do
//
//     PassIoBinding.hpp data = nlohmann::json::parse(jsonString);

#pragma once

#include <boost/optional.hpp>
#include <nlohmann/json.hpp>
#include "helper.hpp"

namespace sgns {
    using nlohmann::json;

    class PassIoBinding {
        public:
        PassIoBinding() :
            source_constraint(boost::none, boost::none, boost::none, boost::none, boost::none, boost::none, std::string("^(input|output|internal|parameter):[a-zA-Z][a-zA-Z0-9_]*$")),
            target_constraint(boost::none, boost::none, boost::none, boost::none, boost::none, boost::none, std::string("^(output|internal):[a-zA-Z][a-zA-Z0-9_]*$"))
        {}
        virtual ~PassIoBinding() = default;

        private:
        std::string name;
        boost::optional<std::string> source;
        ClassMemberConstraints source_constraint;
        boost::optional<std::string> target;
        ClassMemberConstraints target_constraint;

        public:
        /**
         * Binding point name
         */
        const std::string & get_name() const { return name; }
        std::string & get_mutable_name() { return name; }
        void set_name(const std::string & value) { this->name = value; }

        /**
         * Data source using prefix notation
         */
        const boost::optional<std::string> & get_source() const { return source; }
        boost::optional<std::string> & get_mutable_source() { return source; }
        void set_source(const boost::optional<std::string> & value) { CheckConstraint("source", source_constraint, value); this->source = value; }

        /**
         * Data target using prefix notation
         */
        const boost::optional<std::string> & get_target() const { return target; }
        boost::optional<std::string> & get_mutable_target() { return target; }
        void set_target(const boost::optional<std::string> & value) { CheckConstraint("target", target_constraint, value); this->target = value; }
    };
}
