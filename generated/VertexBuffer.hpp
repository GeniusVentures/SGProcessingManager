//  To parse this JSON data, first install
//
//      Boost     http://www.boost.org
//      json.hpp  https://github.com/nlohmann/json
//
//  Then include this file, and then do
//
//     VertexBuffer.hpp data = nlohmann::json::parse(jsonString);

#pragma once

#include <boost/optional.hpp>
#include <nlohmann/json.hpp>
#include "helper.hpp"

namespace sgns {
    /**
     * Buffer binding supplying vertex attribute data referenced by vertex_layout (D-16
     * Amendment)
     *
     * Buffer binding that supplies vertex attribute data for vertex_layout entries, using the
     * same prefix-notation convention as pass_io_binding
     */

    using nlohmann::json;

    /**
     * Buffer binding supplying vertex attribute data referenced by vertex_layout (D-16
     * Amendment)
     *
     * Buffer binding that supplies vertex attribute data for vertex_layout entries, using the
     * same prefix-notation convention as pass_io_binding
     */
    class VertexBuffer {
        public:
        VertexBuffer() :
            source_constraint(boost::none, boost::none, boost::none, boost::none, boost::none, boost::none, std::string("^(input|output|internal|parameter):[a-zA-Z][a-zA-Z0-9_]*$"))
        {}
        virtual ~VertexBuffer() = default;

        private:
        std::string source;
        ClassMemberConstraints source_constraint;

        public:
        /**
         * Data source using prefix notation
         */
        const std::string & get_source() const { return source; }
        std::string & get_mutable_source() { return source; }
        void set_source(const std::string & value) { CheckConstraint("source", source_constraint, value); this->source = value; }
    };
}
