//  To parse this JSON data, first install
//
//      Boost     http://www.boost.org
//      json.hpp  https://github.com/nlohmann/json
//
//  Then include this file, and then do
//
//     VertexLayoutEntry.hpp data = nlohmann::json::parse(jsonString);

#pragma once

#include <boost/optional.hpp>
#include <nlohmann/json.hpp>
#include "helper.hpp"

namespace sgns {
    enum class VertexLayoutFormat : int;
}

namespace sgns {
    using nlohmann::json;

    class VertexLayoutEntry {
        public:
        VertexLayoutEntry() :
            offset_constraint(boost::none, boost::none, boost::none, boost::none, boost::none, boost::none, boost::none)
        {}
        virtual ~VertexLayoutEntry() = default;

        private:
        VertexLayoutFormat format;
        std::string name;
        int64_t offset;
        ClassMemberConstraints offset_constraint;

        public:
        /**
         * Vertex attribute component format
         */
        const VertexLayoutFormat & get_format() const { return format; }
        VertexLayoutFormat & get_mutable_format() { return format; }
        void set_format(const VertexLayoutFormat & value) { this->format = value; }

        /**
         * Vertex attribute name
         */
        const std::string & get_name() const { return name; }
        std::string & get_mutable_name() { return name; }
        void set_name(const std::string & value) { this->name = value; }

        /**
         * Byte offset within the vertex; stride is auto-computed from the tightly-packed sum of
         * attribute sizes, not schema-configurable
         */
        const int64_t & get_offset() const { return offset; }
        int64_t & get_mutable_offset() { return offset; }
        void set_offset(const int64_t & value) { CheckConstraint("offset", offset_constraint, value); this->offset = value; }
    };
}
