//  To parse this JSON data, first install
//
//      Boost     http://www.boost.org
//      json.hpp  https://github.com/nlohmann/json
//
//  Then include this file, and then do
//
//     IndexBuffer.hpp data = nlohmann::json::parse(jsonString);

#pragma once

#include <boost/optional.hpp>
#include <nlohmann/json.hpp>
#include "helper.hpp"

namespace sgns {
    enum class IndexType : int;
}

namespace sgns {
    /**
     * Index buffer binding + index type for render passes
     *
     * Index buffer binding for render passes; index_type is schema-configurable per D-17
     */

    using nlohmann::json;

    /**
     * Index buffer binding + index type for render passes
     *
     * Index buffer binding for render passes; index_type is schema-configurable per D-17
     */
    class IndexBuffer {
        public:
        IndexBuffer() :
            source_constraint(boost::none, boost::none, boost::none, boost::none, boost::none, boost::none, std::string("^(input|output|internal|parameter):[a-zA-Z][a-zA-Z0-9_]*$"))
        {}
        virtual ~IndexBuffer() = default;

        private:
        boost::optional<IndexType> index_type;
        boost::optional<std::string> source;
        ClassMemberConstraints source_constraint;

        public:
        boost::optional<IndexType> get_index_type() const { return index_type; }
        void set_index_type(boost::optional<IndexType> value) { this->index_type = value; }

        /**
         * Data source using prefix notation
         */
        boost::optional<std::string> get_source() const { return source; }
        void set_source(boost::optional<std::string> value) { if (value) CheckConstraint("source", source_constraint, *value); this->source = value; }
    };
}
