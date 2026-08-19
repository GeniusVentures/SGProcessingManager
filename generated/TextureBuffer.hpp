//  To parse this JSON data, first install
//
//      Boost     http://www.boost.org
//      json.hpp  https://github.com/nlohmann/json
//
//  Then include this file, and then do
//
//     TextureBuffer.hpp data = nlohmann::json::parse(jsonString);

#pragma once

#include <boost/optional.hpp>
#include <nlohmann/json.hpp>
#include "helper.hpp"

namespace sgns {
    enum class TextureFilter : int;
}

namespace sgns {
    /**
     * Buffer binding supplying a sampled texture image for render passes (Phase 17 D-05)
     *
     * Raw RGBA8 image bytes for a render pass's sampled texture input (Phase 17 D-05), using
     * the same input:-prefix convention as vertex_buffer -- structurally unrelated to
     * DataType::TEXTURE2_D's MNN-chunking shape (block_len/chunk_stride), which is
     * purpose-built for CPU-side tensor processing, not a single flat sampled image.
     */

    using nlohmann::json;

    /**
     * Buffer binding supplying a sampled texture image for render passes (Phase 17 D-05)
     *
     * Raw RGBA8 image bytes for a render pass's sampled texture input (Phase 17 D-05), using
     * the same input:-prefix convention as vertex_buffer -- structurally unrelated to
     * DataType::TEXTURE2_D's MNN-chunking shape (block_len/chunk_stride), which is
     * purpose-built for CPU-side tensor processing, not a single flat sampled image.
     */
    class TextureBuffer {
        public:
        TextureBuffer() :
            height_constraint(1, 8192, boost::none, boost::none, boost::none, boost::none, boost::none),
            source_constraint(boost::none, boost::none, boost::none, boost::none, boost::none, boost::none, std::string("^(input|output|internal|parameter):[a-zA-Z][a-zA-Z0-9_]*$")),
            width_constraint(1, 8192, boost::none, boost::none, boost::none, boost::none, boost::none)
        {}
        virtual ~TextureBuffer() = default;

        private:
        boost::optional<TextureFilter> filter;
        int64_t height;
        ClassMemberConstraints height_constraint;
        std::string source;
        ClassMemberConstraints source_constraint;
        int64_t width;
        ClassMemberConstraints width_constraint;

        public:
        boost::optional<TextureFilter> get_filter() const { return filter; }
        void set_filter(boost::optional<TextureFilter> value) { this->filter = value; }

        const int64_t & get_height() const { return height; }
        int64_t & get_mutable_height() { return height; }
        void set_height(const int64_t & value) { CheckConstraint("height", height_constraint, value); this->height = value; }

        /**
         * Data source using prefix notation
         */
        const std::string & get_source() const { return source; }
        std::string & get_mutable_source() { return source; }
        void set_source(const std::string & value) { CheckConstraint("source", source_constraint, value); this->source = value; }

        const int64_t & get_width() const { return width; }
        int64_t & get_mutable_width() { return width; }
        void set_width(const int64_t & value) { CheckConstraint("width", width_constraint, value); this->width = value; }
    };
}
