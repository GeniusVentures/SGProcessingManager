//  To parse this JSON data, first install
//
//      Boost     http://www.boost.org
//      json.hpp  https://github.com/nlohmann/json
//
//  Then include this file, and then do
//
//     Dimensions.hpp data = nlohmann::json::parse(jsonString);

#pragma once

#include <boost/optional.hpp>
#include <nlohmann/json.hpp>
#include "helper.hpp"

namespace sgns {
    /**
     * Optional dimensions specification
     */

    using nlohmann::json;

    /**
     * Optional dimensions specification
     */
    class Dimensions {
        public:
        Dimensions() :
            batch_constraint(1, boost::none, boost::none, boost::none, boost::none, boost::none, boost::none),
            block_len_constraint(1, boost::none, boost::none, boost::none, boost::none, boost::none, boost::none),
            block_line_stride_constraint(1, boost::none, boost::none, boost::none, boost::none, boost::none, boost::none),
            block_stride_constraint(0, boost::none, boost::none, boost::none, boost::none, boost::none, boost::none),
            chunk_count_constraint(1, boost::none, boost::none, boost::none, boost::none, boost::none, boost::none),
            chunk_line_stride_constraint(1, boost::none, boost::none, boost::none, boost::none, boost::none, boost::none),
            chunk_offset_constraint(0, boost::none, boost::none, boost::none, boost::none, boost::none, boost::none),
            chunk_stride_constraint(1, boost::none, boost::none, boost::none, boost::none, boost::none, boost::none),
            chunk_subchunk_height_constraint(1, boost::none, boost::none, boost::none, boost::none, boost::none, boost::none),
            chunk_subchunk_width_constraint(1, boost::none, boost::none, boost::none, boost::none, boost::none, boost::none),
            height_constraint(1, boost::none, boost::none, boost::none, boost::none, boost::none, boost::none),
            width_constraint(1, boost::none, boost::none, boost::none, boost::none, boost::none, boost::none)
        {}
        virtual ~Dimensions() = default;

        private:
        boost::optional<int64_t> batch;
        ClassMemberConstraints batch_constraint;
        boost::optional<int64_t> block_len;
        ClassMemberConstraints block_len_constraint;
        boost::optional<int64_t> block_line_stride;
        ClassMemberConstraints block_line_stride_constraint;
        boost::optional<int64_t> block_stride;
        ClassMemberConstraints block_stride_constraint;
        boost::optional<int64_t> chunk_count;
        ClassMemberConstraints chunk_count_constraint;
        boost::optional<int64_t> chunk_line_stride;
        ClassMemberConstraints chunk_line_stride_constraint;
        boost::optional<int64_t> chunk_offset;
        ClassMemberConstraints chunk_offset_constraint;
        boost::optional<int64_t> chunk_stride;
        ClassMemberConstraints chunk_stride_constraint;
        boost::optional<int64_t> chunk_subchunk_height;
        ClassMemberConstraints chunk_subchunk_height_constraint;
        boost::optional<int64_t> chunk_subchunk_width;
        ClassMemberConstraints chunk_subchunk_width_constraint;
        boost::optional<int64_t> height;
        ClassMemberConstraints height_constraint;
        boost::optional<int64_t> width;
        ClassMemberConstraints width_constraint;

        public:
        const boost::optional<int64_t> & get_batch() const { return batch; }
        boost::optional<int64_t> & get_mutable_batch() { return batch; }
        void set_batch(const boost::optional<int64_t> & value) { CheckConstraint("batch", batch_constraint, value); this->batch = value; }

        const boost::optional<int64_t> & get_block_len() const { return block_len; }
        boost::optional<int64_t> & get_mutable_block_len() { return block_len; }
        void set_block_len(const boost::optional<int64_t> & value) { CheckConstraint("block_len", block_len_constraint, value); this->block_len = value; }

        const boost::optional<int64_t> & get_block_line_stride() const { return block_line_stride; }
        boost::optional<int64_t> & get_mutable_block_line_stride() { return block_line_stride; }
        void set_block_line_stride(const boost::optional<int64_t> & value) { CheckConstraint("block_line_stride", block_line_stride_constraint, value); this->block_line_stride = value; }

        const boost::optional<int64_t> & get_block_stride() const { return block_stride; }
        boost::optional<int64_t> & get_mutable_block_stride() { return block_stride; }
        void set_block_stride(const boost::optional<int64_t> & value) { CheckConstraint("block_stride", block_stride_constraint, value); this->block_stride = value; }

        const boost::optional<int64_t> & get_chunk_count() const { return chunk_count; }
        boost::optional<int64_t> & get_mutable_chunk_count() { return chunk_count; }
        void set_chunk_count(const boost::optional<int64_t> & value) { CheckConstraint("chunk_count", chunk_count_constraint, value); this->chunk_count = value; }

        const boost::optional<int64_t> & get_chunk_line_stride() const { return chunk_line_stride; }
        boost::optional<int64_t> & get_mutable_chunk_line_stride() { return chunk_line_stride; }
        void set_chunk_line_stride(const boost::optional<int64_t> & value) { CheckConstraint("chunk_line_stride", chunk_line_stride_constraint, value); this->chunk_line_stride = value; }

        const boost::optional<int64_t> & get_chunk_offset() const { return chunk_offset; }
        boost::optional<int64_t> & get_mutable_chunk_offset() { return chunk_offset; }
        void set_chunk_offset(const boost::optional<int64_t> & value) { CheckConstraint("chunk_offset", chunk_offset_constraint, value); this->chunk_offset = value; }

        const boost::optional<int64_t> & get_chunk_stride() const { return chunk_stride; }
        boost::optional<int64_t> & get_mutable_chunk_stride() { return chunk_stride; }
        void set_chunk_stride(const boost::optional<int64_t> & value) { CheckConstraint("chunk_stride", chunk_stride_constraint, value); this->chunk_stride = value; }

        const boost::optional<int64_t> & get_chunk_subchunk_height() const { return chunk_subchunk_height; }
        boost::optional<int64_t> & get_mutable_chunk_subchunk_height() { return chunk_subchunk_height; }
        void set_chunk_subchunk_height(const boost::optional<int64_t> & value) { CheckConstraint("chunk_subchunk_height", chunk_subchunk_height_constraint, value); this->chunk_subchunk_height = value; }

        const boost::optional<int64_t> & get_chunk_subchunk_width() const { return chunk_subchunk_width; }
        boost::optional<int64_t> & get_mutable_chunk_subchunk_width() { return chunk_subchunk_width; }
        void set_chunk_subchunk_width(const boost::optional<int64_t> & value) { CheckConstraint("chunk_subchunk_width", chunk_subchunk_width_constraint, value); this->chunk_subchunk_width = value; }

        const boost::optional<int64_t> & get_height() const { return height; }
        boost::optional<int64_t> & get_mutable_height() { return height; }
        void set_height(const boost::optional<int64_t> & value) { CheckConstraint("height", height_constraint, value); this->height = value; }

        const boost::optional<int64_t> & get_width() const { return width; }
        boost::optional<int64_t> & get_mutable_width() { return width; }
        void set_width(const boost::optional<int64_t> & value) { CheckConstraint("width", width_constraint, value); this->width = value; }
    };
}
