//  To parse this JSON data, first install
//
//      Boost     http://www.boost.org
//      json.hpp  https://github.com/nlohmann/json
//
//  Then include this file, and then do
//
//     RenderTarget.hpp data = nlohmann::json::parse(jsonString);

#pragma once

#include <boost/optional.hpp>
#include <nlohmann/json.hpp>
#include "helper.hpp"

namespace sgns {
    enum class ColorFormat : int;
    enum class DepthFormat : int;
}

namespace sgns {
    /**
     * Offscreen framebuffer (color+depth) config for render passes
     *
     * Offscreen render-target/framebuffer config - all fields required, no schema defaults
     */

    using nlohmann::json;

    /**
     * Offscreen framebuffer (color+depth) config for render passes
     *
     * Offscreen render-target/framebuffer config - all fields required, no schema defaults
     */
    class RenderTarget {
        public:
        RenderTarget() :
            clear_depth_constraint(boost::none, boost::none, boost::none, 1, boost::none, boost::none, boost::none),
            height_constraint(1, boost::none, boost::none, boost::none, boost::none, boost::none, boost::none),
            width_constraint(1, boost::none, boost::none, boost::none, boost::none, boost::none, boost::none)
        {}
        virtual ~RenderTarget() = default;

        private:
        std::vector<double> clear_color;
        double clear_depth;
        ClassMemberConstraints clear_depth_constraint;
        ColorFormat color_format;
        DepthFormat depth_format;
        int64_t height;
        ClassMemberConstraints height_constraint;
        int64_t width;
        ClassMemberConstraints width_constraint;

        public:
        /**
         * RGBA clear color
         */
        const std::vector<double> & get_clear_color() const { return clear_color; }
        std::vector<double> & get_mutable_clear_color() { return clear_color; }
        void set_clear_color(const std::vector<double> & value) { this->clear_color = value; }

        const double & get_clear_depth() const { return clear_depth; }
        double & get_mutable_clear_depth() { return clear_depth; }
        void set_clear_depth(const double & value) { CheckConstraint("clear_depth", clear_depth_constraint, value); this->clear_depth = value; }

        /**
         * Color attachment format
         */
        const ColorFormat & get_color_format() const { return color_format; }
        ColorFormat & get_mutable_color_format() { return color_format; }
        void set_color_format(const ColorFormat & value) { this->color_format = value; }

        /**
         * Depth attachment format
         */
        const DepthFormat & get_depth_format() const { return depth_format; }
        DepthFormat & get_mutable_depth_format() { return depth_format; }
        void set_depth_format(const DepthFormat & value) { this->depth_format = value; }

        const int64_t & get_height() const { return height; }
        int64_t & get_mutable_height() { return height; }
        void set_height(const int64_t & value) { CheckConstraint("height", height_constraint, value); this->height = value; }

        const int64_t & get_width() const { return width; }
        int64_t & get_mutable_width() { return width; }
        void set_width(const int64_t & value) { CheckConstraint("width", width_constraint, value); this->width = value; }
    };
}
