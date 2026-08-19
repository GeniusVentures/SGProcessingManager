//  To parse this JSON data, first install
//
//      Boost     http://www.boost.org
//      json.hpp  https://github.com/nlohmann/json
//
//  Then include this file, and then do
//
//     PipelineState.hpp data = nlohmann::json::parse(jsonString);

#pragma once

#include <boost/optional.hpp>
#include <nlohmann/json.hpp>
#include "helper.hpp"

namespace sgns {
    enum class BlendFactor : int;
    enum class CullMode : int;
    enum class DepthTest : int;
    enum class FrontFace : int;
    enum class Topology : int;
}

namespace sgns {
    /**
     * Fixed-function pipeline state for render passes
     *
     * Curated, minimal v1 fixed-function pipeline state subset (D-13); depth compare op is
     * fixed at 'less', not schema-configurable (D-14); blend state added Phase 17 (D-05)
     */

    using nlohmann::json;

    /**
     * Fixed-function pipeline state for render passes
     *
     * Curated, minimal v1 fixed-function pipeline state subset (D-13); depth compare op is
     * fixed at 'less', not schema-configurable (D-14); blend state added Phase 17 (D-05)
     */
    class PipelineState {
        public:
        PipelineState() = default;
        virtual ~PipelineState() = default;

        private:
        boost::optional<BlendFactor> blend_dst_factor;
        boost::optional<bool> blend_enable;
        boost::optional<BlendFactor> blend_src_factor;
        boost::optional<CullMode> cull_mode;
        boost::optional<DepthTest> depth_test;
        boost::optional<FrontFace> front_face;
        boost::optional<Topology> topology;

        public:
        boost::optional<BlendFactor> get_blend_dst_factor() const { return blend_dst_factor; }
        void set_blend_dst_factor(boost::optional<BlendFactor> value) { this->blend_dst_factor = value; }

        boost::optional<bool> get_blend_enable() const { return blend_enable; }
        void set_blend_enable(boost::optional<bool> value) { this->blend_enable = value; }

        boost::optional<BlendFactor> get_blend_src_factor() const { return blend_src_factor; }
        void set_blend_src_factor(boost::optional<BlendFactor> value) { this->blend_src_factor = value; }

        boost::optional<CullMode> get_cull_mode() const { return cull_mode; }
        void set_cull_mode(boost::optional<CullMode> value) { this->cull_mode = value; }

        boost::optional<DepthTest> get_depth_test() const { return depth_test; }
        void set_depth_test(boost::optional<DepthTest> value) { this->depth_test = value; }

        boost::optional<FrontFace> get_front_face() const { return front_face; }
        void set_front_face(boost::optional<FrontFace> value) { this->front_face = value; }

        boost::optional<Topology> get_topology() const { return topology; }
        void set_topology(boost::optional<Topology> value) { this->topology = value; }
    };
}
