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
        const boost::optional<BlendFactor> & get_blend_dst_factor() const { return blend_dst_factor; }
        boost::optional<BlendFactor> & get_mutable_blend_dst_factor() { return blend_dst_factor; }
        void set_blend_dst_factor(const boost::optional<BlendFactor> & value) { this->blend_dst_factor = value; }

        const boost::optional<bool> & get_blend_enable() const { return blend_enable; }
        boost::optional<bool> & get_mutable_blend_enable() { return blend_enable; }
        void set_blend_enable(const boost::optional<bool> & value) { this->blend_enable = value; }

        const boost::optional<BlendFactor> & get_blend_src_factor() const { return blend_src_factor; }
        boost::optional<BlendFactor> & get_mutable_blend_src_factor() { return blend_src_factor; }
        void set_blend_src_factor(const boost::optional<BlendFactor> & value) { this->blend_src_factor = value; }

        const boost::optional<CullMode> & get_cull_mode() const { return cull_mode; }
        boost::optional<CullMode> & get_mutable_cull_mode() { return cull_mode; }
        void set_cull_mode(const boost::optional<CullMode> & value) { this->cull_mode = value; }

        const boost::optional<DepthTest> & get_depth_test() const { return depth_test; }
        boost::optional<DepthTest> & get_mutable_depth_test() { return depth_test; }
        void set_depth_test(const boost::optional<DepthTest> & value) { this->depth_test = value; }

        const boost::optional<FrontFace> & get_front_face() const { return front_face; }
        boost::optional<FrontFace> & get_mutable_front_face() { return front_face; }
        void set_front_face(const boost::optional<FrontFace> & value) { this->front_face = value; }

        const boost::optional<Topology> & get_topology() const { return topology; }
        boost::optional<Topology> & get_mutable_topology() { return topology; }
        void set_topology(const boost::optional<Topology> & value) { this->topology = value; }
    };
}
