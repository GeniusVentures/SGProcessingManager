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
     * fixed at 'less', not schema-configurable (D-14)
     */

    using nlohmann::json;

    /**
     * Fixed-function pipeline state for render passes
     *
     * Curated, minimal v1 fixed-function pipeline state subset (D-13); depth compare op is
     * fixed at 'less', not schema-configurable (D-14)
     */
    class PipelineState {
        public:
        PipelineState() = default;
        virtual ~PipelineState() = default;

        private:
        boost::optional<CullMode> cull_mode;
        boost::optional<DepthTest> depth_test;
        boost::optional<FrontFace> front_face;
        boost::optional<Topology> topology;

        public:
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
