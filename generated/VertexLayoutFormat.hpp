//  To parse this JSON data, first install
//
//      Boost     http://www.boost.org
//      json.hpp  https://github.com/nlohmann/json
//
//  Then include this file, and then do
//
//     VertexLayoutFormat.hpp data = nlohmann::json::parse(jsonString);

#pragma once

#include <boost/optional.hpp>
#include <nlohmann/json.hpp>
#include "helper.hpp"

namespace sgns {
    /**
     * Vertex attribute component format
     */

    using nlohmann::json;

    /**
     * Vertex attribute component format
     */
    enum class VertexLayoutFormat : int { FLOAT16, FLOAT32, INT32 };
}
