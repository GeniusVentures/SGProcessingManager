//  To parse this JSON data, first install
//
//      Boost     http://www.boost.org
//      json.hpp  https://github.com/nlohmann/json
//
//  Then include this file, and then do
//
//     DepthFormat.hpp data = nlohmann::json::parse(jsonString);

#pragma once

#include <boost/optional.hpp>
#include <nlohmann/json.hpp>
#include "helper.hpp"

namespace sgns {
    /**
     * Depth attachment format
     */

    using nlohmann::json;

    /**
     * Depth attachment format
     */
    enum class DepthFormat : int { D24_UNORM_S8_UINT, D32_SFLOAT };
}
