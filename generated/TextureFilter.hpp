//  To parse this JSON data, first install
//
//      Boost     http://www.boost.org
//      json.hpp  https://github.com/nlohmann/json
//
//  Then include this file, and then do
//
//     TextureFilter.hpp data = nlohmann::json::parse(jsonString);

#pragma once

#include <boost/optional.hpp>
#include <nlohmann/json.hpp>
#include "helper.hpp"

namespace sgns {
    /**
     * Sampler filter mode for a render pass's sampled texture input
     */

    using nlohmann::json;

    /**
     * Sampler filter mode for a render pass's sampled texture input
     */
    enum class TextureFilter : int { LINEAR, NEAREST };
}
