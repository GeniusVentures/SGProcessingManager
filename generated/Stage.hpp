//  To parse this JSON data, first install
//
//      Boost     http://www.boost.org
//      json.hpp  https://github.com/nlohmann/json
//
//  Then include this file, and then do
//
//     Stage.hpp data = nlohmann::json::parse(jsonString);

#pragma once

#include <boost/optional.hpp>
#include <nlohmann/json.hpp>
#include "helper.hpp"

namespace sgns {
    /**
     * Which pipeline stage this shader source targets
     */

    using nlohmann::json;

    /**
     * Which pipeline stage this shader source targets
     */
    enum class Stage : int { FRAGMENT, VERTEX };
}
