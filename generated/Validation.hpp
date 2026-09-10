//  To parse this JSON data, first install
//
//      Boost     http://www.boost.org
//      json.hpp  https://github.com/nlohmann/json
//
//  Then include this file, and then do
//
//     Validation.hpp data = nlohmann::json::parse(jsonString);

#pragma once

#include <boost/optional.hpp>
#include <nlohmann/json.hpp>
#include "helper.hpp"

namespace sgns {
    /**
     * Work-item result validation mode. v1.0 implements none only; exact/redundant parse but
     * are refused by the ProcessingManager C++ gate as unimplemented
     */

    using nlohmann::json;

    /**
     * Work-item result validation mode. v1.0 implements none only; exact/redundant parse but
     * are refused by the ProcessingManager C++ gate as unimplemented
     */
    enum class Validation : int { EXACT, NONE, REDUNDANT };
}
