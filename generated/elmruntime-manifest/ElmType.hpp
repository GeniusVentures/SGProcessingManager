//  To parse this JSON data, first install
//
//      Boost     http://www.boost.org
//      json.hpp  https://github.com/nlohmann/json
//
//  Then include this file, and then do
//
//     ElmType.hpp data = nlohmann::json::parse(jsonString);

#pragma once

#include <boost/optional.hpp>
#include <nlohmann/json.hpp>
#include "helper.hpp"

namespace sgns {
    /**
     * v1.0 implements causal_lm only (same enum as the Elm work item)
     */

    using nlohmann::json;

    /**
     * v1.0 implements causal_lm only (same enum as the Elm work item)
     */
    enum class ElmType : int { CAUSAL_LM };
}
