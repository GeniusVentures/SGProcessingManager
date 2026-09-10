//  To parse this JSON data, first install
//
//      Boost     http://www.boost.org
//      json.hpp  https://github.com/nlohmann/json
//
//  Then include this file, and then do
//
//     JobType.hpp data = nlohmann::json::parse(jsonString);

#pragma once

#include <boost/optional.hpp>
#include <nlohmann/json.hpp>
#include "helper.hpp"

namespace sgns {
    /**
     * Discriminator for the job payload. Absent means a legacy shader/inference job governed by
     * passes/inputs/outputs. v1.0 implements elm_processing only; unknown strings reject at
     * parse via the generated enum chain
     */

    using nlohmann::json;

    /**
     * Discriminator for the job payload. Absent means a legacy shader/inference job governed by
     * passes/inputs/outputs. v1.0 implements elm_processing only; unknown strings reject at
     * parse via the generated enum chain
     */
    enum class JobType : int { ELM_PROCESSING };
}
