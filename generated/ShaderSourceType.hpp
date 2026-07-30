//  To parse this JSON data, first install
//
//      Boost     http://www.boost.org
//      json.hpp  https://github.com/nlohmann/json
//
//  Then include this file, and then do
//
//     ShaderSourceType.hpp data = nlohmann::json::parse(jsonString);

#pragma once

#include <boost/optional.hpp>
#include <nlohmann/json.hpp>
#include "helper.hpp"

namespace sgns {
    /**
     * Shader source language, validated before it ever reaches the driver
     */

    using nlohmann::json;

    /**
     * Shader source language, validated before it ever reaches the driver
     */
    enum class ShaderSourceType : int { GLSL, SPIRV };
}
