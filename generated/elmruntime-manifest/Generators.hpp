//  To parse this JSON data, first install
//
//      Boost     http://www.boost.org
//      json.hpp  https://github.com/nlohmann/json
//
//  Then include this file, and then do
//
//     Generators.hpp data = nlohmann::json::parse(jsonString);

#pragma once

#include <boost/optional.hpp>
#include <nlohmann/json.hpp>
#include "helper.hpp"

#include "ElmModelManifest.hpp"
#include "ElmModelRuntime.hpp"
#include "ElmType.hpp"
#include "ElmModelArtifact.hpp"

namespace sgns {
    void from_json(const json & j, ElmModelArtifact & x);
    void to_json(json & j, const ElmModelArtifact & x);

    void from_json(const json & j, ElmModelRuntime & x);
    void to_json(json & j, const ElmModelRuntime & x);

    void from_json(const json & j, ElmModelManifest & x);
    void to_json(json & j, const ElmModelManifest & x);

    void from_json(const json & j, ElmType & x);
    void to_json(json & j, const ElmType & x);

    inline void from_json(const json & j, ElmModelArtifact& x) {
        x.set_name(j.at("name").get<std::string>());
        x.set_sha256(j.at("sha256").get<std::string>());
        x.set_size_bytes(j.at("size_bytes").get<int64_t>());
        x.set_uri(j.at("uri").get<std::string>());
    }

    inline void to_json(json & j, const ElmModelArtifact & x) {
        j = json::object();
        j["name"] = x.get_name();
        j["sha256"] = x.get_sha256();
        j["size_bytes"] = x.get_size_bytes();
        j["uri"] = x.get_uri();
    }

    inline void from_json(const json & j, ElmModelRuntime& x) {
        x.set_required_memory_bytes(get_stack_optional<int64_t>(j, "required_memory_bytes"));
    }

    inline void to_json(json & j, const ElmModelRuntime & x) {
        j = json::object();
        j["required_memory_bytes"] = x.get_required_memory_bytes();
    }

    inline void from_json(const json & j, ElmModelManifest& x) {
        x.set_artifacts(j.at("artifacts").get<std::vector<ElmModelArtifact>>());
        x.set_elm_type(j.at("elm_type").get<ElmType>());
        x.set_model_format(j.at("model_format").get<std::string>());
        x.set_quantization(get_stack_optional<std::string>(j, "quantization"));
        x.set_runtime(get_stack_optional<ElmModelRuntime>(j, "runtime"));
        x.set_schema_version(j.at("schema_version").get<int64_t>());
    }

    inline void to_json(json & j, const ElmModelManifest & x) {
        j = json::object();
        j["artifacts"] = x.get_artifacts();
        j["elm_type"] = x.get_elm_type();
        j["model_format"] = x.get_model_format();
        j["quantization"] = x.get_quantization();
        j["runtime"] = x.get_runtime();
        j["schema_version"] = x.get_schema_version();
    }

    inline void from_json(const json & j, ElmType & x) {
        if (j == "causal_lm") x = ElmType::CAUSAL_LM;
        else { throw std::runtime_error("Input JSON does not conform to schema!"); }
    }

    inline void to_json(json & j, const ElmType & x) {
        switch (x) {
            case ElmType::CAUSAL_LM: j = "causal_lm"; break;
            default: throw std::runtime_error("Unexpected value in enumeration \"ElmType\": " + std::to_string(static_cast<int>(x)));
        }
    }
}
