//  To parse this JSON data, first install
//
//      Boost     http://www.boost.org
//      json.hpp  https://github.com/nlohmann/json
//
//  Then include this file, and then do
//
//     ElmModelManifest.hpp data = nlohmann::json::parse(jsonString);

#pragma once

#include <boost/optional.hpp>
#include <nlohmann/json.hpp>
#include "helper.hpp"

#include "ElmModelArtifact.hpp"
#include "ElmModelRuntime.hpp"

namespace sgns {
    enum class ElmType : int;
}

namespace sgns {
    /**
     * Dedicated generation source for the ELM model manifest types. The same three definitions
     * live in gnus-processing-schema.json (shape documentation), but quicktype's schema
     * reachability rules drop definitions not referenced from root properties -- this file
     * exists solely so ElmModelManifest and its dependencies regenerate through the same
     * quicktype pipeline (D-05: one pipeline, zero hand edits to generated/)
     *
     * Content-addressed manifest of an ELM model bundle. Its bytes are hash-pinned by the work
     * item's model_manifest_hash before parsing (SC-1). model_format is deliberately a
     * pattern-constrained STRING, not an enum: sgns::ModelFormat is already a live generated
     * enum from model_config.format with in-repo consumers
     */

    using nlohmann::json;

    /**
     * Dedicated generation source for the ELM model manifest types. The same three definitions
     * live in gnus-processing-schema.json (shape documentation), but quicktype's schema
     * reachability rules drop definitions not referenced from root properties -- this file
     * exists solely so ElmModelManifest and its dependencies regenerate through the same
     * quicktype pipeline (D-05: one pipeline, zero hand edits to generated/)
     *
     * Content-addressed manifest of an ELM model bundle. Its bytes are hash-pinned by the work
     * item's model_manifest_hash before parsing (SC-1). model_format is deliberately a
     * pattern-constrained STRING, not an enum: sgns::ModelFormat is already a live generated
     * enum from model_config.format with in-repo consumers
     */
    class ElmModelManifest {
        public:
        ElmModelManifest() :
            model_format_constraint(boost::none, boost::none, boost::none, boost::none, boost::none, boost::none, std::string("^mnn$")),
            quantization_constraint(boost::none, boost::none, boost::none, boost::none, 1, boost::none, boost::none),
            schema_version_constraint(1, 1, boost::none, boost::none, boost::none, boost::none, boost::none)
        {}
        virtual ~ElmModelManifest() = default;

        private:
        std::vector<ElmModelArtifact> artifacts;
        ElmType elm_type;
        std::string model_format;
        ClassMemberConstraints model_format_constraint;
        boost::optional<std::string> quantization;
        ClassMemberConstraints quantization_constraint;
        boost::optional<ElmModelRuntime> runtime;
        int64_t schema_version;
        ClassMemberConstraints schema_version_constraint;

        public:
        /**
         * Bundle artifacts. Non-empty is enforced by the ElmManifest C++ gate (quicktype drops
         * minItems)
         */
        const std::vector<ElmModelArtifact> & get_artifacts() const { return artifacts; }
        std::vector<ElmModelArtifact> & get_mutable_artifacts() { return artifacts; }
        void set_artifacts(const std::vector<ElmModelArtifact> & value) { this->artifacts = value; }

        /**
         * v1.0 implements causal_lm only (same enum as the Elm work item)
         */
        const ElmType & get_elm_type() const { return elm_type; }
        ElmType & get_mutable_elm_type() { return elm_type; }
        void set_elm_type(const ElmType & value) { this->elm_type = value; }

        /**
         * v1.0 mandates MNN. Pattern is documentation-only post-codegen; the C++ semantic gate
         * fail-closes on any value != mnn (MANIFEST_INVALID)
         */
        const std::string & get_model_format() const { return model_format; }
        std::string & get_mutable_model_format() { return model_format; }
        void set_model_format(const std::string & value) { CheckConstraint("model_format", model_format_constraint, value); this->model_format = value; }

        /**
         * Quantization descriptor (informational metadata only in v1.0)
         */
        boost::optional<std::string> get_quantization() const { return quantization; }
        void set_quantization(boost::optional<std::string> value) { if (value) CheckConstraint("quantization", quantization_constraint, *value); this->quantization = value; }

        /**
         * Runtime resource requirements (local capability preflight only)
         */
        boost::optional<ElmModelRuntime> get_runtime() const { return runtime; }
        void set_runtime(boost::optional<ElmModelRuntime> value) { this->runtime = value; }

        /**
         * Manifest shape version; fixed at 1 (gnus_spec_version style const)
         */
        const int64_t & get_schema_version() const { return schema_version; }
        int64_t & get_mutable_schema_version() { return schema_version; }
        void set_schema_version(const int64_t & value) { CheckConstraint("schema_version", schema_version_constraint, value); this->schema_version = value; }
    };
}
