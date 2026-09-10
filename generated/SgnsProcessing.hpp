//  To parse this JSON data, first install
//
//      Boost     http://www.boost.org
//      json.hpp  https://github.com/nlohmann/json
//
//  Then include this file, and then do
//
//     SgnsProcessing.hpp data = nlohmann::json::parse(jsonString);

#pragma once

#include <boost/optional.hpp>
#include <nlohmann/json.hpp>
#include "helper.hpp"

#include "Elm.hpp"
#include "ElmFunding.hpp"
#include "IoDeclaration.hpp"
#include "Parameter.hpp"
#include "Pass.hpp"

namespace sgns {
    enum class JobType : int;
    enum class Validation : int;
}

namespace sgns {
    /**
     * Schema for defining AI inference and retraining workflows with shader passes
     */

    using nlohmann::json;

    /**
     * Schema for defining AI inference and retraining workflows with shader passes
     */
    class SgnsProcessing {
        public:
        SgnsProcessing() :
            gnus_spec_version_constraint(boost::none, boost::none, 1, 1, boost::none, boost::none, boost::none),
            name_constraint(boost::none, boost::none, boost::none, boost::none, boost::none, boost::none, std::string("^[A-Za-z0-9_-]+$")),
            version_constraint(boost::none, boost::none, boost::none, boost::none, boost::none, boost::none, std::string("^\\d+\\.\\d+(\\.\\d+)?$"))
        {}
        virtual ~SgnsProcessing() = default;

        private:
        boost::optional<std::string> author;
        boost::optional<std::string> description;
        boost::optional<std::vector<Elm>> elms;
        boost::optional<ElmFunding> funding;
        double gnus_spec_version;
        ClassMemberConstraints gnus_spec_version_constraint;
        boost::optional<std::vector<IoDeclaration>> inputs;
        boost::optional<JobType> job_type;
        boost::optional<std::map<std::string, nlohmann::json>> metadata;
        std::string name;
        ClassMemberConstraints name_constraint;
        boost::optional<std::vector<IoDeclaration>> outputs;
        boost::optional<std::vector<Parameter>> parameters;
        boost::optional<std::vector<Pass>> passes;
        boost::optional<std::vector<std::string>> tags;
        boost::optional<Validation> validation;
        std::string version;
        ClassMemberConstraints version_constraint;

        public:
        /**
         * Author of this processing definition
         */
        boost::optional<std::string> get_author() const { return author; }
        void set_author(boost::optional<std::string> value) { this->author = value; }

        /**
         * Human-readable description of what this processing definition does
         */
        boost::optional<std::string> get_description() const { return description; }
        void set_description(boost::optional<std::string> value) { this->description = value; }

        /**
         * ELM work items for elm_processing jobs. Non-empty enforced by the ProcessingManager C++
         * gate (quicktype drops minItems)
         */
        boost::optional<std::vector<Elm>> get_elms() const { return elms; }
        void set_elms(boost::optional<std::vector<Elm>> value) { this->elms = value; }

        /**
         * Funding envelope for elm_processing jobs (maximum processing hours)
         */
        boost::optional<ElmFunding> get_funding() const { return funding; }
        void set_funding(boost::optional<ElmFunding> value) { this->funding = value; }

        /**
         * Version of the GNUS processing definition specification
         */
        const double & get_gnus_spec_version() const { return gnus_spec_version; }
        double & get_mutable_gnus_spec_version() { return gnus_spec_version; }
        void set_gnus_spec_version(const double & value) { CheckConstraint("gnus_spec_version", gnus_spec_version_constraint, value); this->gnus_spec_version = value; }

        /**
         * Declares the external inputs this process requires
         */
        boost::optional<std::vector<IoDeclaration>> get_inputs() const { return inputs; }
        void set_inputs(boost::optional<std::vector<IoDeclaration>> value) { this->inputs = value; }

        /**
         * Discriminator for the job payload. Absent means a legacy shader/inference job governed by
         * passes/inputs/outputs. v1.0 implements elm_processing only; unknown strings reject at
         * parse via the generated enum chain
         */
        boost::optional<JobType> get_job_type() const { return job_type; }
        void set_job_type(boost::optional<JobType> value) { this->job_type = value; }

        /**
         * Additional metadata for the processing definition
         */
        boost::optional<std::map<std::string, nlohmann::json>> get_metadata() const { return metadata; }
        void set_metadata(boost::optional<std::map<std::string, nlohmann::json>> value) { this->metadata = value; }

        /**
         * Unique name for this processing definition
         */
        const std::string & get_name() const { return name; }
        std::string & get_mutable_name() { return name; }
        void set_name(const std::string & value) { CheckConstraint("name", name_constraint, value); this->name = value; }

        /**
         * Declares the final outputs this process will produce
         */
        boost::optional<std::vector<IoDeclaration>> get_outputs() const { return outputs; }
        void set_outputs(boost::optional<std::vector<IoDeclaration>> value) { this->outputs = value; }

        /**
         * Overridable parameters with defaults
         */
        boost::optional<std::vector<Parameter>> get_parameters() const { return parameters; }
        void set_parameters(boost::optional<std::vector<Parameter>> value) { this->parameters = value; }

        /**
         * Array of processing passes to execute
         */
        boost::optional<std::vector<Pass>> get_passes() const { return passes; }
        void set_passes(boost::optional<std::vector<Pass>> value) { this->passes = value; }

        /**
         * Tags for categorizing this definition
         */
        boost::optional<std::vector<std::string>> get_tags() const { return tags; }
        void set_tags(boost::optional<std::vector<std::string>> value) { this->tags = value; }

        /**
         * Work-item result validation mode. v1.0 implements none only; exact/redundant parse but
         * are refused by the ProcessingManager C++ gate as unimplemented
         */
        boost::optional<Validation> get_validation() const { return validation; }
        void set_validation(boost::optional<Validation> value) { this->validation = value; }

        /**
         * Version of this processing definition
         */
        const std::string & get_version() const { return version; }
        std::string & get_mutable_version() { return version; }
        void set_version(const std::string & value) { CheckConstraint("version", version_constraint, value); this->version = value; }
    };
}
