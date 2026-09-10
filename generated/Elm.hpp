//  To parse this JSON data, first install
//
//      Boost     http://www.boost.org
//      json.hpp  https://github.com/nlohmann/json
//
//  Then include this file, and then do
//
//     Elm.hpp data = nlohmann::json::parse(jsonString);

#pragma once

#include <boost/optional.hpp>
#include <nlohmann/json.hpp>
#include "helper.hpp"

#include "ElmGeneration.hpp"

namespace sgns {
    enum class ElmType : int;
}

namespace sgns {
    /**
     * One ELM (causal-LM) work item. work_item_id charset is constrained now so later phases
     * never interpolate arbitrary strings into filesystem paths
     */

    using nlohmann::json;

    /**
     * One ELM (causal-LM) work item. work_item_id charset is constrained now so later phases
     * never interpolate arbitrary strings into filesystem paths
     */
    class Elm {
        public:
        Elm() :
            input_uri_constraint(boost::none, boost::none, boost::none, boost::none, 1, boost::none, boost::none),
            model_manifest_hash_constraint(boost::none, boost::none, boost::none, boost::none, 1, boost::none, boost::none),
            model_manifest_uri_constraint(boost::none, boost::none, boost::none, boost::none, 1, boost::none, boost::none),
            work_item_id_constraint(boost::none, boost::none, boost::none, boost::none, boost::none, boost::none, std::string("^[A-Za-z0-9_-]+$"))
        {}
        virtual ~Elm() = default;

        private:
        ElmType elm_type;
        boost::optional<ElmGeneration> generation;
        std::string input_uri;
        ClassMemberConstraints input_uri_constraint;
        std::string model_manifest_hash;
        ClassMemberConstraints model_manifest_hash_constraint;
        std::string model_manifest_uri;
        ClassMemberConstraints model_manifest_uri_constraint;
        std::string work_item_id;
        ClassMemberConstraints work_item_id_constraint;

        public:
        /**
         * v1.0 implements causal_lm only
         */
        const ElmType & get_elm_type() const { return elm_type; }
        ElmType & get_mutable_elm_type() { return elm_type; }
        void set_elm_type(const ElmType & value) { this->elm_type = value; }

        /**
         * Generation settings; defaults documented per-field are applied by ProcessingManager C++
         * normalization (quicktype never applies JSON Schema defaults)
         */
        boost::optional<ElmGeneration> get_generation() const { return generation; }
        void set_generation(boost::optional<ElmGeneration> value) { this->generation = value; }

        /**
         * URI of the input data for this work item
         */
        const std::string & get_input_uri() const { return input_uri; }
        std::string & get_mutable_input_uri() { return input_uri; }
        void set_input_uri(const std::string & value) { CheckConstraint("input_uri", input_uri_constraint, value); this->input_uri = value; }

        /**
         * Expected hash of the model manifest content
         */
        const std::string & get_model_manifest_hash() const { return model_manifest_hash; }
        std::string & get_mutable_model_manifest_hash() { return model_manifest_hash; }
        void set_model_manifest_hash(const std::string & value) { CheckConstraint("model_manifest_hash", model_manifest_hash_constraint, value); this->model_manifest_hash = value; }

        /**
         * URI of the model manifest for this work item
         */
        const std::string & get_model_manifest_uri() const { return model_manifest_uri; }
        std::string & get_mutable_model_manifest_uri() { return model_manifest_uri; }
        void set_model_manifest_uri(const std::string & value) { CheckConstraint("model_manifest_uri", model_manifest_uri_constraint, value); this->model_manifest_uri = value; }

        /**
         * Unique identifier for this work item within the job (same charset as root name)
         */
        const std::string & get_work_item_id() const { return work_item_id; }
        std::string & get_mutable_work_item_id() { return work_item_id; }
        void set_work_item_id(const std::string & value) { CheckConstraint("work_item_id", work_item_id_constraint, value); this->work_item_id = value; }
    };
}
