//  To parse this JSON data, first install
//
//      Boost     http://www.boost.org
//      json.hpp  https://github.com/nlohmann/json
//
//  Then include this file, and then do
//
//     ElmGeneration.hpp data = nlohmann::json::parse(jsonString);

#pragma once

#include <boost/optional.hpp>
#include <nlohmann/json.hpp>
#include "helper.hpp"

namespace sgns {
    /**
     * Generation settings; defaults documented per-field are applied by ProcessingManager C++
     * normalization (quicktype never applies JSON Schema defaults)
     *
     * Generation settings for an ELM work item. Defaults (max_output_tokens unset = model
     * default, temperature 1.0, top_p 1.0, seed unset) are applied by ProcessingManager C++
     * normalization, not by codegen
     */

    using nlohmann::json;

    /**
     * Generation settings; defaults documented per-field are applied by ProcessingManager C++
     * normalization (quicktype never applies JSON Schema defaults)
     *
     * Generation settings for an ELM work item. Defaults (max_output_tokens unset = model
     * default, temperature 1.0, top_p 1.0, seed unset) are applied by ProcessingManager C++
     * normalization, not by codegen
     */
    class ElmGeneration {
        public:
        ElmGeneration() :
            max_output_tokens_constraint(boost::none, boost::none, boost::none, boost::none, boost::none, boost::none, boost::none),
            seed_constraint(boost::none, boost::none, boost::none, boost::none, boost::none, boost::none, boost::none),
            temperature_constraint(boost::none, boost::none, boost::none, boost::none, boost::none, boost::none, boost::none),
            top_p_constraint(boost::none, boost::none, boost::none, boost::none, boost::none, boost::none, boost::none)
        {}
        virtual ~ElmGeneration() = default;

        private:
        boost::optional<int64_t> max_output_tokens;
        ClassMemberConstraints max_output_tokens_constraint;
        boost::optional<int64_t> seed;
        ClassMemberConstraints seed_constraint;
        boost::optional<double> temperature;
        ClassMemberConstraints temperature_constraint;
        boost::optional<double> top_p;
        ClassMemberConstraints top_p_constraint;

        public:
        /**
         * Maximum tokens to generate (minimum 1)
         */
        boost::optional<int64_t> get_max_output_tokens() const { return max_output_tokens; }
        void set_max_output_tokens(boost::optional<int64_t> value) { if (value) CheckConstraint("max_output_tokens", max_output_tokens_constraint, *value); this->max_output_tokens = value; }

        /**
         * Sampling seed for deterministic generation
         */
        boost::optional<int64_t> get_seed() const { return seed; }
        void set_seed(boost::optional<int64_t> value) { if (value) CheckConstraint("seed", seed_constraint, *value); this->seed = value; }

        /**
         * Sampling temperature, default 1.0
         */
        boost::optional<double> get_temperature() const { return temperature; }
        void set_temperature(boost::optional<double> value) { if (value) CheckConstraint("temperature", temperature_constraint, *value); this->temperature = value; }

        /**
         * Nucleus sampling threshold, default 1.0. The >0 exclusive half is enforced by the
         * ProcessingManager C++ gate (exclusiveMinimum does not survive quicktype codegen)
         */
        boost::optional<double> get_top_p() const { return top_p; }
        void set_top_p(boost::optional<double> value) { if (value) CheckConstraint("top_p", top_p_constraint, *value); this->top_p = value; }
    };
}
