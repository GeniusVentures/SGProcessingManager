//  To parse this JSON data, first install
//
//      Boost     http://www.boost.org
//      json.hpp  https://github.com/nlohmann/json
//
//  Then include this file, and then do
//
//     ElmFunding.hpp data = nlohmann::json::parse(jsonString);

#pragma once

#include <boost/optional.hpp>
#include <nlohmann/json.hpp>
#include "helper.hpp"

namespace sgns {
    /**
     * Funding envelope for elm_processing jobs (maximum processing hours)
     *
     * Funding envelope for an elm_processing job. maximum_processing_hours defaults to 1.0 via
     * ProcessingManager C++ normalization; the >0 exclusive half of minimum is enforced by the
     * C++ gate
     */

    using nlohmann::json;

    /**
     * Funding envelope for elm_processing jobs (maximum processing hours)
     *
     * Funding envelope for an elm_processing job. maximum_processing_hours defaults to 1.0 via
     * ProcessingManager C++ normalization; the >0 exclusive half of minimum is enforced by the
     * C++ gate
     */
    class ElmFunding {
        public:
        ElmFunding() :
            maximum_processing_hours_constraint(boost::none, boost::none, boost::none, boost::none, boost::none, boost::none, boost::none)
        {}
        virtual ~ElmFunding() = default;

        private:
        boost::optional<double> maximum_processing_hours;
        ClassMemberConstraints maximum_processing_hours_constraint;

        public:
        /**
         * Wall-clock processing cap in hours, default 1.0, hard cap 24
         */
        boost::optional<double> get_maximum_processing_hours() const { return maximum_processing_hours; }
        void set_maximum_processing_hours(boost::optional<double> value) { if (value) CheckConstraint("maximum_processing_hours", maximum_processing_hours_constraint, *value); this->maximum_processing_hours = value; }
    };
}
