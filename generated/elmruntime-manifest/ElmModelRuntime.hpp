//  To parse this JSON data, first install
//
//      Boost     http://www.boost.org
//      json.hpp  https://github.com/nlohmann/json
//
//  Then include this file, and then do
//
//     ElmModelRuntime.hpp data = nlohmann::json::parse(jsonString);

#pragma once

#include <boost/optional.hpp>
#include <nlohmann/json.hpp>
#include "helper.hpp"

namespace sgns {
    /**
     * Runtime resource requirements (local capability preflight only)
     *
     * Runtime resource block of an ELM model manifest. Feeds the local CapabilityValidator
     * preflight (plan 02-02); never advertised on the network
     */

    using nlohmann::json;

    /**
     * Runtime resource requirements (local capability preflight only)
     *
     * Runtime resource block of an ELM model manifest. Feeds the local CapabilityValidator
     * preflight (plan 02-02); never advertised on the network
     */
    class ElmModelRuntime {
        public:
        ElmModelRuntime() :
            required_memory_bytes_constraint(boost::none, boost::none, boost::none, boost::none, boost::none, boost::none, boost::none)
        {}
        virtual ~ElmModelRuntime() = default;

        private:
        boost::optional<int64_t> required_memory_bytes;
        ClassMemberConstraints required_memory_bytes_constraint;

        public:
        /**
         * Host RAM the loaded model requires; 0/absent = no requirement (the number bound does not
         * survive codegen on optionals -- the C++ gate re-checks >= 0)
         */
        boost::optional<int64_t> get_required_memory_bytes() const { return required_memory_bytes; }
        void set_required_memory_bytes(boost::optional<int64_t> value) { if (value) CheckConstraint("required_memory_bytes", required_memory_bytes_constraint, *value); this->required_memory_bytes = value; }
    };
}
