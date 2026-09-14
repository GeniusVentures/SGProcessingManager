//  To parse this JSON data, first install
//
//      Boost     http://www.boost.org
//      json.hpp  https://github.com/nlohmann/json
//
//  Then include this file, and then do
//
//     ElmModelArtifact.hpp data = nlohmann::json::parse(jsonString);

#pragma once

#include <boost/optional.hpp>
#include <nlohmann/json.hpp>
#include "helper.hpp"

namespace sgns {
    /**
     * One artifact of an ELM model bundle. The name is a ROLE from a closed set (never a
     * filesystem path); the runtime materializes each role at MNN's default filename via
     * ElmRuntime RoleFileName
     */

    using nlohmann::json;

    /**
     * One artifact of an ELM model bundle. The name is a ROLE from a closed set (never a
     * filesystem path); the runtime materializes each role at MNN's default filename via
     * ElmRuntime RoleFileName
     */
    class ElmModelArtifact {
        public:
        ElmModelArtifact() :
            name_constraint(boost::none, boost::none, boost::none, boost::none, boost::none, boost::none, std::string("^(llm_config|llm_model|llm_weight|tokenizer_file|context_file|embedding_file)$")),
            sha256_constraint(boost::none, boost::none, boost::none, boost::none, boost::none, boost::none, std::string("^[0-9a-fA-F]{64}$")),
            size_bytes_constraint(boost::none, boost::none, boost::none, boost::none, boost::none, boost::none, boost::none),
            uri_constraint(boost::none, boost::none, boost::none, boost::none, 1, boost::none, boost::none)
        {}
        virtual ~ElmModelArtifact() = default;

        private:
        std::string name;
        ClassMemberConstraints name_constraint;
        std::string sha256;
        ClassMemberConstraints sha256_constraint;
        int64_t size_bytes;
        ClassMemberConstraints size_bytes_constraint;
        std::string uri;
        ClassMemberConstraints uri_constraint;

        public:
        /**
         * Artifact role: llm_config, llm_model, llm_weight, tokenizer_file, context_file, or
         * embedding_file
         */
        const std::string & get_name() const { return name; }
        std::string & get_mutable_name() { return name; }
        void set_name(const std::string & value) { CheckConstraint("name", name_constraint, value); this->name = value; }

        /**
         * sha256 of the artifact bytes, 64 hex characters
         */
        const std::string & get_sha256() const { return sha256; }
        std::string & get_mutable_sha256() { return sha256; }
        void set_sha256(const std::string & value) { CheckConstraint("sha256", sha256_constraint, value); this->sha256 = value; }

        /**
         * Exact artifact size in bytes; drives cache LRU accounting and the disk preflight
         */
        const int64_t & get_size_bytes() const { return size_bytes; }
        int64_t & get_mutable_size_bytes() { return size_bytes; }
        void set_size_bytes(const int64_t & value) { CheckConstraint("size_bytes", size_bytes_constraint, value); this->size_bytes = value; }

        /**
         * URI the artifact is fetched from (production: FileManager loaders only)
         */
        const std::string & get_uri() const { return uri; }
        std::string & get_mutable_uri() { return uri; }
        void set_uri(const std::string & value) { CheckConstraint("uri", uri_constraint, value); this->uri = value; }
    };
}
