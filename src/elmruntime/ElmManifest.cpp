#include <elmruntime/ElmManifest.hpp>

#include <util/sgprocmgr-logger.hpp>

#include <cctype>
#include <cstdio>
#include <nlohmann/json.hpp>
#include <set>
#include <string_view>

// CRITICAL lifetime note (copied in substance from ProcessingManager.cpp:636-641):
// the quicktype getters (get_runtime, get_quantization -- and in the root set,
// get_elms etc.) return boost::optional<T> BY VALUE. Dereferencing the call
// result directly (*m.get_runtime()) yields a reference into a temporary
// optional that dies at the end of the full expression -- using it is UB.
// Always materialize the optional into a named local first.
//
// The manifest types come from the A1-fallback generator set
// (generated/elmruntime-manifest/) -- SELF-CONTAINED: its headers include
// their own "helper.hpp"/"ElmType.hpp" siblings via quoted includes, so this
// must be a quoted include resolved against that directory ONLY (a
// global-include <ElmModelManifest.hpp> would work, but the quoted include
// keeps the subdir set from mixing with the root generated/ set in any TU
// that includes both -- their helper.hpp files redefine the same classes).
#include "elmruntime-manifest/Generators.hpp"

namespace sgns::elmruntime
{
    namespace
    {
        sgns::sgprocmanager::Logger ManifestLogger()
        {
            return sgns::sgprocmanager::createLogger( "ElmManifest" );
        }

        /// Closed artifact role set: the manifest `name` is a ROLE, materialized at
        /// MNN's default filename via RoleFileName -- never used as a filesystem path.
        /// (llmconfig.hpp:110-156 filenames; Llm::load() unconditionally checks
        /// llm_config.json, llm.mnn, llm.mnn.weight, and tokenizer.txt.)
        const char *const kRoleFilenames[] = {
            "llm_config", "llm_config.json",        //
            "llm_model", "llm.mnn",                 //
            "llm_weight", "llm.mnn.weight",         //
            "tokenizer_file", "tokenizer.txt",      //
            "context_file", "context.json",         //
            "embedding_file", "embeddings_bf16.bin" //
        };

        /// The roles MNN's Llm::load() requires unconditionally (llm.cpp:265-283);
        /// context_file is optional extra material.
        const char *const kRequiredRoles[] = { "llm_config", "llm_model", "llm_weight", "tokenizer_file" };

        bool IsValidRole( const std::string &role )
        {
            // kRoleFilenames interleaves role/filename pairs; ONLY the even
            // indexes are roles. Comparing against every entry would let a
            // FILENAME (e.g. "llm.mnn") pass the closed-role gate.
            for ( size_t i = 0; i < sizeof( kRoleFilenames ) / sizeof( kRoleFilenames[0] ); i += 2 )
            {
                if ( role == kRoleFilenames[i] )
                {
                    return true;
                }
            }
            return false;
        }

        bool IsHexChar( char c )
        {
            return ( c >= '0' && c <= '9' ) || ( c >= 'a' && c <= 'f' ) || ( c >= 'A' && c <= 'F' );
        }

        /// Normalize a declared hash: strip an optional "sha256:" prefix, require
        /// exactly 64 hex chars afterwards, and lowercase them. Returns false for
        /// anything else (the caller fails with MANIFEST_INVALID).
        bool NormalizeDeclaredHash( const std::string &declared, std::string &outLowerHex )
        {
            std::string_view view( declared );
            constexpr std::string_view kPrefix = "sha256:";
            if ( view.size() >= kPrefix.size() && view.substr( 0, kPrefix.size() ) == kPrefix )
            {
                view.remove_prefix( kPrefix.size() );
            }
            if ( view.size() != 64 )
            {
                return false;
            }
            outLowerHex.clear();
            outLowerHex.reserve( view.size() );
            for ( char c : view )
            {
                if ( !IsHexChar( c ) )
                {
                    return false;
                }
                outLowerHex.push_back( static_cast<char>( std::tolower( static_cast<unsigned char>( c ) ) ) );
            }
            return true;
        }
    } // namespace

    std::string ComputeManifestHexDigest( const std::vector<uint8_t> &bytes )
    {
        const auto digest = sgprocmanagersha::sha256( bytes.data(), bytes.size() );
        // %02x loop per DeriveExecutorId (capability_validator.cpp:168-178)
        std::string hex;
        hex.reserve( digest.size() * 2 );
        for ( uint8_t byte : digest )
        {
            char buf[3];
            std::snprintf( buf, sizeof( buf ), "%02x", byte );
            hex += buf;
        }
        return hex;
    }

    outcome::result<sgns::ElmModelManifest> ParseAndVerifyManifest( const std::vector<uint8_t> &bytes,
                                                                   const std::string         &declaredHash )
    {
        const auto logger = ManifestLogger();

        // (a) DoS ceiling before any parsing work.
        if ( bytes.size() > kMaxManifestBytes )
        {
            logger->error( "ElmManifest: manifest is {} bytes, over the {} byte ceiling", bytes.size(), kMaxManifestBytes );
            return outcome::failure( ElmRuntimeError::MANIFEST_INVALID );
        }

        // (b) Declared-hash normalization: optional "sha256:" prefix, then exactly 64 hex.
        std::string declaredLower;
        if ( !NormalizeDeclaredHash( declaredHash, declaredLower ) )
        {
            logger->error( "ElmManifest: declared model_manifest_hash is not sha256:-prefixed 64-hex" );
            return outcome::failure( ElmRuntimeError::MANIFEST_INVALID );
        }

        // (c) Hash-verify BEFORE parse (SC-1): untrusted bytes never reach the JSON
        // parser until pinned by the declared hash. There is no bypass for this.
        const std::string computed = ComputeManifestHexDigest( bytes );
        if ( computed != declaredLower )
        {
            logger->error( "ElmManifest: sha256 mismatch (computed {}, declared {})", computed, declaredLower );
            return outcome::failure( ElmRuntimeError::MANIFEST_HASH_MISMATCH );
        }

        // (d) Parse via the generated from_json (the A1-fallback generator set).
        sgns::ElmModelManifest manifest;
        try
        {
            const auto j = nlohmann::json::parse( bytes.begin(), bytes.end() );
            j.get_to( manifest );
        }
        catch ( const std::exception &e )
        {
            logger->error( "ElmManifest: manifest JSON parse/constraint failure: {}", e.what() );
            return outcome::failure( ElmRuntimeError::MANIFEST_INVALID );
        }

        // (e) Semantic gates, each log-then-fail in the CheckElmValidity idiom.
        // The schema pattern ^mnn$ does not survive codegen (model_format is a
        // plain string by design -- see schema comment); this gate is authoritative.
        if ( manifest.get_model_format() != "mnn" )
        {
            logger->error( "ElmManifest: model_format \"{}\" is not mnn (v1.0 mandates MNN)", manifest.get_model_format() );
            return outcome::failure( ElmRuntimeError::MANIFEST_INVALID );
        }

        // quicktype drops minItems: non-empty is ours to enforce.
        const auto &artifacts = manifest.get_artifacts();
        if ( artifacts.empty() )
        {
            logger->error( "ElmManifest: manifest declares no artifacts" );
            return outcome::failure( ElmRuntimeError::MANIFEST_INVALID );
        }
        if ( artifacts.size() > kMaxArtifacts )
        {
            logger->error( "ElmManifest: manifest declares {} artifacts, over the ceiling of {}", artifacts.size(), kMaxArtifacts );
            return outcome::failure( ElmRuntimeError::MANIFEST_INVALID );
        }

        // Unique roles + closed role set + per-artifact field gates.
        std::set<std::string> seenRoles;
        for ( const auto &artifact : artifacts )
        {
            const auto &role = artifact.get_name();
            if ( !seenRoles.insert( role ).second )
            {
                logger->error( "ElmManifest: duplicate artifact role: {}", role );
                return outcome::failure( ElmRuntimeError::MANIFEST_INVALID );
            }
            if ( !IsValidRole( role ) )
            {
                logger->error( "ElmManifest: unknown artifact role: {}", role );
                return outcome::failure( ElmRuntimeError::MANIFEST_INVALID );
            }
            if ( artifact.get_uri().empty() )
            {
                logger->error( "ElmManifest: artifact role {} has an empty uri", role );
                return outcome::failure( ElmRuntimeError::MANIFEST_INVALID );
            }
            // size_bytes is int64_t in the generated type: the schema minimum 0 does
            // not survive codegen, so negative values parse -- gate them here.
            if ( artifact.get_size_bytes() < 0 )
            {
                logger->error( "ElmManifest: artifact role {} has negative size_bytes {}", role, artifact.get_size_bytes() );
                return outcome::failure( ElmRuntimeError::MANIFEST_INVALID );
            }
        }

        // Required-role gate, fail-closed on ALL FOUR (research Rec. 5 / risk A2):
        // MNN's Llm::load() unconditionally checks tokenizer.txt, llm.mnn, AND
        // llm.mnn.weight -- requiring exactly these matches the engine's real
        // behavior; context_file is optional extra material.
        for ( const auto *required : kRequiredRoles )
        {
            if ( seenRoles.find( required ) == seenRoles.end() )
            {
                logger->error( "ElmManifest: required artifact role {} is missing", required );
                return outcome::failure( ElmRuntimeError::MANIFEST_INVALID );
            }
        }

        return outcome::success( std::move( manifest ) );
    }

    const char *RoleFileName( const std::string &role )
    {
        const size_t count = sizeof( kRoleFilenames ) / sizeof( kRoleFilenames[0] );
        for ( size_t i = 0; i < count; i += 2 )
        {
            if ( role == kRoleFilenames[i] )
            {
                return kRoleFilenames[i + 1];
            }
        }
        return nullptr;
    }

    uint64_t TotalArtifactBytes( const sgns::ElmModelManifest &manifest )
    {
        uint64_t total = 0;
        for ( const auto &artifact : manifest.get_artifacts() )
        {
            total += static_cast<uint64_t>( artifact.get_size_bytes() );
        }
        return total;
    }
} // namespace sgns::elmruntime
