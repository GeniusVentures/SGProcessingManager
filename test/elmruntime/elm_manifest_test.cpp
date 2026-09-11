// ELM model manifest verification-matrix tests (Plan 02-01, Task 3).
//
// Locks the MCHE-01 / SC-1 front door: manifest bytes are sha256-verified
// against the declared model_manifest_hash BEFORE any parse, then pass the
// semantic gates (mnn-only format, closed role set, unique + 4 required
// roles, DoS ceilings) -- every bad path returns a structured ElmRuntimeError.
// The production-fetch leg exercises MakeFileManagerFetchFn offline through
// file:// fixtures only (D-08: no network, no IPFS).
//
// Pure in-memory JSON + temp files. Every case drives
// sgns::elmruntime::ParseAndVerifyManifest or the FetchFn seam directly.

#include <gtest/gtest.h>

#include <elmruntime/ElmArtifactFetcher.hpp>
#include <elmruntime/ElmManifest.hpp>

#include <cstdint>
#include <cctype>
#include <algorithm>
#include <cstdio>
#include <filesystem>
#include <fstream>
#include <sstream>
#include <string>
#include <vector>

namespace {

    // Bytes of an ASCII string (manifest JSON is UTF-8/ASCII).
    std::vector<uint8_t> ToBytes( const std::string &s )
    {
        return std::vector<uint8_t>( s.begin(), s.end() );
    }

    // 64 lowercase-hex of the sha256 of `payload` (reuse the digester under test).
    std::string Sha256Hex( const std::string &payload )
    {
        return sgns::elmruntime::ComputeManifestHexDigest( ToBytes( payload ) );
    }

    // A single artifact entry; sha256 is computed over a per-role pseudo
    // payload so every field is schema-valid by construction.
    std::string ArtifactJson( const std::string &role, int64_t sizeBytes = 100, const std::string &uri = "ipfs://artifact" )
    {
        std::ostringstream oss;
        oss << "{\"name\": \"" << role << "\", "
            << "\"uri\": \"" << uri << "\", "
            << "\"sha256\": \"" << Sha256Hex( role + "-bytes" ) << "\", "
            << "\"size_bytes\": " << sizeBytes << "}";
        return oss.str();
    }

    // A fully-valid manifest (all four required roles) with computed hash
    // fields. `artifactsJson` replaces the default artifact array when
    // non-empty; `extras` appends top-level fields (e.g. quantization).
    std::string BuildManifestJson( const std::string &artifactsJson = "", const std::string &extras = "" )
    {
        const std::string defaultArtifacts = "[" + ArtifactJson( "llm_config" ) + ", "    //
                                             + ArtifactJson( "llm_model" ) + ", "        //
                                             + ArtifactJson( "llm_weight" ) + ", "       //
                                             + ArtifactJson( "tokenizer_file" ) + "]";
        const std::string &artifacts = artifactsJson.empty() ? defaultArtifacts : artifactsJson;
        return "{"                                            //
               "\"schema_version\": 1,"                       //
               "\"elm_type\": \"causal_lm\","                 //
               "\"model_format\": \"mnn\","                   //
               "\"artifacts\": " + artifacts + extras + "}";
    }

    using sgns::elmruntime::ElmRuntimeError;

    // Run ParseAndVerifyManifest and return the error code (fails the test
    // with a readable message if the manifest unexpectedly verifies).
    ElmRuntimeError ExpectVerifyFailure( const std::string &manifestJson, const std::string &declaredHash )
    {
        auto result = sgns::elmruntime::ParseAndVerifyManifest( ToBytes( manifestJson ), declaredHash );
        if ( result )
        {
            ADD_FAILURE() << "Expected ParseAndVerifyManifest to reject the manifest, but it succeeded";
            return ElmRuntimeError::MANIFEST_INVALID; // unreachable in practice
        }
        return static_cast<ElmRuntimeError>( result.error().value() );
    }

    // Uppercase a hex string (F5 normalization coverage).
    std::string ToUpper( const std::string &s )
    {
        std::string out = s;
        for ( auto &c : out )
        {
            c = static_cast<char>( ::toupper( static_cast<unsigned char>( c ) ) );
        }
        return out;
    }

} // namespace

// ---------------------------------------------------------------------------
// Valid manifests parse (SC-1 happy path)
// ---------------------------------------------------------------------------

TEST( ElmManifestTest, ValidManifestParsesAndTotalsBytes )
{
    const std::string json = BuildManifestJson();
    auto result = sgns::elmruntime::ParseAndVerifyManifest( ToBytes( json ), Sha256Hex( json ) );
    ASSERT_TRUE( result ) << "A fully-valid manifest must verify: " << result.error().message();
    EXPECT_EQ( result.value().get_model_format(), "mnn" );
    EXPECT_EQ( result.value().get_artifacts().size(), 4u );
    // 4 artifacts x 100 bytes each = 400.
    EXPECT_EQ( sgns::elmruntime::TotalArtifactBytes( result.value() ), 400u );
}

TEST( ElmManifestTest, Sha256PrefixAndUppercaseHashNormalizes )
{
    // F5: "sha256:" prefix + UPPERCASE hex both normalize to the same digest.
    const std::string json = BuildManifestJson();
    const std::string declared = "sha256:" + ToUpper( Sha256Hex( json ) );
    auto result = sgns::elmruntime::ParseAndVerifyManifest( ToBytes( json ), declared );
    ASSERT_TRUE( result ) << "sha256:-prefixed uppercase declared hash must normalize: " << result.error().message();
}

TEST( ElmManifestTest, ContextFileExtraRoleIsFine )
{
    const std::string artifacts = "["                                                        //
                                  + ArtifactJson( "llm_config" ) + ", "                     //
                                  + ArtifactJson( "llm_model" ) + ", "                      //
                                  + ArtifactJson( "llm_weight" ) + ", "                     //
                                  + ArtifactJson( "tokenizer_file" ) + ", "                 //
                                  + ArtifactJson( "context_file" ) + "]";
    const std::string json = BuildManifestJson( artifacts );
    auto result = sgns::elmruntime::ParseAndVerifyManifest( ToBytes( json ), Sha256Hex( json ) );
    ASSERT_TRUE( result );
    EXPECT_EQ( result.value().get_artifacts().size(), 5u );
    EXPECT_EQ( sgns::elmruntime::TotalArtifactBytes( result.value() ), 500u );
}

TEST( ElmManifestTest, QuantizationAndRuntimeOptionalBlocksParse )
{
    const std::string json = BuildManifestJson(
        "", R"(, "quantization": "q4_0", "runtime": { "required_memory_bytes": 4096 })" );
    auto result = sgns::elmruntime::ParseAndVerifyManifest( ToBytes( json ), Sha256Hex( json ) );
    ASSERT_TRUE( result );
    // Optional accessors return boost::optional BY VALUE -- materialize first.
    const auto quantization = result.value().get_quantization();
    ASSERT_TRUE( quantization.is_initialized() );
    EXPECT_EQ( *quantization, "q4_0" );
    const auto runtime = result.value().get_runtime();
    ASSERT_TRUE( runtime.is_initialized() );
    EXPECT_EQ( runtime->get_required_memory_bytes().get_value_or( 0 ), 4096 );
}

// ---------------------------------------------------------------------------
// Hash gates: verify happens BEFORE parse (SC-1 / T-02-01-01)
// ---------------------------------------------------------------------------

TEST( ElmManifestTest, OneFlippedByteFailsAsHashMismatchNotParse )
{
    // The manifest JSON is ALSO malformed after the flip (a missing brace at
    // the end): if the implementation parsed before verifying, this would
    // surface as a parse error instead of MANIFEST_HASH_MISMATCH. The
    // mismatch code proves untrusted bytes never reached the parser.
    std::string json = BuildManifestJson().substr( 0, 20 ); // truncated -> unparseable
    const std::string declaredOfDifferentBytes = Sha256Hex( BuildManifestJson() );
    EXPECT_EQ( ExpectVerifyFailure( json, declaredOfDifferentBytes ), ElmRuntimeError::MANIFEST_HASH_MISMATCH );
}

TEST( ElmManifestTest, DeclaredHashNotHex64Rejects )
{
    const std::string json = BuildManifestJson();
    // "sha256:abc" -- the Phase 1 fixture value: not 64 hex chars.
    EXPECT_EQ( ExpectVerifyFailure( json, "sha256:abc" ), ElmRuntimeError::MANIFEST_INVALID );
}

TEST( ElmManifestTest, DeclaredHashWithBadCharsetRejects )
{
    const std::string json = BuildManifestJson();
    // 64 chars but not hex.
    EXPECT_EQ( ExpectVerifyFailure( json, std::string( 64, 'z' ) ), ElmRuntimeError::MANIFEST_INVALID );
}

TEST( ElmManifestTest, EmptyDeclaredHashRejects )
{
    const std::string json = BuildManifestJson();
    EXPECT_EQ( ExpectVerifyFailure( json, "" ), ElmRuntimeError::MANIFEST_INVALID );
}

// ---------------------------------------------------------------------------
// Semantic gates (T-02-01-02 / T-02-01-03)
// ---------------------------------------------------------------------------

TEST( ElmManifestTest, ModelFormatOnnxRejects )
{
    // Hand-built: model_format "onnx" -- the schema pattern ^mnn$ does not
    // survive codegen, so this gate is owned by the C++ check.
    const std::string artifacts = "["                                                    //
                                  + ArtifactJson( "llm_config" ) + ", "                 //
                                  + ArtifactJson( "llm_model" ) + ", "                  //
                                  + ArtifactJson( "llm_weight" ) + ", "                 //
                                  + ArtifactJson( "tokenizer_file" ) + "]";
    const std::string json = "{"                            //
                             "\"schema_version\": 1,"        //
                             "\"elm_type\": \"causal_lm\","  //
                             "\"model_format\": \"onnx\","   //
                             "\"artifacts\": " + artifacts + "}";
    EXPECT_EQ( ExpectVerifyFailure( json, Sha256Hex( json ) ), ElmRuntimeError::MANIFEST_INVALID );
}

TEST( ElmManifestTest, EmptyArtifactsArrayRejects )
{
    EXPECT_EQ( ExpectVerifyFailure( BuildManifestJson( "[]" ), Sha256Hex( BuildManifestJson( "[]" ) ) ),
               ElmRuntimeError::MANIFEST_INVALID );
}

TEST( ElmManifestTest, OverCeilingArtifactCountRejects )
{
    // 33 artifacts (ceiling is 32): four required roles + 29 context extras.
    std::string artifacts = "["                                                              //
        + ArtifactJson( "llm_config" ) + ", "                                                //
        + ArtifactJson( "llm_model" ) + ", "                                                 //
        + ArtifactJson( "llm_weight" ) + ", "                                                //
        + ArtifactJson( "tokenizer_file" );
    for ( int i = 0; i < 29; ++i )
    {
        artifacts += ", " + ArtifactJson( "context_file" );
    }
    artifacts += "]";
    const std::string json = BuildManifestJson( artifacts );
    EXPECT_EQ( ExpectVerifyFailure( json, Sha256Hex( json ) ), ElmRuntimeError::MANIFEST_INVALID );
}

TEST( ElmManifestTest, DuplicateRoleRejects )
{
    const std::string artifacts = "["                                                 //
                                  + ArtifactJson( "llm_config" ) + ", "              //
                                  + ArtifactJson( "llm_model" ) + ", "               //
                                  + ArtifactJson( "llm_model" ) + ", "               // duplicate
                                  + ArtifactJson( "llm_weight" ) + ", "              //
                                  + ArtifactJson( "tokenizer_file" ) + "]";
    const std::string json = BuildManifestJson( artifacts );
    EXPECT_EQ( ExpectVerifyFailure( json, Sha256Hex( json ) ), ElmRuntimeError::MANIFEST_INVALID );
}

TEST( ElmManifestTest, MissingRequiredRoleRejects )
{
    // Drop llm_weight: MNN's Llm::load() needs it unconditionally.
    const std::string artifacts = "["                                              //
                                  + ArtifactJson( "llm_config" ) + ", "           //
                                  + ArtifactJson( "llm_model" ) + ", "            //
                                  + ArtifactJson( "tokenizer_file" ) + "]";
    const std::string json = BuildManifestJson( artifacts );
    EXPECT_EQ( ExpectVerifyFailure( json, Sha256Hex( json ) ), ElmRuntimeError::MANIFEST_INVALID );
}

TEST( ElmManifestTest, UnknownRoleNameRejects )
{
    // "llm.mnn" is a FILENAME, not a role -- the closed-role gate must reject
    // it (and never confuse filenames with roles).
    const std::string artifacts = "["                                              //
                                  + ArtifactJson( "llm_config" ) + ", "           //
                                  + ArtifactJson( "llm.mnn" ) + ", "              //
                                  + ArtifactJson( "llm_weight" ) + ", "           //
                                  + ArtifactJson( "tokenizer_file" ) + "]";
    const std::string json = BuildManifestJson( artifacts );
    // The generated pattern constraint may reject first (as a parse/constraint
    // exception mapped to MANIFEST_INVALID); either way the code is the same.
    EXPECT_EQ( ExpectVerifyFailure( json, Sha256Hex( json ) ), ElmRuntimeError::MANIFEST_INVALID );
}

TEST( ElmManifestTest, ArtifactSha256NotHex64Rejects )
{
    // Hand-built artifact with a non-hex64 sha256: the generated constraint
    // throws at parse; the gate maps it to MANIFEST_INVALID.
    const std::string badArtifact = "{\"name\": \"llm_config\", \"uri\": \"ipfs://a\", "
                                    "\"sha256\": \"nothex\", \"size_bytes\": 10}";
    const std::string artifacts = "[" + badArtifact + ", "     //
                                  + ArtifactJson( "llm_model" ) + ", "   //
                                  + ArtifactJson( "llm_weight" ) + ", "  //
                                  + ArtifactJson( "tokenizer_file" ) + "]";
    const std::string json = BuildManifestJson( artifacts );
    EXPECT_EQ( ExpectVerifyFailure( json, Sha256Hex( json ) ), ElmRuntimeError::MANIFEST_INVALID );
}

TEST( ElmManifestTest, NegativeSizeBytesRejects )
{
    const std::string artifacts = "["                                              //
                                  + ArtifactJson( "llm_config", -1 ) + ", "        // negative
                                  + ArtifactJson( "llm_model" ) + ", "             //
                                  + ArtifactJson( "llm_weight" ) + ", "            //
                                  + ArtifactJson( "tokenizer_file" ) + "]";
    const std::string json = BuildManifestJson( artifacts );
    // The C++ gate owns this: the schema minimum-0 did not survive codegen.
    EXPECT_EQ( ExpectVerifyFailure( json, Sha256Hex( json ) ), ElmRuntimeError::MANIFEST_INVALID );
}

TEST( ElmManifestTest, EmptyUriRejects )
{
    const std::string artifacts = "["                                                    //
                                  + ArtifactJson( "llm_config", 100, "" ) + ", "         // empty uri
                                  + ArtifactJson( "llm_model" ) + ", "                   //
                                  + ArtifactJson( "llm_weight" ) + ", "                  //
                                  + ArtifactJson( "tokenizer_file" ) + "]";
    const std::string json = BuildManifestJson( artifacts );
    EXPECT_EQ( ExpectVerifyFailure( json, Sha256Hex( json ) ), ElmRuntimeError::MANIFEST_INVALID );
}

TEST( ElmManifestTest, OversizedManifestBytesReject )
{
    // > 1 MiB of padded bytes: rejects at the size ceiling before any
    // normalization or parsing (T-02-01-03).
    const std::string declared = Sha256Hex( "anything" );
    std::string padded( sgns::elmruntime::kMaxManifestBytes + 1, ' ' );
    auto result = sgns::elmruntime::ParseAndVerifyManifest( ToBytes( padded ), declared );
    ASSERT_FALSE( result );
    EXPECT_EQ( static_cast<ElmRuntimeError>( result.error().value() ), ElmRuntimeError::MANIFEST_INVALID );
}

TEST( ElmManifestTest, MalformedJsonRejectsAfterHashVerifies )
{
    // Hash-verified garbage: parse failure maps to MANIFEST_INVALID (the
    // hash passed, so this exercises gate (d), not (c)).
    const std::string json = "not json at all";
    EXPECT_EQ( ExpectVerifyFailure( json, Sha256Hex( json ) ), ElmRuntimeError::MANIFEST_INVALID );
}

// ---------------------------------------------------------------------------
// RoleFileName mapping table (T-02-01-02: names are roles, never paths)
// ---------------------------------------------------------------------------

TEST( ElmManifestTest, RoleFileNameMapsAllFiveRoles )
{
    using sgns::elmruntime::RoleFileName;
    EXPECT_STREQ( RoleFileName( "llm_config" ), "llm_config.json" );
    EXPECT_STREQ( RoleFileName( "llm_model" ), "llm.mnn" );
    EXPECT_STREQ( RoleFileName( "llm_weight" ), "llm.mnn.weight" );
    EXPECT_STREQ( RoleFileName( "tokenizer_file" ), "tokenizer.txt" );
    EXPECT_STREQ( RoleFileName( "context_file" ), "context.json" );
    EXPECT_EQ( RoleFileName( "llm.mnn" ), nullptr ); // filenames are not roles
    EXPECT_EQ( RoleFileName( "../evil" ), nullptr ); // never a path
}

// ---------------------------------------------------------------------------
// ComputeManifestHexDigest shape (P2-4 cache dir-name source)
// ---------------------------------------------------------------------------

TEST( ElmManifestTest, DigestIs64LowercaseHex )
{
    const std::string digest = sgns::elmruntime::ComputeManifestHexDigest( ToBytes( "abc" ) );
    EXPECT_EQ( digest.size(), 64u );
    for ( char c : digest )
    {
        EXPECT_TRUE( ( c >= '0' && c <= '9' ) || ( c >= 'a' && c <= 'f' ) ) << "digest: " << digest;
    }
    // Known sha256("abc").
    EXPECT_EQ( digest, "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad" );
}

// ---------------------------------------------------------------------------
// Production FetchFn, offline (D-08: file:// only, no network / IPFS)
// ---------------------------------------------------------------------------

TEST( ElmManifestFetcherTest, FileUriRoundTripsBytes )
{
    // Write a small payload to a temp file, fetch via the production seam.
    // Unique-ify the name with the case's own address so parallel/renamed
    // runs never collide.
    const std::string payload = "elm-model-payload-0123456789";
    const auto tempPath = ( std::filesystem::temp_directory_path()
                            / ( "elmruntime_test_"
                                + std::to_string( reinterpret_cast<uintptr_t>( &payload ) )
                                + ".bin" ) )
                              .string();
    {
        std::ofstream out( tempPath, std::ios::binary );
        out.write( payload.data(), payload.size() );
    }

    auto fetch = sgns::elmruntime::MakeFileManagerFetchFn();
    auto result = fetch( std::string( "file://" ) + tempPath );
    ASSERT_TRUE( result ) << "file:// fetch must succeed: " << result.error().message();
    const std::vector<uint8_t> &bytes = result.value();
    EXPECT_EQ( bytes.size(), payload.size() );
    EXPECT_TRUE( std::equal( bytes.begin(), bytes.end(), payload.begin() ) );

    std::error_code ec;
    std::filesystem::remove( tempPath, ec ); // best-effort cleanup
}

TEST( ElmManifestFetcherTest, NonexistentFileFailsWithFetchFailed )
{
    auto fetch = sgns::elmruntime::MakeFileManagerFetchFn();
    auto result = fetch( "file://Z:/definitely/not/a/real/path/xyz.bin" );
    ASSERT_FALSE( result );
    EXPECT_EQ( static_cast<ElmRuntimeError>( result.error().value() ), ElmRuntimeError::FETCH_FAILED );
}

TEST( ElmManifestFetcherTest, UnregisteredPrefixConvertsThrowToFetchFailed )
{
    // LoadASync throws std::range_error for unknown prefixes; the seam must
    // convert it to FETCH_FAILED, never let the exception cross the boundary.
    auto fetch = sgns::elmruntime::MakeFileManagerFetchFn();
    auto result = fetch( "bogus://x" );
    ASSERT_FALSE( result );
    EXPECT_EQ( static_cast<ElmRuntimeError>( result.error().value() ), ElmRuntimeError::FETCH_FAILED );
}
