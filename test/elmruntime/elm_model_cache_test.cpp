// ELM model cache lifecycle tests (Plan 02-03, Task 3).
//
// Locks MCHE-02/MCHE-03 (SC-2..SC-5): publish path materializes the MNN bundle
// layout under <root>/<manifest-digest>/ via stage->verify->smoke->rename; the
// hit path is size-first (zero re-fetch); tampered entries quarantine (.bad-<hex>)
// and refuse, retry re-downloads cleanly (SC-3); exactly-one download under
// concurrency with both threads pinning the same entry (SC-4); partial downloads
// recovered + LRU eviction under the byte cap (SC-5); restart rebuilds state from
// the directory scan (D-06). No network, no IPFS, no real model (Q2 resolution:
// failure-path assertions only; the gated garbage-bundle leg FAILS by design).
//
// Infrastructure: TempCacheRoot RAII fixture, an in-memory counting FetchFn
// serving real sha256 digests over tiny payloads, and a stub smoke lambda
// ([](dir){ return success; }) -- the real MakeMnnLlmSmokeCheck runs only in the
// SGPROC_HAS_MNN_LLM-gated garbage leg at the bottom.

#include <gtest/gtest.h>

#include <elmruntime/ElmManifest.hpp>
#include <elmruntime/ElmModelCache.hpp>
#include <elmruntime/ElmSmokeCheck.hpp>

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cstdio>
#include <filesystem>
#include <fstream>
#include <functional>
#include <future>
#include <map>
#include <random>
#include <set>
#include <sstream>
#include <string>
#include <thread>
#include <vector>

namespace fs = std::filesystem;

namespace
{
    using sgns::elmruntime::ElmCachePin;
    using sgns::elmruntime::ElmModelCache;
    using sgns::elmruntime::ElmRuntimeError;

    std::vector<uint8_t> ToBytes( const std::string &s )
    {
        return std::vector<uint8_t>( s.begin(), s.end() );
    }

    std::string Sha256Hex( const std::string &payload )
    {
        return sgns::elmruntime::ComputeManifestHexDigest( ToBytes( payload ) );
    }

    // ------------------------------------------------------------------
    // Fixture: RAII temp cache root (the TEST's own temp use; the cache itself
    // must never write outside the injected root -- TempDirCleanliness).
    // ------------------------------------------------------------------
    std::string NewTestUuidSuffix()
    {
        static std::mt19937 gen( static_cast<unsigned>( std::random_device{}() ) );
        std::uniform_int_distribution<int> dist( 0, 0xFFFF );
        std::ostringstream                oss;
        oss << std::hex << dist( gen ) << dist( gen ) << dist( gen ) << dist( gen );
        return oss.str();
    }

    class TempCacheRoot
    {
    public:
        TempCacheRoot()
            : path_( fs::temp_directory_path() / ( "sgproc_elm_cache_test_" + NewTestUuidSuffix() ) )
        {
            fs::create_directories( path_ );
        }

        ~TempCacheRoot()
        {
            std::error_code ec;
            fs::remove_all( path_, ec ); // best-effort
        }

        const std::string &Str() const
        {
            return str_;
        }

        const fs::path &Path() const
        {
            return path_;
        }

    private:
        fs::path   path_;
        std::string str_ = path_.string();
    };

    // ------------------------------------------------------------------
    // In-memory bundle: a manifest + artifact payloads with REAL sha256s.
    // ------------------------------------------------------------------
    struct TestBundle
    {
        std::string              manifestUri;
        std::string              manifestJson;
        std::string              declaredHash; // bare hex64 of the manifest bytes
        std::map<std::string, std::string> artifacts; // uri -> payload
    };

    // Builds a valid 4-artifact bundle (all required roles) whose artifact
    // payloads hash to their manifest sha256 fields by construction.
    TestBundle BuildBundle( const std::string &tag = "bundle" )
    {
        const std::vector<std::pair<std::string, std::string>> rolesAndPayloads = {
            { "llm_config", tag + "-config-json-payload-0123456789abcdef" },
            { "llm_model", tag + "-model-graph-payload-0123456789abcdef0123456789abcdef" },
            { "llm_weight", tag + "-weights-payload-0123456789abcdef0123456789abcdef0123" },
            { "tokenizer_file", tag + "-tokenizer-vocab-payload-0123456789" },
        };

        std::map<std::string, std::string> artifacts; // uri -> payload
        std::ostringstream                 artifactsJson;
        artifactsJson << "[";
        bool first = true;
        for ( const auto &[role, payload] : rolesAndPayloads )
        {
            const std::string uri = "mem://" + tag + "/" + role;
            artifacts[uri] = payload;
            if ( !first )
            {
                artifactsJson << ", ";
            }
            first = false;
            artifactsJson << "{\"name\": \"" << role << "\", "
                          << "\"uri\": \"" << uri << "\", "
                          << "\"sha256\": \"" << Sha256Hex( payload ) << "\", "
                          << "\"size_bytes\": " << payload.size() << "}";
        }
        artifactsJson << "]";

        std::ostringstream manifest;
        manifest << "{"                                             //
                 << "\"schema_version\": 1,"                        //
                 << "\"elm_type\": \"causal_lm\","                  //
                 << "\"model_format\": \"mnn\","                    //
                 << "\"artifacts\": " << artifactsJson.str() << "}";

        TestBundle bundle;
        bundle.manifestUri  = "mem://" + tag + "/manifest.json";
        bundle.manifestJson = manifest.str();
        bundle.declaredHash = Sha256Hex( bundle.manifestJson );
        bundle.artifacts    = artifacts; // 4 artifact URIs + ...
        bundle.artifacts[ bundle.manifestUri ] = bundle.manifestJson; // ...the manifest itself
        return bundle;
    }

    // Counting in-memory FetchFn: serves the bundle's bytes per URI.
    struct CountingFetcher
    {
        std::map<std::string, std::string> uris; // uri -> payload
        std::atomic<int>                   calls{ 0 };
        std::function<void()>              onCall; // optional barrier hook

        sgns::elmruntime::FetchFn Fn()
        {
            return [this]( const std::string &uri ) -> outcome::result<std::vector<uint8_t>> {
                ++calls;
                if ( onCall )
                {
                    onCall();
                }
                const auto it = uris.find( uri );
                if ( it == uris.end() )
                {
                    return outcome::failure( ElmRuntimeError::FETCH_FAILED );
                }
                return ToBytes( it->second );
            };
        }

        void AddBundle( const TestBundle &bundle )
        {
            for ( const auto &[uri, payload] : bundle.artifacts )
            {
                uris[uri] = payload;
            }
        }
    };

    sgns::elmruntime::SmokeCheckFn OkSmoke()
    {
        return []( const std::string & ) -> outcome::result<void> { return outcome::success(); };
    }

    sgns::elmruntime::SmokeCheckFn FailingSmoke()
    {
        return []( const std::string & ) -> outcome::result<void> {
            return outcome::failure( ElmRuntimeError::SMOKE_CHECK_FAILED );
        };
    }

    ElmRuntimeError ErrorCodeOf( const outcome::result<ElmCachePin> &r )
    {
        return static_cast<ElmRuntimeError>( r.error().value() );
    }

    bool HasTmpDirs( const fs::path &root )
    {
        std::error_code ec;
        for ( fs::directory_iterator it( root, ec ), end; !ec && it != end; it.increment( ec ) )
        {
            if ( it->path().filename().string().rfind( ".tmp-", 0 ) == 0 )
            {
                return true;
            }
        }
        return false;
    }

    bool HasBadDir( const fs::path &root, const std::string &hex )
    {
        return fs::exists( root / ( ".bad-" + hex ) );
    }
} // namespace

// ---------------------------------------------------------------------------
// Publish happy path (SC-2): MNN bundle layout at <root>/<digest>/
// ---------------------------------------------------------------------------

TEST( ElmModelCacheTest, PublishHappyPath )
{
    TempCacheRoot  root;
    TestBundle     bundle = BuildBundle();
    CountingFetcher fetcher;
    fetcher.AddBundle( bundle );

    auto cacheResult = ElmModelCache::Create( root.Str(), fetcher.Fn(), OkSmoke() );
    ASSERT_TRUE( cacheResult ) << cacheResult.error().message();
    auto cache = cacheResult.value();

    auto pin = cache->Acquire( bundle.manifestUri, bundle.declaredHash );
    ASSERT_TRUE( pin ) << pin.error().message();

    // Entry directory = root/<digest>/ with the trailing slash on the pin.
    const std::string &dir = pin.value().GetDir();
    EXPECT_EQ( pin.value().GetHash(), bundle.declaredHash );
    ASSERT_FALSE( dir.empty() );
    EXPECT_TRUE( dir.back() == '/' || dir.back() == '\\' );

    const fs::path entryDir( dir );
    EXPECT_TRUE( fs::exists( entryDir / "elm_manifest.json" ) );
    EXPECT_TRUE( fs::exists( entryDir / "llm_config.json" ) );
    EXPECT_TRUE( fs::exists( entryDir / "llm.mnn" ) );
    EXPECT_TRUE( fs::exists( entryDir / "llm.mnn.weight" ) );
    EXPECT_TRUE( fs::exists( entryDir / "tokenizer.txt" ) );

    // Exactly one fetch per artifact + one for the manifest.
    EXPECT_EQ( fetcher.calls.load(), 5 );
    // No staging residue.
    EXPECT_FALSE( HasTmpDirs( root.Path() ) );
}

// ---------------------------------------------------------------------------
// Publish failure paths: artifact hash mismatch + smoke failure (SC-2)
// ---------------------------------------------------------------------------

TEST( ElmModelCacheTest, ArtifactHashMismatchLeavesNoEntry )
{
    TempCacheRoot  root;
    TestBundle     bundle = BuildBundle();
    CountingFetcher fetcher;
    fetcher.AddBundle( bundle );
    // Corrupt the model payload AFTER hashing made the manifest -- same SIZE as
    // the original (so the pre-hash size check passes) but different bytes: the
    // fetched bytes no longer match the declared sha256.
    const std::string original = fetcher.uris[ "mem://bundle/llm_model" ];
    std::string       tampered( original.size(), 'X' ); // same length, different bytes
    ASSERT_FALSE( original.empty() );
    fetcher.uris[ "mem://bundle/llm_model" ] = tampered;

    auto cacheResult = ElmModelCache::Create( root.Str(), fetcher.Fn(), OkSmoke() );
    ASSERT_TRUE( cacheResult );
    auto cache = cacheResult.value();

    auto pin = cache->Acquire( bundle.manifestUri, bundle.declaredHash );
    ASSERT_FALSE( pin );
    EXPECT_EQ( ErrorCodeOf( pin ), ElmRuntimeError::ARTIFACT_HASH_MISMATCH );

    // No final-path entry, no staging residue (Q4 cleanup).
    EXPECT_FALSE( fs::exists( root.Path() / bundle.declaredHash ) );
    EXPECT_FALSE( HasTmpDirs( root.Path() ) );
}

TEST( ElmModelCacheTest, SmokeCheckFailsLeavesNoEntry )
{
    TempCacheRoot  root;
    TestBundle     bundle = BuildBundle();
    CountingFetcher fetcher;
    fetcher.AddBundle( bundle );

    auto cacheResult = ElmModelCache::Create( root.Str(), fetcher.Fn(), FailingSmoke() );
    ASSERT_TRUE( cacheResult );
    auto cache = cacheResult.value();

    auto pin = cache->Acquire( bundle.manifestUri, bundle.declaredHash );
    ASSERT_FALSE( pin );
    EXPECT_EQ( ErrorCodeOf( pin ), ElmRuntimeError::SMOKE_CHECK_FAILED );
    EXPECT_FALSE( fs::exists( root.Path() / bundle.declaredHash ) );
    EXPECT_FALSE( HasTmpDirs( root.Path() ) );
}

TEST( ElmModelCacheTest, ManifestHashMismatchPropagatesFromGate )
{
    TempCacheRoot  root;
    TestBundle     bundle = BuildBundle();
    CountingFetcher fetcher;
    fetcher.AddBundle( bundle );

    auto cacheResult = ElmModelCache::Create( root.Str(), fetcher.Fn(), OkSmoke() );
    ASSERT_TRUE( cacheResult );
    auto cache = cacheResult.value();

    // Declared hash of DIFFERENT bytes: the 02-01 front door rejects before parse.
    const std::string wrongHash = Sha256Hex( "other bytes entirely" );
    auto pin = cache->Acquire( bundle.manifestUri, wrongHash );
    ASSERT_FALSE( pin );
    EXPECT_EQ( ErrorCodeOf( pin ), ElmRuntimeError::MANIFEST_HASH_MISMATCH );
    EXPECT_FALSE( fs::exists( root.Path() / wrongHash ) );
}

// ---------------------------------------------------------------------------
// Hit path: no re-fetch (D-01), tamper -> quarantine -> retry (SC-3)
// ---------------------------------------------------------------------------

TEST( ElmModelCacheTest, HitNoRefetch )
{
    TempCacheRoot  root;
    TestBundle     bundle = BuildBundle();
    CountingFetcher fetcher;
    fetcher.AddBundle( bundle );

    auto cacheResult = ElmModelCache::Create( root.Str(), fetcher.Fn(), OkSmoke() );
    ASSERT_TRUE( cacheResult );
    auto cache = cacheResult.value();

    {
        auto pin = cache->Acquire( bundle.manifestUri, bundle.declaredHash );
        ASSERT_TRUE( pin ) << pin.error().message();
    }
    EXPECT_EQ( fetcher.calls.load(), 5 );

    {
        auto pin = cache->Acquire( bundle.manifestUri, bundle.declaredHash );
        ASSERT_TRUE( pin ) << pin.error().message();
        EXPECT_EQ( pin.value().GetHash(), bundle.declaredHash );
    }
    // Hit performed ZERO additional fetches.
    EXPECT_EQ( fetcher.calls.load(), 5 );
}

TEST( ElmModelCacheTest, TamperQuarantineRetry )
{
    TempCacheRoot  root;
    TestBundle     bundle = BuildBundle();
    CountingFetcher fetcher;
    fetcher.AddBundle( bundle );

    auto cacheResult = ElmModelCache::Create( root.Str(), fetcher.Fn(), OkSmoke() );
    ASSERT_TRUE( cacheResult );
    auto cache = cacheResult.value();

    {
        auto pin = cache->Acquire( bundle.manifestUri, bundle.declaredHash );
        ASSERT_TRUE( pin ) << pin.error().message();
    }

    // Tamper: truncate llm.mnn on disk.
    {
        const fs::path model = root.Path() / bundle.declaredHash / "llm.mnn";
        std::ofstream out( model, std::ios::binary | std::ios::trunc );
        out << "x"; // 1 byte instead of the declared size
    }

    // Reuse fails with CACHE_ENTRY_QUARANTINED; entry moved to .bad-<hex>.
    auto quarantined = cache->Acquire( bundle.manifestUri, bundle.declaredHash );
    ASSERT_FALSE( quarantined );
    EXPECT_EQ( ErrorCodeOf( quarantined ), ElmRuntimeError::CACHE_ENTRY_QUARANTINED );
    EXPECT_TRUE( HasBadDir( root.Path(), bundle.declaredHash ) );
    EXPECT_FALSE( fs::exists( root.Path() / bundle.declaredHash ) );

    // SC-3 retry: fresh Acquire re-downloads cleanly and succeeds; .bad- retained (D-07).
    auto retry = cache->Acquire( bundle.manifestUri, bundle.declaredHash );
    ASSERT_TRUE( retry ) << retry.error().message();
    EXPECT_TRUE( fs::exists( fs::path( retry.value().GetDir() ) / "llm.mnn" ) );
    EXPECT_TRUE( HasBadDir( root.Path(), bundle.declaredHash ) );
}

// ---------------------------------------------------------------------------
// Single-flight (SC-4): exactly one manifest+artifact download under concurrency
// ---------------------------------------------------------------------------

TEST( ElmModelCacheTest, SingleFlightSameHash )
{
    TempCacheRoot  root;
    TestBundle     bundle = BuildBundle();
    CountingFetcher fetcher;
    fetcher.AddBundle( bundle );

    // Start-barrier latch: both threads must reach Acquire before either fetch
    // completes, forcing genuine single-flight overlap (no sleep-poll loops).
    std::promise<void> releaseFetch;
    auto               releaseOnce = releaseFetch.get_future().share();
    fetcher.onCall = [releaseOnce]() { releaseOnce.wait(); };
    std::atomic<int> firstCall{ 0 };
    // The FIRST fetch call blocks until we release; later calls pass through.

    auto cacheResult = ElmModelCache::Create( root.Str(), fetcher.Fn(), OkSmoke() );
    ASSERT_TRUE( cacheResult );
    auto cache = cacheResult.value();

    auto runAcquire = [cache, &bundle]() -> outcome::result<ElmCachePin> {
        return cache->Acquire( bundle.manifestUri, bundle.declaredHash );
    };

    // Two threads race; the first fetch blocks on the latch until both are in.
    auto f1 = std::async( std::launch::async, runAcquire );
    // Give thread 1 time to become the publisher and block inside its first fetch.
    std::this_thread::sleep_for( std::chrono::milliseconds( 100 ) );
    auto f2 = std::async( std::launch::async, runAcquire );
    std::this_thread::sleep_for( std::chrono::milliseconds( 100 ) );
    releaseFetch.set_value(); // unblock: both threads are now inside Acquire

    auto r1 = f1.get();
    auto r2 = f2.get();
    ASSERT_TRUE( r1 ) << r1.error().message();
    ASSERT_TRUE( r2 ) << r2.error().message();
    EXPECT_EQ( r1.value().GetDir(), r2.value().GetDir() );
    EXPECT_EQ( r1.value().GetHash(), r2.value().GetHash() );

    // Exactly ONE full download (manifest + 4 artifacts).
    EXPECT_EQ( fetcher.calls.load(), 5 );
    (void) firstCall;
}

TEST( ElmModelCacheTest, ConcurrentDifferentHashes )
{
    TempCacheRoot root;
    TestBundle    bundleA = BuildBundle( "alpha" );
    TestBundle    bundleB = BuildBundle( "beta" );
    ASSERT_NE( bundleA.declaredHash, bundleB.declaredHash );

    CountingFetcher fetcher;
    fetcher.AddBundle( bundleA );
    fetcher.AddBundle( bundleB );

    auto cacheResult = ElmModelCache::Create( root.Str(), fetcher.Fn(), OkSmoke() );
    ASSERT_TRUE( cacheResult );
    auto cache = cacheResult.value();

    auto f1 = std::async( std::launch::async, [cache, &bundleA] { return cache->Acquire( bundleA.manifestUri, bundleA.declaredHash ); } );
    auto f2 = std::async( std::launch::async, [cache, &bundleB] { return cache->Acquire( bundleB.manifestUri, bundleB.declaredHash ); } );

    auto r1 = f1.get();
    auto r2 = f2.get();
    ASSERT_TRUE( r1 ) << r1.error().message();
    ASSERT_TRUE( r2 ) << r2.error().message();
    EXPECT_NE( r1.value().GetDir(), r2.value().GetDir() ); // no cross-hash blocking
    EXPECT_EQ( fetcher.calls.load(), 10 ); // both downloaded in full
}

// ---------------------------------------------------------------------------
// Pin blocks eviction + LRU order (SC-5, D-04)
// ---------------------------------------------------------------------------

TEST( ElmModelCacheTest, PinBlocksEvictionAndLruOrder )
{
    TempCacheRoot root;
    TestBundle    bundleA = BuildBundle( "alpha" );
    TestBundle    bundleB = BuildBundle( "beta" );

    CountingFetcher fetcher;
    fetcher.AddBundle( bundleA );
    fetcher.AddBundle( bundleB );

    // Cap = ONE bundle's bytes: the second publish must evict the first.
    const uint64_t oneBundleBytes = sgns::elmruntime::TotalArtifactBytes(
        sgns::elmruntime::ParseAndVerifyManifest( ToBytes( bundleA.manifestJson ), bundleA.declaredHash ).value() );
    const uint64_t cap = oneBundleBytes;

    auto cacheResult = ElmModelCache::Create( root.Str(), fetcher.Fn(), OkSmoke(), cap );
    ASSERT_TRUE( cacheResult );
    auto cache = cacheResult.value();

    // Publish A and PIN it (scoped so the pin releases deterministically).
    {
        auto pinA = cache->Acquire( bundleA.manifestUri, bundleA.declaredHash );
        ASSERT_TRUE( pinA ) << pinA.error().message();
        EXPECT_TRUE( fs::exists( root.Path() / bundleA.declaredHash ) );

        // Publish B under pressure: A is pinned so it must be RETAINED (over-cap
        // tolerated while everything is pinned); B present.
        auto pinB = cache->Acquire( bundleB.manifestUri, bundleB.declaredHash );
        ASSERT_TRUE( pinB ) << pinB.error().message();
        EXPECT_TRUE( fs::exists( root.Path() / bundleA.declaredHash ) ) << "pinned A must survive";
        EXPECT_TRUE( fs::exists( root.Path() / bundleB.declaredHash ) );
    } // both pins released: A is now the oldest unpinned entry

    // Trigger an eviction pass via SetByteCap (same cap).
    cache->SetByteCap( cap );

    // A (oldest, unpinned) evicted; B retained (newest).
    EXPECT_FALSE( fs::exists( root.Path() / bundleA.declaredHash ) ) << "released A must be evicted";
    EXPECT_TRUE( fs::exists( root.Path() / bundleB.declaredHash ) ) << "B must survive as the newest";
}

// ---------------------------------------------------------------------------
// Restart rebuild (D-06) + orphan sweep (SC-5)
// ---------------------------------------------------------------------------

TEST( ElmModelCacheTest, RestartRebuildHitsWithZeroFetches )
{
    TempCacheRoot root;
    TestBundle    bundle = BuildBundle();

    {
        CountingFetcher fetcher;
        fetcher.AddBundle( bundle );
        auto cacheResult = ElmModelCache::Create( root.Str(), fetcher.Fn(), OkSmoke() );
        ASSERT_TRUE( cacheResult );
        auto cache = cacheResult.value();
        auto pin = cache->Acquire( bundle.manifestUri, bundle.declaredHash );
        ASSERT_TRUE( pin ) << pin.error().message();
    }

    // Fresh instance over the populated root: hit with ZERO fetch calls.
    CountingFetcher fetcher2;
    fetcher2.AddBundle( bundle );
    auto cacheResult = ElmModelCache::Create( root.Str(), fetcher2.Fn(), OkSmoke() );
    ASSERT_TRUE( cacheResult );
    auto cache = cacheResult.value();
    auto pin = cache->Acquire( bundle.manifestUri, bundle.declaredHash );
    ASSERT_TRUE( pin ) << "restart scan must rebuild the entry: " << pin.error().message();
    EXPECT_EQ( fetcher2.calls.load(), 0 );
}

TEST( ElmModelCacheTest, TmpOrphanSweepedAtConstruction )
{
    TempCacheRoot root;
    TestBundle    bundle = BuildBundle();

    // Pre-create an orphaned staging dir with junk (a crashed prior run).
    const fs::path orphan = root.Path() / ".tmp-deadbeef";
    fs::create_directories( orphan );
    { std::ofstream out( orphan / "junk.bin", std::ios::binary ); out << "partial"; }

    CountingFetcher fetcher;
    fetcher.AddBundle( bundle );
    auto cacheResult = ElmModelCache::Create( root.Str(), fetcher.Fn(), OkSmoke() );
    ASSERT_TRUE( cacheResult );
    EXPECT_FALSE( fs::exists( orphan ) ) << "orphaned .tmp- must be swept at construction";

    auto cache = cacheResult.value();
    auto pin = cache->Acquire( bundle.manifestUri, bundle.declaredHash );
    ASSERT_TRUE( pin ) << pin.error().message();
}

TEST( ElmModelCacheTest, BadDirsIgnoredForever )
{
    TempCacheRoot root;
    TestBundle    bundle = BuildBundle();

    // A pre-existing quarantine dir for the same digest: construction ignores it,
    // publish proceeds (the .bad- dir never blocks re-download, D-07).
    const fs::path bad = root.Path() / ( ".bad-" + bundle.declaredHash );
    fs::create_directories( bad );
    { std::ofstream out( bad / "evidence.txt", std::ios::binary ); out << "forensic"; }

    CountingFetcher fetcher;
    fetcher.AddBundle( bundle );
    auto cacheResult = ElmModelCache::Create( root.Str(), fetcher.Fn(), OkSmoke() );
    ASSERT_TRUE( cacheResult );
    auto cache = cacheResult.value();
    auto pin = cache->Acquire( bundle.manifestUri, bundle.declaredHash );
    ASSERT_TRUE( pin ) << pin.error().message();
    EXPECT_TRUE( fs::exists( bad ) ) << ".bad- is kept forever (D-07)";
}

// ---------------------------------------------------------------------------
// Fail-closed root (D-03)
// ---------------------------------------------------------------------------

TEST( ElmModelCacheTest, EmptyRootFailsClosed )
{
    auto cacheResult = ElmModelCache::Create( "", []( const std::string & ) {
        return outcome::failure( ElmRuntimeError::FETCH_FAILED );
    }, OkSmoke() );
    ASSERT_FALSE( cacheResult );
    EXPECT_EQ( static_cast<ElmRuntimeError>( cacheResult.error().value() ), ElmRuntimeError::CACHE_DIR_UNSET );
}

// ---------------------------------------------------------------------------
// %TEMP% cleanliness (SC-5): the cache never writes outside its injected root
// ---------------------------------------------------------------------------

TEST( ElmModelCacheTest, TempDirCleanliness )
{
    std::set<std::string> before;
    {
        std::error_code ec;
        for ( fs::directory_iterator it( fs::temp_directory_path(), ec ), end; !ec && it != end; it.increment( ec ) )
        {
            const std::string name = it->path().filename().string();
            if ( name.rfind( "sgproc_elm", 0 ) == 0 )
            {
                before.insert( name );
            }
        }
    }

    TempCacheRoot  root;
    TestBundle     bundle = BuildBundle( "clean" );
    CountingFetcher fetcher;
    fetcher.AddBundle( bundle );

    auto cacheResult = ElmModelCache::Create( root.Str(), fetcher.Fn(), OkSmoke() );
    ASSERT_TRUE( cacheResult );
    auto cache = cacheResult.value();
    auto pin = cache->Acquire( bundle.manifestUri, bundle.declaredHash );
    ASSERT_TRUE( pin ) << pin.error().message();

    std::set<std::string> after;
    {
        std::error_code ec;
        for ( fs::directory_iterator it( fs::temp_directory_path(), ec ), end; !ec && it != end; it.increment( ec ) )
        {
            const std::string name = it->path().filename().string();
            if ( name.rfind( "sgproc_elm", 0 ) == 0 )
            {
                after.insert( name );
            }
        }
    }

    // The only sgproc_elm* dir in %TEMP% is this test's own fixture root.
    EXPECT_EQ( after.size(), before.size() + 1 );
    EXPECT_TRUE( after.count( root.Path().filename().string() ) );
}

// ---------------------------------------------------------------------------
// Declared-hash normalization at the Acquire boundary
// ---------------------------------------------------------------------------

TEST( ElmModelCacheTest, Sha256PrefixAndUppercaseHashAcquire )
{
    TempCacheRoot  root;
    TestBundle     bundle = BuildBundle();
    CountingFetcher fetcher;
    fetcher.AddBundle( bundle );

    auto cacheResult = ElmModelCache::Create( root.Str(), fetcher.Fn(), OkSmoke() );
    ASSERT_TRUE( cacheResult );
    auto cache = cacheResult.value();

    std::string upper = bundle.declaredHash;
    std::transform( upper.begin(), upper.end(), upper.begin(), []( unsigned char c ) {
        return static_cast<char>( std::toupper( c ) );
    } );

    auto pin = cache->Acquire( bundle.manifestUri, "sha256:" + upper );
    ASSERT_TRUE( pin ) << "normalized declared hash must acquire: " << pin.error().message();
    EXPECT_EQ( pin.value().GetHash(), bundle.declaredHash );
}

TEST( ElmModelCacheTest, MalformedDeclaredHashRejects )
{
    TempCacheRoot  root;
    TestBundle     bundle = BuildBundle();
    CountingFetcher fetcher;
    fetcher.AddBundle( bundle );

    auto cacheResult = ElmModelCache::Create( root.Str(), fetcher.Fn(), OkSmoke() );
    ASSERT_TRUE( cacheResult );
    auto cache = cacheResult.value();

    auto pin = cache->Acquire( bundle.manifestUri, "sha256:abc" ); // Phase 1 fixture value
    ASSERT_FALSE( pin );
    EXPECT_EQ( ErrorCodeOf( pin ), ElmRuntimeError::MANIFEST_INVALID );
}

// ---------------------------------------------------------------------------
// Pin release determinism: destructor returns refcount to zero, then eviction works
// ---------------------------------------------------------------------------

TEST( ElmModelCacheTest, PinReleaseDeterministic )
{
    TempCacheRoot root;
    TestBundle    bundle = BuildBundle( "pinrel" );

    CountingFetcher fetcher;
    fetcher.AddBundle( bundle );

    auto cacheResult = ElmModelCache::Create( root.Str(), fetcher.Fn(), OkSmoke(), 0 /* cap 0 */ );
    ASSERT_TRUE( cacheResult );
    auto cache = cacheResult.value();

    {
        auto pin = cache->Acquire( bundle.manifestUri, bundle.declaredHash );
        ASSERT_TRUE( pin ) << pin.error().message();
        // Cap 0 but the only entry is pinned: retained (over-cap-with-pins tolerated).
        EXPECT_TRUE( fs::exists( root.Path() / bundle.declaredHash ) );
    } // pin destructor: refcount -> 0

    // Trigger an eviction pass: with cap 0 and nothing pinned, the entry goes.
    cache->SetByteCap( 0 );
    EXPECT_FALSE( fs::exists( root.Path() / bundle.declaredHash ) );
}

// ---------------------------------------------------------------------------
// Gated real-smoke leg (SGPROC_HAS_MNN_LLM only): a garbage bundle (valid hashes
// over non-MNN bytes) must fail SMOKE_CHECK_FAILED from inside MNN's load() --
// proves the end-to-end probe wiring without a real model. FAILS BY DESIGN.
// ---------------------------------------------------------------------------

#if defined( SGPROC_HAS_MNN_LLM )

TEST( ElmModelCacheMnnSmokeTest, GarbageBundleFailsRealSmokeCheck )
{
    TempCacheRoot  root;
    TestBundle     bundle = BuildBundle( "garbage" ); // valid hashes, non-MNN bytes
    CountingFetcher fetcher;
    fetcher.AddBundle( bundle );

    auto cacheResult = ElmModelCache::Create( root.Str(), fetcher.Fn(), sgns::elmruntime::MakeMnnLlmSmokeCheck() );
    ASSERT_TRUE( cacheResult );
    auto cache = cacheResult.value();

    auto pin = cache->Acquire( bundle.manifestUri, bundle.declaredHash );
    ASSERT_FALSE( pin ) << "garbage bytes must NOT pass the real MNN load probe";
    const auto code = static_cast<ElmRuntimeError>( pin.error().value() );
    EXPECT_TRUE( code == ElmRuntimeError::SMOKE_CHECK_FAILED || code == ElmRuntimeError::SMOKE_CHECK_UNAVAILABLE )
        << "actual: " << pin.error().message();
    // No final-path entry from a failed probe.
    EXPECT_FALSE( fs::exists( root.Path() / bundle.declaredHash ) );
    EXPECT_FALSE( HasTmpDirs( root.Path() ) );
}

#endif // SGPROC_HAS_MNN_LLM
