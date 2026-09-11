#include <elmruntime/ElmModelCache.hpp>

#include <elmruntime/ElmManifest.hpp>

#include <util/sgprocmgr-logger.hpp>

#include "FileManager.hpp"

// Boost uuid includes follow the vendored boost 1.85 layout (boost/uuid, no
// trailing "s" -- the same set processing_tasksplit.cpp includes): uuid.hpp,
// uuid_generators.hpp (basic_random_generator), uuid_io.hpp (to_string).
#include <boost/uuid/random_generator.hpp>
#include <boost/uuid/uuid.hpp>
#include <boost/uuid/uuid_generators.hpp>
#include <boost/uuid/uuid_io.hpp>

#include <algorithm>
#include <chrono>
#include <cctype>
#include <cstdio>
#include <filesystem>
#include <fstream>
#include <future>
#include <mutex>
#include <random>
#include <regex>
#include <system_error>
#include <unordered_map>
#include <vector>

namespace fs = std::filesystem;

namespace sgns::elmruntime
{
    namespace
    {
        sgns::sgprocmanager::Logger CacheLogger()
        {
            return sgns::sgprocmanager::createLogger( "ElmModelCache" );
        }

        /// D-01 hit-path result carrier through the single-flight future (P2-7:
        /// the promise is ALWAYS resolved by value -- never set_exception).
        struct AcquireOutcome
        {
            bool        success = false;
            std::string hashHex; // the entry digest on success
            int         errorValue = 0; // ElmRuntimeError value on failure
            std::string errorMessage;
        };

        /// Normalize a declared hash exactly the way ParseAndVerifyManifest does
        /// (optional "sha256:" prefix + 64 hex, lowercased) so the single-flight
        /// map keys "sha256:ABC" and "abc" identically. Returns the bare hex or "".
        std::string NormalizeDeclaredHashKey( const std::string &declared )
        {
            std::string_view view( declared );
            constexpr std::string_view kPrefix = "sha256:";
            if ( view.size() >= kPrefix.size() && view.substr( 0, kPrefix.size() ) == kPrefix )
            {
                view.remove_prefix( kPrefix.size() );
            }
            if ( view.size() != 64 )
            {
                return "";
            }
            std::string out;
            out.reserve( view.size() );
            for ( char c : view )
            {
                const char lower = static_cast<char>( std::tolower( static_cast<unsigned char>( c ) ) );
                if ( !( ( lower >= '0' && lower <= '9' ) || ( lower >= 'a' && lower <= 'f' ) ) )
                {
                    return "";
                }
                out.push_back( lower );
            }
            return out;
        }

        bool IsLowerHex64( const std::string &s )
        {
            if ( s.size() != 64 )
            {
                return false;
            }
            return std::all_of( s.begin(), s.end(), []( char c ) {
                return ( c >= '0' && c <= '9' ) || ( c >= 'a' && c <= 'f' );
            } );
        }

        /// Staging directory name: .tmp-<uuid> (A24 tasksplit pattern, seeded from
        /// std::random_device -- NEVER timestamps, P2-6/A7 collision lesson).
        std::string NewStagingDirName()
        {
            static std::mt19937 gen( static_cast<std::mt19937::result_type>( std::random_device{}() ) );
            boost::uuids::basic_random_generator<std::mt19937> uuidGen( gen );
            return ".tmp-" + boost::uuids::to_string( uuidGen() );
        }

        std::string WithTrailingSlash( const fs::path &dir )
        {
            std::string s = dir.string();
            if ( !s.empty() && s.back() != '/' && s.back() != '\\' )
            {
                s += '/';
            }
            return s;
        }

        std::time_t NowTimeT()
        {
            return std::chrono::system_clock::to_time_t( std::chrono::system_clock::now() );
        }
    } // namespace

    class ElmModelCache::Impl
    {
    public:
        Impl( std::string rootDir, FetchFn fetch, SmokeCheckFn smokeCheck, uint64_t byteCap )
            : root_( std::move( rootDir ) ), fetch_( std::move( fetch ) ), smokeCheck_( std::move( smokeCheck ) ),
              byteCap_( byteCap )
        {
        }

        std::string                 root_;
        FetchFn                     fetch_;
        SmokeCheckFn                smokeCheck_;
        uint64_t                    byteCap_;
        uint64_t                    accountedBytes_ = 0;
        std::mutex                  mutex_; // guards entries_ AND inFlight_
        std::unordered_map<std::string, EntryState> entries_;
        std::unordered_map<std::string, std::shared_future<AcquireOutcome>> inFlight_;
    };

    // ------------------------------------------------------------------
    // ElmCachePin
    // ------------------------------------------------------------------

    ElmCachePin::ElmCachePin( std::string dir, std::string hashHex, std::shared_ptr<ElmModelCache> cache )
        : dir_( std::move( dir ) ), hashHex_( std::move( hashHex ) ), cache_( std::move( cache ) )
    {
    }

    ElmCachePin::~ElmCachePin()
    {
        if ( cache_ )
        {
            cache_->ReleasePin( hashHex_ );
        }
    }

    ElmCachePin::ElmCachePin( ElmCachePin &&other ) noexcept
        : dir_( std::move( other.dir_ ) ), hashHex_( std::move( other.hashHex_ ) ), cache_( std::move( other.cache_ ) )
    {
        other.cache_.reset(); // moved-from releases nothing
    }

    ElmCachePin &ElmCachePin::operator=( ElmCachePin &&other ) noexcept
    {
        if ( this != &other )
        {
            if ( cache_ )
            {
                cache_->ReleasePin( hashHex_ );
            }
            dir_     = std::move( other.dir_ );
            hashHex_ = std::move( other.hashHex_ );
            cache_   = std::move( other.cache_ );
            other.cache_.reset();
        }
        return *this;
    }

    // ------------------------------------------------------------------
    // Construction / restart scan (D-06)
    // ------------------------------------------------------------------

    outcome::result<std::shared_ptr<ElmModelCache>> ElmModelCache::Create( const std::string &cacheRootDir,
                                                                           FetchFn            fetch,
                                                                           SmokeCheckFn       smokeCheck,
                                                                           uint64_t           byteCap )
    {
        const auto logger = CacheLogger();

        if ( cacheRootDir.empty() )
        {
            logger->error( "ElmModelCache: cache root directory is empty -- failing closed (D-03)" );
            return outcome::failure( ElmRuntimeError::CACHE_DIR_UNSET );
        }
        if ( !fetch )
        {
            logger->error( "ElmModelCache: null FetchFn" );
            return outcome::failure( ElmRuntimeError::FETCH_FAILED );
        }
        if ( !smokeCheck )
        {
            logger->error( "ElmModelCache: null SmokeCheckFn" );
            return outcome::failure( ElmRuntimeError::SMOKE_CHECK_FAILED );
        }

        // shared_from_this is safe from birth only via a factory-constructed
        // shared_ptr (Pitfall 15.1): build the raw object, scan, then wrap.
        auto cache = std::shared_ptr<ElmModelCache>( new ElmModelCache( cacheRootDir, std::move( fetch ),
                                                                        std::move( smokeCheck ), byteCap ) );

        std::error_code ec;
        fs::create_directories( cache->impl_->root_, ec );
        if ( ec )
        {
            logger->error( "ElmModelCache: cannot create cache root {}: {}", cache->impl_->root_, ec.message() );
            return outcome::failure( ElmRuntimeError::CACHE_DIR_UNSET );
        }

        // Restart scan (D-06): rebuild entries/LRU/accounting from the directory
        // listing; pin counts start at zero. Runs on the constructor caller's
        // (worker) thread -- Pitfall 14.
        const std::regex hex64Regex( "^[0-9a-f]{64}$" );
        for ( fs::directory_iterator it( cache->impl_->root_, ec ), end; !ec && it != end; it.increment( ec ) )
        {
            const std::string name = it->path().filename().string();
            if ( !it->is_directory( ec ) )
            {
                continue;
            }
            if ( name.rfind( ".tmp-", 0 ) == 0 )
            {
                // Orphaned staging from a crashed run: sweep (SC-5).
                fs::remove_all( it->path(), ec );
                if ( ec )
                {
                    logger->warn( "ElmModelCache: failed to sweep orphaned staging {}: {}", name, ec.message() );
                }
                else
                {
                    logger->info( "ElmModelCache: swept orphaned staging dir {}", name );
                }
                continue;
            }
            if ( name.rfind( ".bad-", 0 ) == 0 )
            {
                continue; // D-07: kept forever, excluded from accounting, never loaded
            }
            if ( !std::regex_match( name, hex64Regex ) )
            {
                // Foreign directory: not quarantined (it was never verified bad at
                // reuse -- just not ours); leave untouched and untracked.
                logger->warn( "ElmModelCache: ignoring non-entry directory {}", name );
                continue;
            }

            // Entry: read its stored manifest for accounting (bytes) -- the manifest
            // was hash-verified at publish time; a missing/unreadable copy means the
            // entry cannot be size-checked later, so skip + leave untouched.
            const fs::path manifestPath = it->path() / "elm_manifest.json";
            std::ifstream  manifestIn( manifestPath, std::ios::binary );
            if ( !manifestIn )
            {
                logger->warn( "ElmModelCache: entry {} has no readable elm_manifest.json -- untracked", name );
                continue;
            }
            std::vector<uint8_t> manifestBytes( ( std::istreambuf_iterator<char>( manifestIn ) ),
                                                std::istreambuf_iterator<char>() );
            manifestIn.close();

            // The stored digest must match the directory name (self-consistency);
            // parse with the entry's own name as the declared hash.
            auto parsed = ParseAndVerifyManifest( manifestBytes, name );
            if ( !parsed )
            {
                logger->warn( "ElmModelCache: entry {} failed re-verification at startup -- untracked: {}",
                              name,
                              parsed.error().message() );
                continue;
            }

            EntryState state;
            state.bytes    = TotalArtifactBytes( parsed.value() );
            std::error_code mtimeEc;
            const auto     lastWrite = fs::last_write_time( it->path(), mtimeEc );
            state.lastUse  = mtimeEc ? NowTimeT()
                                     : std::chrono::duration_cast<std::chrono::seconds>(
                                           lastWrite.time_since_epoch() )
                                           .count();
            state.pinCount = 0;

            std::lock_guard<std::mutex> lock( cache->impl_->mutex_ );
            cache->impl_->accountedBytes_ += state.bytes;
            cache->impl_->entries_[ name ] = state;
        }

        return cache;
    }

    outcome::result<std::shared_ptr<ElmModelCache>> ElmModelCache::CreateProductionElmModelCache( SmokeCheckFn smokeCheck )
    {
        const auto  logger    = CacheLogger();
        std::string cacheDir;
        try
        {
            cacheDir = FileManager::GetInstance().getCacheDir();
        }
        catch ( const std::exception &e )
        {
            logger->error( "ElmModelCache: getCacheDir() threw: {}", e.what() );
            return outcome::failure( ElmRuntimeError::CACHE_DIR_UNSET );
        }
        if ( cacheDir.empty() )
        {
            // D-03 / P2-1: empty in every standalone process (set only via
            // setBitswap) -- NEVER guess a default.
            logger->error( "ElmModelCache: FileManager cache dir is unset -- failing closed (D-03)" );
            return outcome::failure( ElmRuntimeError::CACHE_DIR_UNSET );
        }
        return Create( ( fs::path( cacheDir ) / "elmruntime" ).string(),
                       MakeFileManagerFetchFn(),
                       std::move( smokeCheck ) );
    }

    ElmModelCache::ElmModelCache( std::string cacheRootDir, FetchFn fetch, SmokeCheckFn smokeCheck, uint64_t byteCap )
        : impl_( std::make_unique<Impl>( std::move( cacheRootDir ), std::move( fetch ), std::move( smokeCheck ), byteCap ) )
    {
    }

    // ------------------------------------------------------------------
    // Acquire
    // ------------------------------------------------------------------

    outcome::result<ElmCachePin> ElmModelCache::Acquire( const std::string &manifestUri,
                                                         const std::string &manifestHash )
    {
        const auto logger = CacheLogger();

        const std::string declaredHex = NormalizeDeclaredHashKey( manifestHash );
        if ( declaredHex.empty() )
        {
            logger->error( "ElmModelCache: declared manifest hash is not sha256:-prefixed 64-hex" );
            return outcome::failure( ElmRuntimeError::MANIFEST_INVALID );
        }

        // Single-flight lookup keyed by the normalized declared hash (SC-4). The
        // promise/future carries the outcome BY VALUE (P2-7 -- never set_exception:
        // awaiting callers must see the structured error, not an exception_ptr).
        std::shared_future<AcquireOutcome> flight;
        std::promise<AcquireOutcome>       publisherPromise; // valid only for the publisher
        bool                               amPublisher = false;
        {
            std::lock_guard<std::mutex> lock( impl_->mutex_ );
            auto                        it = impl_->inFlight_.find( declaredHex );
            if ( it != impl_->inFlight_.end() )
            {
                flight = it->second; // join the in-flight download
            }
            else
            {
                flight      = publisherPromise.get_future().share();
                amPublisher = true;
                impl_->inFlight_.emplace( declaredHex, flight );
            }
        }

        if ( !amPublisher )
        {
            // Waiter: block until the publisher resolves (downloads run on the
            // publisher's thread -- no pool exhaustion, T-02-03-03).
            flight.wait();
            const AcquireOutcome outcome = flight.get();
            if ( !outcome.success )
            {
                return outcome::failure( static_cast<ElmRuntimeError>( outcome.errorValue ) );
            }
            // Both threads pin the SAME entry (SC-4): run the hit path.
            return TryHitOrPublish( manifestUri, declaredHex, true /*joinCompletedPublish*/ );
        }

        // Publisher: run the full pipeline; resolve the promise BY VALUE on every
        // path (the guard erases the map entry even if the pipeline throws).
        auto           pinResult = TryHitOrPublish( manifestUri, declaredHex, false );
        AcquireOutcome outcome;
        if ( pinResult )
        {
            outcome.success = true;
            outcome.hashHex = pinResult.value().GetHash();
        }
        else
        {
            outcome.success      = false;
            outcome.errorValue   = pinResult.error().value();
            outcome.errorMessage = pinResult.error().message();
        }
        publisherPromise.set_value( outcome ); // by value, never set_exception (P2-7)

        {
            std::lock_guard<std::mutex> lock( impl_->mutex_ );
            impl_->inFlight_.erase( declaredHex );
        }

        return pinResult;
    }

    // ------------------------------------------------------------------
    // Hit path (D-01) + publish path (miss) + quarantine (D-02)
    // ------------------------------------------------------------------

    outcome::result<ElmCachePin> ElmModelCache::TryHitOrPublish( const std::string &manifestUri,
                                                                 const std::string &declaredHex,
                                                                 bool               joinCompletedPublish )
    {
        const auto logger = CacheLogger();
        const fs::path entryDir( fs::path( impl_->root_ ) / declaredHex );

        // ---------------- Hit path (D-01): size-first, no sha256 re-hash --------
        std::error_code ec;
        if ( fs::exists( entryDir, ec ) && fs::is_directory( entryDir, ec ) )
        {
            // Re-verify the STORED manifest against the declared hash: the small
            // manifest bytes may be hashed (this is the only hash on the hit path).
            std::ifstream manifestIn( entryDir / "elm_manifest.json", std::ios::binary );
            if ( !manifestIn )
            {
                return QuarantineEntry( declaredHex, "elm_manifest.json unreadable" );
            }
            std::vector<uint8_t> manifestBytes( ( std::istreambuf_iterator<char>( manifestIn ) ),
                                                std::istreambuf_iterator<char>() );
            manifestIn.close();

            auto parsed = ParseAndVerifyManifest( manifestBytes, declaredHex );
            if ( !parsed )
            {
                return QuarantineEntry( declaredHex, "stored manifest failed re-verification: " + parsed.error().message() );
            }

            // Stat every role file vs the manifest's declared size_bytes.
            for ( const auto &artifact : parsed.value().get_artifacts() )
            {
                const char *fileName = RoleFileName( artifact.get_name() );
                if ( fileName == nullptr )
                {
                    return QuarantineEntry( declaredHex, "unknown role in stored manifest: " + artifact.get_name() );
                }
                const fs::path artifactPath = entryDir / fileName;
                if ( !fs::exists( artifactPath, ec ) )
                {
                    return QuarantineEntry( declaredHex, std::string( "missing artifact file: " ) + fileName );
                }
                const uintmax_t actualSize = fs::file_size( artifactPath, ec );
                if ( ec )
                {
                    return QuarantineEntry( declaredHex, std::string( "cannot stat artifact file: " ) + fileName );
                }
                if ( actualSize != static_cast<uintmax_t>( artifact.get_size_bytes() ) )
                {
                    return QuarantineEntry( declaredHex,
                                            std::string( "artifact size mismatch: " ) + fileName + " declared "
                                                + std::to_string( artifact.get_size_bytes() ) + " actual "
                                                + std::to_string( actualSize ) );
                }
            }

            // Sizes all match: LRU touch (directory mtime), pin, return.
            fs::last_write_time( entryDir, fs::file_time_type::clock::now(), ec );
            if ( ec )
            {
                logger->warn( "ElmModelCache: LRU touch failed for {}: {}", declaredHex, ec.message() );
            }

            {
                std::lock_guard<std::mutex> lock( impl_->mutex_ );
                auto                        it = impl_->entries_.find( declaredHex );
                if ( it == impl_->entries_.end() )
                {
                    // Restart-scanned entries are present; a foreign-but-valid entry
                    // (published by another process instance) adopts here.
                    EntryState state;
                    state.bytes    = TotalArtifactBytes( parsed.value() );
                    state.lastUse  = NowTimeT();
                    it             = impl_->entries_.emplace( declaredHex, state ).first;
                    impl_->accountedBytes_ += state.bytes;
                }
                it->second.lastUse = NowTimeT();
                ++it->second.pinCount;
            }

            return ElmCachePin( WithTrailingSlash( entryDir ), declaredHex, shared_from_this() );
        }

        if ( joinCompletedPublish )
        {
            // Another thread completed the publish; but the entry dir is absent --
            // the publisher must have failed AFTER our wait resolved. Surface the
            // state as a quarantine-class miss (extremely defensive; normally the
            // waiter's hit path above succeeds).
            logger->error( "ElmModelCache: joined publish for {} but no entry exists", declaredHex );
            return outcome::failure( ElmRuntimeError::CACHE_ENTRY_QUARANTINED );
        }

        // ---------------- Publish path (miss): stage -> verify -> smoke -> rename --
        const std::string stagingName = NewStagingDirName();
        const fs::path    stagingDir( fs::path( impl_->root_ ) / stagingName );
        fs::create_directories( stagingDir, ec );
        if ( ec )
        {
            logger->error( "ElmModelCache: cannot create staging dir {}: {}", stagingDir.string(), ec.message() );
            return outcome::failure( ElmRuntimeError::FETCH_FAILED );
        }

        // RAII staging cleanup on EVERY abort path (Q4: staging failures clean up;
        // quarantine is reserved for reuse-time mismatch per D-02's letter).
        struct StagingGuard
        {
            fs::path                  dir;
            sgns::sgprocmanager::Logger logger;

            ~StagingGuard()
            {
                if ( !dir.empty() )
                {
                    std::error_code rmEc;
                    fs::remove_all( dir, rmEc );
                    if ( rmEc )
                    {
                        logger->warn( "ElmModelCache: staging cleanup failed for {}: {}", dir.string(), rmEc.message() );
                    }
                }
            }
        } stagingGuard{ stagingDir, logger };

        // (1) Fetch manifest bytes through the D-08 seam.
        auto manifestFetch = impl_->fetch_( manifestUri );
        if ( !manifestFetch )
        {
            logger->error( "ElmModelCache: manifest fetch failed for {}: {}", manifestUri,
                           manifestFetch.error().message() );
            return outcome::failure( ElmRuntimeError::MANIFEST_FETCH_FAILED );
        }
        const std::vector<uint8_t> &manifestBytes = manifestFetch.value();

        // (2) The verified-manifest front door (02-01): hash gate precedes parse;
        //     a declared-vs-computed mismatch is MANIFEST_HASH_MISMATCH here.
        auto manifestResult = ParseAndVerifyManifest( manifestBytes, declaredHex );
        if ( !manifestResult )
        {
            logger->error( "ElmModelCache: manifest verification failed for {}: {}", manifestUri,
                           manifestResult.error().message() );
            return outcome::failure( static_cast<ElmRuntimeError>( manifestResult.error().value() ) );
        }
        const sgns::ElmModelManifest &manifest = manifestResult.value();

        // The computed digest IS the directory name (P2-4) -- recompute rather
        // than trusting the declared string for the path.
        const std::string computedHex = ComputeManifestHexDigest( manifestBytes );
        if ( computedHex != declaredHex )
        {
            // ParseAndVerifyManifest already enforces this; defensive double-check.
            logger->error( "ElmModelCache: computed digest {} != declared {}", computedHex, declaredHex );
            return outcome::failure( ElmRuntimeError::MANIFEST_HASH_MISMATCH );
        }

        // (3) Per artifact: fetch -> sha256 verify -> write to staging at the
        //     role's fixed filename. Hash verification precedes every write.
        for ( const auto &artifact : manifest.get_artifacts() )
        {
            auto artifactFetch = impl_->fetch_( artifact.get_uri() );
            if ( !artifactFetch )
            {
                logger->error( "ElmModelCache: artifact fetch failed for {}: {}", artifact.get_uri(),
                               artifactFetch.error().message() );
                return outcome::failure( ElmRuntimeError::ARTIFACT_FETCH_FAILED );
            }
            const std::vector<uint8_t> &artifactBytes = artifactFetch.value();

            // Optional declared-size pre-write check (cheap, catches truncation
            // before hashing): mismatch is ARTIFACT_SIZE_MISMATCH.
            if ( artifactBytes.size() != static_cast<size_t>( artifact.get_size_bytes() ) )
            {
                logger->error( "ElmModelCache: artifact {} fetched {} bytes but manifest declares {}",
                               artifact.get_name(),
                               artifactBytes.size(),
                               artifact.get_size_bytes() );
                return outcome::failure( ElmRuntimeError::ARTIFACT_SIZE_MISMATCH );
            }

            const std::string artifactHex = ComputeManifestHexDigest( artifactBytes );
            std::string       declaredArtifactHex
                = NormalizeDeclaredHashKey( artifact.get_sha256() );
            if ( declaredArtifactHex.empty() || artifactHex != declaredArtifactHex )
            {
                logger->error( "ElmModelCache: artifact {} sha256 mismatch (computed {}, declared {})",
                               artifact.get_name(),
                               artifactHex,
                               artifact.get_sha256() );
                return outcome::failure( ElmRuntimeError::ARTIFACT_HASH_MISMATCH );
            }

            const char *fileName = RoleFileName( artifact.get_name() );
            if ( fileName == nullptr )
            {
                logger->error( "ElmModelCache: artifact role {} has no fixed filename", artifact.get_name() );
                return outcome::failure( ElmRuntimeError::MANIFEST_INVALID );
            }
            const fs::path artifactPath = stagingDir / fileName;
            std::ofstream  out( artifactPath, std::ios::binary );
            if ( !out )
            {
                logger->error( "ElmModelCache: cannot open staging artifact for write: {}", artifactPath.string() );
                return outcome::failure( ElmRuntimeError::FETCH_FAILED );
            }
            out.write( reinterpret_cast<const char *>( artifactBytes.data() ),
                       static_cast<std::streamsize>( artifactBytes.size() ) );
            out.close();
            if ( !out )
            {
                logger->error( "ElmModelCache: staging artifact write failed: {}", artifactPath.string() );
                return outcome::failure( ElmRuntimeError::FETCH_FAILED );
            }
        }

        // (4) Store the verified manifest bytes VERBATIM (D-01/D-06 source).
        {
            const fs::path manifestPath = stagingDir / "elm_manifest.json";
            std::ofstream  out( manifestPath, std::ios::binary );
            if ( !out )
            {
                logger->error( "ElmModelCache: cannot open staging manifest for write: {}", manifestPath.string() );
                return outcome::failure( ElmRuntimeError::FETCH_FAILED );
            }
            out.write( reinterpret_cast<const char *>( manifestBytes.data() ),
                       static_cast<std::streamsize>( manifestBytes.size() ) );
            out.close();
            if ( !out )
            {
                logger->error( "ElmModelCache: staging manifest write failed: {}", manifestPath.string() );
                return outcome::failure( ElmRuntimeError::FETCH_FAILED );
            }
        }

        // (5) Smoke check against the STAGING dir, pre-rename (A5): a failed check
        //     aborts the publish -- no final-path entry can exist unverified.
        auto smoke = impl_->smokeCheck_( WithTrailingSlash( stagingDir ) );
        if ( !smoke )
        {
            logger->error( "ElmModelCache: smoke check failed for {}: {}", manifestUri, smoke.error().message() );
            return outcome::failure( static_cast<ElmRuntimeError>( smoke.error().value() ) );
        }

        // (6) Atomic publish: quarantine any pre-existing target first (Windows
        //     rename-onto-existing fails, P2-2), then rename staging -> final.
        if ( fs::exists( entryDir, ec ) )
        {
            logger->warn( "ElmModelCache: entry {} already exists at publish -- quarantining stale", declaredHex );
            QuarantineExistingDir( declaredHex, entryDir, "superseded at publish" );
        }
        fs::rename( stagingDir, entryDir, ec );
        if ( ec )
        {
            logger->error( "ElmModelCache: publish rename failed {} -> {}: {}", stagingDir.string(),
                           entryDir.string(), ec.message() );
            return outcome::failure( ElmRuntimeError::FETCH_FAILED );
        }
        stagingGuard.dir.clear(); // published: staging no longer exists

        // (7) Record entry + eviction pass (D-04).
        {
            std::lock_guard<std::mutex> lock( impl_->mutex_ );
            EntryState                 state;
            state.bytes    = TotalArtifactBytes( manifest );
            state.lastUse  = NowTimeT();
            state.pinCount = 1; // this Acquire's pin
            const auto [it, inserted] = impl_->entries_.emplace( declaredHex, state );
            if ( !inserted )
            {
                it->second.lastUse = state.lastUse;
                ++it->second.pinCount;
            }
            else
            {
                impl_->accountedBytes_ += state.bytes;
            }
        }
        RunEvictionPass();

        return ElmCachePin( WithTrailingSlash( entryDir ), declaredHex, shared_from_this() );
    }

    outcome::result<ElmCachePin> ElmModelCache::QuarantineEntry( const std::string &hashHex, const std::string &reason )
    {
        const auto  logger   = CacheLogger();
        const fs::path entryDir( fs::path( impl_->root_ ) / hashHex );

        QuarantineExistingDir( hashHex, entryDir, reason );

        // Drop from tracking (no auto-refetch in this call -- D-02: the caller
        // retries; a poisoned entry cannot livelock because it was renamed away).
        {
            std::lock_guard<std::mutex> lock( impl_->mutex_ );
            const auto                  it = impl_->entries_.find( hashHex );
            if ( it != impl_->entries_.end() )
            {
                impl_->accountedBytes_ -= it->second.bytes;
                impl_->entries_.erase( it );
            }
        }

        return outcome::failure( ElmRuntimeError::CACHE_ENTRY_QUARANTINED );
    }

    void ElmModelCache::QuarantineExistingDir( const std::string &hashHex, const fs::path &entryDir,
                                                const std::string &reason )
    {
        const auto       logger = CacheLogger();
        const fs::path   badDir( fs::path( impl_->root_ ) / ( ".bad-" + hashHex ) );
        std::error_code  ec;

        // Remove a pre-existing quarantine of the same digest first (Windows
        // rename-onto-existing fails, P2-2; D-07 keeps the LATEST evidence).
        if ( fs::exists( badDir, ec ) )
        {
            fs::remove_all( badDir, ec );
            if ( ec )
            {
                logger->warn( "ElmModelCache: removing stale quarantine {} failed: {}", badDir.string(), ec.message() );
            }
        }
        fs::rename( entryDir, badDir, ec );
        if ( ec )
        {
            // Quarantine failure: remove outright so the poisoned entry still
            // leaves the load path (fail-closed, SC-3 -- no livelock either way).
            logger->error( "ElmModelCache: quarantine rename failed, removing entry {} instead: {}", hashHex,
                           ec.message() );
            fs::remove_all( entryDir, ec );
        }
        // D-07 telemetry line -- the quarantine event.
        logger->warn( "ElmModelCache: quarantined cache entry {}: {}", hashHex, reason );
    }

    // ------------------------------------------------------------------
    // Eviction (D-04) + pin release
    // ------------------------------------------------------------------

    void ElmModelCache::RunEvictionPass()
    {
        const auto logger = CacheLogger();

        for ( ;; )
        {
            std::string oldestHex;
            std::time_t oldestUse = 0;
            bool        found     = false;

            {
                std::lock_guard<std::mutex> lock( impl_->mutex_ );
                if ( impl_->accountedBytes_ <= impl_->byteCap_ )
                {
                    return; // under the cap: done
                }
                for ( const auto &[hex, state] : impl_->entries_ )
                {
                    if ( state.pinCount > 0 )
                    {
                        continue; // pinned entries are never removed
                    }
                    if ( !found || state.lastUse < oldestUse )
                    {
                        found     = true;
                        oldestHex = hex;
                        oldestUse = state.lastUse;
                    }
                }
                if ( !found )
                {
                    // Everything tracked is pinned: over cap but nothing evictable --
                    // tolerated (the cap is a bound, not an invariant breaker).
                    logger->warn( "ElmModelCache: over byte cap ({}/{} bytes) but all entries pinned",
                                  impl_->accountedBytes_,
                                  impl_->byteCap_ );
                    return;
                }
            }

            // Remove the oldest unpinned entry OUTSIDE the mutex (fs work on the
            // caller's worker thread, Pitfall 14). Windows locked-file failures
            // are tolerated: log + skip + retry next pass (P2-3) -- eviction
            // NEVER fails the acquire.
            std::error_code ec;
            fs::remove_all( fs::path( impl_->root_ ) / oldestHex, ec );
            if ( ec )
            {
                logger->warn( "ElmModelCache: eviction of {} failed (will retry next pass): {}", oldestHex, ec.message() );
                return;
            }

            {
                std::lock_guard<std::mutex> lock( impl_->mutex_ );
                const auto                  it = impl_->entries_.find( oldestHex );
                if ( it != impl_->entries_.end() )
                {
                    impl_->accountedBytes_ -= it->second.bytes;
                    impl_->entries_.erase( it );
                }
            }
            logger->info( "ElmModelCache: evicted LRU entry {} (cap {} bytes)", oldestHex, impl_->byteCap_ );
        }
    }

    void ElmModelCache::ReleasePin( const std::string &hashHex )
    {
        std::lock_guard<std::mutex> lock( impl_->mutex_ );
        const auto                  it = impl_->entries_.find( hashHex );
        if ( it == impl_->entries_.end() )
        {
            return; // evicted/quarantined while pinned out of our view: nothing to do
        }
        if ( it->second.pinCount > 0 )
        {
            --it->second.pinCount;
        }
        it->second.lastUse = NowTimeT(); // LRU refresh on release
    }

    void ElmModelCache::SetByteCap( uint64_t byteCap )
    {
        {
            std::lock_guard<std::mutex> lock( impl_->mutex_ );
            impl_->byteCap_ = byteCap;
        }
        // Next eviction pass trims (RQ8): lowering below usage does not evict inline.
        RunEvictionPass();
    }
} // namespace sgns::elmruntime
