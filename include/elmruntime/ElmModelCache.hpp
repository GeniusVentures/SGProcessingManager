#ifndef SGPROCMGR_ELMRUNTIME_MODEL_CACHE_HPP
#define SGPROCMGR_ELMRUNTIME_MODEL_CACHE_HPP

#include <elmruntime/ElmArtifactFetcher.hpp>
#include <elmruntime/ElmRuntimeError.hpp>
#include <elmruntime/ElmSmokeCheck.hpp>

#include <outcome/sgprocmgr-outcome.hpp>

#include <cstdint>
#include <filesystem>
#include <memory>
#include <string>

namespace sgns::elmruntime
{
    /// @brief Default cache byte cap (D-04): mid-range of the 10-20 GB planning
    ///        range (A4) -- 15 GiB, uint64_t bytes throughout (no float GB math).
    inline constexpr uint64_t kDefaultElmCacheByteCap = 15ULL * 1024 * 1024 * 1024;

    class ElmModelCache;

    /// @brief Move-only RAII pin on a published cache entry (the Phase 3 handle).
    ///
    /// A pinned entry is never removed by the eviction pass. Destructor decrements
    /// the pin refcount (mutex-guarded) and refreshes the entry's LRU timestamp --
    /// every terminal path of Acquire runs it, giving deterministic unpin including
    /// cancellation paths in the caller.
    class ElmCachePin
    {
    public:
        ElmCachePin() = default;
        ~ElmCachePin();

        ElmCachePin( const ElmCachePin & )            = delete;
        ElmCachePin &operator=( const ElmCachePin & ) = delete;

        ElmCachePin( ElmCachePin &&other ) noexcept;
        ElmCachePin &operator=( ElmCachePin &&other ) noexcept;

        /// @brief Entry directory WITH a trailing separator -- hand this string
        ///        directly to MNN::Transformer::Llm::createLLM (P2-5).
        /// @return the bundle directory path, trailing-slash terminated
        const std::string &GetDir() const
        {
            return dir_;
        }

        /// @brief The manifest sha256 hex digest (the entry's directory name).
        /// @return 64 lowercase hex characters
        const std::string &GetHash() const
        {
            return hashHex_;
        }

        /// @brief Whether this pin actually references an entry.
        /// @return true after a successful Acquire until moved-from/released
        explicit operator bool() const
        {
            return cache_ != nullptr;
        }

    private:
        friend class ElmModelCache;

        ElmCachePin( std::string dir, std::string hashHex, std::shared_ptr<ElmModelCache> cache );

        std::string                  dir_;    ///< trailing-slash entry directory
        std::string                  hashHex_; ///< 64-hex manifest digest
        std::shared_ptr<ElmModelCache> cache_;  ///< back-pointer for release
    };

    /// @brief Content-addressed model bundle cache (MCHE-02/MCHE-03).
    ///
    /// Entry layout under <cacheRootDir>/:
    ///   <64-hex-manifest-digest>/   published entry == MNN LlmConfig.base_dir
    ///       elm_manifest.json       verified manifest bytes verbatim (D-01/D-06 source)
    ///       llm_config.json, llm.mnn, llm.mnn.weight, tokenizer.txt[, context.json]
    ///   .tmp-<uuid>/                staging -- swept at construction; never loaded
    ///   .bad-<digest>/              quarantine -- kept forever (D-07), excluded from
    ///                               accounting, never loaded
    ///
    /// Publish path (miss): stage -> fetch manifest -> hash-verify (SC-1 gate) ->
    /// per-artifact fetch + sha256 verify + write -> write elm_manifest.json ->
    /// smoke check (pre-rename, A5) -> atomic rename -> eviction pass.
    /// Hit path (D-01): read elm_manifest.json, stat each role file against the
    /// declared size_bytes (NO sha256 re-hash on hit); any mismatch quarantines
    /// (.bad-<hex>) and fails with CACHE_ENTRY_QUARANTINED -- no auto-refetch in
    /// the failing call (D-02).
    ///
    /// Threading: Acquire is SYNCHRONOUS on the caller's (worker) thread; a
    /// second concurrent Acquire of the same manifest joins the first's
    /// shared_future (single-flight, SC-4). v1 has no posted callbacks/timers --
    /// but the class derives enable_shared_from_this and any FUTURE posted
    /// callback must capture weak_from_this (Pitfall 15.1 discipline).
    class ElmModelCache : public std::enable_shared_from_this<ElmModelCache>
    {
    public:
        /// @brief Construct via factory only (shared_from_this safe from birth --
        ///        Pitfall 15.1 discipline).
        /// @param cacheRootDir - cache root directory; empty fails CACHE_DIR_UNSET (D-03)
        /// @param fetch - injectable fetch seam (production: MakeFileManagerFetchFn, D-08)
        /// @param smokeCheck - loadability probe run pre-rename against staging
        /// @param byteCap - eviction cap in bytes (D-04; default 15 GiB)
        /// @return the cache, or an ElmRuntimeError failure
        static outcome::result<std::shared_ptr<ElmModelCache>> Create( const std::string &cacheRootDir,
                                                                       FetchFn            fetch,
                                                                       SmokeCheckFn       smokeCheck,
                                                                       uint64_t           byteCap = kDefaultElmCacheByteCap );

        /// @brief Production factory: root = FileManager::getCacheDir() + "/elmruntime".
        ///
        /// Fail-closed on an empty cache dir -- NEVER guesses a default (D-03).
        /// This is the ONLY function in the library that touches getCacheDir()
        /// (P2-1: unit tests inject a root via Create instead).
        /// @param smokeCheck - loadability probe (MakeMnnLlmSmokeCheck in production)
        /// @return the cache wired with MakeFileManagerFetchFn(), or CACHE_DIR_UNSET
        static outcome::result<std::shared_ptr<ElmModelCache>> CreateProductionElmModelCache( SmokeCheckFn smokeCheck );

        ElmModelCache( const ElmModelCache & )            = delete;
        ElmModelCache &operator=( const ElmModelCache & ) = delete;

        /// @brief Acquire a usable bundle for a manifest (synchronous, worker thread).
        ///
        /// Hit (entry exists + sizes match): pin + return, zero fetches.
        /// Miss: single-flighted download through the stage->verify->smoke->rename
        /// pipeline. Reuse-time mismatch: quarantine + CACHE_ENTRY_QUARANTINED
        /// (retry re-downloads cleanly -- SC-3).
        /// @param manifestUri - URI passed to the FetchFn for the manifest bytes
        /// @param manifestHash - DECLARED model_manifest_hash (sha256:<hex> or bare);
        ///        only ever COMPARED post-normalization, never used as a path (P2-4)
        /// @return a pin, or a structured ElmRuntimeError
        outcome::result<ElmCachePin> Acquire( const std::string &manifestUri, const std::string &manifestHash );

        /// @brief Adjust the eviction byte cap (RQ8). No-op semantics: lowering below
        ///        current usage does not evict inline -- the next eviction pass
        ///        (publish or a later SetByteCap call) trims to the new cap.
        /// @param byteCap - new cap in bytes
        void SetByteCap( uint64_t byteCap );

    private:
        ElmModelCache( std::string cacheRootDir, FetchFn fetch, SmokeCheckFn smokeCheck, uint64_t byteCap );

        struct EntryState
        {
            uint64_t    pinCount = 0;
            uint64_t    bytes    = 0;
            std::time_t lastUse  = 0; ///< LRU ordering (directory mtime mirror)
        };

        friend class ElmCachePin;

        /// Called by ~ElmCachePin: decrement refcount + refresh LRU timestamp.
        void ReleasePin( const std::string &hashHex );

        /// Hit path (D-01 size check) when the entry dir exists, else the full
        /// stage->verify->smoke->rename publish. joinCompletedPublish=true is the
        /// waiter's post-single-flight re-entry (pins the same entry).
        outcome::result<ElmCachePin> TryHitOrPublish( const std::string &manifestUri,
                                                      const std::string &declaredHex,
                                                      bool               joinCompletedPublish );

        /// Quarantine an existing entry (D-02): rename to .bad-<hex>, telemetry,
        /// drop from tracking, return CACHE_ENTRY_QUARANTINED.
        outcome::result<ElmCachePin> QuarantineEntry( const std::string &hashHex, const std::string &reason );

        /// Rename an existing entry dir to .bad-<hex> (removing a stale quarantine
        /// first -- Windows P2-2). Logs the D-07 telemetry line.
        void QuarantineExistingDir( const std::string &hashHex, const std::filesystem::path &entryDir,
                                    const std::string &reason );

        /// LRU eviction pass (D-04): while accounted bytes exceed the cap, remove
        /// the oldest UNPINNED entry; fs failures are tolerated (P2-3).
        void RunEvictionPass();

        class Impl;
        std::unique_ptr<Impl> impl_;
    };
} // namespace sgns::elmruntime

#endif // SGPROCMGR_ELMRUNTIME_MODEL_CACHE_HPP
