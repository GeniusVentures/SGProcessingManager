#ifndef SGPROCMGR_ELMRUNTIME_ARTIFACT_FETCHER_HPP
#define SGPROCMGR_ELMRUNTIME_ARTIFACT_FETCHER_HPP

#include <elmruntime/ElmRuntimeError.hpp>

#include <outcome/sgprocmgr-outcome.hpp>

#include <functional>
#include <string>
#include <vector>

namespace sgns::elmruntime
{
    /// @brief The ONLY fetch seam in the elmruntime layer (D-08, issue #17 mandate).
    ///
    /// Everything the manifest/cache layer fetches (manifest bytes, artifact bytes)
    /// goes through an injectable FetchFn. Production wires MakeFileManagerFetchFn()
    /// exclusively -- FileManager::LoadASync is the only production fetcher; raw
    /// sockets/curl/httplib never appear under src/elmruntime/. Tests inject
    /// in-memory or file://-backed lambdas (no IPFS, no network).
    /// @param uri - URI to fetch (file://, https://, ipfs:// in production)
    /// @return the fetched bytes, or a structured ElmRuntimeError failure
    using FetchFn = std::function<outcome::result<std::vector<uint8_t>>( const std::string &uri )>;

    /// @brief Production FetchFn factory: fetches via FileManager::LoadASync (D-08).
    ///
    /// Per call: InitializeSingletons() (idempotent prefix registration), a fresh
    /// io_context, LoadASync(url, parse=false, save=false, ioc, cb, "file") wrapped
    /// in try/catch -- LoadASync THROWS std::range_error for unregistered prefixes,
    /// which is converted to FETCH_FAILED and never crosses the boundary -- then
    /// ioc->reset(); ioc->run(); drains ON THE CALLING thread (fetch completion and
    /// the caller are the same thread; Pitfall 14). save=false always: the cache does
    /// its own staging writes, never through FileManager.
    /// @return a FetchFn backed by FileManager
    FetchFn MakeFileManagerFetchFn();
} // namespace sgns::elmruntime

#endif // SGPROCMGR_ELMRUNTIME_ARTIFACT_FETCHER_HPP
