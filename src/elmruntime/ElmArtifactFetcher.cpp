#include <elmruntime/ElmArtifactFetcher.hpp>

#include <util/sgprocmgr-logger.hpp>

#include "FileManager.hpp"

#include <boost/asio/io_context.hpp>
#include <condition_variable>
#include <memory>
#include <mutex>
#include <optional>
#include <string>
#include <utility>

namespace sgns::elmruntime
{
    namespace
    {
        sgns::sgprocmanager::Logger FetcherLogger()
        {
            return sgns::sgprocmanager::createLogger( "ElmArtifactFetcher" );
        }
    } // namespace

    FetchFn MakeFileManagerFetchFn()
    {
        return []( const std::string &uri ) -> outcome::result<std::vector<uint8_t>> {
            const auto logger = FetcherLogger();

            // Idempotent singleton init: registers the file/https/ipfs loader
            // prefixes (FileManager.cpp:31-41). Safe to call repeatedly.
            FileManager::GetInstance().InitializeSingletons();

            auto ioc = std::make_shared<boost::asio::io_context>();

            std::mutex mutex;
            std::condition_variable completionSignal;
            // ResultType is an outcome::result -- its default constructor is
            // deleted, so hold it in an optional and set it from the callback.
            std::optional<FileManager::ResultType> result;
            bool completed = false;

            auto callback = [&]( FileManager::ResultType buffers ) {
                std::lock_guard<std::mutex> lock( mutex );
                result    = std::move( buffers );
                completed = true;
                completionSignal.notify_all();
            };

            try
            {
                // save=false ALWAYS: the cache does its own staging writes; the
                // fetch seam never writes through FileManager (D-08).
                FileManager::GetInstance().LoadASync( uri,
                                                      /*parse=*/false,
                                                      /*save=*/false,
                                                      ioc,
                                                      callback,
                                                      "file" );
            }
            catch ( const std::exception &e )
            {
                // LoadASync THROWS std::range_error for unregistered prefixes
                // (FileManager.cpp:57-60) -- convert to a structured failure;
                // the throw never crosses the acquire boundary.
                logger->error( "ElmArtifactFetcher: fetch of {} failed to start: {}", uri, e.what() );
                return outcome::failure( ElmRuntimeError::FETCH_FAILED );
            }

            // Drain ON THE CALLING thread (the ProcessingManager.cpp:2167-2168
            // ioc->reset(); ioc->run() pattern): fetch completion and the caller
            // are the same thread -- Pitfall 14 satisfied structurally.
            ioc->reset();
            ioc->run();

            // The drain is synchronous: LoadASync queues onto ioc and run() blocks
            // until the load completes and the callback has fired (the FileManager
            // posts the completion onto the same ioc). completed is therefore true
            // here in practice; the wait below guards the ordering for readers.
            {
                std::unique_lock<std::mutex> lock( mutex );
                completionSignal.wait( lock, [&] { return completed; } );
            }

            if ( !result )
            {
                logger->error( "ElmArtifactFetcher: fetch of {} produced no result", uri );
                return outcome::failure( ElmRuntimeError::FETCH_FAILED );
            }

            if ( !*result )
            {
                logger->error( "ElmArtifactFetcher: fetch of {} failed: {}", uri, result->error().message() );
                return outcome::failure( ElmRuntimeError::FETCH_FAILED );
            }

            const auto &buffers = result->value();
            if ( !buffers || buffers->second.empty() )
            {
                logger->error( "ElmArtifactFetcher: fetch of {} produced no buffers", uri );
                return outcome::failure( ElmRuntimeError::FETCH_FAILED );
            }

            // Single-file fetch: extract the first buffer's bytes.
            const auto &chars = buffers->second.front();
            return std::vector<uint8_t>( chars.begin(), chars.end() );
        };
    }
} // namespace sgns::elmruntime
