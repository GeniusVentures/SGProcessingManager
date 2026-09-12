#ifndef SGPROCMGR_ELMRUNTIME_ELM_STOP_STRING_STREAMBUF_HPP
#define SGPROCMGR_ELMRUNTIME_ELM_STOP_STRING_STREAMBUF_HPP

// ELM stop-string streambuf (elmbridge Phase 3, plan 03-03, D-05/D-06/D-07).
//
// A std::streambuf subclass installed on the std::ostream handed to
// MNN::Transformer::Llm::response(). MNN flushes the ostream after every
// decoded token, so the xsputn/overflow appends see incremental text; after
// each append the D-06 incremental overlap scan checks whether any stop
// string now completes inside the scan window.
//
// This header contains ZERO MNN includes (include-set convention of the
// elmruntime layer): the cancel trigger is an injectable
// std::function<void()> so this unit is testable everywhere. The processor
// wires [llm]() { llm->cancel(); } -- the D-13 fork patch -- as that hook.
//
// Intent disambiguation (D-09): TWO separate latches, never one --
//   stop-string latch: a stop string matched (envelope reports "stop")
//   cancel latch:      the external cancel poll observed job cancellation
//                      (envelope reports "cancelled")
// The processor reads StopStringMatched() vs CancelRequested() to decide.

#include <atomic>
#include <cstddef>
#include <functional>
#include <streambuf>
#include <string>
#include <string_view>
#include <vector>

namespace sgns::elmruntime
{
    /// @brief Stop-string scanning streambuf (D-05/D-06/D-07).
    class ElmStopStringStreamBuf : public std::streambuf
    {
    public:
        /// @brief Construct the observer.
        /// @param stopStrings - the stop list (empty vector = no stop-string
        ///        scanning; every append still accumulates)
        /// @param onCancelMatch - invoked ONCE when a stop string matches (the
        ///        processor passes llm->cancel(); idempotent here)
        ElmStopStringStreamBuf( std::vector<std::string>           stopStrings,
                                std::function<void()>              onCancelMatch );

        /// @brief Whether a stop string has matched (stop-intent latch).
        bool Matched() const;

        /// @brief Offset of the earliest stop-string match within the
        ///        accumulated text (valid only after Matched()).
        std::size_t MatchOffset() const;

        /// @brief The full accumulated text (including any matched stop string).
        const std::string &AccumulatedText() const;

        /// @brief D-07 exclusion view: [0, MatchOffset()) when matched, else the
        ///        whole accumulation (the stop string is NOT included).
        std::string_view VisibleText() const;

        /// @brief Whether the external cancel poll has requested cancellation
        ///        (cancel-intent latch -- SEPARATE from the stop-string latch).
        bool CancelRequested() const;

        /// @brief Wire the external cancel poll (D-09 seam): invoked at every
        ///        append; a true return latches cancel intent and fires
        ///        onCancelMatch() once. The processor wires
        ///        [&execCtx]() { return execCtx.cancelToken.IsCancelled(); } --
        ///        the Pitfall 4 streambuf-poll resolution (cancel latency is
        ///        one token; zero ProcessManager callback changes).
        void SetExternalCancelPoll( std::function<bool()> poll );

    protected:
        std::streamsize xsputn( const char *s, std::streamsize count ) override;
        int_type        overflow( int_type ch ) override;

    private:
        /// Appends bytes then runs the incremental overlap scan.
        void AppendAndScan( const char *s, std::streamsize count );

        /// Fires the cancel hook once (idempotent across both latch paths).
        void FireCancelOnce();

        /// Backs a byte offset off to the previous UTF-8 lead byte so the scan
        /// window never begins mid-sequence.
        static std::size_t BackOffToUtf8LeadByte( const std::string &text, std::size_t offset );

        std::vector<std::string> stopStrings_;
        std::function<void()>    onCancelMatch_;
        std::function<bool()>    externalCancelPoll_;

        std::string        text_;
        std::atomic<bool>  stopStringMatched_{ false };
        std::atomic<bool>  cancelRequested_{ false };
        std::atomic<bool>  cancelFired_{ false };
        std::size_t        matchOffset_ = 0; // guarded by the single generation thread
    };
} // namespace sgns::elmruntime

#endif // SGPROCMGR_ELMRUNTIME_ELM_STOP_STRING_STREAMBUF_HPP
