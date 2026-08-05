#pragma once
/**
 * Execution context types for Phase 07: Cancellable Execution Context.
 *
 * Defines CancellationToken (callback-based cooperative cancellation),
 * ExecutionContext (bundles cancel token, progress callback, deadline, budgets),
 * ProgressEvent (stage-boundary progress event), and standardized stage enums
 * per processor type (RenderStage, MNNStage).
 *
 * @brief Execution context data contracts
 */
#ifndef SGPROCMGR_EXECUTION_CONTEXT_HPP
#define SGPROCMGR_EXECUTION_CONTEXT_HPP

#include <atomic>
#include <cstdint>
#include <functional>
#include <string>

namespace sgns::sgprocessing
{

    /// Standardized pipeline stages for RenderProcessor (D-12).
    /// Matches the four coarse checkpoints in D-04 for RenderProcessor.
    enum class RenderStage
    {
        COMPILE        = 0,  ///< Shader compilation
        BUILD_PIPELINE = 1,  ///< Pipeline creation
        DRAW           = 2,  ///< Draw submission
        READBACK       = 3   ///< Readback from framebuffer
    };

    /// Standardized pipeline stages for MNN processors (D-12).
    /// Matches the coarse checkpoints in D-04 for MNN processors.
    enum class MNNStage
    {
        LOAD_MODEL     = 0,  ///< Model file loaded / MNN interpreter created
        CREATE_SESSION = 1,  ///< MNN session created
        RUN            = 2,  ///< Inference executed
        READ_OUTPUT    = 3   ///< Output tensor read
    };

    /// Minimal progress event fired at every stage boundary (D-11, D-13).
    /// Carries pass_id, stage name, and percent (0–100 float).
    struct ProgressEvent
    {
        std::string pass_id;               ///< Pass name from schema
        RenderStage render_stage = RenderStage::COMPILE;  ///< Populated for render passes
        MNNStage    mnn_stage    = MNNStage::LOAD_MODEL;  ///< Populated for MNN passes
        float       percent      = 0.0f;   ///< 0.0–100.0, cumulative progress estimate

        /// Factory for render pass progress events.
        static ProgressEvent ForRender( std::string passId, RenderStage stage, float pct )
        {
            ProgressEvent ev;
            ev.pass_id      = std::move( passId );
            ev.render_stage = stage;
            ev.percent      = pct;
            return ev;
        }

        /// Factory for MNN pass progress events.
        static ProgressEvent ForMNN( std::string passId, MNNStage stage, float pct )
        {
            ProgressEvent ev;
            ev.pass_id   = std::move( passId );
            ev.mnn_stage = stage;
            ev.percent   = pct;
            return ev;
        }
    };

    /// Callback-based cooperative cancellation token (D-01, D-02, D-05).
    ///
    /// Thread-safe: Cancel() may be called from deadline timer thread while
    /// IsCancelled() is read on the processing thread. Uses std::atomic<bool>
    /// with acquire/release ordering.
    struct CancellationToken
    {
        CancellationToken() = default;
        CancellationToken( const CancellationToken & )            = delete;
        CancellationToken &operator=( const CancellationToken & ) = delete;
        CancellationToken( CancellationToken && )                 = delete;
        CancellationToken &operator=( CancellationToken && )      = delete;

        /// Invokes the registered cancel callback (if set) and sets the cancelled flag.
        /// The callback is invoked synchronously, at most once.
        void Cancel()
        {
            bool expected = false;
            if ( m_cancelled.compare_exchange_strong( expected, true,
                  std::memory_order_release, std::memory_order_acquire ) )
            {
                if ( m_cancelCallback )
                {
                    m_cancelCallback();
                }
            }
        }

        /// Returns true after Cancel() has been called.
        bool IsCancelled() const
        {
            return m_cancelled.load( std::memory_order_acquire );
        }

        /// Register the cancel callback. Called by ProcessingManager before
        /// passing the token to the processor.
        void SetCallback( std::function<void()> callback )
        {
            m_cancelCallback = std::move( callback );
        }

    private:
        std::function<void()> m_cancelCallback;
        std::atomic<bool>     m_cancelled{ false };
    };

    /// Bundles everything a processor needs for an execution (D-01, D-02, D-06, D-08, D-10).
    ///
    /// One ExecutionContext per job (D-02), shared across all passes in the job's pass graph.
    struct ExecutionContext
    {
        ExecutionContext() = default;
        ExecutionContext( const ExecutionContext & )            = delete;
        ExecutionContext &operator=( const ExecutionContext & ) = delete;
        ExecutionContext( ExecutionContext && )                 = delete;
        ExecutionContext &operator=( ExecutionContext && )      = delete;

        CancellationToken                     cancelToken;           ///< Per-job cancellation token (D-02)
        std::function<void( ProgressEvent )>   progressCallback;     ///< Processor calls at stage boundaries (D-10)
        uint64_t                              deadlineMs            = 0;  ///< Per-pass wall-clock deadline in ms; 0 = no deadline (D-08)
        uint64_t                              gpuMemoryBudget       = 0;  ///< Estimated GPU memory in bytes; 0 = no budget (D-08)
        uint64_t                              maxOutputArtifactBytes = 0; ///< Max output artifact size in bytes; 0 = no budget (D-08)

        /// Returns a fully no-op ExecutionContext as a heap-allocated unique_ptr.
        /// Used in tests. ExecutionContext is non-copyable, non-movable due to
        /// CancellationToken containing std::atomic<bool>.
        static std::unique_ptr<ExecutionContext> NoOp()
        {
            auto ctx = std::make_unique<ExecutionContext>();
            ctx->cancelToken.SetCallback( []() {} );
            ctx->progressCallback = []( const ProgressEvent & ) {};
            return ctx;
        }
    };

} // namespace sgns::sgprocessing

#endif // SGPROCMGR_EXECUTION_CONTEXT_HPP
