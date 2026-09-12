#ifndef SGPROCMGR_ELMRUNTIME_ELM_ENVELOPE_HPP
#define SGPROCMGR_ELMRUNTIME_ELM_ENVELOPE_HPP

// ELM result envelope (elmbridge Phase 3, plan 03-03, RES-01).
//
// A pure unit with ZERO MNN includes (the ElmRuntimeError/ElmResourcePreflight
// include-set convention): testable in every checkout, consumed by the gated
// processor TU. The envelope is the single result shape every ELM work item
// publishes -- the output buffer content the processor produces; its transport
// into SubTask results is Phase 4's, not this layer's.

#include <cstdint>
#include <optional>
#include <string>

namespace sgns::elmruntime
{
    /// @brief Finish reason for a work item (SC-1's enum; D-08..D-11 mapping target).
    ///
    /// Exactly four values. MNN's TIMEOUT maps to Error (D-08: deadline firing
    /// cancels via the job's cancel token -> Cancelled; TIMEOUT means an internal
    /// bound fired unexpectedly -> an execution fault). No fifth enum value.
    enum class ElmFinishReason : uint8_t
    {
        Stop = 0,     ///< EOS stop token, or a stop-string match (processor intent, D-09)
        MaxTokens,    ///< max_output_tokens exhausted
        Cancelled,    ///< job cancel / deadline (partial text + measured counts, D-10)
        Error         ///< internal failure: code + message detail (D-11)
    };

    /// @brief Exact wire strings for ElmFinishReason (SC-1: "stop" | "max_tokens" |
    ///        "cancelled" | "error").
    /// @param reason - the enum value
    /// @return the wire string
    const char *ToString( ElmFinishReason reason );

    /// @brief Parse a wire string back to ElmFinishReason.
    /// @param text - one of the four exact wire strings
    /// @return the enum value, or std::nullopt on any other input
    std::optional<ElmFinishReason> ElmFinishReasonFromString( const std::string &text );

    /// @brief Error detail carried ONLY when finish_reason == Error (D-11).
    struct ElmEnvelopeError
    {
        std::string code;     ///< ElmRuntimeError / ProcessingErrorStage category name
        std::string message;  ///< operational detail (no secrets exist on this path)
    };

    /// @brief The per-work-item result envelope (RES-01).
    ///
    /// Token counts are executor-side and tokenizer-relative to
    /// model_manifest_hash (Pitfall 6: the chat template inflates prompt tokens
    /// vs requestor expectations; no requestor-side equality assertions exist
    /// or should).
    struct ElmEnvelope
    {
        std::string                     work_item_id;       ///< schema-charset id, carried as data
        std::string                     text;               ///< VisibleText on stop-string match (D-07), else full
        int64_t                         prompt_tokens = 0;  ///< LlmContext::prompt_len
        int64_t                         completion_tokens = 0; ///< LlmContext::output_tokens.size() (the authority)
        ElmFinishReason                 finish_reason = ElmFinishReason::Error;
        std::string                     model_manifest_hash; ///< provenance: ElmCachePin::GetHash() (SC-4)
        std::optional<ElmEnvelopeError> error;              ///< present only when finish_reason == Error
    };

    /// @brief Serialize an envelope to its canonical JSON wire form.
    ///
    /// Exact key set: work_item_id, text, prompt_tokens, completion_tokens,
    /// finish_reason, model_manifest_hash -- plus a nested error {code,message}
    /// object ONLY when finish_reason == Error (D-11's detail field; the key is
    /// absent otherwise). Counts serialize as JSON numbers.
    /// @param envelope - the envelope to serialize
    /// @return the JSON document as a string
    std::string ElmEnvelopeToJson( const ElmEnvelope &envelope );
} // namespace sgns::elmruntime

#endif // SGPROCMGR_ELMRUNTIME_ELM_ENVELOPE_HPP
