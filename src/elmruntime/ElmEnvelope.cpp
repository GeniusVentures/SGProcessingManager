#include <elmruntime/ElmEnvelope.hpp>

#include <nlohmann/json.hpp>

namespace sgns::elmruntime
{
    const char *ToString( ElmFinishReason reason )
    {
        switch ( reason )
        {
            case ElmFinishReason::Stop:      return "stop";
            case ElmFinishReason::MaxTokens: return "max_tokens";
            case ElmFinishReason::Cancelled: return "cancelled";
            case ElmFinishReason::Error:     return "error";
        }
        return "error"; // unreachable for the closed enum; fail closed
    }

    std::optional<ElmFinishReason> ElmFinishReasonFromString( const std::string &text )
    {
        if ( text == "stop" )
        {
            return ElmFinishReason::Stop;
        }
        if ( text == "max_tokens" )
        {
            return ElmFinishReason::MaxTokens;
        }
        if ( text == "cancelled" )
        {
            return ElmFinishReason::Cancelled;
        }
        if ( text == "error" )
        {
            return ElmFinishReason::Error;
        }
        return std::nullopt;
    }

    std::string ElmEnvelopeToJson( const ElmEnvelope &envelope )
    {
        nlohmann::json doc;
        doc["work_item_id"]        = envelope.work_item_id;
        doc["text"]                = envelope.text;
        doc["prompt_tokens"]       = envelope.prompt_tokens;
        doc["completion_tokens"]   = envelope.completion_tokens;
        doc["finish_reason"]       = ToString( envelope.finish_reason );
        doc["model_manifest_hash"] = envelope.model_manifest_hash;
        if ( envelope.finish_reason == ElmFinishReason::Error && envelope.error.has_value() )
        {
            doc["error"] = { { "code", envelope.error->code }, { "message", envelope.error->message } };
        }
        return doc.dump();
    }
} // namespace sgns::elmruntime
