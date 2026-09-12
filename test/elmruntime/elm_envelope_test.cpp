// ELM envelope unit tests (elmbridge Phase 3, plan 03-03, Task 1).
//
// RES-01 matrix: the serialized envelope JSON carries EXACTLY work_item_id,
// text, prompt_tokens, completion_tokens, finish_reason, model_manifest_hash
// -- with the nested error {code,message} object present ONLY when
// finish_reason == error. All four finish reasons + the string round-trip.

#include <gtest/gtest.h>

#include <elmruntime/ElmEnvelope.hpp>

#include <nlohmann/json.hpp>

#include <string>

namespace
{
    using sgns::elmruntime::ElmEnvelope;
    using sgns::elmruntime::ElmEnvelopeError;
    using sgns::elmruntime::ElmEnvelopeToJson;
    using sgns::elmruntime::ElmFinishReason;
    using sgns::elmruntime::ElmFinishReasonFromString;
    using sgns::elmruntime::ToString;

    ElmEnvelope MakeEnvelope( ElmFinishReason reason )
    {
        ElmEnvelope envelope;
        envelope.work_item_id        = "work-item-01";
        envelope.text                = "generated text";
        envelope.prompt_tokens      = 12;
        envelope.completion_tokens  = 7;
        envelope.finish_reason      = reason;
        envelope.model_manifest_hash = "0123456789abcdef0123456789abcdef0123456789abcdef0123456789abcdef";
        if ( reason == ElmFinishReason::Error )
        {
            envelope.error = ElmEnvelopeError{ "MANIFEST_HASH_MISMATCH", "declared != actual" };
        }
        return envelope;
    }
} // namespace

// Exact key set: six keys, no error key on a non-error finish reason.
TEST( ElmEnvelopeTest, KeySetExactOnStop )
{
    const auto        json   = nlohmann::json::parse( ElmEnvelopeToJson( MakeEnvelope( ElmFinishReason::Stop ) ) );
    const std::string finish = json.at( "finish_reason" ).get<std::string>();
    EXPECT_EQ( finish, "stop" );
    EXPECT_EQ( json.size(), 6 );
    EXPECT_FALSE( json.contains( "error" ) );
    EXPECT_EQ( json.at( "work_item_id" ).get<std::string>(), "work-item-01" );
    EXPECT_EQ( json.at( "text" ).get<std::string>(), "generated text" );
    EXPECT_TRUE( json.at( "prompt_tokens" ).is_number() );
    EXPECT_EQ( json.at( "prompt_tokens" ).get<int64_t>(), 12 );
    EXPECT_TRUE( json.at( "completion_tokens" ).is_number() );
    EXPECT_EQ( json.at( "completion_tokens" ).get<int64_t>(), 7 );
    EXPECT_EQ( json.at( "model_manifest_hash" ).get<std::string>(),
               "0123456789abcdef0123456789abcdef0123456789abcdef0123456789abcdef" );
}

TEST( ElmEnvelopeTest, NoErrorKeyOnMaxTokens )
{
    const auto json = nlohmann::json::parse( ElmEnvelopeToJson( MakeEnvelope( ElmFinishReason::MaxTokens ) ) );
    EXPECT_EQ( json.at( "finish_reason" ).get<std::string>(), "max_tokens" );
    EXPECT_FALSE( json.contains( "error" ) );
}

TEST( ElmEnvelopeTest, NoErrorKeyOnCancelled )
{
    const auto json = nlohmann::json::parse( ElmEnvelopeToJson( MakeEnvelope( ElmFinishReason::Cancelled ) ) );
    EXPECT_EQ( json.at( "finish_reason" ).get<std::string>(), "cancelled" );
    EXPECT_FALSE( json.contains( "error" ) );
}

// Error finish reason: the nested {code,message} object is present.
TEST( ElmEnvelopeTest, ErrorDetailPresentOnError )
{
    const auto json = nlohmann::json::parse( ElmEnvelopeToJson( MakeEnvelope( ElmFinishReason::Error ) ) );
    EXPECT_EQ( json.at( "finish_reason" ).get<std::string>(), "error" );
    ASSERT_TRUE( json.contains( "error" ) );
    EXPECT_EQ( json.size(), 7 );
    EXPECT_EQ( json.at( "error" ).at( "code" ).get<std::string>(), "MANIFEST_HASH_MISMATCH" );
    EXPECT_EQ( json.at( "error" ).at( "message" ).get<std::string>(), "declared != actual" );
}

// Wire string round-trip: exactly the four values, both directions.
TEST( ElmEnvelopeTest, FinishReasonStringRoundTrip )
{
    EXPECT_STREQ( ToString( ElmFinishReason::Stop ), "stop" );
    EXPECT_STREQ( ToString( ElmFinishReason::MaxTokens ), "max_tokens" );
    EXPECT_STREQ( ToString( ElmFinishReason::Cancelled ), "cancelled" );
    EXPECT_STREQ( ToString( ElmFinishReason::Error ), "error" );

    EXPECT_EQ( ElmFinishReasonFromString( "stop" ), ElmFinishReason::Stop );
    EXPECT_EQ( ElmFinishReasonFromString( "max_tokens" ), ElmFinishReason::MaxTokens );
    EXPECT_EQ( ElmFinishReasonFromString( "cancelled" ), ElmFinishReason::Cancelled );
    EXPECT_EQ( ElmFinishReasonFromString( "error" ), ElmFinishReason::Error );

    EXPECT_FALSE( ElmFinishReasonFromString( "STOP" ).has_value() );
    EXPECT_FALSE( ElmFinishReasonFromString( "timeout" ).has_value() );
    EXPECT_FALSE( ElmFinishReasonFromString( "" ).has_value() );
}
