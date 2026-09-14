// ELM envelope unit tests (elmbridge Phase 3, plan 03-03, Task 1;
// stamps extended in Phase 4 plan 04-01, D-04).
//
// RES-01 matrix: the serialized envelope JSON carries EXACTLY work_item_id,
// text, prompt_tokens, completion_tokens, finish_reason, model_manifest_hash,
// grab_time_usec, finish_time_usec -- with the nested error {code,message}
// object present ONLY when finish_reason == error. All four finish reasons +
// the string round-trip + the D-04 settlement-stamp carriage.

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

// Exact key set: eight keys (six + the two D-04 stamps), no error key on a
// non-error finish reason.
TEST( ElmEnvelopeTest, KeySetExactOnStop )
{
    const auto        json   = nlohmann::json::parse( ElmEnvelopeToJson( MakeEnvelope( ElmFinishReason::Stop ) ) );
    const std::string finish = json.at( "finish_reason" ).get<std::string>();
    EXPECT_EQ( finish, "stop" );
    EXPECT_EQ( json.size(), 8 );
    EXPECT_FALSE( json.contains( "error" ) );
    EXPECT_EQ( json.at( "work_item_id" ).get<std::string>(), "work-item-01" );
    EXPECT_EQ( json.at( "text" ).get<std::string>(), "generated text" );
    EXPECT_TRUE( json.at( "prompt_tokens" ).is_number() );
    EXPECT_EQ( json.at( "prompt_tokens" ).get<int64_t>(), 12 );
    EXPECT_TRUE( json.at( "completion_tokens" ).is_number() );
    EXPECT_EQ( json.at( "completion_tokens" ).get<int64_t>(), 7 );
    EXPECT_EQ( json.at( "model_manifest_hash" ).get<std::string>(),
               "0123456789abcdef0123456789abcdef0123456789abcdef0123456789abcdef" );
    // D-04 stamps: numeric keys, present ALWAYS (defaults serialize as 0).
    EXPECT_TRUE( json.at( "grab_time_usec" ).is_number() );
    EXPECT_TRUE( json.at( "finish_time_usec" ).is_number() );
    EXPECT_EQ( json.at( "grab_time_usec" ).get<int64_t>(), 0 );
    EXPECT_EQ( json.at( "finish_time_usec" ).get<int64_t>(), 0 );
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

// Error finish reason: the nested {code,message} object is present AND the
// D-04 stamps ride the error envelope too (the settlement closes the window
// at finish even on failure).
TEST( ElmEnvelopeTest, ErrorDetailPresentOnError )
{
    const auto json = nlohmann::json::parse( ElmEnvelopeToJson( MakeEnvelope( ElmFinishReason::Error ) ) );
    EXPECT_EQ( json.at( "finish_reason" ).get<std::string>(), "error" );
    ASSERT_TRUE( json.contains( "error" ) );
    EXPECT_EQ( json.size(), 9 );
    EXPECT_EQ( json.at( "error" ).at( "code" ).get<std::string>(), "MANIFEST_HASH_MISMATCH" );
    EXPECT_EQ( json.at( "error" ).at( "message" ).get<std::string>(), "declared != actual" );
    EXPECT_TRUE( json.at( "grab_time_usec" ).is_number() );
    EXPECT_TRUE( json.at( "finish_time_usec" ).is_number() );
}

// D-04 (04-01): worker-attested settlement stamps round-trip through the JSON
// wire form with their values intact.
TEST( ElmEnvelopeTest, SettlementStampsRoundTrip )
{
    ElmEnvelope envelope      = MakeEnvelope( ElmFinishReason::MaxTokens );
    envelope.grab_time_usec   = 1726325698000001;
    envelope.finish_time_usec = 1726325704123456;
    const auto json = nlohmann::json::parse( ElmEnvelopeToJson( envelope ) );
    EXPECT_EQ( json.at( "grab_time_usec" ).get<int64_t>(), 1726325698000001 );
    EXPECT_EQ( json.at( "finish_time_usec" ).get<int64_t>(), 1726325704123456 );
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
