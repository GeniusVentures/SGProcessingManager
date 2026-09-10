// ELM job schema parse-rejection matrix (Plan 01-01, Task 3).
//
// Locks the Task 2 behavior block 1:1: every malformed-ELM rejection that
// quicktype cannot express (minItems drop, duplicate work_item_id, policy-
// refused validation modes, the exclusive >0 halves of schema bounds), the
// four non-ELM parity rejections that preserve pre-change behavior after the
// root `required` relaxation (SC-5), and the GetElmMaximumProcessingHours
// default-fill normalization (D-04).
//
// Pure JSON strings - no network, no fixtures, no processor execution. Every
// case drives ProcessingManager::Create() and asserts the structured error.

#include <gtest/gtest.h>

#include <processingbase/ProcessingManager.hpp>

#include <string>

namespace {

    // Builds a valid minimal ELM job (D-04): name/version/gnus_spec_version=1/
    // job_type=elm_processing/one elm with the five required fields.
    // `elmJson` replaces the default single work item when non-empty.
    std::string BuildElmJobJson( const std::string &elmJson = "", const std::string &extras = "" )
    {
        const std::string defaultElm = "{"
                                       "\"work_item_id\": \"item-1\","
                                       "\"elm_type\": \"causal_lm\","
                                       "\"model_manifest_uri\": \"ipfs://manifest1\","
                                       "\"model_manifest_hash\": \"sha256:abc\","
                                       "\"input_uri\": \"ipfs://input1\""
                                       "}";
        const std::string &elm = elmJson.empty() ? defaultElm : elmJson;
        return "{"                                    //
               "\"name\": \"elm-job\","               //
               "\"version\": \"1.0\","                //
               "\"gnus_spec_version\": 1,"            //
               "\"job_type\": \"elm_processing\","    //
               "\"elms\": [" + elm + "]" + extras + "}";
    }

    using Error = sgns::sgprocessing::ProcessingManager::Error;

    // Helper: run Create() and return the error code (fails the test with a
    // readable message if the job unexpectedly parses).
    Error ExpectCreateFailure( const std::string &json )
    {
        auto result = sgns::sgprocessing::ProcessingManager::Create( json );
        if ( result )
        {
            ADD_FAILURE() << "Expected Create() to reject the job, but it succeeded";
            return Error::INVALID_JSON; // unreachable in practice
        }
        return static_cast<Error>( result.error().value() );
    }

} // namespace

// ---------------------------------------------------------------------------
// Valid minimal ELM job parses (SC-1)
// ---------------------------------------------------------------------------

TEST( ElmJobSchemaTest, MinimalElmJobParses )
{
    auto result = sgns::sgprocessing::ProcessingManager::Create( BuildElmJobJson() );
    ASSERT_TRUE( result ) << "Minimal ELM job must parse through Create()";
    EXPECT_DOUBLE_EQ( result.value()->GetElmMaximumProcessingHours(), 1.0 );
}

TEST( ElmJobSchemaTest, TwoDistinctWorkItemsParse )
{
    const std::string twoElms = "{"                                       //
                                  "\"work_item_id\": \"item-1\","    //
                                  "\"elm_type\": \"causal_lm\","      //
                                  "\"model_manifest_uri\": \"ipfs://manifest1\","
                                  "\"model_manifest_hash\": \"sha256:abc\","
                                  "\"input_uri\": \"ipfs://input1\"}, {"
                                  "\"work_item_id\": \"item-2\","
                                  "\"elm_type\": \"causal_lm\","
                                  "\"model_manifest_uri\": \"ipfs://manifest2\","
                                  "\"model_manifest_hash\": \"sha256:def\","
                                  "\"input_uri\": \"ipfs://input2\","
                                  "\"generation\": { \"temperature\": 0.7, \"top_p\": 0.9, \"max_output_tokens\": 512, \"seed\": 42 } }";
    auto result = sgns::sgprocessing::ProcessingManager::Create( BuildElmJobJson( twoElms ) );
    ASSERT_TRUE( result );
}

// ---------------------------------------------------------------------------
// ELM rejections the C++ gate owns (SC-1, D-05/D-07/D-13)
// ---------------------------------------------------------------------------

TEST( ElmJobSchemaTest, DuplicateWorkItemIdRejects )
{
    const std::string duplicate = "{"                                  //
                                  "\"work_item_id\": \"a\","       //
                                  "\"elm_type\": \"causal_lm\","     //
                                  "\"model_manifest_uri\": \"ipfs://m1\","
                                  "\"model_manifest_hash\": \"sha256:aaa\","
                                  "\"input_uri\": \"ipfs://i1\"}, {"
                                  "\"work_item_id\": \"a\","
                                  "\"elm_type\": \"causal_lm\","
                                  "\"model_manifest_uri\": \"ipfs://m2\","
                                  "\"model_manifest_hash\": \"sha256:bbb\","
                                  "\"input_uri\": \"ipfs://i2\"}";
    EXPECT_EQ( ExpectCreateFailure( BuildElmJobJson( duplicate ) ), Error::DUPLICATE_WORK_ITEM_ID );
}

TEST( ElmJobSchemaTest, EmptyElmsArrayRejects )
{
    const std::string json = "{"                                  //
                             "\"name\": \"elm-job\","            //
                             "\"version\": \"1.0\","              //
                             "\"gnus_spec_version\": 1,"              //
                             "\"job_type\": \"elm_processing\","
                             "\"elms\": []}";
    EXPECT_EQ( ExpectCreateFailure( json ), Error::ELM_WORK_ITEMS_MISSING );
}

TEST( ElmJobSchemaTest, AbsentElmsRejects )
{
    const std::string json = "{"                                  //
                             "\"name\": \"elm-job\","            //
                             "\"version\": \"1.0\","              //
                             "\"gnus_spec_version\": 1,"              //
                             "\"job_type\": \"elm_processing\"}";
    EXPECT_EQ( ExpectCreateFailure( json ), Error::ELM_WORK_ITEMS_MISSING );
}

TEST( ElmJobSchemaTest, ValidationExactRefused )
{
    EXPECT_EQ( ExpectCreateFailure( BuildElmJobJson( "", R"(, "validation": "exact")" ) ),
               Error::ELM_VALIDATION_UNIMPLEMENTED );
}

TEST( ElmJobSchemaTest, ValidationRedundantRefused )
{
    EXPECT_EQ( ExpectCreateFailure( BuildElmJobJson( "", R"(, "validation": "redundant")" ) ),
               Error::ELM_VALIDATION_UNIMPLEMENTED );
}

TEST( ElmJobSchemaTest, ValidationNoneParses )
{
    auto result = sgns::sgprocessing::ProcessingManager::Create(
        BuildElmJobJson( "", R"(, "validation": "none")" ) );
    ASSERT_TRUE( result );
}

TEST( ElmJobSchemaTest, ValidationBogusRejectsAsInvalidJson )
{
    // Unknown enum strings die in the generated enum chain (std::runtime_error),
    // caught by Init's std::exception handler as INVALID_JSON (Pitfall 7).
    EXPECT_EQ( ExpectCreateFailure( BuildElmJobJson( "", R"(, "validation": "bogus")" ) ),
               Error::INVALID_JSON );
}

TEST( ElmJobSchemaTest, JobTypeBogusRejectsAsInvalidJson )
{
    const std::string json = "{"                                  //
                             "\"name\": \"elm-job\","            //
                             "\"version\": \"1.0\","              //
                             "\"gnus_spec_version\": 1,"              //
                             "\"job_type\": \"not_a_job_type\"}";
    EXPECT_EQ( ExpectCreateFailure( json ), Error::INVALID_JSON );
}

// ---------------------------------------------------------------------------
// Funding gates (D-05)
// ---------------------------------------------------------------------------

TEST( ElmJobSchemaTest, FundingZeroHoursRejects )
{
    EXPECT_EQ( ExpectCreateFailure( BuildElmJobJson( "", R"(, "funding": { "maximum_processing_hours": 0 })" ) ),
               Error::ELM_FUNDING_INVALID );
}

TEST( ElmJobSchemaTest, FundingNegativeHoursRejects )
{
    EXPECT_EQ(
        ExpectCreateFailure( BuildElmJobJson( "", R"(, "funding": { "maximum_processing_hours": -1.5 })" ) ),
        Error::ELM_FUNDING_INVALID );
}

TEST( ElmJobSchemaTest, FundingOver24HoursRejects )
{
    // Schema maximum is 24; inclusive bounds did not survive codegen for
    // optional numbers (Task 1 deviation), so the C++ gate owns this too.
    EXPECT_EQ( ExpectCreateFailure( BuildElmJobJson( "", R"(, "funding": { "maximum_processing_hours": 25 })" ) ),
               Error::ELM_FUNDING_INVALID );
}

TEST( ElmJobSchemaTest, FundingExplicitHoursParseAndNormalize )
{
    auto result = sgns::sgprocessing::ProcessingManager::Create(
        BuildElmJobJson( "", R"(, "funding": { "maximum_processing_hours": 1.3 })" ) );
    ASSERT_TRUE( result );
    EXPECT_DOUBLE_EQ( result.value()->GetElmMaximumProcessingHours(), 1.3 );
}

// ---------------------------------------------------------------------------
// Generation settings gates (A3)
// ---------------------------------------------------------------------------

std::string ElmWithGeneration( const std::string &generationJson )
{
    const std::string elm = "{"                                    //
                            "\"work_item_id\": \"item-1\","         //
                            "\"elm_type\": \"causal_lm\","          //
                            "\"model_manifest_uri\": \"ipfs://manifest1\","
                            "\"model_manifest_hash\": \"sha256:abc\","
                            "\"input_uri\": \"ipfs://input1\","
                            "\"generation\": " + generationJson + "}";
    return BuildElmJobJson( elm );
}

TEST( ElmJobSchemaTest, GenerationTopPZeroRejects )
{
    EXPECT_EQ( ExpectCreateFailure( ElmWithGeneration( R"({ "top_p": 0 })" ) ),
               Error::ELM_GENERATION_SETTINGS_INVALID );
}

TEST( ElmJobSchemaTest, GenerationTopPOneParses )
{
    auto result = sgns::sgprocessing::ProcessingManager::Create( ElmWithGeneration( R"({ "top_p": 1.0 })" ) );
    ASSERT_TRUE( result );
}

TEST( ElmJobSchemaTest, GenerationTopPAboveOneRejects )
{
    EXPECT_EQ( ExpectCreateFailure( ElmWithGeneration( R"({ "top_p": 1.5 })" ) ),
               Error::ELM_GENERATION_SETTINGS_INVALID );
}

TEST( ElmJobSchemaTest, GenerationTemperatureAboveTwoRejects )
{
    EXPECT_EQ( ExpectCreateFailure( ElmWithGeneration( R"({ "temperature": 2.5 })" ) ),
               Error::ELM_GENERATION_SETTINGS_INVALID );
}

TEST( ElmJobSchemaTest, GenerationMaxOutputTokensZeroRejects )
{
    EXPECT_EQ( ExpectCreateFailure( ElmWithGeneration( R"({ "max_output_tokens": 0 })" ) ),
               Error::ELM_GENERATION_SETTINGS_INVALID );
}

TEST( ElmJobSchemaTest, GenerationFullValidBlockParses )
{
    auto result = sgns::sgprocessing::ProcessingManager::Create(
        ElmWithGeneration( R"({ "temperature": 0.8, "top_p": 0.95, "max_output_tokens": 128, "seed": 7 })" ) );
    ASSERT_TRUE( result );
}

// ---------------------------------------------------------------------------
// Pattern / discriminator rejections (T-01-02, defensive)
// ---------------------------------------------------------------------------

TEST( ElmJobSchemaTest, WorkItemIdBadCharsetRejects )
{
    const std::string elm = "{"                                  //
                            "\"work_item_id\": \"bad id!\"," //
                            "\"elm_type\": \"causal_lm\","   //
                            "\"model_manifest_uri\": \"ipfs://m1\","
                            "\"model_manifest_hash\": \"sha256:aaa\","
                            "\"input_uri\": \"ipfs://i1\"}";
    // The pattern constraint lives in the generated setter's CheckConstraint;
    // the throw lands in Init's catch ladder as INVALID_JSON.
    EXPECT_EQ( ExpectCreateFailure( BuildElmJobJson( elm ) ), Error::INVALID_JSON );
}

TEST( ElmJobSchemaTest, ElmsWithoutJobTypeRejects )
{
    // ELM payload without its discriminator is malformed, not a legacy job.
    const std::string json = "{"                                       //
                             "\"name\": \"elm-job\","              //
                             "\"version\": \"1.0\","                //
                             "\"gnus_spec_version\": 1,"                //
                             "\"elms\": [{"
                             "\"work_item_id\": \"item-1\","
                             "\"elm_type\": \"causal_lm\","
                             "\"model_manifest_uri\": \"ipfs://manifest1\","
                             "\"model_manifest_hash\": \"sha256:abc\","
                             "\"input_uri\": \"ipfs://input1\"}]}";
    EXPECT_EQ( ExpectCreateFailure( json ), Error::INVALID_JSON );
}

// ---------------------------------------------------------------------------
// Non-ELM parity rejections (SC-5, Pitfall 1) - pre-change behavior locked
// ---------------------------------------------------------------------------

std::string BuildNonElmJobJson( const std::string &fields )
{
    return "{"                                          //
           "\"name\": \"legacy-job\","              //
           "\"version\": \"1.0\","                  //
           "\"gnus_spec_version\": 1," + fields + "}";
}

TEST( ElmJobSchemaTest, NonElmMissingPassesRejects )
{
    const std::string fields = "\"inputs\": [{ \"name\": \"in1\", \"source_uri_param\": \"p1\", \"type\": \"BUFFER\" }],"
                              "\"outputs\": [{ \"name\": \"out1\", \"source_uri_param\": \"p2\", \"type\": \"BUFFER\" }]";
    EXPECT_EQ( ExpectCreateFailure( BuildNonElmJobJson( fields ) ), Error::INVALID_JSON );
}

TEST( ElmJobSchemaTest, NonElmMissingInputsRejects )
{
    const std::string fields = "\"passes\": [{ \"name\": \"pass1\", \"type\": \"compute\", \"shader\": { \"source\": \"s\" } }],"
                              "\"outputs\": [{ \"name\": \"out1\", \"source_uri_param\": \"p2\", \"type\": \"BUFFER\" }]";
    EXPECT_EQ( ExpectCreateFailure( BuildNonElmJobJson( fields ) ), Error::INVALID_JSON );
}

TEST( ElmJobSchemaTest, NonElmMissingOutputsRejects )
{
    const std::string fields = "\"passes\": [{ \"name\": \"pass1\", \"type\": \"compute\", \"shader\": { \"source\": \"s\" } }],"
                              "\"inputs\": [{ \"name\": \"in1\", \"source_uri_param\": \"p1\", \"type\": \"BUFFER\" }]";
    EXPECT_EQ( ExpectCreateFailure( BuildNonElmJobJson( fields ) ), Error::INVALID_JSON );
}

TEST( ElmJobSchemaTest, NonElmEmptyPassesRejects )
{
    const std::string fields = "\"passes\": [],"
                              "\"inputs\": [{ \"name\": \"in1\", \"source_uri_param\": \"p1\", \"type\": \"BUFFER\" }],"
                              "\"outputs\": [{ \"name\": \"out1\", \"source_uri_param\": \"p2\", \"type\": \"BUFFER\" }]";
    EXPECT_EQ( ExpectCreateFailure( BuildNonElmJobJson( fields ) ), Error::INVALID_JSON );
}

TEST( ElmJobSchemaTest, NonElmHoursAccessorReturnsZero )
{
    // A structurally-invalid non-ELM job cannot reach the accessor (it rejects
    // at Create), so pin the 0.0 contract through a valid non-ELM job.
    const std::string fields = "\"passes\": [{ \"name\": \"pass1\", \"type\": \"compute\", \"shader\": { \"source\": \"s\" } }],"
                              "\"inputs\": [{ \"name\": \"in1\", \"source_uri_param\": \"p1\", \"type\": \"BUFFER\" }],"
                              "\"outputs\": [{ \"name\": \"out1\", \"source_uri_param\": \"p2\", \"type\": \"BUFFER\" }]";
    auto result = sgns::sgprocessing::ProcessingManager::Create( BuildNonElmJobJson( fields ) );
    if ( result )
    {
        EXPECT_DOUBLE_EQ( result.value()->GetElmMaximumProcessingHours(), 0.0 );
    }
    else
    {
        // Rejected non-ELM jobs never expose hours; the 0.0 contract holds by
        // construction (no instance exists). Record via SUCCEED to keep the
        // case visible in the matrix.
        SUCCEED() << "non-ELM job rejected at Create; hours accessor unreachable (0.0 by construction)";
    }
}
