# MNN_Tensor FP4_ULTRA unit tests (Phase 04-sgprocessing-integration, Plan 04-02, PROC-02)
# Tests: FP4_ULTRA is recognized (not rejected) and returns a structured,
# correctly-staged ProcessingError -- without ever touching VulkanInitMutex/MNN
# session creation. Also regression-covers that an unrecognized format is
# still rejected exactly as before this plan's changes.

add_executable(mnn_tensor_fp4_test
    mnn_tensor_fp4_test.cpp
)

target_link_libraries(mnn_tensor_fp4_test
    PRIVATE
    SGProcessors
    GTest::gtest_main
)

enable_testing()
add_test(NAME MNNTensorFp4Test COMMAND mnn_tensor_fp4_test)

# MNN_Llm unit tests (Phase 04-sgprocessing-integration, Plan 04-03, PROC-01).
# SGPROC_HAS_MNN_LLM is set (CACHE INTERNAL) by src/processors/CMakeLists.txt's
# configure-time detection of the vendored MNN's MNN_BUILD_LLM support. MNN_Llm's
# StartProcessing() has no compiled definition anywhere when that support is absent
# (see src/processors/CMakeLists.txt), so this test target is excluded entirely in
# checkouts without it, rather than failing to link.
if(SGPROC_HAS_MNN_LLM)
    add_executable(mnn_llm_test
        mnn_llm_test.cpp
    )

    target_link_libraries(mnn_llm_test
        PRIVATE
        SGProcessors
        GTest::gtest_main
    )

    add_test(NAME MNNLlmTest COMMAND mnn_llm_test)
else()
    message(STATUS "MNN LLM engine headers NOT found -- mnn_llm_test target skipped (see src/processors/CMakeLists.txt)")
endif()
