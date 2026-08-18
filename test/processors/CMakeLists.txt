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
