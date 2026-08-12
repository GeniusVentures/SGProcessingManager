# Quantization unit tests (Phase 12, Plan 12-01)
# Tests: QuantizeFloatBuffer IEEE-754 canonicalization + fixed-grid rounding,
# QuantizeByteBuffer identity behavior.

add_executable(quantization_test
    quantization_test.cpp
)

target_link_libraries(quantization_test
    PRIVATE
    sgprocmanagerquant
    GTest::gtest_main
)

enable_testing()
add_test(NAME QuantizationTest COMMAND quantization_test)
