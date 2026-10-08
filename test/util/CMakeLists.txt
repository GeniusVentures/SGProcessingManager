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

# diff_utils unit tests (Phase 15, Plan 15-01)
# Tests: extracted ComputeFloat32Diff/ComputeUint8Diff correctness,
# ResolveChunkElementTypeHint defaults, IsFloatChunkWithinTolerance/
# IsByteChunkWithinTolerance D-03/D-04 pass/fail boundaries.

add_executable(diff_utils_test
    diff_utils_test.cpp
)

target_link_libraries(diff_utils_test
    PRIVATE
    sgprocmanagerdiff
    GTest::gtest_main
)

add_test(NAME DiffUtilsTest COMMAND diff_utils_test)
