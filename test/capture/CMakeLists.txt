# capture_smoke_test (Phase 10, Plan 10-06, CAPT-01).
#
# Thin CTest smoke test proving capture_harness actually builds, runs, and
# produces a well-formed, round-trippable .cap file in ordinary CI -- it
# deliberately does NOT assert cross-machine hash equality (Pattern 5,
# ARCHITECTURE.md); that is Phase 11's manual, multi-machine job.

add_executable(capture_smoke_test
    capture_smoke_test.cpp
)

target_compile_definitions(capture_smoke_test PRIVATE
    CAPTURE_HARNESS_PATH="$<TARGET_FILE:capture_harness>"
    FIXTURE_ROOT_PATH="${CMAKE_CURRENT_SOURCE_DIR}/../../../test/src"
    OUTPUT_DIR_PATH="$<TARGET_FILE_DIR:capture_smoke_test>"
)

target_include_directories(capture_smoke_test PRIVATE
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/../../include>
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/../..>
)

target_link_libraries(capture_smoke_test
    PRIVATE
    GTest::gtest_main
    SGProcessors
    sgproccapture
)

enable_testing()
add_test(NAME CaptureSmokeTest COMMAND capture_smoke_test)
