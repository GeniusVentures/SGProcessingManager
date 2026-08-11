# Phase 10 (capture-harness-diff-tool-quantization-stub) -- standalone CLI tooling.
# Neither capture_harness nor capture_diff is CTest-gated (Pattern 5): a meaningful
# cross-machine pass/fail needs Phase 11's physical machines.

add_library(sgproccapture STATIC
    capture_file_format.cpp
    capture_file_format.hpp
)

target_include_directories(sgproccapture PUBLIC
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/../../include>
    # capture_file_format.cpp includes its own header via the
    # "tools/capture/capture_file_format.hpp" root-relative path (Plan 10-04);
    # add the SGProcessingManager root so that path resolves.
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/../..>
)

target_link_libraries(sgproccapture
    PUBLIC
    sgprocmanagersha
    SGArtifacts
)

sgnus_install(sgproccapture)

add_executable(capture_harness
    capture_harness.cpp
)

target_link_libraries(capture_harness
    PRIVATE
    ProcessingBase
    sgproccapture
    nlohmann_json::nlohmann_json
    ipfs-bitswap-cpp
)

add_executable(capture_diff
    capture_diff.cpp
)

target_link_libraries(capture_diff
    PRIVATE
    sgproccapture
    nlohmann_json::nlohmann_json
)
