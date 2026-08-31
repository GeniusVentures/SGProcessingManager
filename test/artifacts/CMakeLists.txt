# Artifact serializer unit tests (Phase 08, Plan 08-02)
# Tests: Artifact + Manifest round-trip, determinism, little-endian, max fields

add_executable(artifact_serializer_test
    artifact_serializer_test.cpp
)

target_include_directories(artifact_serializer_test PRIVATE
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/../../include>
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/../../generated>
    $<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}/SGProcessingManager/generated>
    $<BUILD_INTERFACE:${Boost_INCLUDE_DIRS}>
)

target_link_libraries(artifact_serializer_test
    PRIVATE
    SGArtifacts
    GTest::gtest_main
    sgprocmanagersha
)

enable_testing()
add_test(NAME ArtifactSerializerTest COMMAND artifact_serializer_test)
