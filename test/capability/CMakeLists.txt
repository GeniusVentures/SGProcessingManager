# CapabilityValidator unit tests (Phase 06, Plan 06-03)
# Tests all five rejection categories + acceptance path + edge cases.

add_executable(capability_validator_test
    capability_validator_test.cpp
)

target_compile_definitions(capability_validator_test PRIVATE SGPROCMGR_TEST_FRIEND)

target_include_directories(capability_validator_test PRIVATE
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/../../include>
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/../../generated>
    $<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}/SGProcessingManager/generated>
    $<BUILD_INTERFACE:${Boost_INCLUDE_DIRS}>
    $<BUILD_INTERFACE:${OPENSSL_INCLUDE_DIR}>
)

target_link_libraries(capability_validator_test
    PRIVATE
    SGCapability
    GTest::gtest_main
    sgprocmanagerlogger
    sgprocmanagersha
    sgprocmanagertypes
)

# Register with CTest
enable_testing()
add_test(NAME CapabilityValidatorTest COMMAND capability_validator_test)

# ELM resource-preflight tests (Phase 02-02, elmbridge).
#
# TWO TUs, one binary: elm_resources_test.cpp (validator legs — mock snapshots
# via SetSnapshotForTest; includes the ROOT generated/ set transitively) and
# elm_resource_extraction_test.cpp (manifest value extraction; includes ONLY
# the generated/elmruntime-manifest/ fallback set). The two quicktype sets
# both define sgns::ClassMemberConstraints/ElmType — a single TU including
# both is a redefinition error, hence the split.
#
# Gating mirrors the sibling target and test/processingbase/CMakeLists.txt:
# unconditional whole-binary add_test + SGPROC_TEST_DISCOVERY-gated per-case
# discovery (undefined when built as a SuperGenius subdirectory). Pure logic,
# milliseconds — no CTest TIMEOUT needed.
add_executable(sgproccapability_elm_resources_test
    elm_resources_test.cpp
    elm_resource_extraction_test.cpp
)

target_compile_definitions(sgproccapability_elm_resources_test PRIVATE SGPROCMGR_TEST_FRIEND)

target_include_directories(sgproccapability_elm_resources_test PRIVATE
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/../../include>
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/../../generated>
    $<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}/SGProcessingManager/generated>
    $<BUILD_INTERFACE:${Boost_INCLUDE_DIRS}>
    $<BUILD_INTERFACE:${OPENSSL_INCLUDE_DIR}>
)

target_link_libraries(sgproccapability_elm_resources_test
    PRIVATE
    SGCapability
    sgprocmanagerelmruntime
    GTest::gtest_main
    sgprocmanagerlogger
    sgprocmanagersha
    sgprocmanagertypes
)

add_test(NAME sgproccapability_elm_resources_test COMMAND sgproccapability_elm_resources_test)
if(SGPROC_TEST_DISCOVERY AND NOT CMAKE_CROSSCOMPILING)
    gtest_discover_tests(sgproccapability_elm_resources_test DISCOVERY_TIMEOUT 60)
endif()
