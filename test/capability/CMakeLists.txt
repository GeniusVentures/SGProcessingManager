# CapabilityValidator unit tests (Phase 06, Plan 06-03)
# Tests all five rejection categories + acceptance path + edge cases.

add_executable(capability_validator_test
    capability_validator_test.cpp
)

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
    GTest::GTest
    GTest::Main
    sgprocmanagerlogger
    sgprocmanagersha
    sgprocmanagertypes
)

# Register with CTest
enable_testing()
add_test(NAME CapabilityValidatorTest COMMAND capability_validator_test)
