# Execution context test targets — Phase 07
#
# Per-case GTest discovery (gtest_discover_tests) is gated on
# SGPROC_TEST_DISCOVERY, which is defined (default ON) only by the STANDALONE
# build (cmake/CommonBuildParameters.cmake). When SGProcessingManager is built
# as a subdirectory of SuperGenius the option is undefined, so only the
# whole-binary add_test() registration below is emitted -- no test binary is
# executed at build time. This sidesteps both failure modes seen in CI:
# cross-compiled binaries cannot run on the host (iOS "Subprocess killed"),
# and host Debug cold-starts can exceed the discovery timeout.
#
# DISCOVERY_TIMEOUT stays raised for standalone builds: the 5s default has
# been exceeded by fresh Debug binaries on the self-hosted Windows runner
# (large PDB cold start); listing itself prints in ms once the process is up.

add_executable(sgprocmanagerexec_cancellation_test cancellation_test.cpp)
target_link_libraries(sgprocmanagerexec_cancellation_test PRIVATE GTest::gtest_main SGExecution)
add_test(NAME sgprocmanagerexec_cancellation_test COMMAND sgprocmanagerexec_cancellation_test)
if(SGPROC_TEST_DISCOVERY AND NOT CMAKE_CROSSCOMPILING)
    gtest_discover_tests(sgprocmanagerexec_cancellation_test DISCOVERY_TIMEOUT 60)
endif()

add_executable(sgprocmanagerexec_timeout_test timeout_test.cpp)
target_link_libraries(sgprocmanagerexec_timeout_test PRIVATE GTest::gtest_main SGExecution)
add_test(NAME sgprocmanagerexec_timeout_test COMMAND sgprocmanagerexec_timeout_test)
if(SGPROC_TEST_DISCOVERY AND NOT CMAKE_CROSSCOMPILING)
    gtest_discover_tests(sgprocmanagerexec_timeout_test DISCOVERY_TIMEOUT 60)
endif()

add_executable(sgprocmanagerexec_budget_test budget_test.cpp)
target_link_libraries(sgprocmanagerexec_budget_test PRIVATE GTest::gtest_main SGExecution)
add_test(NAME sgprocmanagerexec_budget_test COMMAND sgprocmanagerexec_budget_test)
if(SGPROC_TEST_DISCOVERY AND NOT CMAKE_CROSSCOMPILING)
    gtest_discover_tests(sgprocmanagerexec_budget_test DISCOVERY_TIMEOUT 60)
endif()

add_executable(sgprocmanagerexec_progress_test progress_event_test.cpp)
target_link_libraries(sgprocmanagerexec_progress_test PRIVATE GTest::gtest_main SGExecution)
add_test(NAME sgprocmanagerexec_progress_test COMMAND sgprocmanagerexec_progress_test)
if(SGPROC_TEST_DISCOVERY AND NOT CMAKE_CROSSCOMPILING)
    gtest_discover_tests(sgprocmanagerexec_progress_test DISCOVERY_TIMEOUT 60)
endif()

add_executable(sgprocmanagerexec_checkpoint_test checkpoint_test.cpp)
target_link_libraries(sgprocmanagerexec_checkpoint_test PRIVATE GTest::gtest_main SGExecution SGCapability)
add_test(NAME sgprocmanagerexec_checkpoint_test COMMAND sgprocmanagerexec_checkpoint_test)
if(SGPROC_TEST_DISCOVERY AND NOT CMAKE_CROSSCOMPILING)
    gtest_discover_tests(sgprocmanagerexec_checkpoint_test DISCOVERY_TIMEOUT 60)
endif()

add_executable(sgprocmanagerexec_leak_test leak_detection_test.cpp)
target_link_libraries(sgprocmanagerexec_leak_test PRIVATE GTest::gtest_main SGExecution)
add_test(NAME sgprocmanagerexec_leak_test COMMAND sgprocmanagerexec_leak_test)
if(SGPROC_TEST_DISCOVERY AND NOT CMAKE_CROSSCOMPILING)
    gtest_discover_tests(sgprocmanagerexec_leak_test DISCOVERY_TIMEOUT 60)
endif()

add_executable(sgprocmanagerexec_migration_test migration_adapter_test.cpp)
target_link_libraries(sgprocmanagerexec_migration_test PRIVATE GTest::gtest_main SGExecution)
add_test(NAME sgprocmanagerexec_migration_test COMMAND sgprocmanagerexec_migration_test)
if(SGPROC_TEST_DISCOVERY AND NOT CMAKE_CROSSCOMPILING)
    gtest_discover_tests(sgprocmanagerexec_migration_test DISCOVERY_TIMEOUT 60)
endif()
if(NOT CMAKE_CROSSCOMPILING)
    gtest_discover_tests(sgprocmanagerexec_migration_test DISCOVERY_TIMEOUT 60)
endif()
