# Execution context test targets — Phase 07
#
# gtest_discover_tests() EXECUTES each test binary at build time to enumerate
# individual test cases. Cross-compiled device binaries (iOS, Android) cannot
# run on the build host, so discovery aborts the build ("Subprocess killed").
# Keep the whole-binary add_test() registration on every platform and perform
# per-case discovery on host builds only.

add_executable(sgprocmanagerexec_cancellation_test cancellation_test.cpp)
target_link_libraries(sgprocmanagerexec_cancellation_test PRIVATE GTest::gtest_main SGExecution)
add_test(NAME sgprocmanagerexec_cancellation_test COMMAND sgprocmanagerexec_cancellation_test)
if(NOT CMAKE_CROSSCOMPILING)
    gtest_discover_tests(sgprocmanagerexec_cancellation_test)
endif()

add_executable(sgprocmanagerexec_timeout_test timeout_test.cpp)
target_link_libraries(sgprocmanagerexec_timeout_test PRIVATE GTest::gtest_main SGExecution)
add_test(NAME sgprocmanagerexec_timeout_test COMMAND sgprocmanagerexec_timeout_test)
if(NOT CMAKE_CROSSCOMPILING)
    gtest_discover_tests(sgprocmanagerexec_timeout_test)
endif()

add_executable(sgprocmanagerexec_budget_test budget_test.cpp)
target_link_libraries(sgprocmanagerexec_budget_test PRIVATE GTest::gtest_main SGExecution)
add_test(NAME sgprocmanagerexec_budget_test COMMAND sgprocmanagerexec_budget_test)
if(NOT CMAKE_CROSSCOMPILING)
    gtest_discover_tests(sgprocmanagerexec_budget_test)
endif()

add_executable(sgprocmanagerexec_progress_test progress_event_test.cpp)
target_link_libraries(sgprocmanagerexec_progress_test PRIVATE GTest::gtest_main SGExecution)
add_test(NAME sgprocmanagerexec_progress_test COMMAND sgprocmanagerexec_progress_test)
if(NOT CMAKE_CROSSCOMPILING)
    gtest_discover_tests(sgprocmanagerexec_progress_test)
endif()

add_executable(sgprocmanagerexec_checkpoint_test checkpoint_test.cpp)
target_link_libraries(sgprocmanagerexec_checkpoint_test PRIVATE GTest::gtest_main SGExecution SGCapability)
add_test(NAME sgprocmanagerexec_checkpoint_test COMMAND sgprocmanagerexec_checkpoint_test)
if(NOT CMAKE_CROSSCOMPILING)
    gtest_discover_tests(sgprocmanagerexec_checkpoint_test)
endif()

add_executable(sgprocmanagerexec_leak_test leak_detection_test.cpp)
target_link_libraries(sgprocmanagerexec_leak_test PRIVATE GTest::gtest_main SGExecution)
add_test(NAME sgprocmanagerexec_leak_test COMMAND sgprocmanagerexec_leak_test)
if(NOT CMAKE_CROSSCOMPILING)
    gtest_discover_tests(sgprocmanagerexec_leak_test)
endif()

add_executable(sgprocmanagerexec_migration_test migration_adapter_test.cpp)
target_link_libraries(sgprocmanagerexec_migration_test PRIVATE GTest::gtest_main SGExecution)
add_test(NAME sgprocmanagerexec_migration_test COMMAND sgprocmanagerexec_migration_test)
if(NOT CMAKE_CROSSCOMPILING)
    gtest_discover_tests(sgprocmanagerexec_migration_test)
endif()
