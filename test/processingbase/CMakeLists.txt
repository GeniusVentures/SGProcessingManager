# Processing-base test targets — Phase 01-01 (elmbridge)
#
# Per-case GTest discovery (gtest_discover_tests) is gated on
# SGPROC_TEST_DISCOVERY, which is defined (default ON) only by the STANDALONE
# build (cmake/CommonBuildParameters.cmake). When SGProcessingManager is built
# as a subdirectory of SuperGenius the option is undefined, so only the
# whole-binary add_test() registration below is emitted -- no test binary is
# executed at build time. This mirrors test/execution/CMakeLists.txt exactly;
# wrong gating re-breaks the SuperGenius parent build.

add_executable(sgprocbase_elm_job_schema_test elm_job_schema_test.cpp)
target_link_libraries(sgprocbase_elm_job_schema_test PRIVATE GTest::gtest_main ProcessingBase)
add_test(NAME sgprocbase_elm_job_schema_test COMMAND sgprocbase_elm_job_schema_test)
if(SGPROC_TEST_DISCOVERY AND NOT CMAKE_CROSSCOMPILING)
    gtest_discover_tests(sgprocbase_elm_job_schema_test DISCOVERY_TIMEOUT 60)
endif()

# ELM worker-path routing (elmbridge Phase 4, plan 04-02, Task 3): the
# Pattern 1 intercept + Pattern 2 terminal-publication legs. Offline by
# construction (unverifiable manifest URIs -> terminal error envelopes).
# TIMEOUT 120: FileManager singleton init + ioc drain legs (the elmruntime
# sibling convention).
add_executable(sgprocbase_elm_routing_test elm_worker_routing_test.cpp)
target_link_libraries(sgprocbase_elm_routing_test PRIVATE GTest::gtest_main ProcessingBase)
add_test(NAME sgprocbase_elm_routing_test COMMAND sgprocbase_elm_routing_test)
if(SGPROC_TEST_DISCOVERY AND NOT CMAKE_CROSSCOMPILING)
    gtest_discover_tests(sgprocbase_elm_routing_test DISCOVERY_TIMEOUT 60)
endif()
set_tests_properties(sgprocbase_elm_routing_test PROPERTIES TIMEOUT 120)
