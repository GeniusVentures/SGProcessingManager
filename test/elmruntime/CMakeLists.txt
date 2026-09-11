# elmruntime test targets — Phase 02-01 (elmbridge)
#
# Per-case GTest discovery (gtest_discover_tests) is gated on
# SGPROC_TEST_DISCOVERY, which is defined (default ON) only by the STANDALONE
# build (cmake/CommonBuildParameters.cmake). When SGProcessingManager is built
# as a subdirectory of SuperGenius the option is undefined, so only the
# whole-binary add_test() registration below is emitted -- no test binary is
# executed at build time. This mirrors test/processingbase/CMakeLists.txt
# exactly; wrong gating re-breaks the SuperGenius parent build.
#
# TIMEOUT 120 (Pitfall 15.3): the production-fetch leg (FileManager singleton
# init + ioc->run() drain) is the most hang-prone leg in this plan, so it gets
# an explicit bound rather than a silent skip.

add_executable(sgprocelmruntime_manifest_test elm_manifest_test.cpp)
target_link_libraries(sgprocelmruntime_manifest_test PRIVATE GTest::gtest_main sgprocmanagerelmruntime)
add_test(NAME sgprocelmruntime_manifest_test COMMAND sgprocelmruntime_manifest_test)
if(SGPROC_TEST_DISCOVERY AND NOT CMAKE_CROSSCOMPILING)
    gtest_discover_tests(sgprocelmruntime_manifest_test DISCOVERY_TIMEOUT 60)
endif()
set_tests_properties(sgprocelmruntime_manifest_test PROPERTIES TIMEOUT 120)
