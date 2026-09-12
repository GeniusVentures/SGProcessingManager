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

# elm_model_cache_test (Plan 02-03, Task 3): cache lifecycle matrix -- publish,
# hit, tamper->quarantine->retry, single-flight, pin/evict/LRU, restart rescan,
# tmp sweep, temp cleanliness. The gated garbage-bundle leg runs the REAL
# MakeMnnLlmSmokeCheck() (createLLM on garbage bytes fails inside load()), so the
# target needs MNN/Vulkan under the LLM gate (test/processors/CMakeLists.txt:28-48
# pattern). TIMEOUT 300: the MNN-touching leg initializes the engine; generous
# bound, fast in practice (Pitfall 15.2/15.3).
add_executable(sgprocelmruntime_cache_test elm_model_cache_test.cpp)
target_link_libraries(sgprocelmruntime_cache_test PRIVATE GTest::gtest_main sgprocmanagerelmruntime)
if(SGPROC_HAS_MNN_LLM)
    target_compile_definitions(sgprocelmruntime_cache_test PRIVATE SGPROC_HAS_MNN_LLM)
    target_link_libraries(sgprocelmruntime_cache_test PRIVATE MNN::MNN Vulkan::Vulkan)
endif()
add_test(NAME sgprocelmruntime_cache_test COMMAND sgprocelmruntime_cache_test)
if(SGPROC_TEST_DISCOVERY AND NOT CMAKE_CROSSCOMPILING)
    gtest_discover_tests(sgprocelmruntime_cache_test DISCOVERY_TIMEOUT 60)
endif()
set_tests_properties(sgprocelmruntime_cache_test PROPERTIES TIMEOUT 300)

# elm_envelope_test + elm_stop_streambuf_test (elmbridge Phase 3, plan 03-03,
# Task 1): the RES-01 envelope matrix and the D-05/D-06/D-07 stop-string
# streambuf legs -- both PURE units (no MNN touch, no fixture), but bundled
# into one target per the plan's single sgprocelmruntime_envelope_test
# registration. TIMEOUT 120: fast legs, no engine initialization.
add_executable(sgprocelmruntime_envelope_test
    elm_envelope_test.cpp
    elm_stop_streambuf_test.cpp
)
target_link_libraries(sgprocelmruntime_envelope_test PRIVATE GTest::gtest_main sgprocmanagerelmruntime)
add_test(NAME sgprocelmruntime_envelope_test COMMAND sgprocelmruntime_envelope_test)
if(SGPROC_TEST_DISCOVERY AND NOT CMAKE_CROSSCOMPILING)
    gtest_discover_tests(sgprocelmruntime_envelope_test DISCOVERY_TIMEOUT 60)
endif()
set_tests_properties(sgprocelmruntime_envelope_test PROPERTIES TIMEOUT 120)
