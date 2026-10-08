function(disable_clang_tidy target)
  set_target_properties(${target} PROPERTIES
      C_CLANG_TIDY ""
      CXX_CLANG_TIDY ""
      )
endfunction()

# Windows: copy the vendored Vulkan loader DLL next to a test executable.
# Needed for EVERY test exe that links SGProcessors (directly or via
# SGExecution): with static libs the exe imports vulkan-1.dll whenever its
# reference graph pulls a processor obj -- which now always includes the GPU
# probe (HasUsableVulkanDeviceCached -> vk-bootstrap -> vulkan-1.lib) -- and
# both gtest_discover_tests() (run by CMake right after linking) and plain
# ctest execute the exe with no Vulkan SDK on PATH on CI runners. Without the
# copy the process dies with 0xc0000135 before main() (observed:
# sgprocmanagerexec_migration_test, run 35136102223: link ok, discovery
# failed). copy_if_different makes repeat builds no-ops. VULKAN_RUNTIME_DLL
# is resolved in cmake/CommonBuildParameters.cmake. Mirrors SuperGenius's
# addtest() in SuperGenius/cmake/functions.cmake.
function(sgpm_copy_vulkan_runtime target)
  if(WIN32 AND VULKAN_RUNTIME_DLL)
    add_custom_command(TARGET ${target} POST_BUILD
        COMMAND "${CMAKE_COMMAND}" -E copy_if_different
            "${VULKAN_RUNTIME_DLL}" "$<TARGET_FILE_DIR:${target}>/vulkan-1.dll")
  endif()
endfunction()

# conditionally applies flag.
function(add_flag flag)
  check_cxx_compiler_flag(${flag} FLAG_${flag})
  if (FLAG_${flag} EQUAL 1)
    add_compile_options(${flag})
  endif ()
endfunction()

function(print)
  message(STATUS "[${CMAKE_PROJECT_NAME}] ${ARGV}")
endfunction()

### sgnus_install should be called right after add_library(target)
function(sgnus_install target)
    install(TARGETS ${target} EXPORT SGProcessingManagerTargets
        LIBRARY       DESTINATION ${CMAKE_INSTALL_LIBDIR}/SGProcessingManager
        ARCHIVE       DESTINATION ${CMAKE_INSTALL_LIBDIR}/SGProcessingManager
        RUNTIME       DESTINATION ${CMAKE_INSTALL_BINDIR}
        INCLUDES      DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}/SGProcessingManager
        PUBLIC_HEADER DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}/SGProcessingManager
        FRAMEWORK     DESTINATION ${CMAKE_INSTALL_PREFIX}
        BUNDLE        DESTINATION ${CMAKE_INSTALL_BINDIR}
        )
endfunction()

