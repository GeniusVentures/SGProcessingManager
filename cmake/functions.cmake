function(disable_clang_tidy target)
  set_target_properties(${target} PROPERTIES
      C_CLANG_TIDY ""
      CXX_CLANG_TIDY ""
      )
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

