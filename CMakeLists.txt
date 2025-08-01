cmake_minimum_required(VERSION 3.16)

# Project definition
project(SGProcessingManager
    VERSION 1.0.0
    DESCRIPTION "GNUS.ai Processing Definition Schema C++ Headers"
    LANGUAGES C CXX
)

include(cmake/functions.cmake)

set(libp2p_INCLUDE_DIR "${_THIRDPARTY_BUILD_DIR}/libp2p/include")

#add_subdirectory(generated)
add_subdirectory(src)

install(DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}/include" DESTINATION "${CMAKE_INSTALL_INCLUDEDIR}/SGProcessingManager" FILES_MATCHING PATTERN "*.h*")
install(DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}/generated" DESTINATION "${CMAKE_INSTALL_INCLUDEDIR}/SGProcessingManager" FILES_MATCHING PATTERN "*.h*")

set(SGProcessingManager_CONFIG_DESTINATION_DIR "lib/cmake/SGProcessingManager")
install(EXPORT SGProcessingManagerTargets
    FILE SGProcessingManager.cmake
    DESTINATION ${SGProcessingManager_CONFIG_DESTINATION_DIR}
)

include(CMakePackageConfigHelpers)

# generate the config file that is includes the exports
configure_package_config_file(${CMAKE_CURRENT_SOURCE_DIR}/cmake/config.cmake.in
    "${CMAKE_CURRENT_BINARY_DIR}/SGProcessingManagerConfig.cmake"
    INSTALL_DESTINATION "lib/cmake/examples"
    NO_SET_AND_CHECK_MACRO
    NO_CHECK_REQUIRED_COMPONENTS_MACRO
)

# generate the version file for the config file
write_basic_package_version_file(
    "${CMAKE_CURRENT_BINARY_DIR}/SGProcessingManagerConfigVersion.cmake"
    VERSION "${VERSION_STRING}.${SUBVERSION_STRING}"
    COMPATIBILITY AnyNewerVersion
)

# install the configuration file
install(FILES
    ${CMAKE_CURRENT_BINARY_DIR}/SGProcessingManagerConfig.cmake
    DESTINATION ${SGProcessingManager_CONFIG_DESTINATION_DIR}
)
