# BOOST VERSION TO USE
set(BOOST_MAJOR_VERSION "1" CACHE STRING "Boost Major Version")
set(BOOST_MINOR_VERSION "85" CACHE STRING "Boost Minor Version")
set(BOOST_PATCH_VERSION "0" CACHE STRING "Boost Patch Version")

# convenience settings
set(BOOST_VERSION "${BOOST_MAJOR_VERSION}.${BOOST_MINOR_VERSION}.${BOOST_PATCH_VERSION}")
set(BOOST_VERSION_2U "${BOOST_MAJOR_VERSION}_${BOOST_MINOR_VERSION}")

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

if(DEFINED USE_BOOST_INCLUDE_POSTFIX)
        set(BOOST_INCLUDE_POSTFIX "/boost-${BOOST_VERSION_2U}" CACHE STRING "Boost include postfix")
endif()
include(${CMAKE_CURRENT_LIST_DIR}/functions.cmake)
# --------------------------------------------------------
# Set config of GTest
set(GTest_DIR "${_THIRDPARTY_BUILD_DIR}/GTest/lib/cmake/GTest")
set(GTest_INCLUDE_DIR "${_THIRDPARTY_BUILD_DIR}/GTest/include")
find_package(GTest CONFIG REQUIRED)
include_directories(${GTest_INCLUDE_DIR})
add_compile_definitions(CRYPTO3_CODEC_BASE58)

#OpenSSL
set(OpenSSL_DIR "${_THIRDPARTY_BUILD_DIR}/openssl/build/lib/cmake/OpenSSL" CACHE PATH "Path to OpenSSL install folder")
set(OPENSSL_ROOT_DIR "${_THIRDPARTY_BUILD_DIR}/openssl/build" CACHE PATH "Path to OpenSSL install root folder")
set(OPENSSL_USE_STATIC_LIBS ON CACHE BOOL "OpenSSL use static libs")
set(OPENSSL_MSVC_STATIC_RT ON CACHE BOOL "OpenSSL use static RT")
set(OPENSSL_INCLUDE_DIR "${_THIRDPARTY_BUILD_DIR}/openssl/build/include" CACHE PATH "Path to OpenSSL include folder")

find_package(OpenSSL REQUIRED CONFIG)

# VulkanHeaders
set(VulkanHeaders_DIR "${_THIRDPARTY_BUILD_DIR}/Vulkan-Headers/share/cmake/VulkanHeaders" CACHE PATH "Path to Vulkan-Headers install folder")
find_package(VulkanHeaders CONFIG REQUIRED)

# Vulkan
#
# On macOS, create the Vulkan::Vulkan target manually pointing at the MoltenVK
# dylib nested inside the thirdparty-built MoltenVK.xcframework.  MoltenVK is a
# complete Vulkan implementation that exports the full loader API — it can be
# used directly without any ICD plumbing, exactly as MNN's Vulkan backend does.
#
# On other platforms, use the vendored Khronos Vulkan-Loader found via the
# standard find_package(Vulkan) / VULKAN_SDK mechanism.
if(APPLE)
    if(NOT TARGET Vulkan::Vulkan)
        set(_MVK_LIB "${_THIRDPARTY_BUILD_DIR}/MoltenVK/build/lib/MoltenVK.xcframework/macos-arm64_x86_64/libMoltenVK.a")
        add_library(Vulkan::Vulkan STATIC IMPORTED GLOBAL)
        set_target_properties(Vulkan::Vulkan PROPERTIES
            INTERFACE_INCLUDE_DIRECTORIES "${_THIRDPARTY_BUILD_DIR}/Vulkan-Headers/include"
            IMPORTED_LOCATION "${_MVK_LIB}"
        )
        # Frameworks MoltenVK links against; inherited by every consumer of Vulkan::Vulkan.
        target_link_libraries(Vulkan::Vulkan INTERFACE
            "-framework Metal"
            "-framework IOSurface"
            "-framework QuartzCore"
            "-framework Foundation"
            "-framework CoreFoundation"
            "-framework CoreGraphics"
            "-framework IOKit"
        )
    endif()
else()
    find_package(Vulkan)

    if(NOT TARGET Vulkan::Vulkan)
        set(Vulkan_INCLUDE_DIR "${_THIRDPARTY_BUILD_DIR}/Vulkan-Headers/include")
        if(NOT DEFINED ENV{VULKAN_SDK})
            set(ENV{VULKAN_SDK} "${_THIRDPARTY_BUILD_DIR}/Vulkan-Loader")
        endif()

        find_package(Vulkan REQUIRED)
    endif()

    # Override Vulkan::Vulkan to use our vendored Vulkan-Headers on all non-Apple
    # platforms.  vk-bootstrap was built against our headers (v1.4); mixing with
    # system/NDK headers (v1.3 or other versions) causes unknown-type errors in
    # VkBootstrapDispatch.h and VkBootstrapFeatureChain.h.
    set_target_properties(Vulkan::Vulkan PROPERTIES
        INTERFACE_INCLUDE_DIRECTORIES "${_THIRDPARTY_BUILD_DIR}/Vulkan-Headers/include"
    )
endif()

# vk-bootstrap
set(vk-bootstrap_DIR "${_THIRDPARTY_BUILD_DIR}/vk-bootstrap/lib/cmake/vk-bootstrap")
find_package(vk-bootstrap CONFIG REQUIRED)

# SPIRV-Tools — no longer a standalone build.  libshaderc_combined (linked via
# shaderc::shaderc below) statically bundles the exact same SPIRV-Tools code at the
# exact same pinned commit (v2024.3 DEPS).  The spirv-tools include path is folded into
# shaderc::shaderc's INTERFACE_INCLUDE_DIRECTORIES so <spirv-tools/libspirv.hpp> resolves.

# shaderc — installs no CMake package config (confirmed in 02-02-RESEARCH.md against
# github.com/google/shaderc/issues/1369 and github.com/microsoft/vcpkg/issues/23208); hand-written
# IMPORTED target required, mirroring thirdparty/build/CommonTargets.cmake's own target.
# libshaderc_combined statically bundles glslang+SPIRV-Tools and installs spirv-tools
# headers into <prefix>/include/spirv-tools/, so <spirv-tools/libspirv.hpp> resolves
# for consumers that call spvtools::SpirvTools::Validate() directly (SHADER-02).
if(NOT TARGET shaderc::shaderc)
    add_library(shaderc::shaderc STATIC IMPORTED GLOBAL)
    set_target_properties(shaderc::shaderc PROPERTIES
        IMPORTED_LOCATION "${_THIRDPARTY_BUILD_DIR}/shaderc/lib/${CMAKE_STATIC_LIBRARY_PREFIX}shaderc_combined${CMAKE_STATIC_LIBRARY_SUFFIX}"
        INTERFACE_INCLUDE_DIRECTORIES "${_THIRDPARTY_BUILD_DIR}/shaderc/include"
    )
endif()

# for compression, we need snappy
set(Snappy_DIR "${_THIRDPARTY_BUILD_DIR}/snappy/lib/cmake/Snappy")
find_package(Snappy CONFIG REQUIRED)

# rocksdb
set(RocksDB_DIR "${_THIRDPARTY_BUILD_DIR}/rocksdb/lib/cmake/rocksdb")
find_package(RocksDB CONFIG REQUIRED)

#GSL
set(GSL_INCLUDE_DIR "${_THIRDPARTY_BUILD_DIR}/Microsoft.GSL/include")
include_directories(${GSL_INCLUDE_DIR})

# Boost.DI
set(Boost.DI_DIR "${_THIRDPARTY_BUILD_DIR}/Boost.DI/lib/cmake/Boost.DI")
find_package(Boost.DI CONFIG REQUIRED)

# Boost should be loaded before libp2p v0.1.2
# --------------------------------------------------------
# Set config of Boost project
set(_BOOST_ROOT "${_THIRDPARTY_BUILD_DIR}/boost/build/")
set(Boost_LIB_DIR "${_BOOST_ROOT}/lib")
set(Boost_INCLUDE_DIR "${_BOOST_ROOT}/include${BOOST_INCLUDE_POSTFIX}")
set(Boost_DIR "${Boost_LIB_DIR}/cmake/Boost-${BOOST_VERSION}")
set(boost_headers_DIR "${Boost_LIB_DIR}/cmake/boost_headers-${BOOST_VERSION}")
set(boost_random_DIR "${Boost_LIB_DIR}/cmake/boost_random-${BOOST_VERSION}")
set(boost_system_DIR "${Boost_LIB_DIR}/cmake/boost_system-${BOOST_VERSION}")
set(boost_filesystem_DIR "${Boost_LIB_DIR}/cmake/boost_filesystem-${BOOST_VERSION}")
set(boost_program_options_DIR "${Boost_LIB_DIR}/cmake/boost_program_options-${BOOST_VERSION}")
set(boost_date_time_DIR "${Boost_LIB_DIR}/cmake/boost_date_time-${BOOST_VERSION}")
set(boost_regex_DIR "${Boost_LIB_DIR}/cmake/boost_regex-${BOOST_VERSION}")
set(boost_atomic_DIR "${Boost_LIB_DIR}/cmake/boost_atomic-${BOOST_VERSION}")
set(boost_chrono_DIR "${Boost_LIB_DIR}/cmake/boost_chrono-${BOOST_VERSION}")
set(boost_log_DIR "${Boost_LIB_DIR}/cmake/boost_log-${BOOST_VERSION}")
set(boost_log_setup_DIR "${Boost_LIB_DIR}/cmake/boost_log_setup-${BOOST_VERSION}")
set(boost_thread_DIR "${Boost_LIB_DIR}/cmake/boost_thread-${BOOST_VERSION}")
set(boost_unit_test_framework_DIR "${Boost_LIB_DIR}/cmake/boost_unit_test_framework-${BOOST_VERSION}")
set(Boost_USE_MULTITHREADED ON)
set(Boost_USE_STATIC_LIBS ON)
set(Boost_NO_SYSTEM_PATHS ON)
option(Boost_USE_STATIC_RUNTIME "Use static runtimes" ON)

# header only libraries must not be added here
find_package(Boost REQUIRED COMPONENTS date_time filesystem random regex system thread log log_setup program_options)
include_directories(${Boost_INCLUDE_DIRS})

# absl
if(NOT DEFINED absl_DIR)
    set(absl_DIR "${_THIRDPARTY_BUILD_DIR}/protobuf/lib/cmake/absl")
endif()

# utf8_range
if(NOT DEFINED utf8_range_DIR)
    set(utf8_range_DIR "${_THIRDPARTY_BUILD_DIR}/protobuf/lib/cmake/utf8_range")
endif()

# protobuf project
if(NOT DEFINED Protobuf_DIR)
    set(Protobuf_DIR "${_THIRDPARTY_BUILD_DIR}/protobuf/lib/cmake/protobuf")
endif()

if(NOT DEFINED grpc_INCLUDE_DIR)
    set(grpc_INCLUDE_DIR "${_THIRDPARTY_BUILD_DIR}/grpc/include")
endif()

if(NOT DEFINED Protobuf_INCLUDE_DIR)
    set(Protobuf_INCLUDE_DIR "${grpc_INCLUDE_DIR}/google/protobuf")
endif()

find_package(Protobuf CONFIG REQUIRED)

if(NOT DEFINED PROTOC_EXECUTABLE)
    set(PROTOC_EXECUTABLE "${_THIRDPARTY_BUILD_DIR}/protobuf/bin/protoc${CMAKE_EXECUTABLE_SUFFIX}")
endif()

set(Protobuf_PROTOC_EXECUTABLE ${PROTOC_EXECUTABLE} CACHE PATH "Initial cache" FORCE)

if(NOT TARGET protobuf::protoc)
    add_executable(protobuf::protoc IMPORTED)
endif()

if(EXISTS "${Protobuf_PROTOC_EXECUTABLE}")
    set_target_properties(protobuf::protoc PROPERTIES
            IMPORTED_LOCATION ${Protobuf_PROTOC_EXECUTABLE})
endif()

# protoc definition
get_target_property(PROTOC_LOCATION protobuf::protoc IMPORTED_LOCATION)
print("PROTOC_LOCATION: ${PROTOC_LOCATION}")

if(Protobuf_FOUND)
    message(STATUS "Protobuf version : ${Protobuf_VERSION}")
    message(STATUS "Protobuf compiler : ${Protobuf_PROTOC_EXECUTABLE}")
endif()

# SQLiteModernCpp project
set(SQLiteModernCpp_ROOT_DIR "${_THIRDPARTY_BUILD_DIR}/SQLiteModernCpp")
set(SQLiteModernCpp_DIR "${SQLiteModernCpp_ROOT_DIR}/lib/cmake/SQLiteModernCpp")
set(SQLiteModernCpp_LIB_DIR "${SQLiteModernCpp_ROOT_DIR}/lib")
set(SQLiteModernCpp_INCLUDE_DIR "${SQLiteModernCpp_ROOT_DIR}/include")

#json.hpp
set(nlohmann_json_DIR "${_THIRDPARTY_BUILD_DIR}/json/share/cmake/nlohmann_json")
find_package(nlohmann_json CONFIG REQUIRED)

# fmt
set(fmt_DIR "${_THIRDPARTY_BUILD_DIR}/fmt/lib/cmake/fmt")
find_package(fmt CONFIG REQUIRED)

# tsl_hat_trie
set(tsl_hat_trie_DIR "${_THIRDPARTY_BUILD_DIR}/tsl_hat_trie/lib/cmake/tsl_hat_trie")
find_package(tsl_hat_trie CONFIG REQUIRED)

# spdlog
set(spdlog_DIR "${_THIRDPARTY_BUILD_DIR}/spdlog/lib/cmake/spdlog")
find_package(spdlog CONFIG REQUIRED)
add_compile_definitions("SPDLOG_FMT_EXTERNAL")

# soralog
set(soralog_DIR "${_THIRDPARTY_BUILD_DIR}/soralog/lib/cmake/soralog")
find_package(soralog CONFIG REQUIRED)

# yaml-cpp
set(yaml-cpp_DIR "${_THIRDPARTY_BUILD_DIR}/yaml-cpp/lib/cmake/yaml-cpp")
find_package(yaml-cpp CONFIG REQUIRED)

# cares
set(c-ares_DIR "${_THIRDPARTY_BUILD_DIR}/cares/lib/cmake/c-ares" CACHE PATH "Path to c-ares install folder")
set(c-ares_INCLUDE_DIR "${_THIRDPARTY_BUILD_DIR}/cares/include" CACHE PATH "Path to c-ares include folder")

# libp2p
set(libp2p_DIR "${_THIRDPARTY_BUILD_DIR}/libp2p/lib/cmake/libp2p")
set(libp2p_INCLUDE_DIR "${_THIRDPARTY_BUILD_DIR}/libp2p/include")
find_package(libp2p CONFIG REQUIRED)

# ipfs-lite-cpp
set(ipfs-lite-cpp_DIR "${_THIRDPARTY_BUILD_DIR}/ipfs-lite-cpp/lib/cmake/ipfs-lite-cpp")
find_package(ipfs-lite-cpp CONFIG REQUIRED)

# MNN
set(MNN_DIR "${_THIRDPARTY_BUILD_DIR}/MNN/lib/cmake/MNN")
find_package(MNN CONFIG REQUIRED)
set(MNN_INCLUDE_DIR "${_THIRDPARTY_BUILD_DIR}/MNN/include")
message(STATIS "INCLUDE DIR ${MNN_INCLUDE_DIR}")
include_directories(${MNN_INCLUDE_DIR})
if(CMAKE_BUILD_TYPE STREQUAL "Debug")
    get_target_property(MNN_LIB_PATH MNN::MNN IMPORTED_LOCATION_DEBUG)
elseif(CMAKE_BUILD_TYPE STREQUAL "Release")
    get_target_property(MNN_LIB_PATH MNN::MNN IMPORTED_LOCATION_RELEASE)
elseif(CMAKE_BUILD_TYPE STREQUAL "RelWithDebInfo")
    get_target_property(MNN_LIB_PATH MNN::MNN IMPORTED_LOCATION_RELWITHDEBINFO)
endif()

# zlib
set(ZLIB_ROOT "${_THIRDPARTY_BUILD_DIR}/zlib")

# Prefer package config files while loading Libssh2's dependencies.
# Libssh2 config calls `find_dependency(ZLIB)` without `CONFIG`, which can
# otherwise resolve to CMake's FindZLIB module on Windows CI.
set(_SGNS_CMAKE_FIND_PACKAGE_PREFER_CONFIG_WAS_DEFINED FALSE)
if(DEFINED CMAKE_FIND_PACKAGE_PREFER_CONFIG)
    set(_SGNS_CMAKE_FIND_PACKAGE_PREFER_CONFIG_WAS_DEFINED TRUE)
    set(_SGNS_CMAKE_FIND_PACKAGE_PREFER_CONFIG_PREV "${CMAKE_FIND_PACKAGE_PREFER_CONFIG}")
endif()
set(CMAKE_FIND_PACKAGE_PREFER_CONFIG ON)

# libssh2
set(Libssh2_DIR "${_THIRDPARTY_BUILD_DIR}/libssh2/lib/cmake/libssh2")
find_package(Libssh2 CONFIG REQUIRED)

if(_SGNS_CMAKE_FIND_PACKAGE_PREFER_CONFIG_WAS_DEFINED)
    set(CMAKE_FIND_PACKAGE_PREFER_CONFIG "${_SGNS_CMAKE_FIND_PACKAGE_PREFER_CONFIG_PREV}")
else()
    unset(CMAKE_FIND_PACKAGE_PREFER_CONFIG)
endif()
unset(_SGNS_CMAKE_FIND_PACKAGE_PREFER_CONFIG_PREV)
unset(_SGNS_CMAKE_FIND_PACKAGE_PREFER_CONFIG_WAS_DEFINED)

# AsyncioManager
set(AsyncIOManager_INCLUDE_DIR "${_THIRDPARTY_BUILD_DIR}/AsyncIOManager/include")
set(AsyncIOManager_DIR "${_THIRDPARTY_BUILD_DIR}/AsyncIOManager/lib/cmake/AsyncIOManager")
find_package(AsyncIOManager CONFIG REQUIRED)


include_directories(
        "${CMAKE_CURRENT_LIST_DIR}/../include"
)

add_subdirectory(${PROJECT_ROOT}/src ${CMAKE_BINARY_DIR}/src)
add_subdirectory(${PROJECT_ROOT}/test ${CMAKE_BINARY_DIR}/test)

# Install Headers
install(DIRECTORY "${CMAKE_SOURCE_DIR}/include/" DESTINATION "${CMAKE_INSTALL_INCLUDEDIR}/SGProcessingManager" FILES_MATCHING PATTERN "*.h*")
install(DIRECTORY "${CMAKE_SOURCE_DIR}/generated/" DESTINATION "${CMAKE_INSTALL_INCLUDEDIR}/SGProcessingManager/generated" FILES_MATCHING PATTERN "*.h*")
# install(TARGETS ${PROJECT_NAME} EXPORT SGProcessingManagerTargets
        # LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}
        # ARCHIVE DESTINATION ${CMAKE_INSTALL_LIBDIR}
        # RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR}
        # INCLUDES DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}
        # PUBLIC_HEADER DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}
        # FRAMEWORK DESTINATION ${CMAKE_INSTALL_PREFIX}
        # BUNDLE DESTINATION ${CMAKE_INSTALL_BINDIR}
# )

install(
        EXPORT SGProcessingManagerTargets
        DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/SGProcessingManager
        NAMESPACE sgns::
)

include(CMakePackageConfigHelpers)

# generate the config file that is includes the exports
configure_package_config_file(${PROJECT_ROOT}/cmake/config.cmake.in
        "${CMAKE_CURRENT_BINARY_DIR}/SGProcessingManagerConfig.cmake"
        INSTALL_DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/SGProcessingManager
        NO_SET_AND_CHECK_MACRO
        NO_CHECK_REQUIRED_COMPONENTS_MACRO
)

# generate the version file for the config file
write_basic_package_version_file(
        "${CMAKE_CURRENT_BINARY_DIR}/SGProcessingManagerConfigVersion.cmake"
        VERSION "${CPACK_PACKAGE_VERSION_MAJOR}.${CPACK_PACKAGE_VERSION_MINOR}.${CPACK_PACKAGE_VERSION_PATCH}"
        COMPATIBILITY AnyNewerVersion
)

# install the configuration file
install(FILES
        ${CMAKE_CURRENT_BINARY_DIR}/SGProcessingManagerConfigVersion.cmake
        DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/SGProcessingManager
)

install(FILES
        ${CMAKE_CURRENT_BINARY_DIR}/SGProcessingManagerConfig.cmake
        DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/SGProcessingManager
)
