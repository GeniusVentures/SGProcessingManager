# CapabilityValidator — pre-execution capability gate (Phase 06)
# Validates Vulkan device limits, MNN model compatibility, PassType registration,
# GPU memory, and disk space against a cached startup snapshot.

add_library(SGCapability STATIC
    capability_validator.cpp
    ../../include/capability/capability_validator.hpp
    ../../include/capability/capability_types.hpp
)

target_include_directories(SGCapability PUBLIC
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/../../include>
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/../../generated>
    $<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}/SGProcessingManager/generated>
    $<BUILD_INTERFACE:${Boost_INCLUDE_DIRS}>
    $<BUILD_INTERFACE:${OPENSSL_INCLUDE_DIR}>
)

target_link_libraries(SGCapability
    PUBLIC
    sgprocmanagerlogger
    sgprocmanagersha
    sgprocmanagertypes
    Vulkan::Vulkan
    OpenSSL::Crypto
)

sgnus_install(SGCapability)
