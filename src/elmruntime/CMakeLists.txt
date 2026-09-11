# elmruntime — ELM manifest + model cache layer (Phase 02, elmbridge)
#
# sgprocmanagerelmruntime holds the manifest verify/parse/gate front door
# (MCHE-01), the D-08 injectable FetchFn seam (production: FileManager only),
# and (later plans) the content-addressed model cache and smoke check.
#
# AsyncIOManager is linked PRIVATE: only ElmArtifactFetcher.cpp (the production
# FetchFn TU) needs FileManager -- same PRIVATE precedent as sgprocmanagersha's
# OpenSSL::Crypto link in src/util/CMakeLists.txt.

add_library(sgprocmanagerelmruntime STATIC
    ElmRuntimeError.cpp
    ElmManifest.cpp
    ElmArtifactFetcher.cpp
    ../../include/elmruntime/ElmRuntimeError.hpp
    ../../include/elmruntime/ElmManifest.hpp
    ../../include/elmruntime/ElmArtifactFetcher.hpp
)

target_include_directories(sgprocmanagerelmruntime PUBLIC
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/../../include>
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/../../generated>
    $<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}/SGProcessingManager/generated>
    $<BUILD_INTERFACE:${Boost_INCLUDE_DIRS}>
    $<BUILD_INTERFACE:${OPENSSL_INCLUDE_DIR}>
    # elmruntime headers include <outcome/sgprocmgr-outcome.hpp>, which pulls
    # <libp2p/outcome/outcome.hpp> -- expose libp2p PUBLIC the same way
    # sgprocmanagertypes does (src/util/CMakeLists.txt) so consumers of the
    # public headers resolve it.
    $<BUILD_INTERFACE:${libp2p_INCLUDE_DIR}>
    # PRIVATE visibility for the one production FetchFn TU (ElmArtifactFetcher.cpp)
    PRIVATE
    ${AsyncIOManager_INCLUDE_DIR}
)

target_link_libraries(sgprocmanagerelmruntime
    PUBLIC
    sgprocmanagerlogger
    sgprocmanagersha
    nlohmann_json::nlohmann_json
    spdlog::spdlog
    PRIVATE
    AsyncIOManager
)

sgnus_install(sgprocmanagerelmruntime)
