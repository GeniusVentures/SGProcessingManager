# elmruntime — ELM manifest + model cache layer (Phase 02, elmbridge)
#
# sgprocmanagerelmruntime holds the manifest verify/parse/gate front door
# (MCHE-01), the D-08 injectable FetchFn seam (production: FileManager only),
# and (later plans) the content-addressed model cache and smoke check.
#
# AsyncIOManager is linked PRIVATE: only ElmArtifactFetcher.cpp (the production
# FetchFn TU) needs FileManager -- same PRIVATE precedent as sgprocmanagersha's
# OpenSSL::Crypto link in src/util/CMakeLists.txt.

# SGPROC_HAS_MNN_LLM duplicate detection (plan 02-03, Task 1): src/processors/
# CMakeLists.txt:30-56 already sets this CACHE INTERNAL gate, and src/CMakeLists.txt
# adds `processors` before `elmruntime` -- but per RESEARCH RQ3 this elmruntime
# copy is belt-and-braces so the gate never depends on subdirectory ordering
# (a future reordering must not silently compile the probe's fallback out or in).
if(EXISTS "${MNN_INCLUDE_DIR}/llm/llm.hpp")
    set(SGPROC_HAS_MNN_LLM TRUE CACHE INTERNAL "MNN vendored build has MNN_BUILD_LLM support (elmruntime smoke check)")
else()
    set(SGPROC_HAS_MNN_LLM FALSE CACHE INTERNAL "MNN vendored build has MNN_BUILD_LLM support (elmruntime smoke check)")
endif()

add_library(sgprocmanagerelmruntime STATIC
    ElmRuntimeError.cpp
    ElmManifest.cpp
    ElmArtifactFetcher.cpp
    # Content-addressed model cache (plan 02-03, Task 2): publish/hit/single-flight/
    # pin/quarantine/LRU/restart-scan. AsyncIOManager (PRIVATE above) covers the
    # factory's FileManager usage; no new link deps.
    ElmModelCache.cpp
    # Self-gating TU: the whole MNN surface sits inside #if defined(SGPROC_HAS_MNN_LLM);
    # compiled unconditionally so the fail-closed fallback always links.
    ElmSmokeCheck.cpp
    # Result envelope + stop-string streambuf (elmbridge Phase 3, plan 03-03):
    # pure no-MNN units -- the envelope is the RES-01 result shape, the
    # streambuf is the D-05/D-06/D-07 stop-string observer with the injectable
    # cancel hook the gated processor TU wires to Llm::cancel().
    ElmEnvelope.cpp
    ElmStopStringStreamBuf.cpp
    # Plain-value pinned-entry preflight bridge (plan 03-03): holds the
    # fallback generated-set includes so the processor TU (root generated set)
    # never mixes the two quicktype families.
    ElmEntryPreflight.cpp
    ../../include/elmruntime/ElmRuntimeError.hpp
    ../../include/elmruntime/ElmEnvelope.hpp
    ../../include/elmruntime/ElmEntryPreflight.hpp
    ../../include/elmruntime/ElmStopStringStreamBuf.hpp
    ../../include/elmruntime/ElmManifest.hpp
    ../../include/elmruntime/ElmArtifactFetcher.hpp
    ../../include/elmruntime/ElmResourcePreflight.hpp
    ../../include/elmruntime/ElmModelCache.hpp
    ../../include/elmruntime/ElmSmokeCheck.hpp
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

# The gated smoke-check TU (ElmSmokeCheck.cpp) needs the LLM engine + Vulkan
# context only when present -- the same link set SGProcessors uses at
# src/processors/CMakeLists.txt:114 (MNN::MNN already carries the LLM engine).
if(SGPROC_HAS_MNN_LLM)
    target_compile_definitions(sgprocmanagerelmruntime PRIVATE SGPROC_HAS_MNN_LLM)
    target_link_libraries(sgprocmanagerelmruntime PRIVATE MNN::MNN Vulkan::Vulkan)
endif()

sgnus_install(sgprocmanagerelmruntime)
