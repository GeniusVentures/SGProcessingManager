# SGArtifacts — artifact type system + deterministic binary serialization (Phase 08)
# Provides: artifact_types.hpp, execution_manifest.hpp, artifact_serializer.hpp/.cpp

add_library(SGArtifacts STATIC
    artifact_serializer.cpp
)

target_include_directories(SGArtifacts PUBLIC
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/../../include>
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/../../generated>
    $<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}/SGProcessingManager>
)

target_link_libraries(SGArtifacts
    PUBLIC
    sgprocmanagersha
)

sgnus_install(SGArtifacts)
