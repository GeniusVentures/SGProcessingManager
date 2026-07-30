add_library(SGShaderCompiler STATIC
    shader_compiler.cpp
    ../../include/shaders/shader_compiler.hpp
)

target_include_directories(SGShaderCompiler PUBLIC
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/../../include>
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/../../generated>
    $<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}/SGProcessingManager/generated>
)

target_link_libraries(SGShaderCompiler PUBLIC
    sgprocmanagerlogger
    sgprocmanagertypes
    shaderc::shaderc
    SPIRV-Tools::SPIRV-Tools
)

sgnus_install(SGShaderCompiler)
