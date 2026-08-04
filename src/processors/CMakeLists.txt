# Vulkan validation layers toggle (D-24, VVAL-03)
# ON in Debug builds (development convenience), OFF in Release (production).
# Best-effort: instance creation proceeds even if layers are not found (D-20).
string(TOUPPER "${CMAKE_BUILD_TYPE}" CMAKE_BUILD_TYPE_UPPER)
if(CMAKE_BUILD_TYPE_UPPER STREQUAL "DEBUG")
    set(_vulkan_val_default ON)
else()
    set(_vulkan_val_default OFF)
endif()
option(ENABLE_VULKAN_VALIDATION "Enable Vulkan validation layers for RenderProcessor (best-effort)" ${_vulkan_val_default})

add_library(SGProcessors STATIC 
	processing_processor_mnn_image.cpp
	processing_processor_mnn_audio.cpp
	processing_processor_mnn_ml.cpp
	processing_processor_mnn_string.cpp
	processing_processor_mnn_bool.cpp
	processing_processor_mnn_buffer.cpp
	processing_processor_mnn_float.cpp
	processing_processor_mnn_int.cpp
	processing_processor_mnn_mat2.cpp
	processing_processor_mnn_mat3.cpp
	processing_processor_mnn_mat4.cpp
	processing_processor_mnn_vec2.cpp
	processing_processor_mnn_vec3.cpp
	processing_processor_mnn_vec4.cpp
	processing_processor_mnn_tensor.cpp
	processing_processor_mnn_texturecube.cpp
	processing_processor_mnn_texture1d.cpp
	processing_processor_mnn_volume.cpp
	processing_processor_render.cpp
	vulkan_gpu_probe.cpp
	../../include/processors/processing_processor.hpp
	../../include/processors/processing_processor_mnn_audio.hpp
	../../include/processors/processing_processor_mnn_image.hpp
	../../include/processors/processing_processor_mnn_ml.hpp
	../../include/processors/processing_processor_mnn_string.hpp
	../../include/processors/processing_processor_mnn_bool.hpp
	../../include/processors/processing_processor_mnn_buffer.hpp
	../../include/processors/processing_processor_mnn_float.hpp
	../../include/processors/processing_processor_mnn_int.hpp
	../../include/processors/processing_processor_mnn_mat2.hpp
	../../include/processors/processing_processor_mnn_mat3.hpp
	../../include/processors/processing_processor_mnn_mat4.hpp
	../../include/processors/processing_processor_mnn_vec2.hpp
	../../include/processors/processing_processor_mnn_vec3.hpp
	../../include/processors/processing_processor_mnn_vec4.hpp
	../../include/processors/processing_processor_mnn_tensor.hpp
	../../include/processors/processing_processor_mnn_texturecube.hpp
	../../include/processors/processing_processor_mnn_texture1d.hpp
	../../include/processors/processing_processor_mnn_volume.hpp
	../../include/processors/processing_processor_render.hpp
	../../include/processors/vulkan_gpu_probe.hpp
)


target_include_directories(SGProcessors PUBLIC
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/../../include>
	$<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/../../generated>
	$<BUILD_INTERFACE:${Boost_INCLUDE_DIRS}>
	$<BUILD_INTERFACE:${OPENSSL_INCLUDE_DIR}>
	$<BUILD_INTERFACE:${libp2p_INCLUDE_DIR}>
	$<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}/SGProcessingManager/generated>
)
target_link_libraries(
		SGProcessors
		PUBLIC
		nlohmann_json::nlohmann_json
		spdlog::spdlog
		sgprocmanagerlogger
		sgprocmanagertypes
		MNN::MNN
		Vulkan::Vulkan
		vk-bootstrap::vk-bootstrap
		OpenSSL::Crypto
		sgprocmanagersha
)

if(APPLE)
    target_link_libraries(SGProcessors PRIVATE
        "-framework CoreFoundation"
        "-framework CoreGraphics"
        "-framework CoreServices"
        "-framework IOKit"
        "-framework IOSurface"
        "-framework Metal"
        "-framework QuartzCore"
        "-framework Foundation"
    )

    if(CMAKE_SYSTEM_NAME STREQUAL "Darwin")
        # macOS specific
        target_link_libraries(SGProcessors PRIVATE
            "-framework AppKit")
    elseif(CMAKE_SYSTEM_NAME STREQUAL "iOS")
        # iOS specific
        target_link_libraries(SGProcessors PRIVATE
            "-framework UIKit")
    endif()
endif()

# Vulkan validation layers compile definition (D-23, D-24)
if(ENABLE_VULKAN_VALIDATION)
    target_compile_definitions(SGProcessors PUBLIC ENABLE_VULKAN_VALIDATION)
    message(STATUS "Vulkan validation layers: ENABLED (best-effort)")
else()
    message(STATUS "Vulkan validation layers: DISABLED")
endif()

sgnus_install(SGProcessors)
