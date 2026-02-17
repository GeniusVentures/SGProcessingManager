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
	processing_processor_mnn_tensor.cpp
	processing_processor_mnn_texturecube.cpp
	processing_processor_mnn_texture1d.cpp
	processing_processor_mnn_volume.cpp
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
	../../include/processors/processing_processor_mnn_tensor.hpp
	../../include/processors/processing_processor_mnn_texturecube.hpp
	../../include/processors/processing_processor_mnn_texture1d.hpp
	../../include/processors/processing_processor_mnn_volume.hpp
)


target_include_directories(SGProcessors PUBLIC
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/../../include>
	$<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/../../generated>
	$<BUILD_INTERFACE:${Boost_INCLUDE_DIRS}>
	$<BUILD_INTERFACE:${OPENSSL_INCLUDE_DIR}>
	$<BUILD_INTERFACE:${libp2p_INCLUDE_DIR}>
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

sgnus_install(SGProcessors)
