add_library(SGProcessors STATIC 
	processing_processor_mnn_image.cpp
	processing_processor_mnn_audio.cpp
	processing_processor_mnn_ml.cpp
	../../include/processors/processing_processor.hpp
	../../include/processors/processing_processor_mnn_audio.hpp
	../../include/processors/processing_processor_mnn_image.hpp
	../../include/processors/processing_processor_mnn_ml.hpp
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
