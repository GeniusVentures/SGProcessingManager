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
    $<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}/SGProcessingManager/SGProcessors>
)
target_link_libraries(
		SGProcessors
		PUBLIC
		nlohmann_json::nlohmann_json
		spdlog::spdlog
		sgprocmanagerlogger
		MNN::MNN
		Vulkan::Vulkan
		OpenSSL::Crypto
)

sgnus_install(SGProcessors)
