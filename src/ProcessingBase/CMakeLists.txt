add_library(ProcessingBase STATIC ProcessingManager.cpp)


target_include_directories(ProcessingBase PUBLIC
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/../../include>
	$<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/../../generated>
	$<BUILD_INTERFACE:${Boost_INCLUDE_DIRS}>
	$<BUILD_INTERFACE:${OPENSSL_INCLUDE_DIR}>
	$<BUILD_INTERFACE:${AsyncIOManager_INCLUDE_DIR}>
    $<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}/SGProcessingManager/ProcessingBase>
)
target_link_libraries(
		ProcessingBase
		PUBLIC
        nlohmann_json::nlohmann_json
		spdlog::spdlog
		sgprocmanagerlogger
		AsyncIOManager
		SGProcessingProto
)

sgnus_install(ProcessingBase)
