add_library(ProcessingBase STATIC ProcessingManager.cpp)


target_include_directories(ProcessingBase PUBLIC
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/../../include>
	$<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/../../generated>
	$<BUILD_INTERFACE:${Boost_INCLUDE_DIRS}>
    $<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}/SGProcessingManager/ProcessingBase>
)
target_link_libraries(
		ProcessingBase
		PUBLIC
        nlohmann_json::nlohmann_json
		PRIVATE
        Boost::system
)

sgnus_install(ProcessingBase)
