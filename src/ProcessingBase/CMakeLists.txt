add_library(ProcessingBase STATIC ProcessingManager.cpp)
target_include_directories(ProcessingBase PRIVATE ${Boost_INCLUDE_DIRS})

target_include_directories(ProcessingBase PUBLIC
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/../../include>
	$<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/../../generated>
    $<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}/SGProcessingManager/ProcessingBase>
)
target_link_libraries(
		ProcessingBase
		PUBLIC
        Boost::headers
        nlohmann_json::nlohmann_json
)

sgnus_install(ProcessingBase)
