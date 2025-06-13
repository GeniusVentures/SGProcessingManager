add_library(sgprocmanagerlogger
    sgprocmgr-logger.cpp
)
target_include_directories(sgprocmanagerlogger PUBLIC
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/../../include>
    $<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}/SGProcessingManager/util>
)
target_link_libraries(sgprocmanagerlogger
    PUBLIC
    spdlog::spdlog
)
sgnus_install(sgprocmanagerlogger)