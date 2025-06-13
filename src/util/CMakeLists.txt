add_library(sgprocmanagerlogger
    logger.cpp
)
target_link_libraries(sgprocmanagerlogger
    PUBLIC
    spdlog::spdlog
)
sgnus_install(sgprocmanagerlogger)