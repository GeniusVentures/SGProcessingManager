add_library(sgprocmanagerlogger
    sgprocmgr-logger.cpp
	../../include/util/sgprocmgr-logger.hpp
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

add_library(sgprocmanagersha
    sha256.cpp
	../../include/util/sha256.hpp
)
target_include_directories(sgprocmanagersha PUBLIC
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/../../include>
	$<BUILD_INTERFACE:${OPENSSL_INCLUDE_DIR}>
    $<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}/SGProcessingManager/util>
)
target_link_libraries(sgprocmanagersha
    PUBLIC
    blob
    PRIVATE
    OpenSSL::Crypto
)
sgnus_install(sgprocmanagersha)