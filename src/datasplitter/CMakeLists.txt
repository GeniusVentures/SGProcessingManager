add_library(DataSplitter STATIC 
	ImageSplitter.cpp
	../../include/datasplitter/ImageSplitter.hpp
	)


target_include_directories(DataSplitter PUBLIC
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/../../include>
	$<BUILD_INTERFACE:${Boost_INCLUDE_DIRS}>
	$<BUILD_INTERFACE:${OPENSSL_INCLUDE_DIR}>
)
target_link_libraries(
		DataSplitter
		PUBLIC
		spdlog::spdlog
		sgprocmanagerlogger
		OpenSSL::Crypto
)

sgnus_install(DataSplitter)
