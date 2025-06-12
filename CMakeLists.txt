cmake_minimum_required(VERSION 3.16)

# Project definition
project(SGProcessingManager
    VERSION 1.0.0
    DESCRIPTION "GNUS.ai Processing Definition Schema C++ Headers"
    LANGUAGES CXX
)

# Only set C++ standard if this is the main project
if(CMAKE_PROJECT_NAME STREQUAL PROJECT_NAME)
    set(CMAKE_CXX_STANDARD 17)
    set(CMAKE_CXX_STANDARD_REQUIRED ON)
endif()

# Find required dependencies
find_package(Boost REQUIRED)
find_package(nlohmann_json REQUIRED)

# Create interface library (header-only)
add_library(SGProcessingManager INTERFACE)
add_library(SGProcessingManager::SGProcessingManager ALIAS SGProcessingManager)

# Set include directories
target_include_directories(SGProcessingManager
    INTERFACE
        ${CMAKE_CURRENT_SOURCE_DIR}/generated
)

# Link dependencies
target_link_libraries(SGProcessingManager
    INTERFACE
        Boost::boost
        nlohmann_json::nlohmann_json
)

# Require C++17 for consumers
target_compile_features(SGProcessingManager INTERFACE cxx_std_17)