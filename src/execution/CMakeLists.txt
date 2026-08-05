add_library(SGExecution INTERFACE)
target_include_directories(SGExecution INTERFACE ${CMAKE_SOURCE_DIR}/include)
target_link_libraries(SGExecution INTERFACE SGProcessors)
