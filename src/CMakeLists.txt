add_subdirectory(util)
add_subdirectory(datasplitter)
add_subdirectory(processors)
# elmruntime AFTER processors: the SGPROC_HAS_MNN_LLM CACHE INTERNAL detection
# in src/processors/CMakeLists.txt is set before this subdirectory configures
# (belt-and-braces duplication happens in plan 02-03).
add_subdirectory(elmruntime)
add_subdirectory(processingbase)
add_subdirectory(shaders)
add_subdirectory(capability)
add_subdirectory(execution)
add_subdirectory(artifacts)
