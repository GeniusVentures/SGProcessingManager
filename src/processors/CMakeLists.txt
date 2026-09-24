# Vulkan validation layers toggle (D-24, VVAL-03)
# ON in Debug builds (development convenience), OFF in Release (production).
# Best-effort: instance creation proceeds even if layers are not found (D-20).
string(TOUPPER "${CMAKE_BUILD_TYPE}" CMAKE_BUILD_TYPE_UPPER)
if(CMAKE_BUILD_TYPE_UPPER STREQUAL "DEBUG")
    set(_vulkan_val_default ON)
else()
    set(_vulkan_val_default OFF)
endif()
option(ENABLE_VULKAN_VALIDATION "Enable Vulkan validation layers for RenderProcessor (best-effort)" ${_vulkan_val_default})

# PROC-01 (Phase 04-sgprocessing-integration): MNN's autoregressive LLM engine
# (MNN::Transformer::Llm, <llm/llm.hpp>) is a separately-buildable MNN feature
# gated by MNN's own MNN_BUILD_LLM CMake option. The vendored MNN static
# library this target links has MNN_BUILD_LLM=ON in this checkout, but MNN's
# own install rules place the LLM headers at MNN_INCLUDE_DIR/llm/llm.hpp (a
# sibling of MNN_INCLUDE_DIR/MNN/), not the nested MNN_INCLUDE_DIR/MNN/llm/
# path the include convention would suggest -- confirmed by direct inspection
# on both Windows and macOS vendored builds. Detect this at configure time
# (mirroring this codebase's own established find_library(...)+if(TARGET/EXISTS...)
# unconditional-dependency pattern, see GNUS-NEO-SWARM/src/core/CMakeLists.txt's
# SGProcessingManager link block) rather than letting the whole SGProcessors
# target -- and every consumer of this shared submodule branch -- fail to
# build/link the moment this file is added.
#
# CACHE INTERNAL (not a plain directory-scoped set()) so test/processors/CMakeLists.txt
# -- a sibling subdirectory added separately from SGProcessingManager/CMakeLists.txt,
# outside this variable's normal directory-scope inheritance chain -- can read the same
# detection result without duplicating the EXISTS check.
if(EXISTS "${MNN_INCLUDE_DIR}/llm/llm.hpp")
    message(STATUS "MNN LLM engine headers found (${MNN_INCLUDE_DIR}/llm) -- building MNN_Llm processor (PROC-01)")
    set(SGPROC_HAS_MNN_LLM TRUE CACHE INTERNAL "MNN vendored build has MNN_BUILD_LLM support (PROC-01)")
else()
    message(STATUS "MNN LLM engine headers NOT found under MNN_INCLUDE_DIR -- MNN_Llm processor (PROC-01) source is "
                    "excluded from this build. The vendored MNN was built without LLM headers installed in this "
                    "checkout; DataType::LLM falls back to ProcessingManager's existing NO_PROCESSOR error path "
                    "until a future phase's MNN build installs them.")
    set(SGPROC_HAS_MNN_LLM FALSE CACHE INTERNAL "MNN vendored build has MNN_BUILD_LLM support (PROC-01)")
endif()

set(_SGPROC_LLM_SOURCES "")
set(_SGPROC_LLM_HEADERS "")
if(SGPROC_HAS_MNN_LLM)
    list(APPEND _SGPROC_LLM_SOURCES processing_processor_mnn_llm.cpp)
    list(APPEND _SGPROC_LLM_HEADERS ../../include/processors/processing_processor_mnn_llm.hpp)
    # ELM processor (elmbridge Phase 3, plan 03-03): same gate -- the whole
    # MNN surface sits inside #if defined(SGPROC_HAS_MNN_LLM) in the TU with a
    # fail-closed #else, but every leg needs the engine, so the source joins
    # the gated list.
    list(APPEND _SGPROC_LLM_SOURCES processing_processor_elm.cpp)
    list(APPEND _SGPROC_LLM_HEADERS ../../include/processors/processing_processor_elm.hpp)
endif()

# GNUS fork-patch detection (elmbridge Phase 3, D-14): grep the INSTALLED
# llm.hpp for kGnusLlmForkPatchLevel -- the marker the dev_elmruntime fork
# branch carries (sampler seed config key D-12 + Llm::cancel() D-13). A stock
# MNN checkout still builds: without the marker the processor's cancel
# trigger compiles to a documented degraded no-op path.
set(SGPROC_MNN_LLM_FORK_PATCHES FALSE CACHE INTERNAL "Vendored MNN llm.hpp carries the GNUS fork patches (D-14)")
if(SGPROC_HAS_MNN_LLM)
    file(READ "${MNN_INCLUDE_DIR}/llm/llm.hpp" _SGPROC_MNN_LLM_HPP)
    if(_SGPROC_MNN_LLM_HPP MATCHES "kGnusLlmForkPatchLevel")
        set(SGPROC_MNN_LLM_FORK_PATCHES TRUE CACHE INTERNAL "Vendored MNN llm.hpp carries the GNUS fork patches (D-14)")
        message(STATUS "MNN llm.hpp carries kGnusLlmForkPatchLevel -- SGPROC_MNN_LLM_FORK_PATCHES defined (elmbridge D-14; seeded sampling + Llm::cancel available)")
    else()
        message(STATUS "MNN llm.hpp does NOT carry kGnusLlmForkPatchLevel (stock MNN) -- SGPROC_MNN_LLM_FORK_PATCHES undefined; the ELM cancel trigger compiles to the documented degraded no-op path (elmbridge D-14)")
    endif()
endif()

add_library(SGProcessors STATIC
	processing_processor_mnn_image.cpp
	processing_processor_mnn_audio.cpp
	processing_processor_mnn_ml.cpp
	processing_processor_mnn_string.cpp
	processing_processor_mnn_bool.cpp
	processing_processor_mnn_buffer.cpp
	processing_processor_mnn_float.cpp
	processing_processor_mnn_int.cpp
	processing_processor_mnn_mat2.cpp
	processing_processor_mnn_mat3.cpp
	processing_processor_mnn_mat4.cpp
	processing_processor_mnn_vec2.cpp
	processing_processor_mnn_vec3.cpp
	processing_processor_mnn_vec4.cpp
	processing_processor_mnn_tensor.cpp
	processing_processor_mnn_texturecube.cpp
	processing_processor_mnn_texture1d.cpp
	processing_processor_mnn_volume.cpp
	processing_processor_render.cpp
	vulkan_gpu_probe.cpp
	${_SGPROC_LLM_SOURCES}
	../../include/processors/processing_processor.hpp
	../../include/processors/processing_processor_mnn_audio.hpp
	../../include/processors/processing_processor_mnn_image.hpp
	../../include/processors/processing_processor_mnn_ml.hpp
	../../include/processors/processing_processor_mnn_string.hpp
	../../include/processors/processing_processor_mnn_bool.hpp
	../../include/processors/processing_processor_mnn_buffer.hpp
	../../include/processors/processing_processor_mnn_float.hpp
	../../include/processors/processing_processor_mnn_int.hpp
	../../include/processors/processing_processor_mnn_mat2.hpp
	../../include/processors/processing_processor_mnn_mat3.hpp
	../../include/processors/processing_processor_mnn_mat4.hpp
	../../include/processors/processing_processor_mnn_vec2.hpp
	../../include/processors/processing_processor_mnn_vec3.hpp
	../../include/processors/processing_processor_mnn_vec4.hpp
	../../include/processors/processing_processor_mnn_tensor.hpp
	../../include/processors/processing_processor_mnn_texturecube.hpp
	../../include/processors/processing_processor_mnn_texture1d.hpp
	../../include/processors/processing_processor_mnn_volume.hpp
	../../include/processors/processing_processor_render.hpp
	../../include/processors/vulkan_gpu_probe.hpp
	${_SGPROC_LLM_HEADERS}
)

if(SGPROC_HAS_MNN_LLM)
    target_compile_definitions(SGProcessors PUBLIC SGPROC_HAS_MNN_LLM)
endif()

# Fork-patch compile marker (D-14): consumers can #ifdef on
# SGPROC_MNN_LLM_FORK_PATCHES to detect seeded sampling + Llm::cancel.
if(SGPROC_MNN_LLM_FORK_PATCHES)
    target_compile_definitions(SGProcessors PUBLIC SGPROC_MNN_LLM_FORK_PATCHES)
endif()


target_include_directories(SGProcessors PUBLIC
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/../../include>
	$<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/../../generated>
	$<BUILD_INTERFACE:${Boost_INCLUDE_DIRS}>
	$<BUILD_INTERFACE:${OPENSSL_INCLUDE_DIR}>
	$<BUILD_INTERFACE:${libp2p_INCLUDE_DIR}>
	$<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}/SGProcessingManager/generated>
)
target_link_libraries(
		SGProcessors
		PUBLIC
		nlohmann_json::nlohmann_json
		spdlog::spdlog
		sgprocmanagerlogger
		sgprocmanagertypes
		MNN::MNN
		Vulkan::Vulkan
		vk-bootstrap::vk-bootstrap
		OpenSSL::Crypto
		sgprocmanagersha
		sgprocmanagerquant
		sgprocmanagerdiff
		# ELM processor link closure (elmbridge Phase 3 gap, fixed in 04-01):
		# processing_processor_elm.cpp references sgns::elmruntime symbols
		# (ElmEnvelope/ToString, ElmModelCache, ElmEntryPreflight), but only
		# test/processors linked sgprocmanagerelmruntime explicitly -- any
		# OTHER consumer of SGProcessors (e.g. sgprocbase_elm_job_schema_test
		# via ProcessingBase) hit LNK2019. PUBLIC so the closure propagates;
		# no cycle (elmruntime links only util libs + MNN/Vulkan PRIVATE).
		sgprocmanagerelmruntime
)

# MNN registers its Vulkan backend op executors (Raster, Convolution, Eltwise,
# PReLU, BinaryOp, Deconvolution, ...) via static initializers inside each
# backend .obj member of the static MNN archive. The MSVC linker only pulls
# archive members that resolve a referenced symbol, and nothing references
# those registration objects -- so without forcing the whole archive in, the
# runtime creator map stays empty, every op logs "Vulkan don't support", and
# inference silently falls back to CPU despite a real GPU (observed: a ~3s
# GPU-calibrated test took 524s). MNN's own CMake wraps its deps in
# --whole-archive/-force_load for GNU/Clang static builds but has no MSVC
# equivalent, so apply it here. PUBLIC so every executable linking SGProcessors
# (and thereby MNN) gets it on its final link line -- same pattern as
# node_example's /WHOLEARCHIVE:$<TARGET_FILE:genius_node>.
if(ANDROID)
    # Deliberately NO whole-archive on Android: MNN's
    # source/utils/JNIHelper.cpp is globbed unconditionally into libMNN.a and
    # compiles an empty TU everywhere except Android, where it defines
    # JNI_OnLoad (for MNN's optional MNN_JNI java package, which this build
    # never enables -- nothing in MNN references it). The SDK's own
    # local_secure_storage/impl/Android.cpp also defines JNI_OnLoad and rides
    # into every link that pulls sgns_genius_account, so whole-archiving MNN
    # force-includes JNIHelper.o and collides: "ld.lld: error: duplicate
    # symbol: JNI_OnLoad" (run 35148585923: consensus_slot_key /
    # consensus_pending_lifecycle / validator_registry_batch_slot tests, all
    # four Android configs; the same closure would break GeniusWallet Android
    # too). Android CI builds but never runs tests, and Android MNN Vulkan
    # registration was never runtime-verified -- the pre-wholearchive status
    # quo there is CPU-fallback inference. Drop this guard once MNN's dead
    # JNIHelper TU is excluded from the fork (requires a thirdparty rebuild)
    # or the SDK's JNI entry point is renamed.
elseif(MSVC)
    target_link_options(SGProcessors PUBLIC "/WHOLEARCHIVE:$<TARGET_FILE:MNN::MNN>")
elseif(APPLE)
    target_link_options(SGProcessors PUBLIC "-force_load" "$<TARGET_FILE:MNN::MNN>")
else()
    target_link_options(SGProcessors PUBLIC
        "-Wl,--whole-archive"
        "$<TARGET_FILE:MNN::MNN>"
        "-Wl,--no-whole-archive"
    )
endif()

if(APPLE)
    target_link_libraries(SGProcessors PRIVATE
        "-framework CoreFoundation"
        "-framework CoreGraphics"
        "-framework CoreServices"
        "-framework IOKit"
        "-framework IOSurface"
        "-framework Metal"
        "-framework QuartzCore"
        "-framework Foundation"
    )

    if(CMAKE_SYSTEM_NAME STREQUAL "Darwin")
        # macOS specific
        target_link_libraries(SGProcessors PRIVATE
            "-framework AppKit")
    elseif(CMAKE_SYSTEM_NAME STREQUAL "iOS")
        # iOS specific
        target_link_libraries(SGProcessors PRIVATE
            "-framework UIKit")
    endif()
endif()

# Vulkan validation layers compile definition (D-23, D-24)
if(ENABLE_VULKAN_VALIDATION)
    target_compile_definitions(SGProcessors PUBLIC ENABLE_VULKAN_VALIDATION)
    message(STATUS "Vulkan validation layers: ENABLED (best-effort)")
else()
    message(STATUS "Vulkan validation layers: DISABLED")
endif()

sgnus_install(SGProcessors)
