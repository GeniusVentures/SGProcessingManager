# Vulkan Validation Layers Deferral Decision

## Decision

Vulkan-ValidationLayers is explicitly **not vendored** and **not wired** into debug or CI builds for v1.0 of the render pipeline. No validation-layer extension or callback is requested at `vkCreateInstance` time in any code path this phase delivers.

## Rationale

CTX-04 scoped validation-layer vendoring out of Phase 1 to keep the net-new vendoring surface to `vk-bootstrap` only. Validation layers are a development and debugging aid — they validate correct API usage, not runtime-correctness requirements — and this phase's deliverable (CTX-01..03, DISP-01..03) does not depend on them:

- A headless Vulkan context can be created, tested for coexistence safety, and dispatched to without validation layers present.
- This project's CI has zero macOS Vulkan signal today regardless of validation-layer presence (a pre-existing gap tracked for Phase 4).
- Adding validation layers would introduce an additional vendoring step (`thirdparty/Vulkan-ValidationLayers`, its own submodule + ExternalProject_Add build wiring) plus platform-specific library path resolution at runtime (the loader must find `VK_LAYER_KHRONOS_validation` on disk, and the path differs across Windows/Linux/macOS/MoltenVK) — all for a tooling aid that has no effect on the shipped code path.

## Deferred To

Tracked as v2 requirement **VALLAYER-01** in `.planning/workstreams/sgproc-render/REQUIREMENTS.md` ("Vulkan-ValidationLayers wired into debug/CI builds"), to be implemented in v1.x.

## Future Hook Point

If implemented later, validation layers can be enabled via `vkb::InstanceBuilder::request_validation_layers(true)` (vk-bootstrap has first-class support for this). The toggle would be gated behind a new CMake debug-only option — analogous to the existing `SANITIZE_CODE` debug-only toggle pattern already present in `CommonCompilerOptions.cmake` — so it never activates in release/optimized builds.
