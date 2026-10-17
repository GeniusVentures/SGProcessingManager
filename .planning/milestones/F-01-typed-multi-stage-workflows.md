# Feature Milestone F-01 — Typed Multi-Stage Expert Workflows

**Project:** SGProcessingManager  
**Status:** Planned (design stage)  
**Owner:** @itsafuu  
**Tracking issue:** [#21](https://github.com/GeniusVentures/SGProcessingManager/issues/21)  
**Design:** [Typed Multi-Stage Expert Workflows](../../doc/typed-multi-stage-workflows-design.md)  
**Created:** 2026-10-16  
**Target date:** TBD — no scheduling commitment

## Objective

Make the existing processing JSON actually compose multiple small models and safe native C++ operations **within one assigned SuperGenius worker**, with typed intermediate outputs, dynamic batching, conditional execution, and end-to-end observability. Prove it using a document OCR expert and preserve v1 job compatibility.

## Dependencies — existing work, not duplicated

- #8 pass/input correctness; #9 multi-pass dependency-graph execution; #10 strict typed named bindings.
- #11 pass/backend dispatch, #12 capability validation, #13 execution context controls.
- #14 structured artifacts/manifests; #15 conformance suites.

A prerequisite isn't declared complete merely because the code has types or a prior ticket exists. Each F-01 wave must verify the runtime path and tests.

## Waves and gates

### F-01.1 — Contract and JSON v2 grammar

- [ ] Document v1 vs v2 behavior, compatibility guarantees, type and shape contracts and errors.
- [ ] Define operator descriptors and typed `internal:` artifacts with provenance.
- [ ] Define bounded branch and dynamic fan-out/batching grammar, stable source IDs, declared output joins.
- [ ] Resolve how generated QuickType headers and version negotiation change.
- **Gate:** v1 fixtures unchanged; v2 graph validation rejects unsupported operators, cycles and mismatched bindings.

### F-01.2 — Safe native operators

- [ ] C++ allowlist registry and deterministic operation contracts.
- [ ] Image preprocessing, ROI cropping/coordinate propagation, width grouping, CTC decode, JSON/Markdown assembly.
- [ ] No arbitrary Python/shell/JS or uncontrolled `custom` execution.
- **Gate:** independent operator fixture tests, limits and cancellation.

### F-01.3 — Dynamic batches and internal artifacts

- [ ] Reuse #9/#10 runtime for one-to-many crops, shape-bucket batching, multi-model links.
- [ ] Use named, typed in-memory artifacts; explicit host/device moves and Vulkan synchronization.
- [ ] Stable reassembly into original page/line order with resource lifetime tracking.
- **Gate:** three dependent models and intermediate C++ operators run in one local job without file round trips.

### F-01.4 — Conditional flow and observability

- [ ] Typed condition predicates with skip/fallback semantics and bounded fan-out.
- [ ] Per-stage cancellation, deadline/memory/byte budgets, status, progress and provenance.
- [ ] Explicit failure policy, deterministic join behavior, hashes and manifest data.
- **Gate:** low-confidence branch and negative/security tests verified.

### F-01.5 — OCR reference, benchmarking and integration

- [ ] Independent MNN detector, recognizer and reading-order/layout models.
- [ ] Native PDF/text-layer handling where appropriate, and Markdown/structured JSON output.
- [ ] CTC and coordinate accuracy, empty page and ordering tests.
- [ ] ONNX Runtime vs MNN CPU/Vulkan FP32/FP16 vs eligible SGFP4 v2 measurements (accuracy, speed and memory).
- [ ] Integration through existing SuperGenius job execution, no new worker scheduler.
- **Gate:** 3-model end-to-end conformance and reproducible benchmark report; backward compatibility preserved.

## Done means

- One JSON v2 job performs end-to-end structured OCR on a processor node.
- Three MNN inference passes with native C++ operators; bounded variable line batches; one conditional route.
- Strict validation, typed intermediate resources, accurate artifact metadata and deterministic final assembly.
- Passing regression and security/conformance tests; no regression to legacy single-pass jobs.
- Documentation matches measured capability and specifies unsupported operators and cases.

## Exclusions

Distributed microstage scheduling, custom scripting, assumptions that all ONNX models convert to MNN, and assumption that SGFP4 improves OCR speed. Distillation/training a proprietary OCR model is a later workstream; proving workflow execution comes first.
