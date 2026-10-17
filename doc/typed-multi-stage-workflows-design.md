# Typed Multi-Stage Expert Workflows — SGProcessingManager

**Status:** Proposed / implementation not started  
**Date:** 2026-10-16  
**Feature milestone:** [F-01](../.planning/milestones/F-01-typed-multi-stage-workflows.md)  
**Issue:** [#21](https://github.com/GeniusVentures/SGProcessingManager/issues/21)  
**Owner:** @itsafuu

## Purpose and boundaries

Enable JSON-defined, **native, typed in-node workflows** that compose multiple small MNN models with safe C++ preprocessing and postprocessing operations. The reference use case is OCR/document extraction, but the capability must support image, audio, embedding, reranking and other expert processors.

This does **not** introduce a GCS scheduler. GCS defines/funds a job; SuperGenius owns node participation, task assignment, and processing grid; SGProcessingManager executes local processing steps on the chosen node. Avoid cross-node transfers of intermediate page/line tensors.

## Verified current baseline

- `gnus-processing-schema.json` declares `passes[]`, typed job I/O, model `input_nodes`/`output_nodes` with `input:`, `internal:`, and `output:` references, a `data_transform` pass kind and transform descriptions.
- `ProcessingManager::ProcessInternal()` currently resolves `model.get_source()` through `GetInputIndex()`, selects `processing_.get_passes()[index]`, and runs **one** `m_processor->StartProcessing(...)`; it does not execute the declared list as a dependency graph.
- Render input bindings explicitly reject cross-pass `internal:` and `output:` references, noting missing dependency resolution.
- `MNN_Tensor` implements splitting/window-based overlap stitching **within one pass**, not model-to-model workflow composition.
- `ExecutionContext` already supports cancellation, progress, deadlines and resource budgets. `ProcessOutput` contains typed `Artifact` records and an `ExecutionManifest`, but some metadata is inferred from the pass input, not the producer's actual output.
- Main branch exposes `MNN` inference; ONNX and other formats are recognized in the schema but aren't executable without conversion. SGFP4 v2 weights and `FP4_ULTRA` input tensors are different formats. Do not mix their semantics.
- Tests validating that a JSON has multiple passes **do not prove** end-to-end graph execution.

## Related issues and anti-duplication

| Issue | Responsibility | Relationship |
| --- | --- | --- |
| [#8](https://github.com/GeniusVentures/SGProcessingManager/issues/8) | Pass/input correctness | Existing foundation |
| [#9](https://github.com/GeniusVentures/SGProcessingManager/issues/9) | Ordered pass graph, branch and merge execution | **Prerequisite; do not duplicate** |
| [#10](https://github.com/GeniusVentures/SGProcessingManager/issues/10) | Named typed input/output bindings, internal resources | **Prerequisite; do not duplicate** |
| [#11](https://github.com/GeniusVentures/SGProcessingManager/issues/11) | Executor and backend dispatch | Reuse |
| [#12](https://github.com/GeniusVentures/SGProcessingManager/issues/12) | Capability validation | Reuse |
| [#13](https://github.com/GeniusVentures/SGProcessingManager/issues/13) | Cancellation, deadlines, budgets and progress | Reuse |
| [#14](https://github.com/GeniusVentures/SGProcessingManager/issues/14) | Typed artifacts and manifests | Extend output fidelity |
| [#15](https://github.com/GeniusVentures/SGProcessingManager/issues/15) | Conformance tests | Add workflow cases |

F-01 fills the **application-level processing gaps remaining after** #9/#10: safe native operations, variable-sized fan-out, batching, conditional execution, result assembly, and OCR integration. It must consume the existing graph/bindings implementation rather than introducing a second one.

## Proposed execution model

```text
SuperGenius funded task / local subtask
  └── SGProcessingManager JSON workflow
       ├── validate schema and capabilities
       ├── compile typed dependency graph (reuse #9/#10)
       ├── allocate scoped internal artifacts/buffers
       ├── run registered C++ operators and MNN inference
       ├── branch on bounded typed predicates / fan out to batches
       ├── join results by stable source IDs
       ├── enforce per-stage and job budgets
       └── return typed artifacts, provenance and manifest
```

### 1. Contract, versioning, and validation

Extend the **existing** `gnus-processing-schema.json` as an opt-in v2 grammar; retain v1 single-pass processing unmodified. `passes[]` remains canonical; no parallel `stages[]` engine. Normalize reference syntax and distinguish pass IDs, input names, resource IDs and output names. Version the additional workflow fields. Schema changes must regenerate QuickType headers using the repository's documented mechanism and pass compatibility fixtures.

Validate before loading models or invoking drivers: unique names, defined producer/consumer references, data type/precision/layout/shape compatibility (including `-1` dynamic dimensions), executable operators and backends, acyclic dependencies, deterministic branch destinations, bounds and resource limits. Incompatible conversions need explicit `data_transform` passes. Reject unknown `custom` functions.

### 2. Typed intermediate resources

Adopt #10 named `internal:` bindings as the resource contract. Introduce an internal run-scoped resource table keyed by stable producer/output identity. An entry exposes actual producer type, shape, dtype, strides/layout, format, residency (host/device), byte size, provenance and consumer reference count. Track memory ownership, read/write permissions, and last-use cleanup; never assume the input dtype describes the output. Handle Vulkan synchronization and CPU/GPU transfers explicitly. Preserve buffers between adjacent on-node stages; only persist declared final outputs (or explicitly requested checkpoints). Build on #14 artifact metadata and hashing for durable outputs.

### 3. Safe native operation registry

Registry: `operation_name + version -> contract validator + C++ implementation`. It should be compile-time/allowlist controlled, deterministic for fixed inputs where feasible, resource-limited, and independent of Python. No arbitrary shell, JS, Python, dynamically supplied shared libraries, or executable `custom` operations.

Initial operations:
- image decode/color normalization/deskew/rotation
- crop or ROI extraction from detector bounding boxes; coordinate transforms
- group/sort line crops by width and dynamic batching
- CTC greedy decoding (later optional beam decoding) into UTF-8 with confidence
- reading-order and document text/layout assembly into typed JSON and Markdown
- explicit tensor layout/cast/reshape/transfer as required

PDF handling: expose a native PDF/text-layer adapter with a carefully constrained file/URL access policy. OCR bypass for trusted text layers is an optimization, not an unconditional substitute for text verification.

### 4. Dynamic fan-out, batching and join

One detector pass can emit variable-length bounding boxes; a native crop operator fans out **bounded** line tensors; a batcher groups by width/model constraints; a recognizer executes one or more batches; a join reconstructs results by `page_id`, `line_id`, box and original order. This is local data-parallel work, not a new SuperGenius worker selection algorithm. Define maximum pages, regions, input pixel budgets, batch length, and intermediate bytes.

Use explicit shapes, masks/padding and length tensors for recognizers that require them. Reuse loaded MNN sessions when safe. Batch grouping must not reorder final output.

### 5. Conditional execution and failure policy

Add deterministic, allowlisted typed predicates, e.g. `page.has_trusted_text_layer`, `recognize.confidence < threshold`, or `document.contains_math`. Conditions may select an existing pass/subgraph or skip it, never dynamically load arbitrary instructions. Unsupported predicates fail validation. Define a typed `skipped` state and downstream dependencies/optional outputs. Limit retry count, total executed stages, fan-out, and recursion (initially forbid cycles and recursion). Fail closed or route a low-confidence region to a configured fallback without mutating existing outputs silently.

### 6. Execution lifecycle, security and observability

Inherit #12/#13 capability gates, cancellation and deadlines. Add typed stage results, exact stage failure IDs, progress per stage and aggregate job progress, retries only for declared idempotent operations, deterministic hashes of inputs/models/config/outputs, and immutable output provenance. Bound model file size, decoded image size, crop count, tensor dimensions, memory, per-stage time, output bytes, and document/page counts; prevent decompression bombs and resource exhaustion. Enforce existing input fetch/sandbox capabilities and privacy rules. No file writes outside declared destinations. Validate model manifests and checksums through existing content-addressed mechanisms if/when implemented—do not assume cache features from plans are production.

### 7. Proposed JSON extension — illustrative, not executable today

Retain existing required root `name`, `version`, `gnus_spec_version`, `inputs`, `outputs` and `passes`. The snippet below emphasizes **new fields**; exact serialized binding and model configurations must be reconciled with generated C++ schema types during F-01.1.

```json
{
  "name": "document_ocr",
  "version": "2.0.0",
  "gnus_spec_version": 2.0,
  "inputs": [{ "name": "page", "type": "texture2D",
               "source_uri_param": "file://page.rgba" }],
  "outputs": [{ "name": "document", "type": "buffer",
                "source_uri_param": "file://out/document.json" }],
  "passes": [
    {
      "name": "detect",
      "type": "inference",
      "model": {
        "format": "MNN",
        "source_uri_param": "ipfs://detector-cid",
        "input_nodes": [{ "name": "image", "type": "texture2D",
                          "source": "input:page" }],
        "output_nodes": [{ "name": "boxes", "type": "tensor",
                           "target": "internal:boxes" }]
      }
    },
    {
      "name": "crop_and_batch",
      "type": "data_transform",
      "operator": "ocr.crop_and_batch.v1",
      "inputs": [
        { "name": "page", "source": "input:page" },
        { "name": "boxes", "source": "internal:boxes" }
      ],
      "outputs": [{ "name": "crops", "target": "internal:crops" }],
      "batching": { "max_batch": 64, "group_by": "width" },
      "fanout_limit": 4096
    },
    {
      "name": "recognize",
      "type": "inference",
      "model": {
        "format": "MNN",
        "source_uri_param": "ipfs://recognizer-cid",
        "input_nodes": [{ "name": "images", "type": "tensor",
                          "source": "internal:crops", "shape": [-1, 1, 32, -1] }],
        "output_nodes": [{ "name": "logits", "type": "tensor",
                           "target": "internal:logits" }]
      }
    },
    {
      "name": "decode",
      "type": "data_transform",
      "operator": "ocr.ctc_decode.v1",
      "inputs": [{ "name": "logits", "source": "internal:logits" }],
      "outputs": [{ "name": "lines", "target": "internal:lines" }]
    },
    {
      "name": "assemble",
      "type": "data_transform",
      "operator": "ocr.assemble.v1",
      "inputs": [{ "name": "lines", "source": "internal:lines" }],
      "outputs": [{ "name": "document", "target": "output:document" }]
    }
  ]
}
```

This initial walkthrough intentionally omits optional orientation, CJK routing, layout-model stages, native PDF fast path and conditional fallback: those are explicit milestones, not implied by the minimal example. All missing required dimensions/formats and output schemas must be finalized before implementing the v2 grammar. Never call this sample valid v1 JSON.

### 8. Model and quantization

A model bundle contains independent compatible models (detector, CTC recognizer, layout/order), each with immutable model hash, declared I/O, revision, permissible license and intended quantization. Validate that MNN supports each converted ONNX graph including variable-width CTC output. SGFP4 v2 applies to supported weight operators, distinct from `FP4_ULTRA` packed *input* data. Quantization acceptance requires end-to-end OCR metrics and confusion/character accuracy checks, not just model-size wins; tiny models may be faster in CPU FP16 than GPU SGFP4 at batch one.

Benchmark ONNX Runtime CPU, ONNX Runtime GPU where available, MNN CPU/Vulkan FP32/FP16, and eligible SGFP4 v2 models. Time PDF decoding, detector, crops, batch setup, inference, CTC, assembly, transfers, cold/warm model load. Report throughput, p50/p95 latency, peak memory and accuracy. Do not presume distillation permissions from code license; Indra's model weights have separate terms.

## Implementation waves (F-01)

1. **F-01.1 contracts:** finalize schema extensions/types, serialization, validation, backward compatibility and reference fixtures; reuse #9/#10.
2. **F-01.2 native operators:** registry + bounded native image/crop/CTC/assembly operators; error and dtype/shape contracts.
3. **F-01.3 variable batches:** dynamic shapes, bounds, grouping/fan-out and stable deterministic joins; shared in-node buffers.
4. **F-01.4 conditional execution:** typed predicate grammar, explicit skip/fallback semantics, storage lifetime, cancellation, progress and budgets.
5. **F-01.5 reference OCR integration:** 3+ MNN inference passes, native preprocessing/postprocessing, structured outputs, CPU/Vulkan fixtures and benchmark report.

## Tests and exit criteria

- A single JSON execution invokes **at least 3 separate MNN models** end-to-end on one node; native operators connect detector, recognizer, and reading-order model with no intermediate file/IPFS transfers.
- Output is deterministic ordered JSON/Markdown with UTF-8 text, page/line IDs, bounding boxes and calibrated/defined confidence fields.
- Batching and joins preserve IDs, order, and boxes across varying crop counts/shapes; empty pages and no-text documents are supported.
- Condition path for trustworthy PDF text layer or low-confidence OCR escalation is tested with exact skipped-stage metadata.
- Missing dependencies, cycles, duplicate writers, format/shape mismatch, unsupported operators, excess fan-out, GPU OOM/budget, cancellations, deadlines, and failed stages result in defined errors with no leaked resources.
- CPU baseline and Vulkan results are compared where hardware permits, including repeatable accuracy and performance measurements.
- Existing v1 configurations and processing-conformance suites pass unmodified. The JSON schema and regenerated C++ headers round-trip.
- Tests establish the actual behavior: schema parsing/multiple-pass existence alone is insufficient.

## Non-goals

- Python execution on workers; arbitrary user-code plugins.
- Distributed microstage routing or a new GCS scheduler.
- Automatic commercial rights to third-party OCR model weights.
- Automatic FP4 conversion or an unsupported ONNX runtime.
- Full PDF authoring, handwriting support, or generalized vision-language understanding as acceptance prerequisites.

## Risks / open decisions for F-01.1

- How to version `gnus_spec_version` and generated enums compatibly.
- Whether predicates are attached to `passes[]` or a new typed `when` field with no dynamic code.
- How to represent variable-rank `boxes` and nested line batches without forcing them into flat 1D tensor assumptions.
- Whether backend session caching and zero-copy tensor handoff are safe across MNN/Vulkan lifecycle boundaries.
- How to scope native PDF decoders and third-party model licensing.

**Decision request:** implement one additive processing-graph runtime that serves all expert types; use OCR as the proving workload, not a special-purpose Python-equivalent OCR scripting system.
