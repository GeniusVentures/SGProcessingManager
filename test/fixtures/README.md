# ELM Test Fixtures

## elm-test-model — Qwen2.5-0.5B-Instruct MNN bundle

Real-model fixture for the ELM processor conformance tests
(`test/processors/elm_processor_test.cpp`, elmbridge Phase 3 plan 03-04).
Staged LOCALLY via plain HTTP (zero new build dependencies); binaries are
git-ignored, only this README (with recorded provenance) is committed.

### Staging

Set `SGPROC_ELM_TEST_MODEL_DIR` to this directory's absolute path and ensure
the six required files below are present:

| File | Bytes | sha256 |
|------|-------|--------|
| llm_config.json | 272 | ec05709b4261d59b510a0b7a636c6dcb6c5635c08fee7eb3c4f04188b509694b |
| llm.mnn | 566264 | 480da511e603bd82f8d4af4e1f778ad72baadf8307f3585465ad9a94daca1a88 |
| llm.mnn.json | 2808932 | 245ce4289f456dcb371a8f8deabf75c3c4ee75f34b19e0d9723ba09b2fbacf8c |
| llm.mnn.weight | 277967498 | 7ed0f4dcdd31dca15fcb548d2fc8b63b0014031fbd5f627508435726f90c75da |
| embeddings_bf16.bin | 272269312 | 4e96b0df6d274768cbb7e72404011853d23349999b658dc2f4dfb3c431ea223f |
| tokenizer.txt | 3193477 | b86f1298a0d6a1b2f312946c2f674f883f1d134ccabc79c42dd4c6b5beadf37b |

Total ~557 MB — verify sizes after download; a truncated `llm.mnn.weight`
fails later inside `Llm::load()` with INTERNAL_ERROR.

**Source:** ModelScope repo `MNN/Qwen2.5-0.5B-Instruct-MNN` (official Alibaba
MNN org conversion, Apache-2.0), fetched per file from
`https://modelscope.cn/models/MNN/Qwen2.5-0.5B-Instruct-MNN/resolve/master/<file>`
with `curl.exe -L -o`. HuggingFace mirror fallback:
`https://huggingface.co/Qwen/Qwen2.5-0.5B-Instruct-MNN/resolve/main/<file>`
(same file set; may need an auth token).

**Re-staging on any dev machine** (PowerShell):

```powershell
$dir = "SuperGenius/SGProcessingManager/test/fixtures/elm-test-model"
New-Item -ItemType Directory -Force -Path $dir | Out-Null
$base = "https://modelscope.cn/models/MNN/Qwen2.5-0.5B-Instruct-MNN/resolve/master/"
"llm_config.json","llm.mnn","llm.mnn.json","llm.mnn.weight","embeddings_bf16.bin","tokenizer.txt" |
    ForEach-Object { curl.exe -L -sS -o "$dir/$_" ($base + $_) }
# Verify hashes against the table above:
Get-ChildItem $dir -File | ForEach-Object { "{0} {1}" -f $_.Name, (Get-FileHash $_.FullName -Algorithm SHA256).Hash.ToLower() }
$env:SGPROC_ELM_TEST_MODEL_DIR = (Resolve-Path $dir).Path
```

A synthesized `elm_manifest.json` (Phase 2 manifest format over the four
required roles plus the optional `embedding_file` role) also lives in this
directory; the test's FetchFn serves the bundle through the normal
`ElmModelCache::Acquire` verification pipeline — the fixture is never loaded
raw (hash-verified manifest + artifacts, the same path as production).
Regenerate it by re-running the hash-verification step and rebuilding the
JSON per the manifest schema (`schema_version` 1, `elm_type` `causal_lm`,
`model_format` `mnn`, artifacts with name/uri/sha256/size_bytes), then
recompute `sha256(elm_manifest.json)` if a test needs the declared hash.

### Env var convention

`SGPROC_ELM_TEST_MODEL_DIR` = absolute path to this directory. When unset,
fixture legs report `GTEST_SKIP` with a cross-reference to this file — never
a silent pass.

### Embedding role (resolved by D-03, Phase 4 plan 04-01)

This model requires `embeddings_bf16.bin` at runtime (MNN's `DiskEmbedding`
opens it by default; the bundle's `llm_config.json` declares no
`tie_embeddings`). Formerly a known gap — the test injected the file into
the pinned entry after `Acquire` — the role set now includes the optional
`embedding_file` role (materialized at `embeddings_bf16.bin`), so the cache
publishes it as a hash-verified artifact and the injection workaround is
retired. `llm.mnn.json` (LoRA/GPTQ material `Llm::load` never reads) is no
longer declared or fetched.
