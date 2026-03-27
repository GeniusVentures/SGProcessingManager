# Processing JSON Guide

This guide describes how to write a processing definition JSON, and lists required and optional fields per data type. The authoritative schema is in [gnus-processing-schema.json](gnus-processing-schema.json).

## Quick Start
A processing definition contains top-level metadata, inputs/outputs, parameters, and passes. The minimum viable shape is:

```json
{
  "name": "example-processing",
  "version": "1.0.0",
  "gnus_spec_version": 1.0,
  "inputs": [
    {
      "name": "inputVolume",
      "source_uri_param": "file://path/to/input.raw",
      "type": "texture3D",
      "dimensions": {
        "width": 253,
        "height": 253,
        "chunk_count": 94,
        "chunk_subchunk_width": 96,
        "chunk_subchunk_height": 96,
        "block_len": 96,
        "chunk_stride": 48,
        "chunk_line_stride": 48,
        "block_stride": 48
      },
      "format": "FLOAT32"
    }
  ],
  "outputs": [
    {
      "name": "segmentationOutput",
      "source_uri_param": "file://path/to/output.raw",
      "type": "tensor"
    }
  ],
  "parameters": [
    {
      "name": "modelUri",
      "type": "uri",
      "default": "file://path/to/model.mnn"
    },
    {
      "name": "volumeLayout",
      "type": "string",
      "default": "HWD"
    }
  ],
  "passes": [
    {
      "name": "inference",
      "type": "inference",
      "model": {
        "source_uri_param": "file://path/to/model.mnn",
        "format": "MNN",
        "input_nodes": [
          {
            "name": "input",
            "type": "tensor",
            "source": "input:inputVolume",
            "shape": [1, 1, 96, 96, 96]
          }
        ],
        "output_nodes": [
          {
            "name": "output",
            "type": "tensor",
            "target": "output:segmentationOutput",
            "shape": [1, 2, 96, 96, 96]
          }
        ]
      }
    }
  ]
}
```

## Top-Level Fields
Required:
- `name`
- `version`
- `gnus_spec_version`
- `inputs`
- `outputs`
- `passes`

Optional:
- `author`
- `description`
- `tags`
- `parameters`
- `metadata`

## Inputs and Outputs
Each element in `inputs` and `outputs` must include:
- `name`
- `source_uri_param`
- `type`

Optional:
- `description`
- `dimensions`
- `format`

Notes:
- `source_uri_param` is currently treated as a URL, not a parameter name.
- `format` supports: `FLOAT32`, `FLOAT16`, `INT32`, `INT16`, `INT8`, `RGB8`, `RGBA8`. Only FLOAT32/FLOAT16 are currently used for texture3D.

## Parameters
Parameters allow optional configuration such as model URI or layout. Each parameter includes:
- `name`
- `type`

Optional:
- `default`
- `description`
- `constraints`

Recommended parameters for texture3D:
- `volumeLayout` (string): one of `HWD`, `HDW`, `WHD`, `WDH`, `DHW`, `DWH`.

The processor also recognizes `inputNameLayout` and `inputName_layout` where `inputName` is the input field name.

## Data Type Requirements
This section describes required and optional fields by `type`. If a type is unimplemented, a placeholder is included so the schema remains forward-compatible.

### texture3D (implemented)
Required:
- `dimensions.width`
- `dimensions.height`
- `dimensions.chunk_count` (depth)
- `dimensions.chunk_subchunk_width` (patch width)
- `dimensions.chunk_subchunk_height` (patch height)
- `dimensions.block_len` (patch depth)

Optional:
- `dimensions.chunk_stride` (stride x)
- `dimensions.chunk_line_stride` (stride y)
- `dimensions.block_stride` (stride z)
- `format` (`FLOAT32` or `FLOAT16`)
- Parameter `volumeLayout` or `inputNameLayout`

Notes:
- If stride fields are omitted, the processor defaults to non-overlapping patches (stride = patch size).
- The input buffer is assumed to be a contiguous 3D array in the specified layout.

### texture2D (implemented)
Required:
- `dimensions.width`
- `dimensions.height`
- `dimensions.chunk_count`
- `dimensions.block_len`
- `dimensions.block_line_stride`
- `dimensions.block_stride`
- `dimensions.chunk_line_stride`
- `dimensions.chunk_offset`
- `dimensions.chunk_stride`
- `dimensions.chunk_subchunk_height`
- `dimensions.chunk_subchunk_width`

Optional:
- `format` (typically `RGB8` or `RGBA8`)

Notes:
- This is used by the image processor and uses the current image splitter logic.

### textureCube (implemented)
Required:
- `dimensions.width` (face width)
- `dimensions.height` (face height)

Optional:
- `format` (`RGB8`, `RGBA8`, `FLOAT32`, or `FLOAT16`)
- `cubeLayout` (string): `faces_in_order` (default) or `atlas_3x2`
- texture2D chunk fields (same fields as texture2D) to enable chunking for RGB/RGBA inputs

Notes:
- `faces_in_order` expects 6 faces concatenated in order, each face is width x height.
- `atlas_3x2` expects a 3x2 atlas of faces using the same width/height per face.
- Chunking is ignored for float formats.

### string (implemented)
Required:
- No additional fields beyond `name`, `source_uri_param`, `type`.

Optional:
- Parameters:
  - `tokenizerMode` (string): required by validation, supported values currently `token_ids` or `raw_text`.
  - `vocabUri` (uri): required if `tokenizerMode` is `raw_text`.
  - `maxLength` (int): optional, defaults to 128 inside the processor.

Notes:
- If the input text parses as space-separated integers, they are treated as token ids.

### tensor (implemented)
Required:
- `dimensions.width` (element count)

Optional:
- `dimensions.block_len` (patch length)
- `dimensions.chunk_stride` (stride)
- `format` (`FLOAT32`, `FLOAT16`, `INT32`, `INT16`, or `INT8`)

Notes:
- Tensor input is treated as a flat 1D buffer.
- Integer inputs are converted to float internally for model inference.
- If patch fields are omitted, the processor defaults to a single window covering the full length.

### bool (implemented)
Required:
- `dimensions.width` (length)

Optional:
- `dimensions.block_len` (patch length)
- `dimensions.chunk_stride` (stride)
- `format` (`FLOAT32`, `FLOAT16`, or `INT8`)

Notes:
- A scalar bool is represented by length 1.

### vec2 (implemented)
Required:
- `dimensions.width` (vector count)

Optional:
- `dimensions.block_len` (patch length, in vectors)
- `dimensions.chunk_stride` (stride, in vectors)
- `format` (`FLOAT32` or `FLOAT16`)

Notes:
- Input data is a contiguous array of vec2 values (2 floats per vector, X and Y components).
- The processor treats `width` as the number of vectors, not the number of floats.
- If patch fields are omitted, the processor defaults to a single window covering the full length.

### vec3 (implemented)
Required:
- `dimensions.width` (vector count)

Optional:
- `dimensions.block_len` (patch length, in vectors)
- `dimensions.chunk_stride` (stride, in vectors)
- `format` (`FLOAT32` or `FLOAT16`)

Notes:
- Input data is a contiguous array of vec3 values (3 floats per vector, X, Y, and Z components).
- The processor treats `width` as the number of vectors, not the number of floats.
- If patch fields are omitted, the processor defaults to a single window covering the full length.

### vec4 (implemented)
Required:
- `dimensions.width` (vector count)

Optional:
- `dimensions.block_len` (patch length, in vectors)
- `dimensions.chunk_stride` (stride, in vectors)
- `format` (`FLOAT32` or `FLOAT16`)

Notes:
- Input data is a contiguous array of vec4 values (4 floats per vector, X, Y, Z, and W components).
- The processor treats `width` as the number of vectors, not the number of floats.
- If patch fields are omitted, the processor defaults to a single window covering the full length.

### mat2 (implemented)
Required:
- `dimensions.width` (matrix count)

Optional:
- `dimensions.block_len` (patch length, in matrices)
- `dimensions.chunk_stride` (stride, in matrices)
- `format` (`FLOAT32` or `FLOAT16`)

Notes:
- Input data is a contiguous array of mat2 values in row-major order (4 floats per matrix).
- The processor treats `width` as the number of matrices, not the number of floats.
- If patch fields are omitted, the processor defaults to a single window covering the full length.

### mat3 (implemented)
Required:
- `dimensions.width` (matrix count)

Optional:
- `dimensions.block_len` (patch length, in matrices)
- `dimensions.chunk_stride` (stride, in matrices)
- `format` (`FLOAT32` or `FLOAT16`)

Notes:
- Input data is a contiguous array of mat3 values in row-major order (9 floats per matrix).
- The processor treats `width` as the number of matrices, not the number of floats.
- If patch fields are omitted, the processor defaults to a single window covering the full length.

### mat4 (implemented)
Required:
- `dimensions.width` (matrix count)

Optional:
- `dimensions.block_len` (patch length, in matrices)
- `dimensions.chunk_stride` (stride, in matrices)
- `format` (`FLOAT32` or `FLOAT16`)

Notes:
- Input data is a contiguous array of mat4 values in row-major order (16 floats per matrix).
- The processor treats `width` as the number of matrices, not the number of floats.
- If patch fields are omitted, the processor defaults to a single window covering the full length.

### buffer (implemented)
Required:
- `dimensions.width` (length)

Optional:
- `dimensions.block_len` (patch length)
- `dimensions.chunk_stride` (stride)
- `format` (`INT8` only)

### float (implemented)
Required:
- `dimensions.width` (length)

Optional:
- `dimensions.block_len` (patch length)
- `dimensions.chunk_stride` (stride)
- `format` (`FLOAT32` or `FLOAT16`)

Notes:
- float data type handles 1D floating-point vectors.
- If patch fields are omitted, the processor defaults to a single window covering the full length.

### int (implemented)
Required:
- `dimensions.width` (length)

Optional:
- `dimensions.block_len` (patch length)
- `dimensions.chunk_stride` (stride)
- `format` (`INT32`, `INT16`, or `INT8`)

Notes:
- int data type handles 1D integer vectors.
- Integer inputs are converted to float internally for model inference.
- If patch fields are omitted, the processor defaults to a single window covering the full length.

### texture1D (implemented)
Required:
- `dimensions.width`

Optional:
- `dimensions.block_len` (patch length)
- `dimensions.chunk_stride` (stride)
- `format` (`FLOAT32` or `FLOAT16`)
- Parameter `volumeLayout` or `inputNameLayout` (accepted but currently ignored for 1D)

Notes:
- If patch fields are omitted, the processor defaults to a single window covering the full length.

### texture1D_array, texture2D_array, texture3D_array, textureCube (placeholder)
Required:
- TBD.

Optional:
- `dimensions` and `format` as needed by future texture processors.

## Model Passes
Inference passes require a `model` object with:
- `source_uri_param`
- `format`
- `input_nodes`
- `output_nodes`

Each `input_nodes` item uses `source` with `input:` or `parameter:` prefix. Each `output_nodes` item uses `target` with `output:` or `internal:` prefix.

## Common Pitfalls
- Ensure `source_uri_param` values are valid URLs (e.g., `file://...`).
- For texture3D, the input size must be `width * height * chunk_count * sizeof(element)`.
- If you change layout or format, update parameters and input files accordingly.
