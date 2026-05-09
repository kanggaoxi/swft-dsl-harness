# SWFT DSL Development Flow

Use this document as the minimum context for agents implementing SWFT DSL.

The harness uses route B: PyTorch is the trusted semantic source. Earlier
stages export and validate `model_ir.json`, `node_manifest.json`,
`weight_map.json`, and `torch_runner.py`; golden bins are captured directly
from PyTorch. DSL agents should consume those validated artifacts and should
not re-derive model semantics from the full PyTorch source unless a validation
report explicitly asks for that.

## Repository Areas

- `python/swft/api/`: Python DSL APIs such as movement and compute operators.
- `python/swft/core/`: trace capture and source-to-source compilation,
  including `@sub_kernel` and `compile_kernel`.
- `python/swft/runtime/`: runtime helpers, especially `exec_kernel`.
- `python/swft/utils/`: code generation helpers for C++ drivers and binding
  code.
- `op_test/`: executable examples and validation utilities.

## Compile and Run Chain

The standard `op_test/math/tanh.py` flow is:

1. Generate input and golden output bins.
2. Build Tensor placeholders.
3. Call the `@sub_kernel` function to record DSL trace.
4. Call `compile_kernel(...)` to generate `<op>.cce`.
5. Call `exec_kernel(...)` to generate `main.cpp`.
6. `exec_kernel(...)` compiles `<op>.cce` and `main.cpp`, links an executable,
   runs it, and writes actual output bins.
7. Compare actual bins against golden bins.

`compile_kernel` does not execute the NPU kernel. It emits CCE source.
`exec_kernel` is the step that builds and runs the generated test executable.

## Development Rule

First make the skeleton work:

```text
GM input -> UB -> GM output
```

Only after compile, run, file IO, and comparison pass should the agent add real
subgraph computation. Add one partition at a time and compare immediately.
