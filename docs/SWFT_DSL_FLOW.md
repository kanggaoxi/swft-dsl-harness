# SWFT DSL Development Flow

Use this document as the minimum context for agents implementing SWFT DSL.

The harness uses route B: PyTorch is the trusted semantic source. Earlier
stages export and validate `model_ir.json`, `node_manifest.json`,
`weight_map.json`, `input_spec.json`, and `torch_runner.py`; golden bins are
captured directly from PyTorch. DSL agents should consume those validated
artifacts and should not re-derive model semantics from the full PyTorch
source unless a validation report explicitly asks for that.

Precision and dtype contract:

- weight dtypes follow the tensors stored in the `.pth` checkpoint; do not
  silently force weights to fp32
- model entry input dtypes come from `input_spec.json`
- golden bins are produced by the PyTorch model runtime following checkpoint
  dtypes and `input_spec.json`
- from the full-model perspective, DSL model input and output dtypes must match
  the golden model input and output dtypes
- final full-model output relative error must satisfy the configured
  `final_comparison_rtol`
- partition comparisons use the configured default tolerance, but documented
  local overrides are allowed; final full-model accuracy is the hard gate

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
2. Build GM Tensor placeholders.
3. Call the `@sub_kernel` function to record DSL trace.
4. Call `compile_kernel(...)` to generate `<op>.cce`.
5. Call `exec_kernel(...)` to generate `main.cpp`.
6. `exec_kernel(...)` compiles `<op>.cce` and `main.cpp`, links an executable,
   runs it, and writes actual output bins.
7. Compare actual bins against golden bins.

`compile_kernel` does not execute the NPU kernel. It emits CCE source.
`exec_kernel` is the step that builds and runs the generated test executable.

## GM Tensors and exec_kernel

GM Tensors connect kernel parameters with `exec_kernel(inputs=..., outputs=...)`:

- the kernel function must access inputs, weights, and outputs through GM Tensor parameters
- `inputs` and `outputs` contain GM Tensor variable names
- those variable names must exist in the `locals()` passed to `exec_kernel`
- before `exec_kernel`, the DSL should already have called `compile_kernel` and the kernel function should have been called to record trace

Recommended order:

```text
define GM Tensors
define @sub_kernel function
compile_kernel(...)
call the kernel function to record trace
exec_kernel(...)
```

## Formal Performance Path

`exec_kernel` is the correctness path. Performance work should not rely only on
Python-side profiling. After full correctness passes, stage `07_perf` should
move the generated `.cce`, input/output data, and host entry into the formal
interface path, then build and run it outside Python to collect latency.

## Development Rule

First make the skeleton work:

```text
GM input -> UB -> GM output
```

Only after compile, run, file IO, and comparison pass should the agent add real
subgraph computation. Add one partition at a time and compare immediately.
