## Changes

This document serves the purpose of detailing the changes made to the project in further depth.

## How it works

To use the custom backend (referred to as the *shared backend*, as it comes from the `triton-shared` project), you need to set an environment variable:

```bash
export TRITON_USE_SHARED_BACKEND=1
```

This enables the shared backend instead of the default CPU backend. After setting the variable, you can simply run your Python code as usual.

#e How `triton-shared` is integrated into the project

`triton-shared` is designed to be installed as a Triton plugin. See [the official repo](https://github.com/microsoft/triton-shared?tab=readme-ov-file#usage) for details.

To integrate it, I added the `TRITON_PLUGIN_DIRS` environment variable in the `README.md`. This ensures that `triton-shared` is recognized as a backend when installing the `triton-cpu` Python package.

## Background on how the shared backend works

The shared backend works by taking the TTIR and converting it to Linalg using the `triton-shared-opt` binary, which must be specified via the `TRITON_SHARED_OPT_PATH` environment variable. It is then lowered to LLVM by calling `mlir-opt`, whose path should be set using the `LLVM_BINARY_DIR` environment variable.

## `triton-shared` version

Since the OpenEuler `triton-cpu` package is based on version 3.0.0, I needed to use a compatible version of `triton-shared`. In this case, the commit `89286b4` is the latest one that supports Triton 3.0.0.

## How the division by zero bug was solved

By comparing my version of `triton-shared` with the upstream version, I noticed that the MLIR code generated was nearly identical. This led me to rule out incorrect code generation as the cause.

However, upon inspecting how the launcher functions were generated in `driver.py`, I noticed a difference: the upstream version passed more parameters, including constants as arguments.

To fix the bug, I modified the shared backend's `driver.py` to also pass constants as arguments.

## Adding Support for ARM SVE

A new method called `_sve_transform` was added to the `compiler.py` file of the shared backend to add optimizations through the transform dialect that enable SVE. This implementation is based on the [LLVM example](https://github.com/llvm/llvm-project/blob/llvmorg-19.1.7/mlir/test/Integration/Dialect/Linalg/CPU/ArmSVE/matmul.mlir). Additionally, several passes were added to the pipeline executed by `mlir-opt`.

## Bug with SVE on LLVM 19

LLVM 19 contains a bug where using SVE in the transform dialect without first loading the dialect via an SVE-related pass in `mlir-opt` causes a crash. One straightforward (though hacky) workaround is to include an SVE-related pass in the `mlir-opt` pipeline, even if it's not strictly necessary.

In the process of investigating this issue, I gained a deep understanding of the underlying bug and was able to fix it for OpenEuler (branch `dev_19.1.7`). I identified the fix in LLVM 20 [here](https://github.com/llvm/llvm-project/commit/b9d3a644c2716e651b388f9fff660b12fdba577c#diff-291d3da22331ded009a6178f86f517984d6ac0aeb27fe6907bbd793d41670340R110).

The root of the issue is that SVE was not registered as an extension in the transform dialect. Until the referenced commit, there were no native SVE transform ops in the transform dialect, so there was no reason to register the extension. That commit fixes the issue by introducing the first native SVE transform op, and as a result, it also registers SVE as a transform dialect extension. In our case, however, the crash still occurs even without native SVE ops, because SVE (or even SME) can be invoked indirectly via `apply_registered_pass`.

Therefore, the fix for the OpenEuler LLVM 19 branch consists of registering the SVE extension in the transform dialect **without** adding any native transform ops. The transform op introduced in LLVM 20 is written using LLVM 20 features and is not compatible with LLVM 19, so it is intentionally omitted.


# Adding Support for ARM SME

Similar to SVE, a new method called `_sme_transform` was added to the `compiler.py` file of the shared backend to add optimizations through the transform dialect that enable SME. This is based on the [LLVM example](https://github.com/llvm/llvm-project/blob/llvmorg-19.1.7/mlir/test/Integration/Dialect/Linalg/CPU/ArmSME/matmul.mlir). Additional passes were also added to the pipeline run by `mlir-opt`.

