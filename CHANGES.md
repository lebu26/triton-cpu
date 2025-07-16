## Changes

This document serves the purpose of detailing the changes made to the project in further depth.

## How it works

To use the custom backend (referred to as the *shared backend*, as it comes from the `triton-shared` project), you need to set an environment variable:

```bash
export TRITON_USE_SHARED_BACKEND=1
```

This enables the shared backend instead of the default CPU backend. After setting the variable, you can simply run your Python code as usual.

## How `triton-shared` is integrated into the project

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


