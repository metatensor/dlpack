# DLPack integration for Rust

This crate contains a direct translation of the C header for
[DLPack](https://github.com/dmlc/dlpack), which is intended to be used inside
Rust projects that want to offer a C API.

We also provide some tools to convert to and from Rust types, through the
following cargo features:

- `ndarray` to convert from and to `ndarray::Array`
- `pyo3` to convert from and to Python data, following the [python specification
  for dlpack](https://dmlc.github.io/dlpack/latest/python_spec.html)
