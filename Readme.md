# saker-rs

Fast SIMD-accelerated linear algebra for neural network inference, written in Rust.

Named after the [Saker Falcon](https://en.wikipedia.org/wiki/Saker_falcon) — a fast bird of prey native to the Carpathian region.

## Features

- Blocked GEMM with cache-aware tiling
- AVX-512 and AVX2 micro-kernels with FMA
- Scalar fallback for tail tiles
- Fused bias + activation (SiLU, Sigmoid)
- Parallel execution via Rayon

## Usage

```toml
[dependencies]
saker-rs = "0.1.0"
```

```rust
use saker_rs::linarg::operations::sgemm_bias_parallel;
use saker_rs::activations::Activation;

sgemm_bias_parallel(
    m, n, k,
    &weights,
    &input,
    Some(&bias),
    &mut output,
    Activation::Silu,
);
```

## Requirements

- x86_64 with AVX-512F + FMA for the fast path

## License

MIT