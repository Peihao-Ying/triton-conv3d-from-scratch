# triton-conv3d-from-scratch

A high-performance 3D convolution (Conv3d) GPU kernel written in Triton, built from first principles. Features implicit im2col fusion, Winograd transform, grouped/depthwise dispatch, and autotuning — benchmarked against cuDNN.

## Project Structure

| Phase | Directory | Content |
|-------|-----------|---------|
| 1 | `01_convolution_basics/` | Convolution fundamentals, im2col decomposition, correctness verification |
| 2 | `02_triton_basics/` | Triton programming model — tiling, K-loop, pointer arithmetic |
| 3 | `03_conv3d_naive/` | Baseline: explicit im2col + Triton matmul |
| 4 | `04_conv3d_implicit/` | Implicit im2col Conv3d kernel — fused address computation |
| 5 | `05_optimization/` | 6 optimizations + ultimate kernel with 3 dispatch paths |

## Core Idea

Conv3d computation can essentially be decomposed into two steps:

1. **im2col**: Flatten each local region the kernel slides over into a vector, and arrange them into a matrix
2. **GEMM**: Weight matrix × Input matrix = Output matrix

```
weight_matrix:  (C_out, C_in × kD × kH × kW)
input_matrix:   (C_in × kD × kH × kW, D_out × H_out × W_out)
output_matrix:  (C_out, D_out × H_out × W_out)
```

The key breakthrough (Phase 4) is **implicit im2col**: computing input addresses on-the-fly inside the Triton matmul kernel's K-loop, so the intermediate im2col matrix is never materialized.

## Final Kernel

The ultimate kernel in `05_optimization/conv3d_ultimate.py` combines all optimizations with three dispatch paths:

- **Winograd** — for 3×3×3 kernels, stride=1, dilation=1 (3.375× fewer multiplies)
- **Depthwise** — for groups == C_in == C_out (specialized dot-product reduction)
- **General optimized** — 3D grid (tiles, batch, groups), ~20 autotune configs, constexpr kernel dims, split K-loop for memory coalescing

Supports arbitrary batch size, stride, padding, dilation, and groups. **25/25 tests pass.**

## Technical Insights

1. **im2col → GEMM decomposition** is the conceptual foundation. Once you see Conv3d as a matrix multiplication, everything else follows.

2. **Implicit im2col** is the core design decision — fusing address computation into the K-loop eliminates the memory-hungry intermediate matrix.

3. **Loop restructuring > micro-optimization.** Splitting the K-loop (outer spatial, inner channel) improves cache behavior more than any amount of faster integer division.

4. **Triton's `tl.constexpr`** enables loop unrolling, constant folding, and dead code elimination — the compiler does the hard work when you give it compile-time constants.

5. **Dispatch is an optimization.** Winograd, depthwise, and general im2col are fundamentally different algorithms — a single kernel would be slower than three specialized ones.

6. **cuDNN is hard to beat on small problems** due to kernel launch overhead and hardware-specific assembly. Custom Triton kernels shine on larger, non-standard workloads (stride > 1, large spatial dims).

## Benchmarks

Benchmarks were run on different GPUs during development (RTX 5070 and RTX 3080). Absolute timings are not comparable across phases — use the speedup ratios (Triton vs cuDNN) for evaluation.

## Requirements

- Python 3.10+
- PyTorch 2.0+
- Triton 2.0+ (requires NVIDIA GPU)

## Note

Developed with AI-assisted development tools.

## References

- [Implicit GEMM Convolution](https://github.com/NVIDIA/cutlass/blob/main/media/docs/implicit_gemm_convolution.md) — NVIDIA CUTLASS documentation on fusing im2col into GEMM
- [cuDNN: Efficient Primitives for Deep Learning](https://arxiv.org/abs/1410.0759) — Chetlur et al., 2014
- [Triton Language API](https://triton-lang.org/main/python-api/triton.language.html)
- [Triton Tutorials](https://triton-lang.org/main/getting-started/tutorials/index.html)
- [PyTorch Conv3d Docs](https://docs.pytorch.org/docs/stable/generated/torch.nn.Conv3d.html)
