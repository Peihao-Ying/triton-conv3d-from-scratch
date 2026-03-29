# triton-conv3d-from-scratch

Implementing a 3D convolution (Conv3d) Triton GPU kernel from scratch — documenting the full learning journey from "what is convolution" to "writing a high-performance GPU kernel by hand."

## Learning Roadmap

| Phase | Directory | Content | Status |
|-------|-----------|---------|--------|
| 1 | `01_convolution_basics/` | Understanding convolution, im2col, and matrix multiplication | ✅ |
| 2 | `02_triton_basics/` | Read and understand Triton tutorials | ✅ |
| 3 | `03_conv3d_naive/` | Verification: explicit im2col + Triton matmul | ✅ |
| 4 | `04_conv3d_implicit/` | Implicit im2col Conv3d kernel — fused address computation | ✅ |
| 5 | `05_optimization/` | Performance optimization: batch, autotuning, Winograd, groups | ✅ |

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

## What I Learned

1. **im2col → GEMM decomposition** is the conceptual foundation. Once you see Conv3d as a matrix multiplication, everything else follows.

2. **Implicit im2col** is the real breakthrough — fusing address computation into the K-loop eliminates the memory-hungry intermediate matrix.

3. **Loop restructuring > micro-optimization.** Splitting the K-loop (outer spatial, inner channel) improves cache behavior more than any amount of faster integer division.

4. **Triton's `tl.constexpr`** enables loop unrolling, constant folding, and dead code elimination — the compiler does the hard work when you give it compile-time constants.

5. **Dispatch is an optimization.** Winograd, depthwise, and general im2col are fundamentally different algorithms — a single kernel would be slower than three specialized ones.

6. **cuDNN is hard to beat on small problems** due to kernel launch overhead and hardware-specific assembly. Custom Triton kernels shine on larger, non-standard workloads (stride > 1, large spatial dims).

## Requirements

- Python 3.10+
- PyTorch 2.0+
- Triton 2.0+ (requires NVIDIA GPU)

## Note

Code in this project is written with AI assistance (Claude / Claude Code).

## References

- [Triton Language API](https://triton-lang.org/main/python-api/triton.language.html)
- [Triton Tutorials](https://triton-lang.org/main/getting-started/tutorials/index.html)
- [PyTorch Conv3d Docs](https://docs.pytorch.org/docs/stable/generated/torch.nn.Conv3d.html)
