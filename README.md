# triton-conv3d-from-scratch

Implementing a 3D convolution (Conv3d) Triton GPU kernel from scratch — documenting the full learning journey from "what is convolution" to "writing a high-performance GPU kernel by hand."

## Learning Roadmap

| Phase | Directory | Content | Status |
|-------|-----------|---------|--------|
| 1 | `01_convolution_basics/` | Understanding convolution, im2col, and matrix multiplication | ✅ |
| 2 | `02_triton_basics/` | Triton intro: vector add and matmul kernels | ⬜ |
| 3 | `03_conv3d_naive/` | Explicit im2col + Triton matmul to implement Conv3d | ⬜ |
| 4 | `04_conv3d_implicit/` | Implicit im2col: fusing address computation into the kernel | ⬜ |

## Core Idea

Conv3d computation can essentially be decomposed into two steps:

1. **im2col**: Flatten each local region the kernel slides over into a vector, and arrange them into a matrix
2. **GEMM**: Weight matrix × Input matrix = Output matrix

```
weight_matrix:  (C_out, C_in × kD × kH × kW)
input_matrix:   (C_in × kD × kH × kW, D_out × H_out × W_out)
output_matrix:  (C_out, D_out × H_out × W_out)
```

Triton's job is to efficiently perform this matrix multiplication on the GPU.

## Requirements

- Python 3.10+
- PyTorch 2.0+
- Triton 2.0+ (requires NVIDIA GPU)

## References

- [Triton Language API](https://triton-lang.org/main/python-api/triton.language.html)
- [Triton Tutorials](https://triton-lang.org/main/getting-started/tutorials/index.html)
- [PyTorch Conv3d Docs](https://docs.pytorch.org/docs/stable/generated/torch.nn.Conv3d.html)
