# 03 - Naive Conv3d (Verification Step)

## Goal

Quickly verify that im2col + Triton matmul produces correct Conv3d results. This is a stepping stone to the final implementation, not a final product.

> **Why not stop here?** Explicit im2col constructs a full intermediate matrix in GPU memory. For large inputs, this wastes significant memory. The real goal is phase 04, where im2col is fused into the kernel itself.

## Approach

1. Use PyTorch for explicit im2col (unfold input into a matrix)
2. Use a Triton kernel for matrix multiplication
3. Verify correctness against `torch.nn.Conv3d`

## TODO

- [ ] Write the im2col function
- [ ] Use Triton matmul kernel for the GEMM step
- [ ] Combine into a complete conv3d_triton function
- [ ] Verify correctness against torch.nn.Conv3d
