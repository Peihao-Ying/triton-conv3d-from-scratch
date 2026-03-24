# 03 - Naive Conv3d (Verification Step)

## Goal

Quickly verify that im2col + Triton matmul produces correct Conv3d results. This is a stepping stone to the final implementation, not a final product.

> **Why not stop here?** Explicit im2col constructs a full intermediate matrix in GPU memory. For large inputs, this wastes significant memory. The real goal is phase 04, where im2col is fused into the kernel itself.

## Approach

1. Use PyTorch for explicit im2col (unfold input into a matrix)
2. Use a Triton kernel for matrix multiplication
3. Verify correctness against `torch.nn.Conv3d`

## Files

- `im2col.py` — im2col function that converts 3D input patches into a column matrix
- `conv3d_triton.py` — combines im2col + Triton matmul into a complete conv3d function
- `test_conv3d.py` — correctness test comparing against `torch.nn.Conv3d`

## Constraints

- stride=1, padding=0, dilation=1, groups=1
- batch_size=1 only
- float16 intermediate (Triton matmul), float32 input/output
- No performance optimization

## Usage

```bash
cd 03_conv3d_naive
python test_conv3d.py
```

## Completed

- [x] Write the im2col function
- [x] Use Triton matmul kernel for the GEMM step
- [x] Combine into a complete conv3d_triton function
- [x] Verify correctness against torch.nn.Conv3d
