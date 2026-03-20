# 03 - Naive Conv3d Implementation

## Goal

Implement Conv3d with Triton in two steps:
1. Use PyTorch for explicit im2col (unfold input into a matrix)
2. Use a Triton kernel for matrix multiplication

## TODO

- [ ] Write the im2col function
- [ ] Write the Triton matmul kernel
- [ ] Combine into a complete conv3d_triton function
- [ ] Verify correctness against torch.nn.Conv3d
- [ ] Benchmark performance
