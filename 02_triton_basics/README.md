# 02 - Triton Basics

## Goal

- Understand Triton's programming model: program, block, grid
- Implement a vector add kernel (simplest starting point)
- Implement a matrix multiplication kernel (the core of Conv3d)

## Key Concepts

- `@triton.jit`: Marks a function as a GPU kernel
- `tl.program_id(axis)`: Current block's index
- `tl.arange(start, end)`: Generate indices
- `tl.load / tl.store`: Read/write GPU memory
- `tl.dot`: Block-level matrix multiplication

## TODO

- [ ] Run the Triton vector add tutorial
- [ ] Run the Triton matmul tutorial
- [ ] Understand tiling and BLOCK_SIZE concepts
