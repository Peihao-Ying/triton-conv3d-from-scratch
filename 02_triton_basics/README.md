# 02 - Triton Basics

## Goal

Read and understand the official Triton tutorials (vector add and matmul). The goal is NOT to implement kernels from scratch — it's to understand the structure and patterns that every Triton kernel follows.

## The 5-Step Pattern

Every Triton kernel follows the same structure:

1. **Get program_id** — which block of work is this thread responsible for?
2. **Compute addresses** — calculate memory pointers for the data this block needs
3. **Load data** — read from GPU memory into registers
4. **Compute** — perform the actual operation
5. **Store results** — write back to GPU memory

## Key Concepts from Vector Add Tutorial

- `tl.program_id(axis)`: Each kernel instance gets a unique block index
- `tl.arange(start, end)`: Generate a range of indices within a block
- `tl.load / tl.store`: Read/write GPU memory
- **Mask for bounds checking**: When the total work isn't a perfect multiple of BLOCK_SIZE, use a mask to avoid out-of-bounds memory access

## Key Concepts from Matmul Tutorial

- **2D tiling**: The output matrix is divided into 2D blocks, each handled by one program instance
- **K-dimension loop**: Each block iterates along the shared K dimension, accumulating partial results
- `tl.dot`: Block-level matrix multiplication (computes a small matmul per iteration)
- **Pointer arithmetic with stride**: Navigate multi-dimensional tensors using stride values to compute correct memory addresses

## Note

> Address computation details (pointer arithmetic, stride calculations) can be handled by AI. Focus on understanding the overall architecture — the 5-step pattern, tiling strategy, and how the K-dimension loop accumulates results.

## Completion Checklist

- [x] Read and understand the Triton vector add tutorial
- [x] Read and understand the Triton matmul tutorial
- [x] Understand tiling and BLOCK_SIZE concepts
- [x] Identify the 5-step pattern in both tutorials
- [x] Run matmul tutorial successfully (verified on RTX 5070)
