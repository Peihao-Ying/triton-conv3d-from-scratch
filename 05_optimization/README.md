# 05 - Performance Optimization

Phase 4 gave us a correct implicit im2col Conv3d kernel. This phase focuses on making it faster.

## Where We Are

Our kernel is competitive with cuDNN on large inputs but loses on small/medium workloads. The main bottlenecks:

1. **Redundant global memory reads** — neighboring output positions share most of their input data, but we reload everything from global memory
2. **Limited parallelism** — batch_size=1 only, no batch dimension parallelism
3. **Suboptimal autotuning** — only 5 configs, cuDNN searches hundreds
4. **Integer arithmetic overhead** — on-the-fly index decomposition (divisions, modulos) in every K-loop iteration

## Optimization Steps

### Step 1: Batch parallelism
- Support batch_size > 1
- Add batch dimension to the kernel grid (2D grid: batch × tile)
- Low effort, immediately useful for real workloads

### Step 2: Expanded autotuning
- Add more autotune configs (different BLOCK_SIZE combos)
- Tune num_stages and num_warps per config
- Add configs optimized for small M (small C_out)

### Step 3: Reduce index computation overhead
- Pre-compute `kD*kH*kW` and `kH*kW` as kernel constants
- Explore making kD, kH, kW `tl.constexpr` (trades compilation time for speed)
- Consider lookup tables for k → (ci, kd, kh, kw) decomposition

### Step 4: Shared memory / data reuse
- Adjacent output positions in the N-tile share overlapping input regions
- Load shared input data into shared memory once, reuse across the tile
- This is the biggest potential win — reduces global memory bandwidth pressure

### Step 5: Explore alternative algorithms
- **Winograd transform** — reduces arithmetic ops for small kernels (3×3), trades multiplications for additions
- **FFT-based convolution** — efficient for large kernels, uses convolution theorem (conv in spatial domain = pointwise multiply in frequency domain)
- These are fundamentally different approaches, not just optimizations of im2col

### Step 6: Groups / depthwise convolution
- Support groups parameter (common in modern architectures like MobileNet)
- Depthwise convolution (groups=C_in) is a special case worth optimizing separately

## Benchmark Targets

Goal: match or beat cuDNN across all problem sizes on our hardware (RTX 3080).

| Case | Current | Target |
|------|---------|--------|
| Small first layer | 0.32x | ≥ 0.8x |
| Medium mid-net | 0.92x | ≥ 1.0x |
| Large deeper | 0.69x | ≥ 1.0x |
| Stride=2 downsample | 1.31x | ≥ 1.3x |
| Large high-res | 1.14x | ≥ 1.2x |

## TODO

- [ ] Step 1: Batch parallelism (batch_size > 1)
- [ ] Step 2: Expanded autotuning
- [ ] Step 3: Reduce index computation overhead
- [ ] Step 4: Shared memory / data reuse
- [ ] Step 5: Explore Winograd / FFT
- [ ] Step 6: Groups / depthwise convolution
