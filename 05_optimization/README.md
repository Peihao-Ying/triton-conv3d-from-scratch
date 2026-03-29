# 05 - Performance Optimization

Phase 4 gave us a correct implicit im2col Conv3d kernel. This phase makes it faster, more general, and production-ready.

## Approach

Six independent optimizations, each implemented in isolation on top of the Phase 4 baseline. A final **ultimate kernel** combines them all with smart dispatch.

```
05_optimization/
├── 01_batch_parallel/     ← batch_size > 1
├── 02_autotuning/         ← ~20 autotune configs
├── 03_reduce_index/       ← constexpr dims + optional LUT
├── 04_shared_memory/      ← split K-loop for data reuse
├── 05_winograd/           ← Winograd F(2×2×2, 3×3×3)
├── 06_groups/             ← grouped + depthwise convolution
├── conv3d_ultimate.py     ← all optimizations combined
├── test_conv3d_ultimate.py
└── benchmark_ultimate.py
```

---

## Optimization 1: Batch Parallelism

**Problem:** Phase 4 only handles batch_size=1. Real workloads always have batches.

**Solution:** Add a second grid dimension for batch.

```python
grid = (num_tiles, N_batch)
batch_idx = tl.program_id(1)
# Offset input/output pointers by batch_idx * batch_stride
```

Each kernel instance handles one (tile, batch) pair. The K-loop and address computation stay identical to Phase 4 — this is a pure parallelism expansion.

---

## Optimization 2: Expanded Autotuning

**Problem:** Phase 4 has only 5 autotune configs. Small C_out workloads have no good match.

**Solution:** Expand to ~20 configs:

| BLOCK_SIZE_M | BLOCK_SIZE_N | BLOCK_SIZE_K | num_stages | num_warps |
|---|---|---|---|---|
| 16, 32 | 64, 128 | 16, 32 | 2, 3 | 2, 4 |
| 64, 128 | 64, 128, 256 | 16, 32, 64 | 3, 4, 5 | 4, 8 |

Small-M configs (16, 32) handle cases where C_out < 64. Deeper pipelines (num_stages=5) overlap more memory latency.

The kernel body is unchanged — autotuning is a free lunch.

---

## Optimization 3: Reduce Index Computation

**Problem:** The K-loop decomposes flat index k into (ci, kd, kh, kw) with 4 integer divisions per iteration. Integer division is expensive on GPUs (~30 cycles vs ~4 for multiply).

**Solution A — constexpr kernel dims:**

```python
@triton.jit
def kernel(..., kD: tl.constexpr, kH: tl.constexpr, kW: tl.constexpr):
    # kH*kW is a compile-time constant → div/mod become bit shifts
    kHW = kH * kW       # folded at compile time
    kDHW = kD * kHW     # folded at compile time
```

**Solution B — LUT path (optional):**

Pre-compute 4 lookup tables on host (ci_lut, kd_lut, kh_lut, kw_lut, each size K). Replace 4 div/mod with 4 `tl.load` calls. The tables are tiny and accessed sequentially, so they stay in L1 cache (~5 cycles per load vs ~30 for division).

A `USE_LUT` constexpr flag selects the path at compile time — no runtime branching.

---

## Optimization 4: Split K-Loop for Data Reuse

**Problem:** Phase 4 iterates over flat K in blocks, mixing channel and spatial dimensions. This causes scattered memory access patterns because adjacent k values may map to completely different spatial positions.

**Solution:** Restructure the K-loop into nested loops:

```python
# Outer: spatial positions (compile-time unrolled)
for kd in range(kD):        # constexpr → unrolled
    for kh in range(kH):    # constexpr → unrolled
        for kw in range(kW):# constexpr → unrolled
            # Bounds check computed ONCE per spatial position
            d_in = d_out * stride + kd * dilation - pad
            valid = (0 <= d_in < D) and (0 <= h_in < H) and (0 <= w_in < W)

            # Inner: contiguous channel sweep
            for ci_block in range(C_in // BLOCK_SIZE_K):
                b = tl.load(input[ci_block, spatial_addr])
                a = tl.load(weight[kd, kh, kw, ci_block])
                acc += tl.dot(a, b)
```

Why this helps:

1. **Memory coalescing** — for fixed (kd, kh, kw), input loads across the N-tile differ only by channel offset, which is contiguous in NCDHW layout
2. **L2 cache reuse** — adjacent output positions share overlapping spatial input regions; the outer loop keeps spatial fixed while sweeping all channels
3. **Hoisted bounds check** — spatial validity is computed once per (kd, kh, kw), not once per channel block

Additional trick: `eviction_policy="evict_last"` on B-side (input) loads to hint the cache to keep input data longer.

---

## Optimization 5: Winograd Transform

**Problem:** For 3×3×3 kernels, direct convolution computes 27 multiplications per output element. This is compute-bound for large channel counts.

**Solution:** Winograd F(2×2×2, 3×3×3) reduces multiplications from 27 to 8 per output element — a 3.375× reduction.

### How it works

1D Winograd F(2, 3) transforms a length-3 filter and a length-4 input tile into a transform domain where element-wise multiplication replaces convolution. The 3D version applies this separably:

```
Filter transform:  (C_out, C_in, 3, 3, 3) → (C_out, C_in, 4, 4, 4)
Input transform:   extract 4×4×4 tiles with stride-2 → transform each
Output transform:  inverse transform → extract 2×2×2 result per tile
```

In the transform domain, each of the 4×4×4 = 64 positions is an independent matrix multiplication:

```python
# Triton kernel — grid: (num_tile_blocks, 64)
winograd_pos = tl.program_id(1)  # which of 64 positions
# Standard matmul: M[:, :, pos] = V[:, :, pos] @ U[:, :, pos].T
```

### Implementation split

- **Host (PyTorch):** filter transform, input tiling + transform, output inverse transform. These are fixed-size operations on small tensors — negligible overhead.
- **Triton kernel:** 64 batched matmuls. This is where compute time is spent.

### Constraints

- Kernel size must be exactly 3×3×3
- stride=(1,1,1), dilation=(1,1,1), groups=1
- Best for large C_in/C_out where the 3.375× multiply reduction dominates the transform overhead

---

## Optimization 6: Grouped and Depthwise Convolution

**Problem:** Phase 4 only supports groups=1. Many modern architectures (ResNeXt, MobileNet, EfficientNet) rely on grouped or depthwise convolutions.

### General grouped (groups >= 1)

Each group operates on a C_in/groups channel slice independently:

```python
grid = (num_tiles, groups)
group_idx = tl.program_id(1)
# Offset input by group_idx * (C_in // groups) channels
# Offset weight by group_idx * (C_out // groups) output filters
# Offset output by group_idx * (C_out // groups) output channels
```

Weight shape is `(C_out, C_in//groups, kD, kH, kW)` — already partitioned by group.

### Depthwise (groups == C_in == C_out)

Each channel has exactly one spatial filter. The "matrix multiplication" degenerates to a dot product over kD×kH×kW positions, so a matmul kernel is wasteful.

Specialized kernel:

```python
grid = (C, cdiv(N_pos, BLOCK_SIZE_N), N_batch)
channel = tl.program_id(0)
# Triple loop over (kd, kh, kw) — fully unrolled (constexpr)
# No tl.dot — just scalar multiply + accumulate
```

A dispatcher selects general grouped or depthwise based on whether `groups == C_in == C_out`.

---

## Ultimate Kernel: `conv3d_ultimate.py`

The final kernel integrates all six optimizations through three dispatch paths:

```
conv3d_ultimate(input, weight, ...)
    │
    ├── 3×3×3 + stride=1 + dilation=1 + groups=1?
    │   └── Path 1: Winograd (opt 5)
    │
    ├── groups == C_in == C_out?
    │   └── Path 2: Depthwise (opt 6)
    │
    └── Otherwise
        └── Path 3: General optimized
            ├── 3D grid (tiles, batch, groups)    ← opt 1 + 6
            ├── ~20 autotune configs              ← opt 2
            ├── constexpr kD, kH, kW              ← opt 3
            └── split K-loop (spatial → channel)  ← opt 4
```

Dispatch happens at Python level — no branches inside kernels. Each path is fully specialized for its use case.

## Test Results

**25/25 tests pass**, covering:

| Category | Tests | What's verified |
|---|---|---|
| Phase 4 regression | 7 | batch=1, groups=1, various stride/pad/dilation |
| Winograd path | 3 | 3×3×3 kernel, stride=1, different channel sizes |
| Batch parallelism | 4 | batch ∈ {2, 4, 8} with various configs |
| Grouped convolution | 3 | groups ∈ {2, 4} |
| Depthwise | 3 | groups == C_in == C_out |
| Combined | 5 | batch + groups + stride + padding together |

Tolerance: `max(2e-2, sqrt(K) * 2e-3)` for general path, `max(5e-2, C_in * 5e-3)` for Winograd (transforms introduce additional rounding).

## Phase 4 → Phase 5 Comparison

| What | Phase 4 | Phase 5 |
|---|---|---|
| Batch support | batch=1 only | Any batch size |
| Autotune configs | 5 | ~20 |
| Index computation | Runtime div/mod | Compile-time constant folding |
| Memory access | Flat K sweep | Split loop: spatial outer, channel inner |
| 3×3×3 special case | Same as general | Winograd: 3.375× fewer multiplies |
| Groups | groups=1 only | General grouped + depthwise |
| Dispatch paths | 1 (general) | 3 (Winograd / depthwise / general) |
| Tests | 7 | 25 |

## Bottleneck Summary

| Bottleneck | Solution | Optimization |
|---|---|---|
| batch_size=1 only | Batch dim in grid | 1 |
| Poor autotune coverage | ~20 configs with small-M | 2 |
| Integer div/mod overhead | constexpr constant folding + LUT | 3 |
| Scattered memory access | Split K-loop for coalescing | 4 |
| 27 muls per output (3×3×3) | Winograd: 8 muls per output | 5 |
| No grouped/depthwise | Group-aware kernel + depthwise specialization | 6 |

## Files

- `conv3d_ultimate.py` — Final combined kernel (3 dispatch paths)
- `test_conv3d_ultimate.py` — 25 correctness tests
- `benchmark_ultimate.py` — Performance benchmarks vs cuDNN
- `01_batch_parallel/` through `06_groups/` — Each optimization in isolation

## What We Learned

1. **Implicit im2col is the core insight.** Computing addresses on-the-fly inside the K-loop eliminates the intermediate matrix. Everything else is polish on top of this idea.

2. **Restructuring loops matters more than micro-optimizing arithmetic.** The split K-loop (opt 4) improves memory access patterns fundamentally — no amount of faster integer division can compensate for cache misses.

3. **Dispatch is an optimization.** A single "do everything" kernel would be slower than three specialized paths. Winograd and depthwise are fundamentally different algorithms, not parameter tweaks.

4. **Autotuning is free performance.** Expanding the config space costs nothing at runtime — Triton benchmarks all configs and picks the winner.

5. **constexpr is the Triton superpower.** Making kernel dimensions compile-time constants enables loop unrolling, constant folding, and dead code elimination — all without manual effort.

6. **cuDNN is hard to beat on small problems.** Kernel launch overhead and hardware-specific optimizations (tensor cores, custom assembly) give cuDNN an edge for small inputs. Custom Triton kernels shine on larger, less-standard workloads (stride > 1, large spatial dims).
