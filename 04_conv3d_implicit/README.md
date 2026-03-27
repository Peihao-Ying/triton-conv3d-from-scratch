# 04 - Implicit im2col Conv3d (Main Deliverable)

**This is the phase that matters most.** Everything before this was preparation.

## Goal

Fuse the im2col address computation into the Triton matmul kernel itself, so no intermediate matrix is ever constructed. This saves GPU memory and reduces memory bandwidth.

## Core Idea

A standard Triton matmul kernel has a K-dimension loop that loads tiles from two pre-existing matrices. In implicit im2col, instead of reading from a pre-built im2col matrix, we **compute the input address on the fly** inside that loop.

For each position in the K-dimension loop (where K = C_in × kD × kH × kW):

```python
# Decompose flat index k into (ci, kd, kh, kw)
ci = k // (kD * kH * kW)
remainder = k % (kD * kH * kW)
kd = remainder // (kH * kW)
kh = (remainder % (kH * kW)) // kW
kw = remainder % kW

# Combined with output position (d_out, h_out, w_out), compute input address:
d_in = d_out * stride_d + kd * dilation_d - pad_d
h_in = h_out * stride_h + kh * dilation_h - pad_h
w_in = w_out * stride_w + kw * dilation_w - pad_w

# Load from original input tensor at (n, ci, d_in, h_in, w_in)
```

This way, the kernel reads directly from the original input — no im2col matrix needed.

## Status: Complete

All three steps implemented and verified.

### Step 1: Simplest case ✅
- stride=1, padding=0, dilation=1, groups=1
- Address computation correct inside K-loop
- Verified against `torch.nn.Conv3d`

### Step 2: General parameters ✅
- stride > 1
- padding > 0 (with bounds checking for padded regions)
- dilation > 1
- 7/7 correctness tests pass

### Step 3: Benchmark ✅

Benchmarked on RTX 3080 against `torch.nn.Conv3d` (cuDNN backend):

| Case | PyTorch | Triton | Speedup |
|------|---------|--------|---------|
| Small first layer | 0.012ms | 0.038ms | 0.32x |
| Medium mid-net | 0.050ms | 0.054ms | 0.92x |
| Large deeper | 0.057ms | 0.084ms | 0.69x |
| Stride=2 downsample | 0.079ms | 0.060ms | **1.31x** |
| Large high-res | 0.198ms | 0.174ms | **1.14x** |

Competitive with cuDNN on larger inputs and stride=2 workloads. Slower on small problems due to kernel launch overhead and cuDNN's hardware-specific optimizations.

## Files

- `conv3d_implicit.py` — Triton kernel + wrapper function
- `test_conv3d_implicit.py` — Correctness tests (7 cases)
- `benchmark.py` — Performance comparison vs PyTorch

## Constraints

- groups=1, batch_size=1
- float16 `tl.dot` with float32 accumulator
