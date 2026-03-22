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

## Implementation Plan

### Step 1: Simplest case
- stride=1, padding=0, dilation=1, groups=1
- Focus on getting the address computation correct inside the K-loop
- Verify against `torch.nn.Conv3d`

### Step 2: Extend to general parameters
- Support stride > 1
- Support padding > 0 (with bounds checking for padded regions)

### Step 3: Benchmark
- Compare performance against `torch.nn.Conv3d`

## TODO

- [ ] Implement the implicit im2col kernel (stride=1, padding=0, dilation=1, groups=1)
- [ ] Verify correctness against torch.nn.Conv3d
- [ ] Support stride > 1
- [ ] Support padding > 0
- [ ] Benchmark against torch.nn.Conv3d
