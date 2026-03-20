# 04 - Implicit im2col Conv3d

## Goal

Fuse the im2col address computation into the Triton kernel itself, eliminating the need to explicitly construct the intermediate matrix.

## Core Idea

Inside the matmul kernel's K-dimension loop, compute `(ci, kd, kh, kw)` from the current k value:

```python
ci = k // (kD * kH * kW)
remainder = k % (kD * kH * kW)
kd = remainder // (kH * kW)
kh = (remainder % (kH * kW)) // kW
kw = remainder % kW
```

Then, combined with the output position `(d_out, h_out, w_out)`, compute the input address and read directly from the original input.

## TODO

- [ ] Implement the implicit im2col kernel
- [ ] Support stride > 1
- [ ] Support padding > 0
- [ ] Benchmark vs the naive version
