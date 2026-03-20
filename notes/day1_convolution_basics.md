# Learning Notes - Day 1

## Concepts Covered Today

### The Essence of Convolution
- A kernel is a template that slides over the input, computing a dot product at each position
- Larger dot product = this position more closely matches the pattern defined by the kernel

### Understanding Channels
- Channels are not independent worlds — they are different attributes at the same position (like different features of a person)
- Position is the anchor, channels are attributes
- A single kernel covers all channels, multiplying and summing everything to produce one number
- Multiple kernels → multiple output channels (different feature detectors)

### Convolution = Matrix Multiplication (im2col)
- The input region for each output position is flattened into a row
- All positions are arranged into the input matrix
- All kernels are flattened into the weight matrix
- A single matrix multiplication computes everything at once

### Shape Meanings
```
Conv3d input:   (N, C_in, D, H, W)
Conv3d weights: (C_out, C_in, kD, kH, kW)
Conv3d output:  (N, C_out, D_out, H_out, W_out)

After im2col:
  input_matrix:  (D_out×H_out×W_out, C_in×kD×kH×kW)
  weight_matrix: (C_out, C_in×kD×kH×kW)
  output:        weight_matrix @ input_matrix.T
```

## Next Steps
- [ ] Learn the Triton vector add tutorial
- [ ] Learn the Triton matmul tutorial
