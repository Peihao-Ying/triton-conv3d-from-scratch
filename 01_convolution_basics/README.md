# 01 - Convolution Basics

## What is Convolution

Convolution = a small "template" (kernel) slides over the input. At each position, the covered region and the kernel are element-wise multiplied and summed to produce a single number.

### 1D Example

```
input  = [1, 3, 5, 7, 9]
kernel = [1, 2, 1]

position 0: 1×1 + 3×2 + 5×1 = 12
position 1: 3×1 + 5×2 + 7×1 = 20
position 2: 5×1 + 7×2 + 9×1 = 28

output = [12, 20, 28]
```

### 3D Convolution (Conv3d)

The input is a "colored cube": at each spatial position `(d, h, w)`, there are `C_in` attribute values (channels).

The kernel is a small colored cube with shape `(C_in, kD, kH, kW)`.

At each position: the entire covered region (all channels, all spatial positions) is dot-producted with the kernel to produce a single number.

**Channels are not independent worlds — they are different attributes at the same position, all weighted and summed together.**

## Output Dimensions

```
D_out = (D - kD) / stride + 1
H_out = (H - kH) / stride + 1
W_out = (W - kW) / stride + 1
```

(Ignoring padding and dilation here; simplest case: stride=1, padding=0)

## im2col: Convolution → Matrix Multiplication

The input region corresponding to each output position is flattened into a row, and all positions are arranged into a matrix:

```
input_matrix:   (D_out×H_out×W_out,  C_in×kD×kH×kW)
weight_matrix:  (C_out,              C_in×kD×kH×kW)

output = weight_matrix @ input_matrix.T
shape:   (C_out, D_out×H_out×W_out)

reshape → (C_out, D_out, H_out, W_out)
```

## Verification Code

`verify_conv3d.py` computes Conv3d using three methods and verifies the results match:

```bash
python verify_conv3d.py
```

Three methods:
1. `torch.nn.Conv3d` (ground truth)
2. 7-nested for loops (most intuitive)
3. im2col + matrix multiplication (this is what Triton will accelerate)
