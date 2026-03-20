"""
The essence of Conv3d: im2col + matrix multiplication

This script has three steps:
1. Compute with PyTorch's Conv3d (ground truth)
2. Compute with brute-force 7-nested for loops (to see every step clearly)
3. Compute with im2col + matrix multiplication (this is what Triton will accelerate)

All three results should be identical.
"""

import torch
import torch.nn as nn

# ============================================================
# Set up a small example so you can see what's happening
# ============================================================

batch_size = 1
C_in = 2      # Input channels (e.g., each point has 2 attributes)
C_out = 3     # Output channels (e.g., 3 kernels extracting 3 features)
D, H, W = 4, 4, 4          # Input spatial size: 4×4×4 cube
kD, kH, kW = 3, 3, 3       # Kernel size: 3×3×3 small cube
stride = 1
padding = 0

# Output dimensions
D_out = D - kD + 1  # = 2
H_out = H - kH + 1  # = 2
W_out = W - kW + 1  # = 2

print(f"Input: ({batch_size}, {C_in}, {D}, {H}, {W})")
print(f"Kernel: ({C_out}, {C_in}, {kD}, {kH}, {kW})")
print(f"Output: ({batch_size}, {C_out}, {D_out}, {H_out}, {W_out})")
print()

# Create input and weights (fixed random seed for reproducibility)
torch.manual_seed(42)
x = torch.randn(batch_size, C_in, D, H, W)
weight = torch.randn(C_out, C_in, kD, kH, kW)
bias = torch.randn(C_out)

# ============================================================
# Method 1: PyTorch Conv3d (ground truth)
# ============================================================

conv = nn.Conv3d(C_in, C_out, (kD, kH, kW), stride=stride, padding=padding, bias=True)
# Copy our own weights and bias so all three methods use the same parameters
with torch.no_grad():
    conv.weight.copy_(weight)
    conv.bias.copy_(bias)

out_pytorch = conv(x)
print("Method 1 - PyTorch Conv3d output shape:", out_pytorch.shape)
print()

# ============================================================
# Method 2: Brute force, 7-nested for loops (to see every step)
# ============================================================

out_naive = torch.zeros(batch_size, C_out, D_out, H_out, W_out)

for n in range(batch_size):          # Iterate over each sample in the batch
    for co in range(C_out):          # Iterate over each output channel (each kernel)
        for d in range(D_out):       # Iterate over output depth positions
            for h in range(H_out):   # Iterate over output height positions
                for w in range(W_out):  # Iterate over output width positions
                    # Start from bias
                    val = bias[co].item()

                    # Sum up products over all channels and positions covered by the kernel
                    for ci in range(C_in):        # Each input channel
                        for kd in range(kD):      # Kernel depth dimension
                            for kh in range(kH):  # Kernel height dimension
                                for kw in range(kW):  # Kernel width dimension
                                    val += (weight[co, ci, kd, kh, kw].item() *
                                            x[n, ci, d+kd, h+kh, w+kw].item())

                    out_naive[n, co, d, h, w] = val

# Verify: does it match PyTorch's result?
diff_naive = (out_naive - out_pytorch).abs().max().item()
print(f"Method 2 - for loops vs PyTorch max difference: {diff_naive:.10f}")
print("(close to 0 means correct)")
print()

# ============================================================
# Method 3: im2col + matrix multiplication (the core that Triton will accelerate)
# ============================================================

# Step 1: im2col
# Flatten each "colored small cube" at every position into a row
# Total positions: D_out × H_out × W_out
# Each small cube has C_in × kD × kH × kW elements

K = C_in * kD * kH * kW    # Flattened length of each small cube
N_positions = D_out * H_out * W_out  # Total number of positions

input_matrix = torch.zeros(N_positions, K)

pos = 0
for d in range(D_out):
    for h in range(H_out):
        for w in range(W_out):
            # Extract the small cube covered at this position
            # shape: (C_in, kD, kH, kW)
            patch = x[0, :, d:d+kD, h:h+kH, w:w+kW]

            # Flatten into a row, length = C_in * kD * kH * kW
            input_matrix[pos] = patch.reshape(-1)
            pos += 1

print(f"im2col input_matrix shape: {input_matrix.shape}")
print(f"  rows = {N_positions} (output positions = {D_out}×{H_out}×{W_out})")
print(f"  cols = {K} (each small cube flattened = {C_in}×{kD}×{kH}×{kW})")
print()

# Step 2: Flatten weights too
# Each kernel (C_in, kD, kH, kW) is flattened into a row
# There are C_out kernels
weight_matrix = weight.reshape(C_out, K)  # shape: (C_out, K)
print(f"weight_matrix shape: {weight_matrix.shape}")
print(f"  rows = {C_out} (output channels, i.e., number of kernels)")
print(f"  cols = {K}")
print()

# Step 3: Matrix multiplication!
# weight_matrix @ input_matrix.T
# (C_out, K) @ (K, N_positions) → (C_out, N_positions)
output_flat = weight_matrix @ input_matrix.T  # shape: (C_out, N_positions)

# Add bias (one bias value per output channel)
output_flat += bias.reshape(C_out, 1)

# Reshape back to (C_out, D_out, H_out, W_out)
out_im2col = output_flat.reshape(1, C_out, D_out, H_out, W_out)

# Verify
diff_im2col = (out_im2col - out_pytorch).abs().max().item()
print(f"Method 3 - im2col+matmul vs PyTorch max difference: {diff_im2col:.10f}")
print("(close to 0 means correct)")
print()

# ============================================================
# Summary
# ============================================================

print("=" * 50)
print("All three methods compute the same thing:")
print(f"  PyTorch Conv3d  output[0,0,0,0,0] = {out_pytorch[0,0,0,0,0].item():.6f}")
print(f"  for loops       output[0,0,0,0,0] = {out_naive[0,0,0,0,0].item():.6f}")
print(f"  im2col+matmul   output[0,0,0,0,0] = {out_im2col[0,0,0,0,0].item():.6f}")
print()
print("Next step: write a Triton GPU kernel to accelerate the matrix multiplication in Method 3")
