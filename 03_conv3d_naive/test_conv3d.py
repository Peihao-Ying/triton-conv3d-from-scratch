"""
Test Conv3d Triton implementation against PyTorch's nn.Conv3d.

Creates a small test case, runs both implementations with identical weights,
and verifies the results match within float16 tolerance.
"""

import torch
import torch.nn as nn

from conv3d_triton import conv3d_triton

# ============================================================
# Test parameters
# ============================================================

batch_size = 1
C_in = 2       # input channels
C_out = 3      # output channels
D, H, W = 4, 4, 4          # input spatial size
kD, kH, kW = 3, 3, 3       # kernel size

# Output dimensions (stride=1, padding=0)
D_out = D - kD + 1  # = 2
H_out = H - kH + 1  # = 2
W_out = W - kW + 1  # = 2

print(f"Input shape:  ({batch_size}, {C_in}, {D}, {H}, {W})")
print(f"Kernel shape: ({C_out}, {C_in}, {kD}, {kH}, {kW})")
print(f"Output shape: ({batch_size}, {C_out}, {D_out}, {H_out}, {W_out})")
print()

# ============================================================
# Create test data (fixed seed for reproducibility)
# ============================================================

torch.manual_seed(42)
x = torch.randn(batch_size, C_in, D, H, W)
weight = torch.randn(C_out, C_in, kD, kH, kW)
bias = torch.randn(C_out)

# ============================================================
# Ground truth: PyTorch nn.Conv3d
# ============================================================

conv = nn.Conv3d(C_in, C_out, (kD, kH, kW), stride=1, padding=0, bias=True)
with torch.no_grad():
    conv.weight.copy_(weight)
    conv.bias.copy_(bias)

out_pytorch = conv(x)
print(f"PyTorch Conv3d output shape: {out_pytorch.shape}")

# ============================================================
# Our implementation: im2col + Triton matmul
# ============================================================

out_triton = conv3d_triton(x, weight, bias)
print(f"Triton Conv3d output shape:  {out_triton.shape}")
print()

# ============================================================
# Compare results
# ============================================================

# Move PyTorch output to same device for comparison
out_pytorch_dev = out_pytorch.detach().to(out_triton.device)

max_diff = (out_triton - out_pytorch_dev).abs().max().item()
print(f"Conv3d Triton vs PyTorch max difference: {max_diff:.5f}")

# Allow tolerance for float16 intermediate computation
atol = 1e-2
if max_diff < atol:
    print("✅ Results match!")
else:
    print(f"❌ Results differ (max diff {max_diff} > tolerance {atol})")
    # Print some values for debugging
    print("\nPyTorch output[0,0]:")
    print(out_pytorch_dev[0, 0])
    print("\nTriton output[0,0]:")
    print(out_triton[0, 0])
