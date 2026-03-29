"""
Test Winograd F(2x2x2, 3x3x3) Conv3D against PyTorch's nn.Conv3d.

All tests use 3x3x3 kernels with stride=1, dilation=1 — the only
configuration Winograd F(2,3) supports.

Higher tolerance than the direct kernel because:
  - Winograd rearranges the arithmetic (different summation order)
  - float16 tensor-core accumulation in the Triton batched matmul
  - Transforms add extra floating-point operations
  - Error scales with C_in (more fp16 products summed), so we use
    a per-test tolerance that accounts for channel count
"""

import torch
import torch.nn as nn

from conv3d_winograd import conv3d_winograd


def run_test(label, batch_size, C_in, C_out, D, H, W, padding=0):
    """Run a single test case: Winograd vs PyTorch nn.Conv3d."""
    if isinstance(padding, int):
        padding = (padding, padding, padding)

    pD, pH, pW = padding
    D_out = D + 2 * pD - 2  # stride=1, dilation=1, kernel=3
    H_out = H + 2 * pH - 2
    W_out = W + 2 * pW - 2

    print(f"--- {label} ---")
    print(f"Input: ({batch_size},{C_in},{D},{H},{W})  "
          f"Kernel: ({C_out},{C_in},3,3,3)  padding={padding}")
    print(f"Output: ({batch_size},{C_out},{D_out},{H_out},{W_out})")

    torch.manual_seed(42)
    x = torch.randn(batch_size, C_in, D, H, W)
    weight = torch.randn(C_out, C_in, 3, 3, 3)
    bias = torch.randn(C_out)

    # Ground truth: PyTorch nn.Conv3d
    conv = nn.Conv3d(C_in, C_out, 3, stride=1, padding=padding, dilation=1, bias=True)
    with torch.no_grad():
        conv.weight.copy_(weight)
        conv.bias.copy_(bias)
    out_pytorch = conv(x)

    # Our Winograd implementation
    out_ours = conv3d_winograd(x, weight, bias, padding=padding)

    # Compare
    out_pytorch_dev = out_pytorch.detach().to(out_ours.device)
    max_diff = (out_ours - out_pytorch_dev).abs().max().item()
    mean_diff = (out_ours - out_pytorch_dev).abs().mean().item()
    print(f"Max diff:  {max_diff:.6f}")
    print(f"Mean diff: {mean_diff:.6f}")

    # Tolerance scales with C_in: fp16 dot accumulates rounding error proportional
    # to the number of products summed.  Base tolerance 5e-2 for C_in <= 4,
    # allow ~5e-3 extra per additional input channel.
    atol = 5e-2 + max(0, C_in - 4) * 5e-3
    if max_diff < atol:
        print(f"PASS (tol={atol:.3f})\n")
        return True
    else:
        print(f"FAIL (max diff {max_diff} > tolerance {atol})\n")
        return False


# ============================================================
# Test cases
# ============================================================

results = []

# Test 1: Small symmetric, no padding
results.append(run_test(
    "Test 1: small symmetric, padding=0",
    batch_size=1, C_in=2, C_out=3,
    D=8, H=8, W=8,
    padding=0))

# Test 2: Small with padding=1 (most common practical config)
results.append(run_test(
    "Test 2: small with padding=1",
    batch_size=1, C_in=2, C_out=3,
    D=8, H=8, W=8,
    padding=1))

# Test 3: Larger channels — stresses the batched matmul
results.append(run_test(
    "Test 3: larger channels",
    batch_size=1, C_in=16, C_out=32,
    D=16, H=16, W=16,
    padding=1))

# Test 4: Non-symmetric spatial dimensions
results.append(run_test(
    "Test 4: non-symmetric spatial dims",
    batch_size=1, C_in=4, C_out=8,
    D=10, H=14, W=12,
    padding=1))

# Test 5: Odd output dimensions (tests tile padding/trimming)
results.append(run_test(
    "Test 5: odd output dims (tile trimming)",
    batch_size=1, C_in=3, C_out=6,
    D=7, H=9, W=11,
    padding=0))

# Test 6: No bias (explicit check that bias=None path works)
print("--- Test 6: no bias ---")
torch.manual_seed(42)
x6 = torch.randn(1, 4, 8, 8, 8)
w6 = torch.randn(8, 4, 3, 3, 3)
conv6 = nn.Conv3d(4, 8, 3, padding=1, bias=False)
with torch.no_grad():
    conv6.weight.copy_(w6)
out_pt6 = conv6(x6)
out_ours6 = conv3d_winograd(x6, w6, bias=None, padding=(1, 1, 1))
out_pt6_dev = out_pt6.detach().to(out_ours6.device)
max_diff6 = (out_ours6 - out_pt6_dev).abs().max().item()
print(f"Max diff: {max_diff6:.6f}")
atol6 = 5e-2  # C_in=4, base tolerance
passed6 = max_diff6 < atol6
print(f"{'PASS' if passed6 else 'FAIL'} (tol={atol6:.3f})\n")
results.append(passed6)

# Test 7: Minimal input — smallest possible (D=H=W=3, no padding -> 1x1x1 output)
results.append(run_test(
    "Test 7: minimal 3x3x3 input, padding=0",
    batch_size=1, C_in=2, C_out=2,
    D=3, H=3, W=3,
    padding=0))

# ============================================================
# Summary
# ============================================================
passed = sum(results)
total = len(results)
print(f"Results: {passed}/{total} tests passed")
