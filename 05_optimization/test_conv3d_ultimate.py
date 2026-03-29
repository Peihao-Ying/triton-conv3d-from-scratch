"""
Comprehensive test suite for the ultimate Conv3D kernel.

Covers all three dispatch paths:
  - General optimized (groups=1, various stride/pad/dil)
  - Winograd (3x3x3, stride=1, dilation=1)
  - Depthwise (groups=C_in=C_out)
  - Grouped convolution (groups=2, 4)
  - Batch sizes: 1, 2, 4, 8
  - Combined: batch + groups + stride + padding
"""

import torch
import torch.nn as nn

from conv3d_ultimate import conv3d_ultimate


def run_test(label, batch_size, C_in, C_out, D, H, W, kD, kH, kW,
             stride=1, padding=0, dilation=1, groups=1):
    if isinstance(stride, int):
        stride = (stride, stride, stride)
    if isinstance(padding, int):
        padding = (padding, padding, padding)
    if isinstance(dilation, int):
        dilation = (dilation, dilation, dilation)

    sD, sH, sW = stride
    pD, pH, pW = padding
    dD, dH, dW = dilation

    D_out = (D + 2 * pD - dD * (kD - 1) - 1) // sD + 1
    H_out = (H + 2 * pH - dH * (kH - 1) - 1) // sH + 1
    W_out = (W + 2 * pW - dW * (kW - 1) - 1) // sW + 1

    print(f"--- {label} ---")
    print(f"Input: ({batch_size},{C_in},{D},{H},{W})  "
          f"Kernel: ({C_out},{C_in // groups},{kD},{kH},{kW})  "
          f"stride={stride} pad={padding} dil={dilation} groups={groups}")
    print(f"Output: ({batch_size},{C_out},{D_out},{H_out},{W_out})")

    torch.manual_seed(42)
    x = torch.randn(batch_size, C_in, D, H, W)
    weight = torch.randn(C_out, C_in // groups, kD, kH, kW)
    bias = torch.randn(C_out)

    # Ground truth: PyTorch nn.Conv3d
    conv = nn.Conv3d(C_in, C_out, (kD, kH, kW),
                     stride=stride, padding=padding, dilation=dilation,
                     groups=groups, bias=True)
    with torch.no_grad():
        conv.weight.copy_(weight)
        conv.bias.copy_(bias)

    out_pytorch = conv(x)

    # Our implementation
    out_ours = conv3d_ultimate(x, weight, bias,
                               stride=stride, padding=padding,
                               dilation=dilation, groups=groups)

    # Compare
    out_pytorch_dev = out_pytorch.detach().to(out_ours.device)
    max_diff = (out_ours - out_pytorch_dev).abs().max().item()
    print(f"Max diff: {max_diff:.6f}")

    # fp16 tl.dot accumulates rounding error proportional to sqrt(K).
    # Winograd adds transform overhead on top of that.
    K = (C_in // groups) * kD * kH * kW
    is_winograd = (kD == kH == kW == 3 and stride == (1, 1, 1)
                   and dilation == (1, 1, 1) and groups == 1)
    if is_winograd:
        atol = max(5e-2, C_in * 5e-3)
    else:
        atol = max(2e-2, K ** 0.5 * 2e-3)

    if max_diff < atol:
        print("PASS\n")
        return True
    else:
        print(f"FAIL (max diff {max_diff} > tolerance {atol})\n")
        return False


# ============================================================
# Test cases
# ============================================================

results = []

# --- Phase 4 regression tests (batch=1, groups=1) ---
print("=" * 60)
print("PHASE 4 REGRESSION (batch=1, groups=1)")
print("=" * 60)

results.append(run_test(
    "Test 1: stride=1, pad=0 (small symmetric)",
    1, 2, 3, 4, 4, 4, 3, 3, 3))

results.append(run_test(
    "Test 2: stride=1, pad=0 (larger non-symmetric)",
    1, 3, 8, 6, 8, 10, 3, 3, 3))

results.append(run_test(
    "Test 3: stride=2",
    1, 3, 4, 8, 8, 8, 3, 3, 3, stride=2))

results.append(run_test(
    "Test 4: padding=1",
    1, 2, 4, 6, 6, 6, 3, 3, 3, padding=1))

results.append(run_test(
    "Test 5: stride=2, padding=1",
    1, 3, 8, 8, 8, 8, 3, 3, 3, stride=2, padding=1))

results.append(run_test(
    "Test 6: dilation=2",
    1, 2, 4, 8, 8, 8, 3, 3, 3, dilation=2))

results.append(run_test(
    "Test 7: stride=2, padding=1, dilation=2",
    1, 3, 8, 10, 10, 10, 3, 3, 3, stride=2, padding=1, dilation=2))

# --- Winograd path tests (3x3x3, stride=1, dilation=1) ---
print("=" * 60)
print("WINOGRAD PATH (3x3x3, stride=1, dil=1)")
print("=" * 60)

results.append(run_test(
    "Test 8: Winograd small, pad=0",
    1, 2, 3, 8, 8, 8, 3, 3, 3))

results.append(run_test(
    "Test 9: Winograd with pad=1",
    1, 4, 8, 8, 8, 8, 3, 3, 3, padding=1))

results.append(run_test(
    "Test 10: Winograd larger channels",
    1, 16, 32, 16, 16, 16, 3, 3, 3, padding=1))

results.append(run_test(
    "Test 11: Winograd non-symmetric spatial",
    1, 4, 8, 6, 10, 14, 3, 3, 3, padding=1))

# --- Batch tests ---
print("=" * 60)
print("BATCH PARALLELISM")
print("=" * 60)

results.append(run_test(
    "Test 12: batch=2, stride=2, pad=1 (general path)",
    2, 3, 4, 8, 8, 8, 3, 3, 3, stride=2, padding=1))

results.append(run_test(
    "Test 13: batch=4, medium (general path)",
    4, 32, 64, 16, 16, 16, 3, 3, 3, stride=2, padding=1))

results.append(run_test(
    "Test 14: batch=8, small (general path)",
    8, 3, 8, 8, 8, 8, 3, 3, 3, stride=2, padding=1))

results.append(run_test(
    "Test 15: batch=4, Winograd path",
    4, 8, 16, 8, 8, 8, 3, 3, 3, padding=1))

# --- Groups tests ---
print("=" * 60)
print("GROUPED CONVOLUTION")
print("=" * 60)

results.append(run_test(
    "Test 16: groups=2",
    1, 4, 8, 8, 8, 8, 3, 3, 3, padding=1, groups=2))

results.append(run_test(
    "Test 17: groups=4",
    1, 8, 16, 8, 8, 8, 3, 3, 3, padding=1, groups=4))

results.append(run_test(
    "Test 18: depthwise (groups=C_in=C_out=16)",
    1, 16, 16, 8, 8, 8, 3, 3, 3, padding=1, groups=16))

results.append(run_test(
    "Test 19: depthwise + stride=2",
    1, 16, 16, 16, 16, 16, 3, 3, 3, stride=2, padding=1, groups=16))

# --- Combined tests ---
print("=" * 60)
print("COMBINED (batch + groups + stride + padding)")
print("=" * 60)

results.append(run_test(
    "Test 20: batch=4, groups=2, stride=2, pad=1",
    4, 8, 16, 16, 16, 16, 3, 3, 3, stride=2, padding=1, groups=2))

results.append(run_test(
    "Test 21: batch=2, depthwise, pad=1",
    2, 32, 32, 8, 8, 8, 3, 3, 3, padding=1, groups=32))

results.append(run_test(
    "Test 22: batch=4, depthwise, stride=2, pad=1",
    4, 16, 16, 16, 16, 16, 3, 3, 3, stride=2, padding=1, groups=16))

# --- Edge cases ---
print("=" * 60)
print("EDGE CASES")
print("=" * 60)

results.append(run_test(
    "Test 23: C_out=1",
    1, 3, 1, 8, 8, 8, 3, 3, 3, stride=2, padding=1))

results.append(run_test(
    "Test 24: tiny spatial (4x4x4, stride=2)",
    1, 4, 8, 4, 4, 4, 3, 3, 3, stride=2, padding=1))

results.append(run_test(
    "Test 25: non-3x3x3 kernel (5x5x5, general path)",
    1, 2, 4, 8, 8, 8, 5, 5, 5, padding=2))

# Summary
passed = sum(results)
total = len(results)
print("=" * 60)
print(f"Results: {passed}/{total} tests passed")
if passed == total:
    print("ALL TESTS PASSED")
else:
    failed = [i + 1 for i, r in enumerate(results) if not r]
    print(f"FAILED tests: {failed}")
