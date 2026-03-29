"""
Test batch-parallel Conv3d against PyTorch's nn.Conv3d.

Test cases:
  Tests 1-7:   Same 7 configs from Phase 4 with batch_size=1 (regression)
  Tests 8-14:  Same 7 configs with batch_size=4
  Test 15:     batch_size=8 (medium config)
"""

import torch
import torch.nn as nn

from conv3d_batch import conv3d_batch


def run_test(label, batch_size, C_in, C_out, D, H, W, kD, kH, kW,
             stride=1, padding=0, dilation=1):
    # Normalize to tuples
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
          f"Kernel: ({C_out},{C_in},{kD},{kH},{kW})  "
          f"stride={stride} pad={padding} dil={dilation}")
    print(f"Output: ({batch_size},{C_out},{D_out},{H_out},{W_out})")

    torch.manual_seed(42)
    x = torch.randn(batch_size, C_in, D, H, W)
    weight = torch.randn(C_out, C_in, kD, kH, kW)
    bias = torch.randn(C_out)

    # Ground truth: PyTorch nn.Conv3d
    conv = nn.Conv3d(C_in, C_out, (kD, kH, kW),
                     stride=stride, padding=padding, dilation=dilation, bias=True)
    with torch.no_grad():
        conv.weight.copy_(weight)
        conv.bias.copy_(bias)

    out_pytorch = conv(x)

    # Our implementation
    out_ours = conv3d_batch(x, weight, bias,
                            stride=stride, padding=padding, dilation=dilation)

    # Compare
    out_pytorch_dev = out_pytorch.detach().to(out_ours.device)
    max_diff = (out_ours - out_pytorch_dev).abs().max().item()
    print(f"Max diff: {max_diff:.6f}")

    # fp16 tl.dot accumulates rounding error proportional to sqrt(K)
    K = C_in * kD * kH * kW
    atol = max(2e-2, K ** 0.5 * 2e-3)
    if max_diff < atol:
        print("PASS\n")
        return True
    else:
        print(f"FAIL (max diff {max_diff} > tolerance {atol})\n")
        return False


# ============================================================
# Test cases — batch_size=1 (regression, same as Phase 4)
# ============================================================

results = []

results.append(run_test(
    "Test 1: B=1, stride=1, pad=0 (small symmetric)",
    1, 2, 3, 4, 4, 4, 3, 3, 3))

results.append(run_test(
    "Test 2: B=1, stride=1, pad=0 (larger non-symmetric)",
    1, 3, 8, 6, 8, 10, 3, 3, 3))

results.append(run_test(
    "Test 3: B=1, stride=2",
    1, 3, 4, 8, 8, 8, 3, 3, 3, stride=2))

results.append(run_test(
    "Test 4: B=1, padding=1",
    1, 2, 4, 6, 6, 6, 3, 3, 3, padding=1))

results.append(run_test(
    "Test 5: B=1, stride=2, padding=1",
    1, 3, 8, 8, 8, 8, 3, 3, 3, stride=2, padding=1))

results.append(run_test(
    "Test 6: B=1, dilation=2",
    1, 2, 4, 8, 8, 8, 3, 3, 3, dilation=2))

results.append(run_test(
    "Test 7: B=1, stride=2, padding=1, dilation=2",
    1, 3, 8, 10, 10, 10, 3, 3, 3, stride=2, padding=1, dilation=2))


# ============================================================
# Test cases — batch_size=4
# ============================================================

results.append(run_test(
    "Test 8: B=4, stride=1, pad=0 (small symmetric)",
    4, 2, 3, 4, 4, 4, 3, 3, 3))

results.append(run_test(
    "Test 9: B=4, stride=1, pad=0 (larger non-symmetric)",
    4, 3, 8, 6, 8, 10, 3, 3, 3))

results.append(run_test(
    "Test 10: B=4, stride=2",
    4, 3, 4, 8, 8, 8, 3, 3, 3, stride=2))

results.append(run_test(
    "Test 11: B=4, padding=1",
    4, 2, 4, 6, 6, 6, 3, 3, 3, padding=1))

results.append(run_test(
    "Test 12: B=4, stride=2, padding=1",
    4, 3, 8, 8, 8, 8, 3, 3, 3, stride=2, padding=1))

results.append(run_test(
    "Test 13: B=4, dilation=2",
    4, 2, 4, 8, 8, 8, 3, 3, 3, dilation=2))

results.append(run_test(
    "Test 14: B=4, stride=2, padding=1, dilation=2",
    4, 3, 8, 10, 10, 10, 3, 3, 3, stride=2, padding=1, dilation=2))


# ============================================================
# Test case — batch_size=8
# ============================================================

results.append(run_test(
    "Test 15: B=8, mid-network (stride=1, pad=1)",
    8, 32, 64, 16, 16, 16, 3, 3, 3, padding=1))


# ============================================================
# Summary
# ============================================================

passed = sum(results)
total = len(results)
print(f"Results: {passed}/{total} tests passed")
