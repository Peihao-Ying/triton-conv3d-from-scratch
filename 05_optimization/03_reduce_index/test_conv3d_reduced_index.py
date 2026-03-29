"""
Test reduced-index Conv3d against PyTorch's nn.Conv3d.

Both USE_LUT=True and USE_LUT=False paths are tested.

Test cases (same 7 as Phase 4):
  1. Small symmetric, stride=1, padding=0
  2. Larger non-symmetric, stride=1, padding=0
  3. Stride=2
  4. Padding=1
  5. Stride=2 + padding=1
  6. Dilation=2
  7. Stride + padding + dilation combined
"""

import torch
import torch.nn as nn

from conv3d_reduced_index import conv3d_reduced_index


def run_test(label, batch_size, C_in, C_out, D, H, W, kD, kH, kW,
             stride=1, padding=0, dilation=1, use_lut=True):
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

    lut_tag = "LUT" if use_lut else "constexpr"
    print(f"--- {label} [{lut_tag}] ---")
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
    out_ours = conv3d_reduced_index(x, weight, bias,
                                    stride=stride, padding=padding,
                                    dilation=dilation, use_lut=use_lut)

    # Compare
    out_pytorch_dev = out_pytorch.detach().to(out_ours.device)
    max_diff = (out_ours - out_pytorch_dev).abs().max().item()
    print(f"Max diff: {max_diff:.6f}")

    atol = 2e-2  # float16 tl.dot accumulates rounding error
    if max_diff < atol:
        print("PASS\n")
        return True
    else:
        print(f"FAIL (max diff {max_diff} > tolerance {atol})\n")
        return False


# ============================================================
# Test cases — both USE_LUT paths
# ============================================================

test_configs = [
    ("Test 1: stride=1, pad=0 (small symmetric)",
     1, 2, 3, 4, 4, 4, 3, 3, 3, {}),
    ("Test 2: stride=1, pad=0 (larger non-symmetric)",
     1, 3, 8, 6, 8, 10, 3, 3, 3, {}),
    ("Test 3: stride=2",
     1, 3, 4, 8, 8, 8, 3, 3, 3, {"stride": 2}),
    ("Test 4: padding=1",
     1, 2, 4, 6, 6, 6, 3, 3, 3, {"padding": 1}),
    ("Test 5: stride=2, padding=1",
     1, 3, 8, 8, 8, 8, 3, 3, 3, {"stride": 2, "padding": 1}),
    ("Test 6: dilation=2",
     1, 2, 4, 8, 8, 8, 3, 3, 3, {"dilation": 2}),
    ("Test 7: stride=2, padding=1, dilation=2",
     1, 3, 8, 10, 10, 10, 3, 3, 3, {"stride": 2, "padding": 1, "dilation": 2}),
]

results = []

for use_lut in [False, True]:
    tag = "USE_LUT=True" if use_lut else "USE_LUT=False (constexpr only)"
    print(f"\n{'='*60}")
    print(f"  Testing path: {tag}")
    print(f"{'='*60}\n")

    for label, bs, ci, co, d, h, w, kd, kh, kw, kwargs in test_configs:
        results.append(run_test(label, bs, ci, co, d, h, w, kd, kh, kw,
                                use_lut=use_lut, **kwargs))

passed = sum(results)
total = len(results)
print(f"{'='*60}")
print(f"Results: {passed}/{total} tests passed")
if passed == total:
    print("All tests passed!")
else:
    print(f"{total - passed} test(s) FAILED")
