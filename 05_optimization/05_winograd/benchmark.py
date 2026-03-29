"""
Benchmark: Winograd F(2x2x2, 3x3x3) vs Phase 4 implicit im2col vs PyTorch nn.Conv3d.

All cases use 3x3x3 kernels with stride=1, dilation=1 — the only config
where Winograd applies.  This lets us do a fair three-way comparison on
the same problem sizes.
"""

import sys
import os

import torch
import torch.nn as nn

from conv3d_winograd import conv3d_winograd, DEVICE

# Import Phase 4 baseline from its directory
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "04_conv3d_implicit"))
from conv3d_implicit import conv3d_implicit


def benchmark_fn(fn, warmup=10, repeat=50):
    """Benchmark a function using CUDA events for accurate GPU timing."""
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()

    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)

    times = []
    for _ in range(repeat):
        start.record()
        fn()
        end.record()
        torch.cuda.synchronize()
        times.append(start.elapsed_time(end))

    times.sort()
    median = times[len(times) // 2]
    return median


def run_benchmark(label, C_in, C_out, D, H, W, padding=1):
    """Three-way benchmark: PyTorch vs Phase 4 implicit im2col vs Winograd."""
    padding_tuple = (padding, padding, padding)

    pD = pH = pW = padding
    D_out = D + 2 * pD - 2
    H_out = H + 2 * pH - 2
    W_out = W + 2 * pW - 2

    print(f"\n{'='*70}")
    print(f"{label}")
    print(f"Input: (1,{C_in},{D},{H},{W})  Kernel: ({C_out},{C_in},3,3,3)  padding={padding}")
    print(f"Output: (1,{C_out},{D_out},{H_out},{W_out})")

    torch.manual_seed(42)
    x = torch.randn(1, C_in, D, H, W, device=DEVICE)
    weight = torch.randn(C_out, C_in, 3, 3, 3, device=DEVICE)
    bias = torch.randn(C_out, device=DEVICE)

    # --- PyTorch nn.Conv3d ---
    conv = nn.Conv3d(C_in, C_out, 3, stride=1, padding=padding, dilation=1, bias=True).to(DEVICE)
    with torch.no_grad():
        conv.weight.copy_(weight)
        conv.bias.copy_(bias)

    pytorch_ms = benchmark_fn(lambda: conv(x))

    # --- Phase 4: implicit im2col (Triton) ---
    implicit_ms = benchmark_fn(
        lambda: conv3d_implicit(
            x, weight, bias,
            stride=(1, 1, 1), padding=padding_tuple, dilation=(1, 1, 1)))

    # --- Winograd (this implementation) ---
    winograd_ms = benchmark_fn(
        lambda: conv3d_winograd(x, weight, bias, padding=padding_tuple))

    # --- Report ---
    print(f"  PyTorch (cuDNN):    {pytorch_ms:8.3f} ms")
    print(f"  Triton implicit:    {implicit_ms:8.3f} ms")
    print(f"  Winograd:           {winograd_ms:8.3f} ms")
    print(f"  Winograd vs cuDNN:  {pytorch_ms / winograd_ms:.2f}x")
    print(f"  Winograd vs impl:   {implicit_ms / winograd_ms:.2f}x")

    return pytorch_ms, implicit_ms, winograd_ms


# ============================================================
# Benchmark suite — 3x3x3 stride=1 configs only
# ============================================================

print("Winograd F(2x2x2, 3x3x3) Conv3D Benchmark")
print(f"Device: {DEVICE}")
print(f"GPU: {torch.cuda.get_device_name()}")

results = []
labels = []

labels.append("Small first layer")
results.append(run_benchmark(
    "Small: typical first layer",
    C_in=1, C_out=16, D=16, H=16, W=16, padding=1))

labels.append("Medium mid-net")
results.append(run_benchmark(
    "Medium: mid-network layer",
    C_in=32, C_out=64, D=16, H=16, W=16, padding=1))

labels.append("Large deeper")
results.append(run_benchmark(
    "Large: deeper layer",
    C_in=64, C_out=128, D=8, H=16, W=16, padding=1))

labels.append("Large high-res")
results.append(run_benchmark(
    "Large input: high resolution",
    C_in=16, C_out=32, D=32, H=32, W=32, padding=1))

labels.append("Wide channels")
results.append(run_benchmark(
    "Wide channels: where Winograd should shine",
    C_in=128, C_out=128, D=8, H=8, W=8, padding=1))

# ============================================================
# Summary table
# ============================================================
print(f"\n{'='*70}")
print("Summary")
print(f"{'Case':<22} {'cuDNN':>9} {'Implicit':>9} {'Winograd':>9} {'W/cuDNN':>8} {'W/Impl':>8}")
print("-" * 70)
for label, (pt, impl, wino) in zip(labels, results):
    print(f"{label:<22} {pt:>7.3f}ms {impl:>7.3f}ms {wino:>7.3f}ms "
          f"{pt/wino:>7.2f}x {impl/wino:>7.2f}x")
