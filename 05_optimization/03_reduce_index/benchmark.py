"""
Benchmark: reduced-index Conv3d variants vs Phase 4 baseline vs PyTorch.

Compares four implementations:
  1. Phase 4 baseline  (no constexpr, no LUT — original implicit im2col)
  2. Constexpr-only    (kD/kH/kW as tl.constexpr, USE_LUT=False)
  3. Constexpr + LUT   (constexpr + pre-computed lookup tables)
  4. PyTorch nn.Conv3d

Same 5 benchmark cases as Phase 4.
"""

import sys
import os

import torch
import torch.nn as nn

# Import Phase 4 baseline
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "04_conv3d_implicit"))
from conv3d_implicit import conv3d_implicit

# Import reduced-index kernel
from conv3d_reduced_index import conv3d_reduced_index, DEVICE


def benchmark_fn(fn, warmup=10, repeat=50):
    """Benchmark a function using CUDA events for accurate GPU timing."""
    # Warmup
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
    # Use median to avoid outliers
    median = times[len(times) // 2]
    return median


def run_benchmark(label, C_in, C_out, D, H, W, kD, kH, kW,
                  stride=1, padding=0, dilation=1):
    if isinstance(stride, int):
        stride = (stride, stride, stride)
    if isinstance(padding, int):
        padding = (padding, padding, padding)
    if isinstance(dilation, int):
        dilation = (dilation, dilation, dilation)

    print(f"\n{'='*70}")
    print(f"{label}")
    print(f"Input: (1,{C_in},{D},{H},{W})  Kernel: ({C_out},{C_in},{kD},{kH},{kW})")
    print(f"stride={stride}  padding={padding}  dilation={dilation}")

    torch.manual_seed(42)
    x = torch.randn(1, C_in, D, H, W, device=DEVICE)
    weight = torch.randn(C_out, C_in, kD, kH, kW, device=DEVICE)
    bias = torch.randn(C_out, device=DEVICE)

    # PyTorch Conv3d
    conv = nn.Conv3d(C_in, C_out, (kD, kH, kW),
                     stride=stride, padding=padding, dilation=dilation, bias=True).to(DEVICE)
    with torch.no_grad():
        conv.weight.copy_(weight)
        conv.bias.copy_(bias)

    pytorch_ms = benchmark_fn(lambda: conv(x))

    # Phase 4 baseline (no constexpr, no LUT)
    baseline_ms = benchmark_fn(
        lambda: conv3d_implicit(x, weight, bias,
                                stride=stride, padding=padding, dilation=dilation))

    # Constexpr-only (USE_LUT=False)
    constexpr_ms = benchmark_fn(
        lambda: conv3d_reduced_index(x, weight, bias,
                                     stride=stride, padding=padding,
                                     dilation=dilation, use_lut=False))

    # Constexpr + LUT (USE_LUT=True)
    lut_ms = benchmark_fn(
        lambda: conv3d_reduced_index(x, weight, bias,
                                     stride=stride, padding=padding,
                                     dilation=dilation, use_lut=True))

    print(f"  PyTorch:        {pytorch_ms:8.3f} ms")
    print(f"  Phase 4:        {baseline_ms:8.3f} ms")
    print(f"  Constexpr:      {constexpr_ms:8.3f} ms  ({baseline_ms/constexpr_ms:.2f}x vs Phase 4)")
    print(f"  Constexpr+LUT:  {lut_ms:8.3f} ms  ({baseline_ms/lut_ms:.2f}x vs Phase 4)")

    return pytorch_ms, baseline_ms, constexpr_ms, lut_ms


# ============================================================
# Benchmark suite
# ============================================================

print("Reduced Index Conv3d Benchmark")
print(f"Device: {DEVICE}")
print(f"GPU: {torch.cuda.get_device_name()}")

results = []

results.append(run_benchmark(
    "Small: typical first layer",
    C_in=1, C_out=16, D=16, H=16, W=16, kD=3, kH=3, kW=3, padding=1))

results.append(run_benchmark(
    "Medium: mid-network layer",
    C_in=32, C_out=64, D=16, H=16, W=16, kD=3, kH=3, kW=3, padding=1))

results.append(run_benchmark(
    "Large: deeper layer",
    C_in=64, C_out=128, D=8, H=16, W=16, kD=3, kH=3, kW=3, padding=1))

results.append(run_benchmark(
    "Stride=2: downsampling layer",
    C_in=32, C_out=64, D=16, H=32, W=32, kD=3, kH=3, kW=3, stride=2, padding=1))

results.append(run_benchmark(
    "Large input: high resolution",
    C_in=16, C_out=32, D=32, H=32, W=32, kD=3, kH=3, kW=3, padding=1))

# Summary table
labels = ["Small first layer", "Medium mid-net", "Large deeper",
          "Stride=2 downsample", "Large high-res"]

print(f"\n{'='*70}")
print("Summary")
print(f"{'Case':<22} {'PyTorch':>9} {'Phase4':>9} {'Constexp':>9} {'Cexp+LUT':>9} {'vs P4':>8} {'vs PT':>8}")
print("-" * 76)
for label, (pt, p4, ce, lut) in zip(labels, results):
    # "vs P4" = speedup of best Triton variant over Phase 4 baseline
    # "vs PT" = speedup of best Triton variant over PyTorch
    best = min(ce, lut)
    print(f"{label:<22} {pt:>7.3f}ms {p4:>7.3f}ms {ce:>7.3f}ms {lut:>7.3f}ms "
          f"{p4/best:>7.2f}x {pt/best:>7.2f}x")
