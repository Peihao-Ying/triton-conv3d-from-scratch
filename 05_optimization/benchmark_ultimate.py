"""
Benchmark: ultimate Conv3D vs PyTorch nn.Conv3d.

Tests all dispatch paths across various problem sizes and batch sizes.
"""

import torch
import torch.nn as nn

from conv3d_ultimate import conv3d_ultimate, DEVICE


def benchmark_fn(fn, warmup=10, repeat=50):
    """Benchmark using CUDA events for accurate GPU timing."""
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
    return times[len(times) // 2]


def run_benchmark(label, batch_size, C_in, C_out, D, H, W, kD, kH, kW,
                  stride=1, padding=0, dilation=1, groups=1):
    if isinstance(stride, int):
        stride = (stride, stride, stride)
    if isinstance(padding, int):
        padding = (padding, padding, padding)
    if isinstance(dilation, int):
        dilation = (dilation, dilation, dilation)

    print(f"\n{'='*60}")
    print(f"{label}")
    print(f"Input: ({batch_size},{C_in},{D},{H},{W})  "
          f"Kernel: ({C_out},{C_in // groups},{kD},{kH},{kW})")
    print(f"stride={stride}  padding={padding}  dilation={dilation}  groups={groups}")

    torch.manual_seed(42)
    x = torch.randn(batch_size, C_in, D, H, W, device=DEVICE)
    weight = torch.randn(C_out, C_in // groups, kD, kH, kW, device=DEVICE)
    bias = torch.randn(C_out, device=DEVICE)

    # PyTorch Conv3d
    conv = nn.Conv3d(C_in, C_out, (kD, kH, kW),
                     stride=stride, padding=padding, dilation=dilation,
                     groups=groups, bias=True).to(DEVICE)
    with torch.no_grad():
        conv.weight.copy_(weight)
        conv.bias.copy_(bias)

    pytorch_ms = benchmark_fn(lambda: conv(x))

    # Our ultimate kernel
    triton_ms = benchmark_fn(
        lambda: conv3d_ultimate(x, weight, bias,
                                stride=stride, padding=padding,
                                dilation=dilation, groups=groups))

    speedup = pytorch_ms / triton_ms
    tag = "(Triton faster)" if speedup > 1 else "(PyTorch faster)"
    print(f"PyTorch:  {pytorch_ms:8.3f} ms")
    print(f"Triton:   {triton_ms:8.3f} ms")
    print(f"Speedup:  {speedup:.2f}x {tag}")

    return label, pytorch_ms, triton_ms


# ============================================================
# Benchmark suite
# ============================================================

print("Ultimate Conv3D Benchmark")
print(f"Device: {DEVICE}")
print(f"GPU: {torch.cuda.get_device_name()}")

results = []

# --- Original Phase 4 benchmarks (batch=1) ---
print("\n" + "#" * 60)
print("# ORIGINAL PHASE 4 SIZES (batch=1, groups=1)")
print("#" * 60)

results.append(run_benchmark(
    "Small first layer (b=1)",
    1, 1, 16, 16, 16, 16, 3, 3, 3, padding=1))

results.append(run_benchmark(
    "Medium mid-net (b=1)",
    1, 32, 64, 16, 16, 16, 3, 3, 3, padding=1))

results.append(run_benchmark(
    "Large deeper (b=1)",
    1, 64, 128, 8, 16, 16, 3, 3, 3, padding=1))

results.append(run_benchmark(
    "Stride=2 downsample (b=1)",
    1, 32, 64, 16, 32, 32, 3, 3, 3, stride=2, padding=1))

results.append(run_benchmark(
    "Large high-res (b=1)",
    1, 16, 32, 32, 32, 32, 3, 3, 3, padding=1))

# --- Batched versions ---
print("\n" + "#" * 60)
print("# BATCHED (batch=4, groups=1)")
print("#" * 60)

results.append(run_benchmark(
    "Medium mid-net (b=4)",
    4, 32, 64, 16, 16, 16, 3, 3, 3, padding=1))

results.append(run_benchmark(
    "Large deeper (b=4)",
    4, 64, 128, 8, 16, 16, 3, 3, 3, padding=1))

results.append(run_benchmark(
    "Stride=2 downsample (b=4)",
    4, 32, 64, 16, 32, 32, 3, 3, 3, stride=2, padding=1))

results.append(run_benchmark(
    "Large high-res (b=4)",
    4, 16, 32, 32, 32, 32, 3, 3, 3, padding=1))

# --- Grouped convolution ---
print("\n" + "#" * 60)
print("# GROUPED CONVOLUTION")
print("#" * 60)

results.append(run_benchmark(
    "Groups=4 (b=4)",
    4, 64, 128, 16, 16, 16, 3, 3, 3, padding=1, groups=4))

results.append(run_benchmark(
    "Groups=32 ResNeXt-style (b=4)",
    4, 128, 128, 8, 16, 16, 3, 3, 3, padding=1, groups=32))

# --- Depthwise ---
print("\n" + "#" * 60)
print("# DEPTHWISE CONVOLUTION")
print("#" * 60)

results.append(run_benchmark(
    "Depthwise C=32 (b=4)",
    4, 32, 32, 16, 16, 16, 3, 3, 3, padding=1, groups=32))

results.append(run_benchmark(
    "Depthwise C=64 large (b=4)",
    4, 64, 64, 16, 32, 32, 3, 3, 3, padding=1, groups=64))

results.append(run_benchmark(
    "Depthwise stride=2 (b=4)",
    4, 32, 32, 16, 32, 32, 3, 3, 3, stride=2, padding=1, groups=32))

# --- Non-3x3x3 (general path) ---
print("\n" + "#" * 60)
print("# NON-3x3x3 KERNEL")
print("#" * 60)

results.append(run_benchmark(
    "5x5x5 kernel (b=1)",
    1, 16, 32, 16, 16, 16, 5, 5, 5, padding=2))

# Summary
print(f"\n{'='*70}")
print("SUMMARY")
print(f"{'Case':<40} {'PyTorch':>10} {'Triton':>10} {'Speedup':>10}")
print("-" * 70)
for label, pt, tr in results:
    speedup = pt / tr
    marker = " *" if speedup >= 1.0 else ""
    print(f"{label:<40} {pt:>8.3f}ms {tr:>8.3f}ms {speedup:>8.2f}x{marker}")
print("-" * 70)
print("* = Triton faster or equal")
