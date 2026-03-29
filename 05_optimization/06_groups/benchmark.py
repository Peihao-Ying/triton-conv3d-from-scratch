"""
Benchmark: grouped and depthwise Conv3d (Triton) vs torch.nn.Conv3d.

Measures execution time across group counts (1, 2, 4, 32) and depthwise.
"""

import torch
import torch.nn as nn

from conv3d_groups import conv3d_groups, DEVICE


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
                  stride=1, padding=0, dilation=1, groups=1):
    if isinstance(stride, int):
        stride = (stride, stride, stride)
    if isinstance(padding, int):
        padding = (padding, padding, padding)
    if isinstance(dilation, int):
        dilation = (dilation, dilation, dilation)

    print(f"\n{'='*60}")
    print(f"{label}")
    print(f"Input: (1,{C_in},{D},{H},{W})  Kernel: ({C_out},{C_in // groups},{kD},{kH},{kW})")
    print(f"stride={stride}  padding={padding}  dilation={dilation}  groups={groups}")

    torch.manual_seed(42)
    x = torch.randn(1, C_in, D, H, W, device=DEVICE)
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

    # Our grouped/depthwise kernel
    triton_ms = benchmark_fn(
        lambda: conv3d_groups(x, weight, bias,
                              stride=stride, padding=padding, dilation=dilation,
                              groups=groups))

    speedup = pytorch_ms / triton_ms
    print(f"PyTorch:  {pytorch_ms:8.3f} ms")
    print(f"Triton:   {triton_ms:8.3f} ms")
    print(f"Speedup:  {speedup:.2f}x {'(Triton faster)' if speedup > 1 else '(PyTorch faster)'}")

    return label, pytorch_ms, triton_ms


# ============================================================
# Benchmark suite
# ============================================================

print("Grouped / Depthwise Conv3d Benchmark")
print(f"Device: {DEVICE}")
print(f"GPU: {torch.cuda.get_device_name()}")

results = []

# --- groups=1 baseline ---
results.append(run_benchmark(
    "groups=1 (baseline)",
    C_in=32, C_out=64, D=16, H=16, W=16, kD=3, kH=3, kW=3, padding=1, groups=1))

# --- groups=2 ---
results.append(run_benchmark(
    "groups=2",
    C_in=32, C_out=64, D=16, H=16, W=16, kD=3, kH=3, kW=3, padding=1, groups=2))

# --- groups=4 ---
results.append(run_benchmark(
    "groups=4",
    C_in=32, C_out=64, D=16, H=16, W=16, kD=3, kH=3, kW=3, padding=1, groups=4))

# --- groups=32 ---
results.append(run_benchmark(
    "groups=32",
    C_in=32, C_out=64, D=16, H=16, W=16, kD=3, kH=3, kW=3, padding=1, groups=32))

# --- Depthwise (groups=C_in=C_out) ---
results.append(run_benchmark(
    "depthwise (C=32)",
    C_in=32, C_out=32, D=16, H=16, W=16, kD=3, kH=3, kW=3, padding=1, groups=32))

results.append(run_benchmark(
    "depthwise (C=64)",
    C_in=64, C_out=64, D=16, H=16, W=16, kD=3, kH=3, kW=3, padding=1, groups=64))

results.append(run_benchmark(
    "depthwise large spatial",
    C_in=32, C_out=32, D=32, H=32, W=32, kD=3, kH=3, kW=3, padding=1, groups=32))

# --- Summary ---
print(f"\n{'='*60}")
print("Summary")
print(f"{'Case':<30} {'PyTorch':>10} {'Triton':>10} {'Speedup':>10}")
print("-" * 60)
for label, pt, tr in results:
    print(f"{label:<30} {pt:>8.3f}ms {tr:>8.3f}ms {pt/tr:>9.2f}x")
