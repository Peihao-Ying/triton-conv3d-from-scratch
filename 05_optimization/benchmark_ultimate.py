"""
Benchmark: ultimate Conv3D (v2) vs PyTorch nn.Conv3d (cuDNN).

Tests general path across batch sizes 1/4/8, plus grouped and depthwise.
100 iterations, median timing.
"""

import torch
import torch.nn as nn

from conv3d_ultimate import conv3d_ultimate, DEVICE


def benchmark_fn(fn, warmup=10, repeat=100):
    """Benchmark using CUDA events. Returns median time in ms."""
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

    torch.manual_seed(42)
    x = torch.randn(batch_size, C_in, D, H, W, device=DEVICE)
    weight = torch.randn(C_out, C_in // groups, kD, kH, kW, device=DEVICE)
    bias = torch.randn(C_out, device=DEVICE)

    conv = nn.Conv3d(C_in, C_out, (kD, kH, kW),
                     stride=stride, padding=padding, dilation=dilation,
                     groups=groups, bias=True).to(DEVICE)
    with torch.no_grad():
        conv.weight.copy_(weight)
        conv.bias.copy_(bias)

    pytorch_ms = benchmark_fn(lambda: conv(x))
    triton_ms = benchmark_fn(
        lambda: conv3d_ultimate(x, weight, bias,
                                stride=stride, padding=padding,
                                dilation=dilation, groups=groups))

    speedup = pytorch_ms / triton_ms
    tag = "Triton" if speedup > 1 else "cuDNN"
    print(f"  {label:<45} {pytorch_ms:>7.3f}ms {triton_ms:>7.3f}ms {speedup:>7.2f}x  {tag}")

    return label, pytorch_ms, triton_ms


print("=" * 90)
print("  Ultimate Conv3D v2 Benchmark")
print(f"  GPU: {torch.cuda.get_device_name()}")
print(f"  Kernel: flat K-loop + expanded autotuning + batch/group parallelism")
print(f"  Timing: CUDA events, 100 iterations, median")
print("=" * 90)

results = []

# ── batch=1, groups=1 ──
print(f"\n  {'Case':<45} {'cuDNN':>8} {'Triton':>8} {'Speedup':>8}  {'Winner'}")
print(f"  {'─'*45} {'─'*8} {'─'*8} {'─'*8}  {'─'*6}")

print(f"\n  ── batch=1, groups=1 ──")
results.append(run_benchmark("S1: tiny 1->16, 16^3", 1, 1, 16, 16, 16, 16, 3, 3, 3, padding=1))
results.append(run_benchmark("S2: small 3->32, 16^3", 1, 3, 32, 16, 16, 16, 3, 3, 3, padding=1))
results.append(run_benchmark("M1: mid-net 32->64, 16^3", 1, 32, 64, 16, 16, 16, 3, 3, 3, padding=1))
results.append(run_benchmark("M2: mid-net 64->128, 8x16x16", 1, 64, 128, 8, 16, 16, 3, 3, 3, padding=1))
results.append(run_benchmark("M3: stride=2, 32->64, 16x32x32", 1, 32, 64, 16, 32, 32, 3, 3, 3, stride=2, padding=1))
results.append(run_benchmark("L1: high-res 16->32, 32^3", 1, 16, 32, 32, 32, 32, 3, 3, 3, padding=1))
results.append(run_benchmark("L2: high-res 16->32, 48^3", 1, 16, 32, 48, 48, 48, 3, 3, 3, padding=1))
results.append(run_benchmark("L3: high-res 16->32, 64^3", 1, 16, 32, 64, 64, 64, 3, 3, 3, padding=1))
results.append(run_benchmark("L4: high-res 64^3 stride=2", 1, 16, 32, 64, 64, 64, 3, 3, 3, stride=2, padding=1))
results.append(run_benchmark("C1: wide 128->128, 16^3", 1, 128, 128, 16, 16, 16, 3, 3, 3, padding=1))
results.append(run_benchmark("C2: wide 128->256, 16^3", 1, 128, 256, 16, 16, 16, 3, 3, 3, padding=1))
results.append(run_benchmark("C3: wide 256->256, 8^3", 1, 256, 256, 8, 8, 8, 3, 3, 3, padding=1))
results.append(run_benchmark("X1: stress 64->128, 32^3", 1, 64, 128, 32, 32, 32, 3, 3, 3, padding=1))
results.append(run_benchmark("X2: stress 64->128, 32^3 stride=2", 1, 64, 128, 32, 32, 32, 3, 3, 3, stride=2, padding=1))

# ── batch=4, groups=1 ──
print(f"\n  ── batch=4, groups=1 ──")
results.append(run_benchmark("M1: mid-net 32->64, 16^3 b=4", 4, 32, 64, 16, 16, 16, 3, 3, 3, padding=1))
results.append(run_benchmark("M3: stride=2, 32->64 b=4", 4, 32, 64, 16, 32, 32, 3, 3, 3, stride=2, padding=1))
results.append(run_benchmark("L1: high-res 16->32, 32^3 b=4", 4, 16, 32, 32, 32, 32, 3, 3, 3, padding=1))
results.append(run_benchmark("L3: high-res 16->32, 64^3 b=4", 4, 16, 32, 64, 64, 64, 3, 3, 3, padding=1))
results.append(run_benchmark("C1: wide 128->128, 16^3 b=4", 4, 128, 128, 16, 16, 16, 3, 3, 3, padding=1))
results.append(run_benchmark("X2: stress 64->128, 32^3 s2 b=4", 4, 64, 128, 32, 32, 32, 3, 3, 3, stride=2, padding=1))

# ── batch=8, groups=1 ──
print(f"\n  ── batch=8, groups=1 ──")
results.append(run_benchmark("M1: mid-net 32->64, 16^3 b=8", 8, 32, 64, 16, 16, 16, 3, 3, 3, padding=1))
results.append(run_benchmark("L1: high-res 16->32, 32^3 b=8", 8, 16, 32, 32, 32, 32, 3, 3, 3, padding=1))
results.append(run_benchmark("X2: stress 64->128, 32^3 s2 b=8", 8, 64, 128, 32, 32, 32, 3, 3, 3, stride=2, padding=1))

# ── grouped convolution ──
print(f"\n  ── grouped convolution ──")
results.append(run_benchmark("groups=2, 32->64, 16^3 b=4", 4, 32, 64, 16, 16, 16, 3, 3, 3, padding=1, groups=2))
results.append(run_benchmark("groups=4, 64->128, 16^3 b=4", 4, 64, 128, 16, 16, 16, 3, 3, 3, padding=1, groups=4))

# ── depthwise ──
print(f"\n  ── depthwise ──")
results.append(run_benchmark("depthwise C=32, 16^3 b=4", 4, 32, 32, 16, 16, 16, 3, 3, 3, padding=1, groups=32))
results.append(run_benchmark("depthwise C=64, 16x32x32 b=4", 4, 64, 64, 16, 32, 32, 3, 3, 3, padding=1, groups=64))

# ── non-standard kernels ──
print(f"\n  ── non-standard kernels ──")
results.append(run_benchmark("5x5x5, 32->64, 16^3 b=1", 1, 32, 64, 16, 16, 16, 5, 5, 5, padding=2))
results.append(run_benchmark("1x1x1 pointwise, 128->256 b=1", 1, 128, 256, 16, 16, 16, 1, 1, 1))

# ── Summary ──
print(f"\n{'='*90}")
print("  Summary")
print(f"{'='*90}")
print(f"  {'Case':<45} {'cuDNN':>8} {'Triton':>8} {'Speedup':>8}")
print(f"  {'─'*45} {'─'*8} {'─'*8} {'─'*8}")

triton_wins = 0
for label, pt, tr in results:
    sp = pt / tr
    marker = " <-" if sp > 1 else ""
    print(f"  {label:<45} {pt:>7.3f}ms {tr:>7.3f}ms {sp:>7.2f}x{marker}")
    if sp > 1:
        triton_wins += 1

print(f"\n  Triton faster in {triton_wins}/{len(results)} cases")
print(f"{'='*90}")
