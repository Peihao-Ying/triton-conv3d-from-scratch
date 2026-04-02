"""
Comprehensive benchmark: Ultimate Conv3D v2 vs PyTorch nn.Conv3d (cuDNN).

Covers 6 dimensions of variation:
  1. Batch size scaling (1 -> 2 -> 4 -> 8)
  2. Spatial resolution scaling (8^3 -> 16^3 -> 32^3 -> 48^3 -> 64^3)
  3. Channel width scaling (16 -> 32 -> 64 -> 128 -> 256)
  4. Stride (1 vs 2)
  5. Kernel size (1x1x1, 3x3x3, 5x5x5)
  6. Groups (1, 2, 4, depthwise)
  + Real-world model configs (ResNet3D, C3D, SlowFast)

100 iterations, median timing.
"""

import torch
import torch.nn as nn

from conv3d_ultimate import conv3d_ultimate, DEVICE


def benchmark_fn(fn, warmup=10, repeat=100):
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


def run(label, B, Ci, Co, D, H, W, kD, kH, kW,
        stride=1, padding=0, dilation=1, groups=1):
    if isinstance(stride, int):
        stride = (stride, stride, stride)
    if isinstance(padding, int):
        padding = (padding, padding, padding)
    if isinstance(dilation, int):
        dilation = (dilation, dilation, dilation)

    torch.manual_seed(42)
    x = torch.randn(B, Ci, D, H, W, device=DEVICE)
    w = torch.randn(Co, Ci // groups, kD, kH, kW, device=DEVICE)
    b = torch.randn(Co, device=DEVICE)

    conv = nn.Conv3d(Ci, Co, (kD, kH, kW), stride=stride, padding=padding,
                     dilation=dilation, groups=groups, bias=True).to(DEVICE)
    with torch.no_grad():
        conv.weight.copy_(w)
        conv.bias.copy_(b)

    pt = benchmark_fn(lambda: conv(x))
    tr = benchmark_fn(lambda: conv3d_ultimate(x, w, b, stride=stride, padding=padding,
                                              dilation=dilation, groups=groups))
    sp = pt / tr
    winner = "Triton" if sp > 1 else "cuDNN"
    marker = " <-" if sp > 1 else ""
    print(f"  {label:<52} {pt:>7.3f}ms {tr:>7.3f}ms {sp:>6.2f}x  {winner}{marker}")
    return label, pt, tr


def section(title):
    print(f"\n  ── {title} ──")


def header():
    print(f"  {'Case':<52} {'cuDNN':>8} {'Triton':>8} {'Speed':>7}  {'Winner'}")
    print(f"  {'─'*52} {'─'*8} {'─'*8} {'─'*7}  {'─'*6}")


print("=" * 92)
print("  Ultimate Conv3D v2 — Comprehensive Benchmark")
print(f"  GPU: {torch.cuda.get_device_name()}")
print(f"  Kernel: flat K-loop + ~20 autotune configs + batch/group parallelism")
print("=" * 92)

results = []
header()

# ================================================================
# 1. Spatial resolution scaling (fixed: B=1, Ci=16, Co=32, 3x3x3)
# ================================================================
section("Spatial scaling (B=1, 16->32, 3x3x3, pad=1)")
for S in [8, 16, 24, 32, 48, 64]:
    results.append(run(f"spatial {S}^3", 1, 16, 32, S, S, S, 3, 3, 3, padding=1))

# ================================================================
# 2. Channel width scaling (fixed: B=1, 16^3, 3x3x3)
# ================================================================
section("Channel scaling (B=1, 16^3, 3x3x3, pad=1)")
results.append(run("ch 3->16", 1, 3, 16, 16, 16, 16, 3, 3, 3, padding=1))
results.append(run("ch 16->32", 1, 16, 32, 16, 16, 16, 3, 3, 3, padding=1))
results.append(run("ch 32->64", 1, 32, 64, 16, 16, 16, 3, 3, 3, padding=1))
results.append(run("ch 64->128", 1, 64, 128, 16, 16, 16, 3, 3, 3, padding=1))
results.append(run("ch 128->256", 1, 128, 256, 16, 16, 16, 3, 3, 3, padding=1))
results.append(run("ch 256->256", 1, 256, 256, 16, 16, 16, 3, 3, 3, padding=1))
results.append(run("ch 256->512", 1, 256, 512, 16, 16, 16, 3, 3, 3, padding=1))

# ================================================================
# 3. Batch scaling (fixed: 32->64, 16^3, 3x3x3)
# ================================================================
section("Batch scaling (32->64, 16^3, 3x3x3, pad=1)")
for B in [1, 2, 4, 8]:
    results.append(run(f"batch={B}, 32->64, 16^3", B, 32, 64, 16, 16, 16, 3, 3, 3, padding=1))

section("Batch scaling (16->32, 32^3, 3x3x3, pad=1)")
for B in [1, 2, 4, 8]:
    results.append(run(f"batch={B}, 16->32, 32^3", B, 16, 32, 32, 32, 32, 3, 3, 3, padding=1))

# ================================================================
# 4. Stride comparison
# ================================================================
section("Stride=1 vs stride=2 (B=1, 32->64, 3x3x3, pad=1)")
results.append(run("32->64, 16x32x32, s=1", 1, 32, 64, 16, 32, 32, 3, 3, 3, stride=1, padding=1))
results.append(run("32->64, 16x32x32, s=2", 1, 32, 64, 16, 32, 32, 3, 3, 3, stride=2, padding=1))
results.append(run("64->128, 32^3, s=1", 1, 64, 128, 32, 32, 32, 3, 3, 3, stride=1, padding=1))
results.append(run("64->128, 32^3, s=2", 1, 64, 128, 32, 32, 32, 3, 3, 3, stride=2, padding=1))

section("Stride=2 at various batch sizes (64->128, 32^3)")
for B in [1, 2, 4, 8]:
    results.append(run(f"64->128, 32^3, s=2, b={B}", B, 64, 128, 32, 32, 32, 3, 3, 3, stride=2, padding=1))

# ================================================================
# 5. Kernel size comparison (B=1, 32->64, 16^3)
# ================================================================
section("Kernel size (B=1, 32->64, 16^3)")
results.append(run("1x1x1 (pointwise)", 1, 32, 64, 16, 16, 16, 1, 1, 1))
results.append(run("3x3x3", 1, 32, 64, 16, 16, 16, 3, 3, 3, padding=1))
results.append(run("5x5x5", 1, 32, 64, 16, 16, 16, 5, 5, 5, padding=2))
results.append(run("1x3x3 (2D-style)", 1, 32, 64, 16, 16, 16, 1, 3, 3, padding=(0, 1, 1)))
results.append(run("3x1x1 (temporal-only)", 1, 32, 64, 16, 16, 16, 3, 1, 1, padding=(1, 0, 0)))

# ================================================================
# 6. Grouped convolution
# ================================================================
section("Grouped conv (B=4, 64->128, 16^3, 3x3x3, pad=1)")
for g in [1, 2, 4, 8, 16]:
    results.append(run(f"groups={g}", 4, 64, 128, 16, 16, 16, 3, 3, 3, padding=1, groups=g))

section("Depthwise (B=4, 3x3x3, pad=1)")
results.append(run("depthwise C=32, 16^3", 4, 32, 32, 16, 16, 16, 3, 3, 3, padding=1, groups=32))
results.append(run("depthwise C=64, 16^3", 4, 64, 64, 16, 16, 16, 3, 3, 3, padding=1, groups=64))
results.append(run("depthwise C=32, 32^3", 4, 32, 32, 32, 32, 32, 3, 3, 3, padding=1, groups=32))
results.append(run("depthwise C=64, 32^3", 4, 64, 64, 32, 32, 32, 3, 3, 3, padding=1, groups=64))

# ================================================================
# 7. Real-world model layer configs
# ================================================================
section("Real-world model layers")
# C3D-style (Tran et al., 2015): all 3x3x3, channels 64->128->256->512
results.append(run("C3D conv2: 64->128, 16x28x28 b=4", 4, 64, 128, 16, 28, 28, 3, 3, 3, padding=1))
results.append(run("C3D conv3: 128->256, 8x14x14 b=4", 4, 128, 256, 8, 14, 14, 3, 3, 3, padding=1))
results.append(run("C3D conv4: 256->512, 4x7x7 b=4", 4, 256, 512, 4, 7, 7, 3, 3, 3, padding=1))

# ResNet3D-style: 3x3x3 with stride=2 downsampling
results.append(run("R3D downsample: 64->128, 16^3 s=2 b=4", 4, 64, 128, 16, 16, 16, 3, 3, 3, stride=2, padding=1))
results.append(run("R3D downsample: 128->256, 8^3 s=2 b=4", 4, 128, 256, 8, 8, 8, 3, 3, 3, stride=2, padding=1))

# SlowFast-style: large spatial, small channels (Slow pathway)
results.append(run("SlowFast slow: 8->64, 8x56x56 b=4", 4, 8, 64, 8, 56, 56, 1, 7, 7, stride=(1, 2, 2), padding=(0, 3, 3)))

# Medical imaging style: large 3D volume
results.append(run("MedImg: 1->32, 64^3 b=1", 1, 1, 32, 64, 64, 64, 3, 3, 3, padding=1))
results.append(run("MedImg: 32->64, 32^3 s=2 b=1", 1, 32, 64, 32, 32, 32, 3, 3, 3, stride=2, padding=1))
results.append(run("MedImg: 64->128, 16^3 b=1", 1, 64, 128, 16, 16, 16, 3, 3, 3, padding=1))

# ================================================================
# Summary
# ================================================================
print(f"\n{'='*92}")
print("  SUMMARY")
print(f"{'='*92}")
print(f"  {'Case':<52} {'cuDNN':>8} {'Triton':>8} {'Speed':>7}")
print(f"  {'─'*52} {'─'*8} {'─'*8} {'─'*7}")

triton_wins = 0
for label, pt, tr in results:
    sp = pt / tr
    marker = " <-" if sp > 1 else ""
    print(f"  {label:<52} {pt:>7.3f}ms {tr:>7.3f}ms {sp:>6.2f}x{marker}")
    if sp > 1:
        triton_wins += 1

print(f"\n  Triton faster in {triton_wins}/{len(results)} cases")

# Per-category stats
categories = {
    "Spatial scaling": results[0:6],
    "Channel scaling": results[6:13],
    "Batch scaling (small spatial)": results[13:17],
    "Batch scaling (large spatial)": results[17:21],
    "Stride comparison": results[21:25],
    "Stride=2 batch scaling": results[25:29],
    "Kernel size": results[29:34],
    "Grouped conv": results[34:39],
    "Depthwise": results[39:43],
    "Real-world models": results[43:],
}

print(f"\n  Per-category win rate:")
for cat, items in categories.items():
    wins = sum(1 for _, pt, tr in items if pt / tr > 1)
    avg_sp = sum(pt / tr for _, pt, tr in items) / len(items)
    print(f"    {cat:<35} {wins}/{len(items)} wins  avg speedup {avg_sp:.2f}x")

print(f"{'='*92}")
