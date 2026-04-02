"""
Extended benchmark: implicit im2col Conv3d (Triton) vs torch.nn.Conv3d (cuDNN).

Covers small → large problem sizes to find where Triton becomes competitive.
Phase 4 kernel: batch_size=1, groups=1.
"""

import torch
import torch.nn as nn

from conv3d_implicit import conv3d_implicit, DEVICE


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


def run_benchmark(label, C_in, C_out, D, H, W, kD, kH, kW,
                  stride=1, padding=0, dilation=1):
    if isinstance(stride, int):
        stride = (stride, stride, stride)
    if isinstance(padding, int):
        padding = (padding, padding, padding)
    if isinstance(dilation, int):
        dilation = (dilation, dilation, dilation)

    sD, sH, sW = stride
    pD, pH, pW = padding
    dD, dH, dW = dilation
    D_out = (D + 2*pD - dD*(kD-1) - 1) // sD + 1
    H_out = (H + 2*pH - dH*(kH-1) - 1) // sH + 1
    W_out = (W + 2*pW - dW*(kW-1) - 1) // sW + 1
    N_pos = D_out * H_out * W_out
    K = C_in * kD * kH * kW

    # GEMM shape for reference
    M = C_out

    print(f"\n{'─'*70}")
    print(f"  {label}")
    print(f"  Input: (1,{C_in},{D},{H},{W})  Kernel: ({C_out},{C_in},{kD},{kH},{kW})")
    print(f"  stride={stride}  padding={padding}  dilation={dilation}")
    print(f"  Output: (1,{C_out},{D_out},{H_out},{W_out})  GEMM: ({M},{K})×({K},{N_pos})")

    torch.manual_seed(42)
    x = torch.randn(1, C_in, D, H, W, device=DEVICE)
    weight = torch.randn(C_out, C_in, kD, kH, kW, device=DEVICE)
    bias = torch.randn(C_out, device=DEVICE)

    conv = nn.Conv3d(C_in, C_out, (kD, kH, kW),
                     stride=stride, padding=padding, dilation=dilation, bias=True).to(DEVICE)
    with torch.no_grad():
        conv.weight.copy_(weight)
        conv.bias.copy_(bias)

    pytorch_ms = benchmark_fn(lambda: conv(x))
    triton_ms = benchmark_fn(
        lambda: conv3d_implicit(x, weight, bias,
                                stride=stride, padding=padding, dilation=dilation))

    speedup = pytorch_ms / triton_ms
    tag = "Triton ✓" if speedup > 1 else "cuDNN ✓"
    print(f"  PyTorch: {pytorch_ms:8.3f} ms  |  Triton: {triton_ms:8.3f} ms  |  {speedup:.2f}x ({tag})")

    return label, pytorch_ms, triton_ms


# ============================================================
# Benchmark suite
# ============================================================

print("=" * 70)
print("  Implicit im2col Conv3d — Extended Benchmark")
print(f"  Device: {torch.cuda.get_device_name()}")
print(f"  batch_size=1, groups=1, dtype=float32 (fp16 dot)")
print("=" * 70)

results = []

# --- Group 1: Small (warm-up territory, cuDNN likely wins) ---

results.append(run_benchmark(
    "S1: tiny first layer",
    C_in=1, C_out=16, D=16, H=16, W=16, kD=3, kH=3, kW=3, padding=1))

results.append(run_benchmark(
    "S2: small first layer",
    C_in=3, C_out=32, D=16, H=16, W=16, kD=3, kH=3, kW=3, padding=1))

# --- Group 2: Medium (typical mid-network) ---

results.append(run_benchmark(
    "M1: mid-net 32→64",
    C_in=32, C_out=64, D=16, H=16, W=16, kD=3, kH=3, kW=3, padding=1))

results.append(run_benchmark(
    "M2: mid-net 64→128",
    C_in=64, C_out=128, D=8, H=16, W=16, kD=3, kH=3, kW=3, padding=1))

results.append(run_benchmark(
    "M3: mid-net stride=2 downsample",
    C_in=32, C_out=64, D=16, H=32, W=32, kD=3, kH=3, kW=3, stride=2, padding=1))

# --- Group 3: Large spatial (high-res input) ---

results.append(run_benchmark(
    "L1: high-res 32³",
    C_in=16, C_out=32, D=32, H=32, W=32, kD=3, kH=3, kW=3, padding=1))

results.append(run_benchmark(
    "L2: high-res 48³",
    C_in=16, C_out=32, D=48, H=48, W=48, kD=3, kH=3, kW=3, padding=1))

results.append(run_benchmark(
    "L3: high-res 64³",
    C_in=16, C_out=32, D=64, H=64, W=64, kD=3, kH=3, kW=3, padding=1))

results.append(run_benchmark(
    "L4: high-res 64³ stride=2",
    C_in=16, C_out=32, D=64, H=64, W=64, kD=3, kH=3, kW=3, stride=2, padding=1))

# --- Group 4: Large channels (compute-heavy GEMM) ---

results.append(run_benchmark(
    "C1: wide 128→128, 16³",
    C_in=128, C_out=128, D=16, H=16, W=16, kD=3, kH=3, kW=3, padding=1))

results.append(run_benchmark(
    "C2: wide 128→256, 16³",
    C_in=128, C_out=256, D=16, H=16, W=16, kD=3, kH=3, kW=3, padding=1))

results.append(run_benchmark(
    "C3: wide 256→256, 8³",
    C_in=256, C_out=256, D=8, H=8, W=8, kD=3, kH=3, kW=3, padding=1))

results.append(run_benchmark(
    "C4: wide 256→512, 8³",
    C_in=256, C_out=512, D=8, H=8, W=8, kD=3, kH=3, kW=3, padding=1))

# --- Group 5: Large both (stress test) ---

results.append(run_benchmark(
    "X1: large 64→128, 32³",
    C_in=64, C_out=128, D=32, H=32, W=32, kD=3, kH=3, kW=3, padding=1))

results.append(run_benchmark(
    "X2: large 128→128, 32³",
    C_in=128, C_out=128, D=32, H=32, W=32, kD=3, kH=3, kW=3, padding=1))

results.append(run_benchmark(
    "X3: large 64→128, 32³ stride=2",
    C_in=64, C_out=128, D=32, H=32, W=32, kD=3, kH=3, kW=3, stride=2, padding=1))

# --- Group 6: Non-3×3×3 kernels ---

results.append(run_benchmark(
    "K1: 5×5×5 kernel, 32→64",
    C_in=32, C_out=64, D=16, H=16, W=16, kD=5, kH=5, kW=5, padding=2))

results.append(run_benchmark(
    "K2: 1×1×1 kernel (pointwise), 128→256",
    C_in=128, C_out=256, D=16, H=16, W=16, kD=1, kH=1, kW=1))

results.append(run_benchmark(
    "K3: 1×3×3 kernel (2D-style), 64→64",
    C_in=64, C_out=64, D=16, H=32, W=32, kD=1, kH=3, kW=3, padding=(0,1,1)))

# ============================================================
# Summary table
# ============================================================

print(f"\n{'='*70}")
print("  Summary")
print(f"{'='*70}")
print(f"  {'Case':<38} {'PyTorch':>9} {'Triton':>9} {'Speedup':>9}")
print(f"  {'─'*38} {'─'*9} {'─'*9} {'─'*9}")

triton_wins = 0
for label, pt, tr in results:
    sp = pt / tr
    marker = " ←" if sp > 1 else ""
    print(f"  {label:<38} {pt:>7.3f}ms {tr:>7.3f}ms {sp:>8.2f}x{marker}")
    if sp > 1:
        triton_wins += 1

print(f"\n  Triton faster in {triton_wins}/{len(results)} cases")
print(f"{'='*70}")
