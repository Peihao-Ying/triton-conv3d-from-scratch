"""
Benchmark: batch-parallel Conv3d (Triton) vs torch.nn.Conv3d.

Same 5 problem sizes from Phase 4, each tested at batch_size in {1, 4, 8}.
"""

import torch
import torch.nn as nn

from conv3d_batch import conv3d_batch, DEVICE


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


def run_benchmark(label, batch_size, C_in, C_out, D, H, W, kD, kH, kW,
                  stride=1, padding=0, dilation=1):
    if isinstance(stride, int):
        stride = (stride, stride, stride)
    if isinstance(padding, int):
        padding = (padding, padding, padding)
    if isinstance(dilation, int):
        dilation = (dilation, dilation, dilation)

    print(f"\n{'='*60}")
    print(f"{label}")
    print(f"Input: ({batch_size},{C_in},{D},{H},{W})  Kernel: ({C_out},{C_in},{kD},{kH},{kW})")
    print(f"stride={stride}  padding={padding}  dilation={dilation}")

    torch.manual_seed(42)
    x = torch.randn(batch_size, C_in, D, H, W, device=DEVICE)
    weight = torch.randn(C_out, C_in, kD, kH, kW, device=DEVICE)
    bias = torch.randn(C_out, device=DEVICE)

    # PyTorch Conv3d
    conv = nn.Conv3d(C_in, C_out, (kD, kH, kW),
                     stride=stride, padding=padding, dilation=dilation, bias=True).to(DEVICE)
    with torch.no_grad():
        conv.weight.copy_(weight)
        conv.bias.copy_(bias)

    pytorch_ms = benchmark_fn(lambda: conv(x))

    # Our batch-parallel kernel
    triton_ms = benchmark_fn(
        lambda: conv3d_batch(x, weight, bias,
                             stride=stride, padding=padding, dilation=dilation))

    speedup = pytorch_ms / triton_ms
    print(f"PyTorch:  {pytorch_ms:8.3f} ms")
    print(f"Triton:   {triton_ms:8.3f} ms")
    print(f"Speedup:  {speedup:.2f}x {'(Triton faster)' if speedup > 1 else '(PyTorch faster)'}")

    return label, batch_size, pytorch_ms, triton_ms


# ============================================================
# Benchmark suite — 5 cases x 3 batch sizes
# ============================================================

print("Batch-parallel Conv3d Benchmark")
print(f"Device: {DEVICE}")
print(f"GPU: {torch.cuda.get_device_name()}")

CASES = [
    ("Small first layer",    1, 16,  16, 16, 16, 3, 3, 3, 1, 1, 1),
    ("Medium mid-net",      32, 64,  16, 16, 16, 3, 3, 3, 1, 1, 1),
    ("Large deeper",        64, 128,  8, 16, 16, 3, 3, 3, 1, 1, 1),
    ("Stride=2 downsample", 32, 64,  16, 32, 32, 3, 3, 3, 2, 1, 1),
    ("Large high-res",      16, 32,  32, 32, 32, 3, 3, 3, 1, 1, 1),
]

BATCH_SIZES = [1, 4, 8]

results = []

for batch_size in BATCH_SIZES:
    for case_label, C_in, C_out, D, H, W, kD, kH, kW, s, p, d in CASES:
        label = f"B={batch_size}: {case_label}"
        results.append(run_benchmark(
            label, batch_size, C_in, C_out, D, H, W, kD, kH, kW,
            stride=s, padding=p, dilation=d))


# ============================================================
# Summary table
# ============================================================

print(f"\n{'='*75}")
print("Summary")
print(f"{'Case':<35} {'Batch':>5} {'PyTorch':>10} {'Triton':>10} {'Speedup':>10}")
print("-" * 75)
for label, bs, pt, tr in results:
    speedup = pt / tr
    print(f"{label:<35} {bs:>5} {pt:>8.3f}ms {tr:>8.3f}ms {speedup:>9.2f}x")
