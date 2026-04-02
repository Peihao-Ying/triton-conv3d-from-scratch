# Benchmark Results

GPU: NVIDIA GeForce RTX 3080 (10GB)
Timing: CUDA events, 100 iterations, median

## Phase 4 Baseline (implicit im2col, batch=1, groups=1)

Kernel: `04_conv3d_implicit/conv3d_implicit.py` — flat K-loop, 5 autotune configs.

| Case | Group | Input | Kernel | Stride | Padding | GEMM (M,K)x(K,N) | cuDNN | Triton | Speedup |
|---|---|---|---|---|---|---|---|---|---|
| S1: tiny first layer | Small | (1,1,16,16,16) | (16,1,3,3,3) | 1 | 1 | (16,27)x(27,4096) | 0.017ms | 0.040ms | 0.43x |
| S2: small first layer | Small | (1,3,16,16,16) | (32,3,3,3,3) | 1 | 1 | (32,81)x(81,4096) | 0.022ms | 0.035ms | 0.62x |
| M1: mid-net 32->64 | Medium | (1,32,16,16,16) | (64,32,3,3,3) | 1 | 1 | (64,864)x(864,4096) | 0.051ms | 0.055ms | 0.92x |
| M2: mid-net 64->128 | Medium | (1,64,8,16,16) | (128,64,3,3,3) | 1 | 1 | (128,1728)x(1728,2048) | 0.058ms | 0.080ms | 0.73x |
| **M3: mid-net stride=2** | **Medium** | **(1,32,16,32,32)** | **(64,32,3,3,3)** | **2** | **1** | **(64,864)x(864,2048)** | **0.084ms** | **0.055ms** | **1.52x** |
| **L1: high-res 32^3** | **Large spatial** | **(1,16,32,32,32)** | **(32,16,3,3,3)** | **1** | **1** | **(32,432)x(432,32768)** | **0.191ms** | **0.172ms** | **1.11x** |
| **L2: high-res 48^3** | **Large spatial** | **(1,16,48,48,48)** | **(32,16,3,3,3)** | **1** | **1** | **(32,432)x(432,110592)** | **0.530ms** | **0.396ms** | **1.34x** |
| **L3: high-res 64^3** | **Large spatial** | **(1,16,64,64,64)** | **(32,16,3,3,3)** | **1** | **1** | **(32,432)x(432,262144)** | **1.200ms** | **0.900ms** | **1.33x** |
| **L4: high-res 64^3 s2** | **Large spatial** | **(1,16,64,64,64)** | **(32,16,3,3,3)** | **2** | **1** | **(32,432)x(432,32768)** | **0.240ms** | **0.174ms** | **1.38x** |
| **C1: wide 128->128** | **Large channel** | **(1,128,16,16,16)** | **(128,128,3,3,3)** | **1** | **1** | **(128,3456)x(3456,4096)** | **0.205ms** | **0.191ms** | **1.07x** |
| **C2: wide 128->256** | **Large channel** | **(1,128,16,16,16)** | **(256,128,3,3,3)** | **1** | **1** | **(256,3456)x(3456,4096)** | **0.345ms** | **0.315ms** | **1.09x** |
| C3: wide 256->256 | Large channel | (1,256,8,8,8) | (256,256,3,3,3) | 1 | 1 | (256,6912)x(6912,512) | 0.173ms | 0.240ms | 0.72x |
| C4: wide 256->512 | Large channel | (1,256,8,8,8) | (512,256,3,3,3) | 1 | 1 | (512,6912)x(6912,512) | 0.222ms | 0.247ms | 0.90x |
| X1: 64->128, 32^3 | Stress | (1,64,32,32,32) | (128,64,3,3,3) | 1 | 1 | (128,1728)x(1728,32768) | 0.596ms | 0.634ms | 0.94x |
| X2: 128->128, 32^3 | Stress | (1,128,32,32,32) | (128,128,3,3,3) | 1 | 1 | (128,3456)x(3456,32768) | 1.098ms | 1.135ms | 0.97x |
| **X3: 64->128, 32^3 s2** | **Stress** | **(1,64,32,32,32)** | **(128,64,3,3,3)** | **2** | **1** | **(128,1728)x(1728,4096)** | **0.199ms** | **0.097ms** | **2.05x** |
| **K1: 5x5x5, 32->64** | **Non-standard** | **(1,32,16,16,16)** | **(64,32,5,5,5)** | **1** | **2** | **(64,4000)x(4000,4096)** | **0.197ms** | **0.175ms** | **1.12x** |
| K2: 1x1x1 pointwise | Non-standard | (1,128,16,16,16) | (256,128,1,1,1) | 1 | 0 | (256,128)x(128,4096) | 0.029ms | 0.061ms | 0.48x |
| K3: 1x3x3 2D-style | Non-standard | (1,64,16,32,32) | (64,64,1,3,3) | 1 | (0,1,1) | (64,576)x(576,16384) | 0.103ms | 0.170ms | 0.61x |

Triton faster in **9/19** cases.

## Phase 5 Individual Optimizations

Each optimization benchmarked in isolation against cuDNN. Where available, also compared against Phase 4 baseline.

### Opt 1: Batch Parallelism

Functional extension (Phase 4 only supports batch=1). Performance at batch=1 matches Phase 4.

| Case | Batch | cuDNN | Triton | Speedup |
|---|---|---|---|---|
| Small first layer | 1 | 0.013ms | 0.039ms | 0.34x |
| Medium mid-net | 1 | 0.055ms | 0.056ms | 0.98x |
| Stride=2 downsample | 1 | 0.077ms | 0.057ms | 1.34x |
| Large high-res | 1 | 0.188ms | 0.158ms | 1.20x |
| Medium mid-net | 4 | 0.124ms | 0.116ms | 1.06x |
| Large high-res | 4 | 0.628ms | 0.483ms | 1.30x |
| Large high-res | 8 | 1.180ms | 0.892ms | 1.32x |

### Opt 2: Expanded Autotuning (~20 configs)

Same kernel body as Phase 4, only config list expanded. **Only proven pure performance win.**

| Case | cuDNN | Triton | Speedup | vs Phase 4 |
|---|---|---|---|---|
| Small first layer | 0.012ms | 0.033ms | 0.38x | ~same |
| Medium mid-net | 0.060ms | 0.056ms | 1.08x | ~same |
| Large deeper | 0.057ms | 0.079ms | 0.73x | ~same |
| Stride=2 downsample | 0.081ms | 0.052ms | 1.55x | ~same |
| **Large high-res** | **0.189ms** | **0.109ms** | **1.73x** | **1.58x vs Phase 4** |

### Opt 3: constexpr + LUT — NEGATIVE

Both constexpr-only and constexpr+LUT regressed performance vs Phase 4.

| Case | Phase 4 | Constexpr | vs Phase 4 | Constexpr+LUT | vs Phase 4 |
|---|---|---|---|---|---|
| Small first layer | 0.034ms | 0.041ms | 0.83x | 0.081ms | 0.42x |
| Medium mid-net | 0.056ms | 0.065ms | 0.87x | 0.100ms | 0.56x |
| Large deeper | 0.078ms | 0.087ms | 0.89x | 0.125ms | 0.62x |
| Stride=2 downsample | 0.054ms | 0.059ms | 0.91x | 0.098ms | 0.55x |
| Large high-res | 0.166ms | 0.173ms | 0.96x | 0.222ms | 0.75x |

### Opt 4: Split K-loop — NEGATIVE

Restructuring K-loop into nested spatial/channel loops caused 5-6x slowdown.

| Case | cuDNN | Triton | Speedup |
|---|---|---|---|
| Small first layer | 0.011ms | 0.067ms | 0.17x |
| Medium mid-net | 0.050ms | 0.341ms | 0.15x |
| Large deeper | 0.057ms | 0.427ms | 0.13x |
| Stride=2 downsample | 0.077ms | 0.331ms | 0.23x |
| Large high-res | 0.188ms | 1.005ms | 0.19x |

### Opt 5: Winograd F(2x2x2, 3x3x3) — NEGATIVE

10-30x slower than both cuDNN and Phase 4. Transform overhead dominates.

| Case | cuDNN | Phase 4 | Winograd | vs cuDNN | vs Phase 4 |
|---|---|---|---|---|---|
| Small first layer | 0.017ms | 0.033ms | 0.355ms | 0.05x | 0.09x |
| Medium mid-net | 0.051ms | 0.055ms | 0.679ms | 0.08x | 0.08x |
| Large deeper | 0.057ms | 0.078ms | 0.701ms | 0.08x | 0.11x |
| Large high-res | 0.190ms | 0.172ms | 2.626ms | 0.07x | 0.07x |
| Wide channels 128->128 | 0.037ms | 0.125ms | 0.421ms | 0.09x | 0.30x |

### Opt 6: Groups / Depthwise

Functional extension. groups=2 won (1.54x), depthwise lost (cuDNN highly optimized for depthwise).

| Case | cuDNN | Triton | Speedup |
|---|---|---|---|
| groups=1 (baseline) | 0.050ms | 0.055ms | 0.91x |
| groups=2 | 0.079ms | 0.051ms | 1.54x |
| groups=4 | 0.062ms | 0.062ms | 0.99x |
| groups=32 | 0.015ms | 0.158ms | 0.10x |
| depthwise C=32 | 0.009ms | 0.028ms | 0.33x |
| depthwise C=64 | 0.015ms | 0.033ms | 0.46x |
| depthwise large spatial | 0.046ms | 0.058ms | 0.79x |

### Optimization Summary

| Optimization | Performance Impact | Kept in Ultimate? |
|---|---|---|
| Batch parallelism | Functional (enables batch>1) | Yes |
| Expanded autotuning | **Positive** (up to 1.58x on large high-res) | Yes |
| constexpr + LUT | **Negative** (4-58% slower) | No |
| Split K-loop | **Negative** (5-6x slower) | No |
| Winograd | **Negative** (10-30x slower) | No |
| Groups / depthwise | Functional (enables groups>1) | Yes |

## Ultimate Kernel v2

Kernel: `05_optimization/conv3d_ultimate.py` — flat K-loop (Phase 4 style) + expanded autotuning + batch/group parallelism. Winograd, split K-loop, and constexpr removed based on benchmark data above.

25/25 correctness tests pass. **Triton faster in 17/29 cases.**

| Case | Batch | cuDNN | Triton | Speedup |
|---|---|---|---|---|
| S1: tiny 1->16, 16^3 | 1 | 0.018ms | 0.035ms | 0.53x |
| S2: small 3->32, 16^3 | 1 | 0.017ms | 0.035ms | 0.50x |
| M1: mid-net 32->64, 16^3 | 1 | 0.050ms | 0.060ms | 0.84x |
| M2: mid-net 64->128, 8x16x16 | 1 | 0.057ms | 0.083ms | 0.69x |
| **M3: stride=2, 32->64** | **1** | **0.078ms** | **0.054ms** | **1.43x** |
| **L1: high-res 16->32, 32^3** | **1** | **0.190ms** | **0.110ms** | **1.74x** |
| **L2: high-res 16->32, 48^3** | **1** | **0.524ms** | **0.317ms** | **1.65x** |
| **L3: high-res 16->32, 64^3** | **1** | **1.186ms** | **0.708ms** | **1.68x** |
| **L4: high-res 64^3 stride=2** | **1** | **0.234ms** | **0.109ms** | **2.15x** |
| **C1: wide 128->128, 16^3** | **1** | **0.203ms** | **0.194ms** | **1.05x** |
| **C2: wide 128->256, 16^3** | **1** | **0.345ms** | **0.271ms** | **1.27x** |
| C3: wide 256->256, 8^3 | 1 | 0.175ms | 0.225ms | 0.78x |
| **X1: stress 64->128, 32^3** | **1** | **0.587ms** | **0.532ms** | **1.10x** |
| **X2: stress 64->128, 32^3 s2** | **1** | **0.197ms** | **0.097ms** | **2.02x** |
| M1: mid-net 32->64, 16^3 | 4 | 0.123ms | 0.127ms | 0.97x |
| M3: stride=2, 32->64 | 4 | 0.074ms | 0.103ms | 0.71x |
| **L1: high-res 16->32, 32^3** | **4** | **0.635ms** | **0.394ms** | **1.61x** |
| L3: high-res 16->32, 64^3 | 4 | 2.023ms | 2.566ms | 0.79x |
| **C1: wide 128->128, 16^3** | **4** | **0.625ms** | **0.564ms** | **1.11x** |
| **X2: stress 64->128, 32^3 s2** | **4** | **0.419ms** | **0.337ms** | **1.24x** |
| M1: mid-net 32->64, 16^3 | 8 | 0.208ms | 0.257ms | 0.81x |
| **L1: high-res 16->32, 32^3** | **8** | **1.187ms** | **0.694ms** | **1.71x** |
| **X2: stress 64->128, 32^3 s2** | **8** | **0.784ms** | **0.599ms** | **1.31x** |
| **groups=2, 32->64, 16^3** | **4** | **0.230ms** | **0.106ms** | **2.16x** |
| **groups=4, 64->128, 16^3** | **4** | **0.402ms** | **0.225ms** | **1.79x** |
| depthwise C=32, 16^3 | 4 | 0.026ms | 0.045ms | 0.57x |
| depthwise C=64, 16x32x32 | 4 | 0.197ms | 0.231ms | 0.85x |
| **5x5x5, 32->64, 16^3** | **1** | **0.196ms** | **0.172ms** | **1.14x** |
| 1x1x1 pointwise, 128->256 | 1 | 0.030ms | 0.052ms | 0.57x |

## Analysis

### Where Triton wins

- **Large spatial** (32^3+): consistent 1.6-1.7x advantage across batch sizes
- **stride=2**: up to **2.15x** — cuDNN's stride>1 path is weaker
- **Grouped convolution** (groups=2,4): **2.16x** — cuDNN grouped dispatch has overhead
- **Wide channels** (128+): 1.05-1.27x when N_pos is large enough
- **5x5x5 kernel**: larger K means more im2col savings

### Where cuDNN wins

- **Small problems** (<0.05ms total): kernel launch overhead dominates
- **Large channel + small spatial** (256ch on 8^3): N_pos=512, GEMM too skinny
- **Depthwise**: cuDNN has heavily optimized depthwise paths
- **1x1x1 pointwise**: pure GEMM, cuDNN calls cuBLAS directly
- **batch=4 + 64^3**: memory pressure — 4 * 16 * 64^3 * 4B = 67MB input, likely L2 thrashing

### Phase 4 -> Ultimate v2 improvement

Comparing same cases at batch=1:

| Case | Phase 4 | Ultimate v2 | Improvement |
|---|---|---|---|
| L1: high-res 32^3 | 0.172ms | 0.110ms | **1.56x** |
| L2: high-res 48^3 | 0.396ms | 0.317ms | **1.25x** |
| L3: high-res 64^3 | 0.900ms | 0.708ms | **1.27x** |
| X1: stress 64->128, 32^3 | 0.634ms | 0.532ms | **1.19x** |

The improvement comes entirely from expanded autotuning — more config choices allow Triton to find better tile sizes for each workload.
