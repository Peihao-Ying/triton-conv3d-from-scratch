# Benchmark Results

GPU: NVIDIA GeForce RTX 3080 (10GB)
Kernel: Phase 4 implicit im2col (`04_conv3d_implicit/conv3d_implicit.py`)
Constraints: batch_size=1, groups=1, dtype=float32 (fp16 dot)
Timing: CUDA events, 100 iterations, median

## Summary

| Case | PyTorch | Triton | Speedup |
|---|---|---|---|
| S1: tiny first layer | 0.017ms | 0.040ms | 0.43x |
| S2: small first layer | 0.022ms | 0.035ms | 0.62x |
| M1: mid-net 32→64 | 0.051ms | 0.055ms | 0.92x |
| M2: mid-net 64→128 | 0.058ms | 0.080ms | 0.73x |
| **M3: mid-net stride=2 downsample** | 0.084ms | 0.055ms | **1.52x** |
| **L1: high-res 32³** | 0.191ms | 0.172ms | **1.11x** |
| **L2: high-res 48³** | 0.530ms | 0.396ms | **1.34x** |
| **L3: high-res 64³** | 1.200ms | 0.900ms | **1.33x** |
| **L4: high-res 64³ stride=2** | 0.240ms | 0.174ms | **1.38x** |
| **C1: wide 128→128, 16³** | 0.205ms | 0.191ms | **1.07x** |
| **C2: wide 128→256, 16³** | 0.345ms | 0.315ms | **1.09x** |
| C3: wide 256→256, 8³ | 0.173ms | 0.240ms | 0.72x |
| C4: wide 256→512, 8³ | 0.222ms | 0.247ms | 0.90x |
| X1: large 64→128, 32³ | 0.596ms | 0.634ms | 0.94x |
| X2: large 128→128, 32³ | 1.098ms | 1.135ms | 0.97x |
| **X3: large 64→128, 32³ stride=2** | 0.199ms | 0.097ms | **2.05x** |
| **K1: 5×5×5 kernel, 32→64** | 0.197ms | 0.175ms | **1.12x** |
| K2: 1×1×1 pointwise, 128→256 | 0.029ms | 0.061ms | 0.48x |
| K3: 1×3×3 2D-style, 64→64 | 0.103ms | 0.170ms | 0.61x |

Triton faster in **9/19** cases.

## Case Details

### Small (kernel launch overhead dominates)

| Case | Input | Kernel | GEMM shape |
|---|---|---|---|
| S1 | (1,1,16,16,16) | (16,1,3,3,3) | (16,27)×(27,4096) |
| S2 | (1,3,16,16,16) | (32,3,3,3,3) | (32,81)×(81,4096) |

### Medium (typical mid-network)

| Case | Input | Kernel | GEMM shape |
|---|---|---|---|
| M1 | (1,32,16,16,16) | (64,32,3,3,3) | (64,864)×(864,4096) |
| M2 | (1,64,8,16,16) | (128,64,3,3,3) | (128,1728)×(1728,2048) |
| M3 | (1,32,16,32,32) | (64,32,3,3,3) stride=2 | (64,864)×(864,2048) |

### Large spatial (high-resolution input)

| Case | Input | Kernel | GEMM shape |
|---|---|---|---|
| L1 | (1,16,32,32,32) | (32,16,3,3,3) | (32,432)×(432,32768) |
| L2 | (1,16,48,48,48) | (32,16,3,3,3) | (32,432)×(432,110592) |
| L3 | (1,16,64,64,64) | (32,16,3,3,3) | (32,432)×(432,262144) |
| L4 | (1,16,64,64,64) | (32,16,3,3,3) stride=2 | (32,432)×(432,32768) |

### Large channels (compute-heavy GEMM)

| Case | Input | Kernel | GEMM shape |
|---|---|---|---|
| C1 | (1,128,16,16,16) | (128,128,3,3,3) | (128,3456)×(3456,4096) |
| C2 | (1,128,16,16,16) | (256,128,3,3,3) | (256,3456)×(3456,4096) |
| C3 | (1,256,8,8,8) | (256,256,3,3,3) | (256,6912)×(6912,512) |
| C4 | (1,256,8,8,8) | (512,256,3,3,3) | (512,6912)×(6912,512) |

### Stress test (large spatial + large channels)

| Case | Input | Kernel | GEMM shape |
|---|---|---|---|
| X1 | (1,64,32,32,32) | (128,64,3,3,3) | (128,1728)×(1728,32768) |
| X2 | (1,128,32,32,32) | (128,128,3,3,3) | (128,3456)×(3456,32768) |
| X3 | (1,64,32,32,32) | (128,64,3,3,3) stride=2 | (128,1728)×(1728,4096) |

### Non-standard kernels

| Case | Input | Kernel | GEMM shape |
|---|---|---|---|
| K1 | (1,32,16,16,16) | (64,32,5,5,5) | (64,4000)×(4000,4096) |
| K2 | (1,128,16,16,16) | (256,128,1,1,1) | (256,128)×(128,4096) |
| K3 | (1,64,16,32,32) | (64,64,1,3,3) | (64,576)×(576,16384) |

## Analysis

### Where Triton wins

- **Large spatial dimensions** (32³+): N_pos is large, Triton tiles are fully utilized. Peak advantage at 48³/64³ (~1.3x).
- **stride=2**: All stride=2 cases favor Triton, up to **2.05x**. cuDNN's stride>1 path is less optimized.
- **5×5×5 kernel**: Larger K dimension means implicit im2col saves more memory bandwidth by not materializing the intermediate matrix.

### Where cuDNN wins

- **Small problems** (S1/S2): Kernel launch overhead dominates when total compute is <0.05ms.
- **Large channels + small spatial** (C3/C4): N_pos=512 is too small — GEMM is too skinny for Triton tiles to be efficient.
- **1×1×1 pointwise**: Pure GEMM with no im2col needed. cuDNN dispatches directly to cuBLAS.
- **1×3×3 2D-style**: cuDNN has specialized 2D convolution paths.

### Key takeaway

The Phase 4 implicit im2col kernel (no batch parallelism, no autotuning expansion, no loop restructuring) already beats cuDNN on workloads with **high spatial resolution** and **stride > 1** — exactly the profile of medical imaging (CT/MRI volumes) and video processing.
