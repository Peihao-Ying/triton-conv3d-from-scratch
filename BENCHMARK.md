# Benchmark Results

GPU: NVIDIA GeForce RTX 3080 (10GB)
Kernel: Phase 4 implicit im2col (`04_conv3d_implicit/conv3d_implicit.py`)
Constraints: batch_size=1, groups=1, dtype=float32 (fp16 dot)
Timing: CUDA events, 100 iterations, median

## Results

| Case | Group | Input | Kernel | Stride | Padding | GEMM (M,K)x(K,N) | PyTorch | Triton | Speedup |
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

## Analysis

### Where Triton wins

- **Large spatial dimensions** (32^3+): N_pos is large, Triton tiles are fully utilized. Peak advantage at 48^3/64^3 (~1.3x).
- **stride=2**: All stride=2 cases favor Triton, up to **2.05x**. cuDNN's stride>1 path is less optimized.
- **5x5x5 kernel**: Larger K dimension means implicit im2col saves more memory bandwidth by not materializing the intermediate matrix.

### Where cuDNN wins

- **Small problems** (S1/S2): Kernel launch overhead dominates when total compute is <0.05ms.
- **Large channels + small spatial** (C3/C4): N_pos=512 is too small — GEMM is too skinny for Triton tiles to be efficient.
- **1x1x1 pointwise**: Pure GEMM with no im2col needed. cuDNN dispatches directly to cuBLAS.
- **1x3x3 2D-style**: cuDNN has specialized 2D convolution paths.

### Key takeaway

The Phase 4 implicit im2col kernel (no batch parallelism, no autotuning expansion, no loop restructuring) already beats cuDNN on workloads with **high spatial resolution** and **stride > 1** — exactly the profile of medical imaging (CT/MRI volumes) and video processing.
