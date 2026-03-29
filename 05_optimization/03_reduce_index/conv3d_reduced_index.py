"""
Conv3d with reduced index computation — two optimizations over Phase 4 baseline:

1. **constexpr kernel dims**: kD, kH, kW are tl.constexpr parameters, so the
   compiler can constant-fold kHW = kH * kW and kDHW = kD * kHW.  The 4
   division/modulo operations in the K-loop become shifts or multiplies by
   compile-time constants (much cheaper than generic integer division).

2. **LUT-based index decomposition** (optional, USE_LUT=True): Pre-compute
   ci, kd, kh, kw for every k in [0, K) on the host and pass them as 4 small
   GPU tensors.  Inside the K-loop the 4 div/mod are replaced by 4 tl.load
   calls.  Because the LUT is tiny and accessed sequentially it lives in L1
   cache — each load is ~5 cycles vs ~30+ cycles for an integer division.

The kernel has both paths; a USE_LUT constexpr flag selects at compile time.

HW_out = H_out * W_out is pre-computed on the host and passed as a regular
parameter to save one multiply per program.

Everything else (tiling, weight load, input address, output store) is
identical to Phase 4.

Supports: arbitrary stride, padding, dilation.  groups=1, batch_size=1.
"""

import torch
import triton
import triton.language as tl

DEVICE = triton.runtime.driver.active.get_active_torch_device()


# ============================================================
# Autotune configs
# ============================================================

def get_autotune_config():
    return [
        triton.Config(
            {"BLOCK_SIZE_M": 64, "BLOCK_SIZE_N": 64, "BLOCK_SIZE_K": 32, "GROUP_SIZE_M": 8},
            num_stages=4, num_warps=4,
        ),
        triton.Config(
            {"BLOCK_SIZE_M": 128, "BLOCK_SIZE_N": 64, "BLOCK_SIZE_K": 32, "GROUP_SIZE_M": 8},
            num_stages=4, num_warps=4,
        ),
        triton.Config(
            {"BLOCK_SIZE_M": 64, "BLOCK_SIZE_N": 128, "BLOCK_SIZE_K": 32, "GROUP_SIZE_M": 8},
            num_stages=4, num_warps=4,
        ),
        triton.Config(
            {"BLOCK_SIZE_M": 32, "BLOCK_SIZE_N": 64, "BLOCK_SIZE_K": 32, "GROUP_SIZE_M": 8},
            num_stages=5, num_warps=2,
        ),
        triton.Config(
            {"BLOCK_SIZE_M": 64, "BLOCK_SIZE_N": 32, "BLOCK_SIZE_K": 32, "GROUP_SIZE_M": 8},
            num_stages=5, num_warps=2,
        ),
    ]


# ============================================================
# Reduced-index Conv3d kernel
# ============================================================

@triton.autotune(
    configs=get_autotune_config(),
    key=["M", "N", "K"],
)
@triton.jit
def conv3d_reduced_index_kernel(
    # Tensor pointers
    input_ptr,
    weight_ptr,
    output_ptr,
    # LUT pointers (only used when USE_LUT=True)
    ci_lut_ptr,
    kd_lut_ptr,
    kh_lut_ptr,
    kw_lut_ptr,
    # Matrix dimensions: C = A @ B where A=(M,K), B=(K,N), C=(M,N)
    M,  # C_out
    N,  # D_out * H_out * W_out  (total output positions)
    K,  # C_in * kD * kH * kW    (flattened kernel volume)
    # Weight (A-side) strides
    stride_wm, stride_wk,
    # Output (C-side) strides
    stride_om, stride_on,
    # Input tensor strides — layout (1, C_in, D, H, W)
    stride_ic, stride_id, stride_ih, stride_iw,
    # Input spatial dims (for bounds checking with padding)
    D, H, W,
    # Pre-computed H_out * W_out (saves one multiply per program)
    HW_out,
    H_out, W_out,
    # Stride, padding, dilation
    stride_d, stride_h, stride_w,
    pad_d, pad_h, pad_w,
    dil_d, dil_h, dil_w,
    # Constexpr kernel dimensions — enables constant folding of div/mod
    kD: tl.constexpr,
    kH: tl.constexpr,
    kW: tl.constexpr,
    # Flag: use LUT path or constexpr-arithmetic path
    USE_LUT: tl.constexpr,
    # Autotune constexprs
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
    BLOCK_SIZE_K: tl.constexpr,
    GROUP_SIZE_M: tl.constexpr,
):
    """Fused im2col + matmul kernel for Conv3d with reduced index computation.

    Two paths controlled by USE_LUT:
      - False: div/mod with compile-time constant kD/kH/kW (fast shifts)
      - True:  4 x tl.load from pre-computed LUT (L1-cache friendly)
    """
    # --- Program ID -> tile (pid_m, pid_n) with grouped ordering ---
    pid = tl.program_id(axis=0)
    num_pid_m = tl.cdiv(M, BLOCK_SIZE_M)
    num_pid_n = tl.cdiv(N, BLOCK_SIZE_N)
    num_pid_in_group = GROUP_SIZE_M * num_pid_n
    group_id = pid // num_pid_in_group
    first_pid_m = group_id * GROUP_SIZE_M
    group_size_m = min(num_pid_m - first_pid_m, GROUP_SIZE_M)
    pid_m = first_pid_m + ((pid % num_pid_in_group) % group_size_m)
    pid_n = (pid % num_pid_in_group) // group_size_m

    # --- Offsets ---
    offs_m = (pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)) % M  # (BLOCK_SIZE_M,)
    offs_n = (pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)) % N  # (BLOCK_SIZE_N,)
    offs_k = tl.arange(0, BLOCK_SIZE_K)                                # (BLOCK_SIZE_K,)

    # --- A-side (weight) pointer setup — loaded normally ---
    weight_ptrs = weight_ptr + (offs_m[:, None] * stride_wm + offs_k[None, :] * stride_wk)

    # --- Pre-loop: decompose offs_n into output spatial coords (done once) ---
    # HW_out is passed as a parameter — saves one multiply vs computing H_out * W_out
    d_out = offs_n // HW_out               # (BLOCK_SIZE_N,)
    h_out = (offs_n % HW_out) // W_out     # (BLOCK_SIZE_N,)
    w_out = offs_n % W_out                  # (BLOCK_SIZE_N,)

    # --- Compile-time constants from constexpr kD, kH, kW ---
    kHW: tl.constexpr = kH * kW
    kDHW: tl.constexpr = kD * kHW

    # --- K-loop: accumulate tiles ---
    accumulator = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)

    for k_block in range(0, tl.cdiv(K, BLOCK_SIZE_K)):
        k_indices = offs_k + k_block * BLOCK_SIZE_K  # (BLOCK_SIZE_K,)
        k_mask = k_indices < K

        # A-side: load weight tile — shape (BLOCK_SIZE_M, BLOCK_SIZE_K)
        a = tl.load(weight_ptrs, mask=k_mask[None, :], other=0.0)

        # B-side: implicit im2col — compute input addresses on-the-fly
        if USE_LUT:
            # LUT path: 4 loads from pre-computed tables
            ci = tl.load(ci_lut_ptr + k_indices, mask=k_mask, other=0)
            kd_val = tl.load(kd_lut_ptr + k_indices, mask=k_mask, other=0)
            kh_val = tl.load(kh_lut_ptr + k_indices, mask=k_mask, other=0)
            kw_val = tl.load(kw_lut_ptr + k_indices, mask=k_mask, other=0)
        else:
            # Constexpr-arithmetic path: div/mod with compile-time constants
            ci = k_indices // kDHW                  # (BLOCK_SIZE_K,)
            rem = k_indices % kDHW
            kd_val = rem // kHW                     # (BLOCK_SIZE_K,)
            kh_val = (rem % kHW) // kW              # (BLOCK_SIZE_K,)
            kw_val = rem % kW                       # (BLOCK_SIZE_K,)

        # Input coords — broadcast to (BLOCK_SIZE_K, BLOCK_SIZE_N)
        d_in = d_out[None, :] * stride_d + kd_val[:, None] * dil_d - pad_d
        h_in = h_out[None, :] * stride_h + kh_val[:, None] * dil_h - pad_h
        w_in = w_out[None, :] * stride_w + kw_val[:, None] * dil_w - pad_w

        # Bounds check for padded regions
        valid = ((d_in >= 0) & (d_in < D)
                 & (h_in >= 0) & (h_in < H)
                 & (w_in >= 0) & (w_in < W))

        # Compute flat address into input tensor
        input_addrs = (ci[:, None] * stride_ic
                       + d_in * stride_id
                       + h_in * stride_ih
                       + w_in * stride_iw)

        b = tl.load(input_ptr + input_addrs, mask=k_mask[:, None] & valid, other=0.0)

        # Cast to float16 for tl.dot (tensor cores require float16 inputs)
        accumulator = tl.dot(a.to(tl.float16), b.to(tl.float16), accumulator)

        # Advance weight pointers
        weight_ptrs += BLOCK_SIZE_K * stride_wk

    # --- Store output tile ---
    c = accumulator.to(tl.float32)
    offs_cm = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    offs_cn = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    c_ptrs = output_ptr + stride_om * offs_cm[:, None] + stride_on * offs_cn[None, :]
    c_mask = (offs_cm[:, None] < M) & (offs_cn[None, :] < N)
    tl.store(c_ptrs, c, mask=c_mask)


# ============================================================
# Wrapper function
# ============================================================

def conv3d_reduced_index(
    input: torch.Tensor,
    weight: torch.Tensor,
    bias: torch.Tensor = None,
    stride: tuple[int, int, int] = (1, 1, 1),
    padding: tuple[int, int, int] = (0, 0, 0),
    dilation: tuple[int, int, int] = (1, 1, 1),
    use_lut: bool = True,
) -> torch.Tensor:
    """Compute 3D convolution using implicit im2col with reduced index computation.

    Args:
        input: Input tensor of shape (N, C_in, D, H, W). Only N=1 supported.
        weight: Weight tensor of shape (C_out, C_in, kD, kH, kW).
        bias: Optional bias tensor of shape (C_out,).
        stride: Convolution stride (sD, sH, sW).
        padding: Zero-padding (pD, pH, pW).
        dilation: Kernel dilation (dD, dH, dW).
        use_lut: If True, use LUT-based index decomposition; else constexpr div/mod.

    Returns:
        Output tensor of shape (N, C_out, D_out, H_out, W_out).
    """
    N_batch, C_in, D, H, W = input.shape
    C_out, C_in_w, kD, kH, kW = weight.shape
    assert C_in == C_in_w, "Input channels must match weight channels"
    assert N_batch == 1, "Only batch_size=1 is supported"

    sD, sH, sW = stride
    pD, pH, pW = padding
    dD, dH, dW = dilation

    # Output spatial dimensions (general formula)
    D_out = (D + 2 * pD - dD * (kD - 1) - 1) // sD + 1
    H_out = (H + 2 * pH - dH * (kH - 1) - 1) // sH + 1
    W_out = (W + 2 * pW - dW * (kW - 1) - 1) // sW + 1

    # Matrix dimensions
    M = C_out
    N_pos = D_out * H_out * W_out
    K = C_in * kD * kH * kW

    # Pre-compute HW_out on host (saves one multiply per program)
    HW_out = H_out * W_out

    # Ensure tensors are on GPU, contiguous, float32
    input_gpu = input.to(device=DEVICE, dtype=torch.float32).contiguous()
    weight_matrix = weight.reshape(C_out, -1).to(device=DEVICE, dtype=torch.float32).contiguous()

    # Pre-compute LUT tensors on GPU (only used when use_lut=True)
    if use_lut:
        kHW = kH * kW
        kDHW = kD * kHW
        k_range = torch.arange(K, device=DEVICE, dtype=torch.int32)
        ci_lut = (k_range // kDHW).to(torch.int32).contiguous()
        rem = k_range % kDHW
        kd_lut = (rem // kHW).to(torch.int32).contiguous()
        kh_lut = ((rem % kHW) // kW).to(torch.int32).contiguous()
        kw_lut = (rem % kW).to(torch.int32).contiguous()
    else:
        # Dummy pointers — not accessed when USE_LUT=False
        ci_lut = torch.empty(1, device=DEVICE, dtype=torch.int32)
        kd_lut = torch.empty(1, device=DEVICE, dtype=torch.int32)
        kh_lut = torch.empty(1, device=DEVICE, dtype=torch.int32)
        kw_lut = torch.empty(1, device=DEVICE, dtype=torch.int32)

    # Allocate output
    output_flat = torch.empty((M, N_pos), device=DEVICE, dtype=torch.float32)

    # Launch kernel
    grid = lambda META: (
        triton.cdiv(M, META["BLOCK_SIZE_M"]) * triton.cdiv(N_pos, META["BLOCK_SIZE_N"]),
    )
    conv3d_reduced_index_kernel[grid](
        input_gpu, weight_matrix, output_flat,
        ci_lut, kd_lut, kh_lut, kw_lut,
        M, N_pos, K,
        weight_matrix.stride(0), weight_matrix.stride(1),
        output_flat.stride(0), output_flat.stride(1),
        input_gpu.stride(1), input_gpu.stride(2), input_gpu.stride(3), input_gpu.stride(4),
        D, H, W,
        HW_out,
        H_out, W_out,
        sD, sH, sW,
        pD, pH, pW,
        dD, dH, dW,
        # Constexpr params (keyword args — Triton specializes per unique tuple)
        kD=kD, kH=kH, kW=kW,
        USE_LUT=use_lut,
    )

    # Add bias
    if bias is not None:
        output_flat += bias.to(DEVICE).reshape(C_out, 1)

    # Reshape to 5D
    return output_flat.reshape(1, C_out, D_out, H_out, W_out)
