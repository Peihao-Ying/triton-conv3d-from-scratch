"""
Conv3d with split K-loop for better data reuse.

Phase 4 baseline iterates over a flat K = C_in * kD * kH * kW in blocks of
BLOCK_SIZE_K.  Adjacent k-indices mix channel and spatial dimensions, causing
scattered memory access.

This kernel restructures the K-loop into:
  - Outer loops over kernel spatial positions (kd, kh, kw)
  - Inner loop over input channel blocks (ci in steps of BLOCK_SIZE_K)

Benefits:
  1. Memory coalescing — for a fixed (kd, kh, kw), the input addresses across
     the N-tile differ only by the channel index, which is contiguous in NCDHW.
  2. L2 cache reuse — adjacent output positions share overlapping spatial input
     regions; the outer loop keeps spatial coords fixed while sweeping channels.
  3. Bounds check hoisted — spatial validity is computed once per (kd, kh, kw),
     not per channel block.

kD, kH, kW are tl.constexpr so the outer triple loop is unrolled at compile
time.  Eviction policy hints keep input data in L2 longer.

Supports: arbitrary stride, padding, dilation. groups=1, batch_size=1.
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
# Split K-loop Conv3d kernel
# ============================================================

@triton.autotune(
    configs=get_autotune_config(),
    key=["M", "N", "C_in"],
)
@triton.jit
def conv3d_shared_kernel(
    # Tensor pointers
    input_ptr,
    weight_ptr,
    output_ptr,
    # Matrix dimensions
    M,      # C_out
    N,      # D_out * H_out * W_out  (total output positions)
    C_in,   # input channels — inner loop bound
    # Weight (A-side) strides
    stride_wm, stride_wk,
    # Output (C-side) strides
    stride_om, stride_on,
    # Input tensor strides — layout (1, C_in, D, H, W)
    stride_ic, stride_id, stride_ih, stride_iw,
    # Input spatial dims (for bounds checking with padding)
    D, H, W,
    # Convolution geometry — constexpr so triple loop is unrolled
    kD: tl.constexpr,
    kH: tl.constexpr,
    kW: tl.constexpr,
    H_out, W_out,
    # Stride, padding, dilation
    stride_d, stride_h, stride_w,
    pad_d, pad_h, pad_w,
    dil_d, dil_h, dil_w,
    # Autotune constexprs
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
    BLOCK_SIZE_K: tl.constexpr,
    GROUP_SIZE_M: tl.constexpr,
):
    """Fused im2col + matmul kernel with split K-loop for data reuse.

    The outer loops iterate over kernel spatial positions (kd, kh, kw) and
    are unrolled at compile time.  The inner loop sweeps input channels in
    blocks of BLOCK_SIZE_K, giving contiguous memory access along C_in.
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

    # --- Pre-loop: decompose offs_n into output spatial coords (done once) ---
    HW_out = H_out * W_out
    d_out = offs_n // HW_out               # (BLOCK_SIZE_N,)
    h_out = (offs_n % HW_out) // W_out     # (BLOCK_SIZE_N,)
    w_out = offs_n % W_out                  # (BLOCK_SIZE_N,)

    # --- Accumulator ---
    accumulator = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)
    kHW = kH * kW
    kDHW = kD * kHW

    # --- Split K-loop: outer over spatial, inner over channels ---
    for kd_val in range(kD):
        for kh_val in range(kH):
            for kw_val in range(kW):
                # Input spatial coords — same for all ci in this iteration
                d_in = d_out * stride_d + kd_val * dil_d - pad_d  # (BLOCK_SIZE_N,)
                h_in = h_out * stride_h + kh_val * dil_h - pad_h  # (BLOCK_SIZE_N,)
                w_in = w_out * stride_w + kw_val * dil_w - pad_w  # (BLOCK_SIZE_N,)

                # Bounds check (computed once per spatial position)
                spatial_valid = ((d_in >= 0) & (d_in < D)
                                 & (h_in >= 0) & (h_in < H)
                                 & (w_in >= 0) & (w_in < W))  # (BLOCK_SIZE_N,)

                # Base address for this spatial position (without channel offset)
                base_addr = d_in * stride_id + h_in * stride_ih + w_in * stride_iw  # (BLOCK_SIZE_N,)

                # Weight k-offset base for this (kd, kh, kw)
                k_spatial_offset = kd_val * kHW + kh_val * kW + kw_val

                # Inner loop: sweep input channels in blocks of BLOCK_SIZE_K
                for ci_block in range(0, tl.cdiv(C_in, BLOCK_SIZE_K)):
                    ci_indices = ci_block * BLOCK_SIZE_K + offs_k  # (BLOCK_SIZE_K,)
                    ci_mask = ci_indices < C_in

                    # B-side: input load — contiguous along channel dim
                    # shape: (BLOCK_SIZE_K, BLOCK_SIZE_N)
                    input_addrs = ci_indices[:, None] * stride_ic + base_addr[None, :]
                    b = tl.load(input_ptr + input_addrs,
                                mask=ci_mask[:, None] & spatial_valid[None, :],
                                other=0.0,
                                eviction_policy="evict_last")

                    # A-side: weight load
                    # weight layout is (C_out, K) where K = C_in * kD * kH * kW
                    # For channel ci and spatial offset (kd, kh, kw):
                    #   k = ci * kDHW + kd * kHW + kh * kW + kw
                    k_offset = ci_indices * kDHW + k_spatial_offset
                    weight_addrs = offs_m[:, None] * stride_wm + k_offset[None, :] * stride_wk
                    a = tl.load(weight_ptr + weight_addrs,
                                mask=ci_mask[None, :],
                                other=0.0)

                    # Accumulate
                    accumulator = tl.dot(a.to(tl.float16), b.to(tl.float16), accumulator)

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

def conv3d_shared(
    input: torch.Tensor,
    weight: torch.Tensor,
    bias: torch.Tensor = None,
    stride: tuple[int, int, int] = (1, 1, 1),
    padding: tuple[int, int, int] = (0, 0, 0),
    dilation: tuple[int, int, int] = (1, 1, 1),
) -> torch.Tensor:
    """Compute 3D convolution with split K-loop for better data reuse.

    Args:
        input: Input tensor of shape (N, C_in, D, H, W). Only N=1 supported.
        weight: Weight tensor of shape (C_out, C_in, kD, kH, kW).
        bias: Optional bias tensor of shape (C_out,).
        stride: Convolution stride (sD, sH, sW).
        padding: Zero-padding (pD, pH, pW).
        dilation: Kernel dilation (dD, dH, dW).

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

    # Ensure tensors are on GPU, contiguous, float32
    input_gpu = input.to(device=DEVICE, dtype=torch.float32).contiguous()
    weight_matrix = weight.reshape(C_out, -1).to(device=DEVICE, dtype=torch.float32).contiguous()

    # Allocate output
    output_flat = torch.empty((M, N_pos), device=DEVICE, dtype=torch.float32)

    # Launch kernel
    grid = lambda META: (
        triton.cdiv(M, META["BLOCK_SIZE_M"]) * triton.cdiv(N_pos, META["BLOCK_SIZE_N"]),
    )
    conv3d_shared_kernel[grid](
        input_gpu, weight_matrix, output_flat,
        M, N_pos, C_in,
        weight_matrix.stride(0), weight_matrix.stride(1),
        output_flat.stride(0), output_flat.stride(1),
        input_gpu.stride(1), input_gpu.stride(2), input_gpu.stride(3), input_gpu.stride(4),
        D, H, W,
        kD, kH, kW,
        H_out, W_out,
        sD, sH, sW,
        pD, pH, pW,
        dD, dH, dW,
    )

    # Add bias
    if bias is not None:
        output_flat += bias.to(DEVICE).reshape(C_out, 1)

    # Reshape to 5D
    return output_flat.reshape(1, C_out, D_out, H_out, W_out)
