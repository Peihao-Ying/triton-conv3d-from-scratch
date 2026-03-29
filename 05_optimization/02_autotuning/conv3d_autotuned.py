"""
Conv3d using implicit im2col with expanded autotuning.

Same kernel body as Phase 4 (04_conv3d_implicit/conv3d_implicit.py) — only the
autotune config list is expanded from 5 to ~20 configs to cover a wider range
of problem sizes (small C_out, balanced, large-M).

Supports: arbitrary stride, padding, dilation. groups=1, batch_size=1.
"""

import torch
import triton
import triton.language as tl

DEVICE = triton.runtime.driver.active.get_active_torch_device()


# ============================================================
# Expanded autotune configs (~20 configs)
# ============================================================

def get_autotune_config():
    return [
        # --- Small-M configs: BLOCK_SIZE_M in {16, 32} for small C_out ---
        triton.Config(
            {"BLOCK_SIZE_M": 16, "BLOCK_SIZE_N": 64, "BLOCK_SIZE_K": 32, "GROUP_SIZE_M": 8},
            num_stages=3, num_warps=4,
        ),
        triton.Config(
            {"BLOCK_SIZE_M": 16, "BLOCK_SIZE_N": 128, "BLOCK_SIZE_K": 32, "GROUP_SIZE_M": 8},
            num_stages=3, num_warps=4,
        ),
        triton.Config(
            {"BLOCK_SIZE_M": 16, "BLOCK_SIZE_N": 256, "BLOCK_SIZE_K": 32, "GROUP_SIZE_M": 4},
            num_stages=2, num_warps=8,
        ),
        triton.Config(
            {"BLOCK_SIZE_M": 32, "BLOCK_SIZE_N": 64, "BLOCK_SIZE_K": 32, "GROUP_SIZE_M": 8},
            num_stages=3, num_warps=4,
        ),
        triton.Config(
            {"BLOCK_SIZE_M": 32, "BLOCK_SIZE_N": 128, "BLOCK_SIZE_K": 32, "GROUP_SIZE_M": 8},
            num_stages=3, num_warps=4,
        ),
        triton.Config(
            {"BLOCK_SIZE_M": 32, "BLOCK_SIZE_N": 256, "BLOCK_SIZE_K": 32, "GROUP_SIZE_M": 4},
            num_stages=2, num_warps=8,
        ),
        triton.Config(
            {"BLOCK_SIZE_M": 32, "BLOCK_SIZE_N": 128, "BLOCK_SIZE_K": 16, "GROUP_SIZE_M": 8},
            num_stages=4, num_warps=4,
        ),

        # --- Balanced configs: BLOCK_SIZE_M in {64, 128} with various N ---
        triton.Config(
            {"BLOCK_SIZE_M": 64, "BLOCK_SIZE_N": 64, "BLOCK_SIZE_K": 32, "GROUP_SIZE_M": 8},
            num_stages=4, num_warps=4,
        ),
        triton.Config(
            {"BLOCK_SIZE_M": 64, "BLOCK_SIZE_N": 128, "BLOCK_SIZE_K": 32, "GROUP_SIZE_M": 8},
            num_stages=4, num_warps=4,
        ),
        triton.Config(
            {"BLOCK_SIZE_M": 64, "BLOCK_SIZE_N": 256, "BLOCK_SIZE_K": 32, "GROUP_SIZE_M": 4},
            num_stages=3, num_warps=8,
        ),
        triton.Config(
            {"BLOCK_SIZE_M": 64, "BLOCK_SIZE_N": 64, "BLOCK_SIZE_K": 64, "GROUP_SIZE_M": 8},
            num_stages=3, num_warps=4,
        ),
        triton.Config(
            {"BLOCK_SIZE_M": 64, "BLOCK_SIZE_N": 128, "BLOCK_SIZE_K": 16, "GROUP_SIZE_M": 8},
            num_stages=5, num_warps=4,
        ),
        triton.Config(
            {"BLOCK_SIZE_M": 128, "BLOCK_SIZE_N": 64, "BLOCK_SIZE_K": 32, "GROUP_SIZE_M": 8},
            num_stages=4, num_warps=4,
        ),
        triton.Config(
            {"BLOCK_SIZE_M": 128, "BLOCK_SIZE_N": 128, "BLOCK_SIZE_K": 32, "GROUP_SIZE_M": 8},
            num_stages=4, num_warps=8,
        ),
        triton.Config(
            {"BLOCK_SIZE_M": 128, "BLOCK_SIZE_N": 256, "BLOCK_SIZE_K": 32, "GROUP_SIZE_M": 4},
            num_stages=3, num_warps=8,
        ),
        triton.Config(
            {"BLOCK_SIZE_M": 128, "BLOCK_SIZE_N": 128, "BLOCK_SIZE_K": 16, "GROUP_SIZE_M": 8},
            num_stages=5, num_warps=4,
        ),
        triton.Config(
            {"BLOCK_SIZE_M": 128, "BLOCK_SIZE_N": 64, "BLOCK_SIZE_K": 64, "GROUP_SIZE_M": 8},
            num_stages=3, num_warps=4,
        ),

        # --- Large-M configs: BLOCK_SIZE_M=128 with deeper pipelines ---
        triton.Config(
            {"BLOCK_SIZE_M": 128, "BLOCK_SIZE_N": 64, "BLOCK_SIZE_K": 32, "GROUP_SIZE_M": 4},
            num_stages=5, num_warps=8,
        ),
        triton.Config(
            {"BLOCK_SIZE_M": 128, "BLOCK_SIZE_N": 128, "BLOCK_SIZE_K": 64, "GROUP_SIZE_M": 4},
            num_stages=3, num_warps=8,
        ),
        triton.Config(
            {"BLOCK_SIZE_M": 128, "BLOCK_SIZE_N": 64, "BLOCK_SIZE_K": 16, "GROUP_SIZE_M": 8},
            num_stages=5, num_warps=4,
        ),
    ]


# ============================================================
# Implicit im2col Conv3d kernel (same body as Phase 4)
# ============================================================

@triton.autotune(
    configs=get_autotune_config(),
    key=["M", "N", "K"],
)
@triton.jit
def conv3d_autotuned_kernel(
    # Tensor pointers
    input_ptr,
    weight_ptr,
    output_ptr,
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
    # Convolution geometry
    kD, kH, kW,
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
    """Fused im2col + matmul kernel for Conv3d.

    Computes output = weight_matrix @ virtual_im2col_matrix
    where the im2col matrix is never materialized — input addresses are
    computed on-the-fly inside the K-loop.
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
    HW_out = H_out * W_out
    d_out = offs_n // HW_out               # (BLOCK_SIZE_N,)
    h_out = (offs_n % HW_out) // W_out     # (BLOCK_SIZE_N,)
    w_out = offs_n % W_out                  # (BLOCK_SIZE_N,)

    # --- K-loop: accumulate tiles ---
    accumulator = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)
    kHW = kH * kW
    kDHW = kD * kHW

    for k_block in range(0, tl.cdiv(K, BLOCK_SIZE_K)):
        k_indices = offs_k + k_block * BLOCK_SIZE_K  # (BLOCK_SIZE_K,)
        k_mask = k_indices < K

        # A-side: load weight tile — shape (BLOCK_SIZE_M, BLOCK_SIZE_K)
        a = tl.load(weight_ptrs, mask=k_mask[None, :], other=0.0)

        # B-side: implicit im2col — compute input addresses on-the-fly
        # Decompose k_indices -> (ci, kd, kh, kw)
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

def conv3d_autotuned(
    input: torch.Tensor,
    weight: torch.Tensor,
    bias: torch.Tensor = None,
    stride: tuple[int, int, int] = (1, 1, 1),
    padding: tuple[int, int, int] = (0, 0, 0),
    dilation: tuple[int, int, int] = (1, 1, 1),
) -> torch.Tensor:
    """Compute 3D convolution using implicit im2col fused into a Triton kernel.

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
    K = C_in * kD * kH * kW

    # Ensure tensors are on GPU, contiguous, float32
    input_gpu = input.to(device=DEVICE, dtype=torch.float32).contiguous()
    weight_matrix = weight.reshape(C_out, -1).to(device=DEVICE, dtype=torch.float32).contiguous()

    # Allocate output
    output_flat = torch.empty((M, N_pos), device=DEVICE, dtype=torch.float32)

    # Launch kernel
    grid = lambda META: (
        triton.cdiv(M, META["BLOCK_SIZE_M"]) * triton.cdiv(N_pos, META["BLOCK_SIZE_N"]),
    )
    conv3d_autotuned_kernel[grid](
        input_gpu, weight_matrix, output_flat,
        M, N_pos, K,
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
