"""
Ultimate Conv3D — combines batch parallelism, expanded autotuning, grouped
convolution, and depthwise specialization.

Two dispatch paths:

  Path 1 — Depthwise:
    For groups == C_in == C_out. Each channel has one filter.
    Simple dot-product reduction (no tl.dot needed).

  Path 2 — General optimized:
    Phase 4 flat K-loop (implicit im2col) with expanded autotuning (~20 configs)
    and batch + group parallelism via 3D grid: (tiles, batch, groups).

Design decisions based on benchmark data (RTX 3080):
  - Flat K-loop beats split K-loop (opt 4 was 5-6x slower)
  - Winograd removed (opt 5 was 10-30x slower on all tested sizes)
  - constexpr kernel dims removed (opt 3 was 4-17% slower)
  - Expanded autotuning is the only pure performance win (~1.7x on large high-res)
"""

import torch
import triton
import triton.language as tl

DEVICE = triton.runtime.driver.active.get_active_torch_device()


# ============================================================
# Expanded autotune configs (~20 configs)
# ============================================================

def get_general_autotune_config():
    return [
        # --- Small-M configs: for small C_out / small M_g ---
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
        # --- Balanced configs ---
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
        # --- Large-M with deeper pipelines ---
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


def get_depthwise_autotune_config():
    return [
        triton.Config({"BLOCK_SIZE_N": 256}, num_stages=3, num_warps=4),
        triton.Config({"BLOCK_SIZE_N": 512}, num_stages=3, num_warps=4),
        triton.Config({"BLOCK_SIZE_N": 1024}, num_stages=3, num_warps=4),
        triton.Config({"BLOCK_SIZE_N": 128}, num_stages=4, num_warps=2),
    ]


# ============================================================
# General kernel: flat K-loop + batch + groups + expanded autotuning
# ============================================================

@triton.autotune(
    configs=get_general_autotune_config(),
    key=["M_g", "N", "K"],
)
@triton.jit
def conv3d_general_kernel(
    # Tensor pointers
    input_ptr,
    weight_ptr,
    output_ptr,
    # Per-group matrix dimensions
    M_g,      # C_out // groups
    N,        # D_out * H_out * W_out  (total output positions)
    K,        # C_in_g * kD * kH * kW  (flattened kernel volume per group)
    # Weight (A-side) strides
    stride_wm, stride_wk,
    # Output (C-side) strides
    stride_ob,  # batch stride in output
    stride_om, stride_on,
    # Input strides — layout (N_batch, C_in, D, H, W)
    stride_ib,  # batch stride in input
    stride_ic, stride_id, stride_ih, stride_iw,
    # Input spatial dims
    D, H, W,
    # Convolution geometry
    kD, kH, kW,
    H_out, W_out,
    # Stride, padding, dilation
    stride_d, stride_h, stride_w,
    pad_d, pad_h, pad_w,
    dil_d, dil_h, dil_w,
    # Group channel offset
    C_in_g,   # C_in // groups
    # Autotune constexprs
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
    BLOCK_SIZE_K: tl.constexpr,
    GROUP_SIZE_M: tl.constexpr,
):
    """Implicit im2col Conv3d with batch + group parallelism.

    3D grid: (tiles, N_batch, groups).
    Uses Phase 4 flat K-loop — proven faster than split K-loop on RTX 3080.
    """
    # --- Batch and group indices from grid ---
    batch_idx = tl.program_id(axis=1)
    g = tl.program_id(axis=2)

    # --- Offset pointers by batch and group ---
    input_ptr = input_ptr + batch_idx * stride_ib + g * C_in_g * stride_ic
    output_ptr = output_ptr + batch_idx * stride_ob + g * M_g * stride_om
    weight_ptr = weight_ptr + g * M_g * stride_wm

    # --- Program ID -> tile (pid_m, pid_n) with grouped ordering ---
    pid = tl.program_id(axis=0)
    num_pid_m = tl.cdiv(M_g, BLOCK_SIZE_M)
    num_pid_n = tl.cdiv(N, BLOCK_SIZE_N)
    num_pid_in_group = GROUP_SIZE_M * num_pid_n
    group_id = pid // num_pid_in_group
    first_pid_m = group_id * GROUP_SIZE_M
    group_size_m = min(num_pid_m - first_pid_m, GROUP_SIZE_M)
    pid_m = first_pid_m + ((pid % num_pid_in_group) % group_size_m)
    pid_n = (pid % num_pid_in_group) // group_size_m

    # --- Offsets ---
    offs_m = (pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)) % M_g
    offs_n = (pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)) % N
    offs_k = tl.arange(0, BLOCK_SIZE_K)

    # --- A-side (weight) pointer setup ---
    weight_ptrs = weight_ptr + (offs_m[:, None] * stride_wm + offs_k[None, :] * stride_wk)

    # --- Pre-loop: decompose offs_n into output spatial coords ---
    HW_out = H_out * W_out
    d_out = offs_n // HW_out
    h_out = (offs_n % HW_out) // W_out
    w_out = offs_n % W_out

    # --- Flat K-loop: accumulate tiles ---
    accumulator = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)
    kHW = kH * kW
    kDHW = kD * kHW

    for k_block in range(0, tl.cdiv(K, BLOCK_SIZE_K)):
        k_indices = offs_k + k_block * BLOCK_SIZE_K
        k_mask = k_indices < K

        # A-side: load weight tile
        a = tl.load(weight_ptrs, mask=k_mask[None, :], other=0.0)

        # B-side: implicit im2col — compute input addresses on-the-fly
        ci = k_indices // kDHW
        rem = k_indices % kDHW
        kd_val = rem // kHW
        kh_val = (rem % kHW) // kW
        kw_val = rem % kW

        d_in = d_out[None, :] * stride_d + kd_val[:, None] * dil_d - pad_d
        h_in = h_out[None, :] * stride_h + kh_val[:, None] * dil_h - pad_h
        w_in = w_out[None, :] * stride_w + kw_val[:, None] * dil_w - pad_w

        valid = ((d_in >= 0) & (d_in < D)
                 & (h_in >= 0) & (h_in < H)
                 & (w_in >= 0) & (w_in < W))

        input_addrs = (ci[:, None] * stride_ic
                       + d_in * stride_id
                       + h_in * stride_ih
                       + w_in * stride_iw)

        b = tl.load(input_ptr + input_addrs, mask=k_mask[:, None] & valid, other=0.0)

        accumulator = tl.dot(a.to(tl.float16), b.to(tl.float16), accumulator)

        weight_ptrs += BLOCK_SIZE_K * stride_wk

    # --- Store output tile ---
    c = accumulator.to(tl.float32)
    offs_cm = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    offs_cn = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    c_ptrs = output_ptr + stride_om * offs_cm[:, None] + stride_on * offs_cn[None, :]
    c_mask = (offs_cm[:, None] < M_g) & (offs_cn[None, :] < N)
    tl.store(c_ptrs, c, mask=c_mask)


# ============================================================
# Depthwise kernel (with batch support)
# ============================================================

@triton.autotune(
    configs=get_depthwise_autotune_config(),
    key=["N_pos"],
)
@triton.jit
def conv3d_depthwise_kernel(
    input_ptr,
    weight_ptr,
    output_ptr,
    C,       # number of channels (= groups = C_in = C_out)
    N_pos,   # D_out * H_out * W_out
    # Input strides — layout (N_batch, C, D, H, W)
    stride_ib,  # batch stride in input
    stride_ic, stride_id, stride_ih, stride_iw,
    # Output strides
    stride_ob,  # batch stride in output
    # Input spatial dims
    D, H, W,
    # Output spatial dims
    HW_out, H_out, W_out,
    # Stride, padding, dilation
    stride_d, stride_h, stride_w,
    pad_d, pad_h, pad_w,
    dil_d, dil_h, dil_w,
    # Kernel size — constexpr for loop unrolling
    kD: tl.constexpr,
    kH: tl.constexpr,
    kW: tl.constexpr,
    # Autotune
    BLOCK_SIZE_N: tl.constexpr,
):
    """Depthwise Conv3d with batch support.

    Grid: (C, cdiv(N_pos, BLOCK_SIZE_N), N_batch).
    Each program: one channel, one output tile, one batch element.
    """
    channel = tl.program_id(0)
    pid_n = tl.program_id(1)
    batch_idx = tl.program_id(2)

    # Offset pointers for this batch element
    input_ptr = input_ptr + batch_idx * stride_ib
    output_ptr = output_ptr + batch_idx * stride_ob

    offs_n = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    n_mask = offs_n < N_pos

    # Decompose linear output index -> (d_out, h_out, w_out)
    d_out = offs_n // HW_out
    h_out = (offs_n % HW_out) // W_out
    w_out = offs_n % W_out

    # Weight base for this channel: weight shape (C, kD*kH*kW)
    kHW: tl.constexpr = kH * kW
    kDHW: tl.constexpr = kD * kHW
    w_base = channel * kDHW

    # Accumulate over kernel volume (unrolled)
    acc = tl.zeros((BLOCK_SIZE_N,), dtype=tl.float32)

    for kd in range(kD):
        for kh in range(kH):
            for kw in range(kW):
                w = tl.load(weight_ptr + w_base + kd * kHW + kh * kW + kw)

                d_in = d_out * stride_d + kd * dil_d - pad_d
                h_in = h_out * stride_h + kh * dil_h - pad_h
                w_in = w_out * stride_w + kw * dil_w - pad_w

                valid = ((d_in >= 0) & (d_in < D)
                         & (h_in >= 0) & (h_in < H)
                         & (w_in >= 0) & (w_in < W))

                inp_addr = (channel * stride_ic
                            + d_in * stride_id
                            + h_in * stride_ih
                            + w_in * stride_iw)
                inp = tl.load(input_ptr + inp_addr, mask=valid & n_mask, other=0.0)

                acc += w * inp

    out_addr = channel * N_pos + offs_n
    tl.store(output_ptr + out_addr, acc, mask=n_mask)


# ============================================================
# Internal wrappers
# ============================================================

def _launch_general(input_gpu, weight, bias, stride, padding, dilation, groups):
    """Launch the general kernel."""
    N_batch, C_in, D, H, W = input_gpu.shape
    C_out = weight.shape[0]
    kD, kH, kW = weight.shape[2], weight.shape[3], weight.shape[4]

    sD, sH, sW = stride
    pD, pH, pW = padding
    dD, dH, dW = dilation

    C_in_g = C_in // groups
    M_g = C_out // groups

    D_out = (D + 2 * pD - dD * (kD - 1) - 1) // sD + 1
    H_out = (H + 2 * pH - dH * (kH - 1) - 1) // sH + 1
    W_out = (W + 2 * pW - dW * (kW - 1) - 1) // sW + 1
    N_pos = D_out * H_out * W_out

    K = C_in_g * kD * kH * kW

    weight_matrix = weight.reshape(C_out, -1).contiguous()

    output_flat = torch.empty((N_batch, C_out, N_pos), device=DEVICE, dtype=torch.float32)

    grid = lambda META: (
        triton.cdiv(M_g, META["BLOCK_SIZE_M"]) * triton.cdiv(N_pos, META["BLOCK_SIZE_N"]),
        N_batch,
        groups,
    )
    conv3d_general_kernel[grid](
        input_gpu, weight_matrix, output_flat,
        M_g, N_pos, K,
        weight_matrix.stride(0), weight_matrix.stride(1),
        output_flat.stride(0), output_flat.stride(1), output_flat.stride(2),
        input_gpu.stride(0),
        input_gpu.stride(1), input_gpu.stride(2), input_gpu.stride(3), input_gpu.stride(4),
        D, H, W,
        kD, kH, kW,
        H_out, W_out,
        sD, sH, sW,
        pD, pH, pW,
        dD, dH, dW,
        C_in_g,
    )

    if bias is not None:
        output_flat += bias.reshape(1, C_out, 1)

    return output_flat.reshape(N_batch, C_out, D_out, H_out, W_out)


def _launch_depthwise(input_gpu, weight, bias, stride, padding, dilation):
    """Launch the depthwise kernel."""
    N_batch, C, D, H, W = input_gpu.shape
    kD, kH, kW = weight.shape[2], weight.shape[3], weight.shape[4]

    sD, sH, sW = stride
    pD, pH, pW = padding
    dD, dH, dW = dilation

    D_out = (D + 2 * pD - dD * (kD - 1) - 1) // sD + 1
    H_out = (H + 2 * pH - dH * (kH - 1) - 1) // sH + 1
    W_out = (W + 2 * pW - dW * (kW - 1) - 1) // sW + 1
    N_pos = D_out * H_out * W_out
    HW_out = H_out * W_out

    weight_flat = weight.reshape(C, -1).contiguous()

    output_flat = torch.empty((N_batch, C, N_pos), device=DEVICE, dtype=torch.float32)

    grid = lambda META: (C, triton.cdiv(N_pos, META["BLOCK_SIZE_N"]), N_batch)
    conv3d_depthwise_kernel[grid](
        input_gpu, weight_flat, output_flat,
        C, N_pos,
        input_gpu.stride(0),
        input_gpu.stride(1), input_gpu.stride(2), input_gpu.stride(3), input_gpu.stride(4),
        output_flat.stride(0),
        D, H, W,
        HW_out, H_out, W_out,
        sD, sH, sW,
        pD, pH, pW,
        dD, dH, dW,
        kD=kD, kH=kH, kW=kW,
    )

    if bias is not None:
        output_flat += bias.reshape(1, C, 1)

    return output_flat.reshape(N_batch, C, D_out, H_out, W_out)


# ============================================================
# Public API
# ============================================================

def conv3d_ultimate(
    input: torch.Tensor,
    weight: torch.Tensor,
    bias: torch.Tensor | None = None,
    stride: tuple[int, int, int] = (1, 1, 1),
    padding: tuple[int, int, int] = (0, 0, 0),
    dilation: tuple[int, int, int] = (1, 1, 1),
    groups: int = 1,
) -> torch.Tensor:
    """Ultimate Conv3D — dispatches to the best kernel for the given params.

    Supports arbitrary batch size, stride, padding, dilation, and groups.

    Dispatch:
      - groups == C_in == C_out -> depthwise kernel
      - otherwise -> general optimized kernel (flat K-loop + expanded autotuning)

    Args:
        input:    (N, C_in, D, H, W) input tensor.
        weight:   (C_out, C_in // groups, kD, kH, kW) weight tensor.
        bias:     optional (C_out,) bias tensor.
        stride:   convolution stride (sD, sH, sW).
        padding:  zero-padding (pD, pH, pW).
        dilation: kernel dilation (dD, dH, dW).
        groups:   number of groups (default 1).

    Returns:
        (N, C_out, D_out, H_out, W_out) output tensor.
    """
    N_batch, C_in, D, H, W = input.shape
    C_out = weight.shape[0]

    assert C_in % groups == 0
    assert C_out % groups == 0
    assert weight.shape[1] == C_in // groups

    input_gpu = input.to(device=DEVICE, dtype=torch.float32).contiguous()
    weight_gpu = weight.to(device=DEVICE, dtype=torch.float32).contiguous()
    bias_gpu = bias.to(device=DEVICE, dtype=torch.float32) if bias is not None else None

    # Depthwise specialization
    if groups == C_in and groups == C_out:
        return _launch_depthwise(input_gpu, weight_gpu, bias_gpu, stride, padding, dilation)

    # General path (handles groups >= 1)
    return _launch_general(input_gpu, weight_gpu, bias_gpu, stride, padding, dilation, groups)
