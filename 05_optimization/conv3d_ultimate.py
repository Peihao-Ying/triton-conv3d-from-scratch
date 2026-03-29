"""
Ultimate Conv3D — combines all six Phase 5 optimizations into one kernel.

Three dispatch paths:

  Path 1 — Winograd F(2x2x2, 3x3x3):
    For 3x3x3 kernels with stride=1, dilation=1, groups=1.
    Reduces multiplies from 27 to 8 per output element.

  Path 2 — Depthwise:
    For groups == C_in == C_out.  Each channel has one filter.
    Simple dot-product reduction (no tl.dot needed).

  Path 3 — General optimized:
    Combines batch parallelism (opt 1), expanded autotuning (opt 2),
    constexpr kernel dims (opt 3), split K-loop for data reuse (opt 4),
    and grouped convolution (opt 6) into one kernel.
    3D grid: (tiles, batch, groups).

All paths support arbitrary batch size.
"""

import math

import torch
import triton
import triton.language as tl

DEVICE = triton.runtime.driver.active.get_active_torch_device()


# ============================================================
# Expanded autotune configs (~20 configs) for the general kernel
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
# Path 3: General optimized kernel
# Combines: batch parallel + expanded autotuning + constexpr dims
#           + split K-loop + groups
# ============================================================

@triton.autotune(
    configs=get_general_autotune_config(),
    key=["M_g", "N", "C_in_g"],
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
    C_in_g,   # C_in // groups (inner loop bound)
    # Weight (A-side) strides
    stride_wm, stride_wk,
    # Output (C-side) strides
    stride_ob,  # batch stride in output
    stride_om, stride_on,
    # Input strides — layout (N_batch, C_in, D, H, W)
    stride_ib,  # batch stride in input
    stride_ic, stride_id, stride_ih, stride_iw,
    # Input spatial dims (for bounds checking with padding)
    D, H, W,
    # Pre-computed H_out * W_out
    HW_out,
    H_out, W_out,
    # Stride, padding, dilation
    stride_d, stride_h, stride_w,
    pad_d, pad_h, pad_w,
    dil_d, dil_h, dil_w,
    # Constexpr kernel dimensions — enables constant folding + loop unrolling
    kD: tl.constexpr,
    kH: tl.constexpr,
    kW: tl.constexpr,
    # Autotune constexprs
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
    BLOCK_SIZE_K: tl.constexpr,
    GROUP_SIZE_M: tl.constexpr,
):
    """Fully optimized Conv3d kernel.

    3D grid: (tiles, N_batch, groups).
    Split K-loop: outer over (kd, kh, kw) unrolled at compile time,
    inner over channel blocks for contiguous memory access.
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

    # --- Pre-loop: decompose offs_n into output spatial coords (done once) ---
    d_out = offs_n // HW_out
    h_out = (offs_n % HW_out) // W_out
    w_out = offs_n % W_out

    # --- Accumulator ---
    accumulator = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)
    kHW: tl.constexpr = kH * kW
    kDHW: tl.constexpr = kD * kHW

    # --- Split K-loop: outer over spatial (unrolled), inner over channels ---
    for kd_val in range(kD):
        for kh_val in range(kH):
            for kw_val in range(kW):
                # Input spatial coords — same for all ci in this iteration
                d_in = d_out * stride_d + kd_val * dil_d - pad_d
                h_in = h_out * stride_h + kh_val * dil_h - pad_h
                w_in = w_out * stride_w + kw_val * dil_w - pad_w

                # Bounds check (computed once per spatial position)
                spatial_valid = ((d_in >= 0) & (d_in < D)
                                 & (h_in >= 0) & (h_in < H)
                                 & (w_in >= 0) & (w_in < W))

                # Base address for this spatial position (without channel offset)
                base_addr = d_in * stride_id + h_in * stride_ih + w_in * stride_iw

                # Weight k-offset base for this (kd, kh, kw)
                k_spatial_offset = kd_val * kHW + kh_val * kW + kw_val

                # Inner loop: sweep input channels in blocks of BLOCK_SIZE_K
                for ci_block in range(0, tl.cdiv(C_in_g, BLOCK_SIZE_K)):
                    ci_indices = ci_block * BLOCK_SIZE_K + offs_k
                    ci_mask = ci_indices < C_in_g

                    # B-side: input load — contiguous along channel dim
                    input_addrs = ci_indices[:, None] * stride_ic + base_addr[None, :]
                    b = tl.load(input_ptr + input_addrs,
                                mask=ci_mask[:, None] & spatial_valid[None, :],
                                other=0.0,
                                eviction_policy="evict_last")

                    # A-side: weight load
                    # k = ci * kDHW + kd * kHW + kh * kW + kw
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
    c_mask = (offs_cm[:, None] < M_g) & (offs_cn[None, :] < N)
    tl.store(c_ptrs, c, mask=c_mask)


# ============================================================
# Path 2: Depthwise kernel (with batch support)
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
# Path 1: Winograd F(2x2x2, 3x3x3)
# ============================================================

# 1D transform matrices
_G = torch.tensor([
    [1.0,  0.0,  0.0],
    [0.5,  0.5,  0.5],
    [0.5, -0.5,  0.5],
    [0.0,  0.0,  1.0],
], dtype=torch.float32)

_BT = torch.tensor([
    [1.0,  0.0, -1.0,  0.0],
    [0.0,  1.0,  1.0,  0.0],
    [0.0, -1.0,  1.0,  0.0],
    [0.0,  1.0,  0.0, -1.0],
], dtype=torch.float32)

_AT = torch.tensor([
    [1.0,  1.0,  1.0,  0.0],
    [0.0,  1.0, -1.0, -1.0],
], dtype=torch.float32)


def _transform_3d(tensor, matrix):
    """Apply a separable transform along the last 3 dims."""
    t = torch.einsum("ia,...ayz->...iyz", matrix, tensor)
    t = torch.einsum("jb,...xbz->...xjz", matrix, t)
    t = torch.einsum("kc,...xyc->...xyk", matrix, t)
    return t


@triton.autotune(
    configs=[
        triton.Config({"BLOCK_M": 64, "BLOCK_N": 64, "BLOCK_K": 32}, num_stages=4, num_warps=4),
        triton.Config({"BLOCK_M": 32, "BLOCK_N": 64, "BLOCK_K": 32}, num_stages=4, num_warps=4),
        triton.Config({"BLOCK_M": 64, "BLOCK_N": 32, "BLOCK_K": 32}, num_stages=4, num_warps=4),
        triton.Config({"BLOCK_M": 32, "BLOCK_N": 32, "BLOCK_K": 32}, num_stages=5, num_warps=2),
    ],
    key=["M_dim", "N_dim", "K_dim"],
)
@triton.jit
def winograd_batched_matmul_kernel(
    U_ptr, V_ptr, M_ptr,
    M_dim, N_dim, K_dim,
    stride_u_m, stride_u_k, stride_u_p,
    stride_v_n, stride_v_k, stride_v_p,
    stride_m_n, stride_m_m, stride_m_p,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    """64 independent matmuls — one per Winograd domain position."""
    pid_p = tl.program_id(axis=1)
    pid_mn = tl.program_id(axis=0)
    num_pid_n = tl.cdiv(N_dim, BLOCK_N)
    pid_m = pid_mn // num_pid_n
    pid_n = pid_mn % num_pid_n

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_k = tl.arange(0, BLOCK_K)

    u_ptrs = U_ptr + (offs_m[:, None] * stride_u_m + offs_k[None, :] * stride_u_k + pid_p * stride_u_p)
    v_ptrs = V_ptr + (offs_n[:, None] * stride_v_n + offs_k[None, :] * stride_v_k + pid_p * stride_v_p)

    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    for k_start in range(0, tl.cdiv(K_dim, BLOCK_K)):
        k_offs = offs_k + k_start * BLOCK_K
        k_mask = k_offs < K_dim
        u = tl.load(u_ptrs, mask=(offs_m[:, None] < M_dim) & k_mask[None, :], other=0.0)
        v = tl.load(v_ptrs, mask=(offs_n[:, None] < N_dim) & k_mask[None, :], other=0.0)
        acc = tl.dot(u.to(tl.float16), tl.trans(v).to(tl.float16), acc)
        u_ptrs += BLOCK_K * stride_u_k
        v_ptrs += BLOCK_K * stride_v_k

    offs_cm = offs_m
    offs_cn = offs_n
    m_ptrs = M_ptr + (offs_cn[None, :] * stride_m_n + offs_cm[:, None] * stride_m_m + pid_p * stride_m_p)
    m_mask = (offs_cm[:, None] < M_dim) & (offs_cn[None, :] < N_dim)
    tl.store(m_ptrs, acc, mask=m_mask)


def _conv3d_winograd(input_dev, weight_dev, bias, padding):
    """Winograd F(2x2x2, 3x3x3) Conv3D. Supports arbitrary batch size."""
    N_batch, C_in, D, H, W = input_dev.shape
    C_out = weight_dev.shape[0]
    pD, pH, pW = padding

    D_out = D + 2 * pD - 2
    H_out = H + 2 * pH - 2
    W_out = W + 2 * pW - 2

    G_dev = _G.to(device=input_dev.device)
    BT_dev = _BT.to(device=input_dev.device)
    AT_dev = _AT.to(device=input_dev.device)

    # Pad input
    if pD > 0 or pH > 0 or pW > 0:
        input_dev = torch.nn.functional.pad(input_dev, (pW, pW, pH, pH, pD, pD))

    # Tile counts
    tile_d = math.ceil(D_out / 2)
    tile_h = math.ceil(H_out / 2)
    tile_w = math.ceil(W_out / 2)
    num_tiles = tile_d * tile_h * tile_w

    # Extra padding so tile extraction doesn't overflow
    D_pad, H_pad, W_pad = D + 2 * pD, H + 2 * pH, W + 2 * pW
    extra_d = max(0, tile_d * 2 + 2 - D_pad)
    extra_h = max(0, tile_h * 2 + 2 - H_pad)
    extra_w = max(0, tile_w * 2 + 2 - W_pad)
    if extra_d > 0 or extra_h > 0 or extra_w > 0:
        input_dev = torch.nn.functional.pad(input_dev, (0, extra_w, 0, extra_h, 0, extra_d))

    # Filter transform (once)
    U = _transform_3d(weight_dev, G_dev)  # (C_out, C_in, 4, 4, 4)

    # Extract tiles and transform
    patches = input_dev.unfold(2, 4, 2).unfold(3, 4, 2).unfold(4, 4, 2)
    patches = patches.contiguous().reshape(N_batch, C_in, num_tiles, 4, 4, 4)
    V = _transform_3d(patches, BT_dev)

    # Batched matmul via Triton
    U_flat = U.reshape(C_out, C_in, 64).contiguous()
    V_perm = V.permute(0, 2, 1, 3, 4, 5).contiguous()
    V_flat = V_perm.reshape(N_batch * num_tiles, C_in, 64).contiguous()
    M_flat = torch.empty((N_batch * num_tiles, C_out, 64), device=input_dev.device, dtype=torch.float32)

    grid = lambda META: (
        triton.cdiv(C_out, META["BLOCK_M"]) * triton.cdiv(N_batch * num_tiles, META["BLOCK_N"]),
        64,
    )
    winograd_batched_matmul_kernel[grid](
        U_flat, V_flat, M_flat,
        C_out, N_batch * num_tiles, C_in,
        U_flat.stride(0), U_flat.stride(1), U_flat.stride(2),
        V_flat.stride(0), V_flat.stride(1), V_flat.stride(2),
        M_flat.stride(0), M_flat.stride(1), M_flat.stride(2),
    )

    # Output transform
    M_tensor = M_flat.reshape(N_batch, num_tiles, C_out, 4, 4, 4)
    M_tensor = M_tensor.permute(0, 2, 1, 3, 4, 5).contiguous()
    Y = _transform_3d(M_tensor, AT_dev)

    # Stitch tiles
    Y = Y.reshape(N_batch, C_out, tile_d, tile_h, tile_w, 2, 2, 2)
    Y = Y.permute(0, 1, 2, 5, 3, 6, 4, 7).contiguous()
    Y = Y.reshape(N_batch, C_out, tile_d * 2, tile_h * 2, tile_w * 2)
    output = Y[:, :, :D_out, :H_out, :W_out]

    if bias is not None:
        output = output + bias.reshape(1, C_out, 1, 1, 1)

    return output.contiguous()


# ============================================================
# Internal wrappers for each path
# ============================================================

def _launch_general(input_gpu, weight, bias, stride, padding, dilation, groups):
    """Launch the general optimized kernel (Path 3)."""
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
    HW_out = H_out * W_out

    weight_matrix = weight.reshape(C_out, -1).contiguous()

    # Output: (N_batch, C_out, N_pos) — groups * M_g = C_out
    output_flat = torch.empty((N_batch, C_out, N_pos), device=DEVICE, dtype=torch.float32)

    grid = lambda META: (
        triton.cdiv(M_g, META["BLOCK_SIZE_M"]) * triton.cdiv(N_pos, META["BLOCK_SIZE_N"]),
        N_batch,
        groups,
    )
    conv3d_general_kernel[grid](
        input_gpu, weight_matrix, output_flat,
        M_g, N_pos, C_in_g,
        weight_matrix.stride(0), weight_matrix.stride(1),
        output_flat.stride(0), output_flat.stride(1), output_flat.stride(2),
        input_gpu.stride(0),
        input_gpu.stride(1), input_gpu.stride(2), input_gpu.stride(3), input_gpu.stride(4),
        D, H, W,
        HW_out, H_out, W_out,
        sD, sH, sW,
        pD, pH, pW,
        dD, dH, dW,
        kD=kD, kH=kH, kW=kW,
    )

    if bias is not None:
        output_flat += bias.reshape(1, C_out, 1)

    return output_flat.reshape(N_batch, C_out, D_out, H_out, W_out)


def _launch_depthwise(input_gpu, weight, bias, stride, padding, dilation):
    """Launch the depthwise kernel (Path 2)."""
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

    # Output: (N_batch, C, N_pos)
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
# Public API: dispatcher
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
    Automatically selects Winograd, depthwise, or general optimized path.

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
    kD, kH, kW = weight.shape[2], weight.shape[3], weight.shape[4]

    # Validate
    assert C_in % groups == 0
    assert C_out % groups == 0
    assert weight.shape[1] == C_in // groups

    # Move to GPU
    input_gpu = input.to(device=DEVICE, dtype=torch.float32).contiguous()
    weight_gpu = weight.to(device=DEVICE, dtype=torch.float32).contiguous()
    bias_gpu = bias.to(device=DEVICE, dtype=torch.float32) if bias is not None else None

    # Path 1: Winograd for 3x3x3, stride=1, dilation=1, groups=1
    if (kD == kH == kW == 3
        and stride == (1, 1, 1)
        and dilation == (1, 1, 1)
        and groups == 1):
        return _conv3d_winograd(input_gpu, weight_gpu, bias_gpu, padding)

    # Path 2: Depthwise specialization
    if groups == C_in and groups == C_out:
        return _launch_depthwise(input_gpu, weight_gpu, bias_gpu, stride, padding, dilation)

    # Path 3: General optimized
    return _launch_general(input_gpu, weight_gpu, bias_gpu, stride, padding, dilation, groups)
