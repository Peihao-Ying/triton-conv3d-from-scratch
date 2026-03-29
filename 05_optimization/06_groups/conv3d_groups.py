"""
Grouped and depthwise Conv3d using implicit im2col in Triton.

Extends the Phase 4 implicit im2col kernel with:
  A) General grouped convolution (groups >= 1)
     - 2D grid: (num_tiles, groups); tl.program_id(1) = group index
     - Per-group slicing of weight, input channels, output channels
  B) Depthwise convolution (groups == C_in == C_out)
     - Specialized kernel: M_g=1 so the matmul degenerates to a dot product
     - Grid over (channels, output position tiles)
     - kD, kH, kW are constexpr so triple loop fully unrolls
  C) Dispatcher that picks the right path

Supports: arbitrary stride, padding, dilation. batch_size=1.
"""

import torch
import triton
import triton.language as tl

DEVICE = triton.runtime.driver.active.get_active_torch_device()


# ============================================================
# Autotune configs (shared by grouped kernel)
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


def get_depthwise_autotune_config():
    return [
        triton.Config({"BLOCK_SIZE_N": 256}, num_stages=3, num_warps=4),
        triton.Config({"BLOCK_SIZE_N": 512}, num_stages=3, num_warps=4),
        triton.Config({"BLOCK_SIZE_N": 1024}, num_stages=3, num_warps=4),
        triton.Config({"BLOCK_SIZE_N": 128}, num_stages=4, num_warps=2),
    ]


# ============================================================
# A) General grouped Conv3d kernel
# ============================================================

@triton.autotune(
    configs=get_autotune_config(),
    key=["M_g", "N", "K_g"],
)
@triton.jit
def conv3d_grouped_kernel(
    # Tensor pointers
    input_ptr,
    weight_ptr,
    output_ptr,
    # Per-group matrix dimensions
    M_g,   # C_out // groups
    N,     # D_out * H_out * W_out  (total output positions)
    K_g,   # (C_in // groups) * kD * kH * kW
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
    # Group info
    C_in_g,  # C_in // groups
    # Autotune constexprs
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
    BLOCK_SIZE_K: tl.constexpr,
    GROUP_SIZE_M: tl.constexpr,
):
    """Fused im2col + matmul kernel for grouped Conv3d.

    2D grid: axis=0 tiles over (M_g, N), axis=1 = group index.
    Each group operates on its own slice of weights, input channels,
    and output channels.
    """
    # --- Group index from 2nd grid axis ---
    g = tl.program_id(axis=1)

    # --- Offset pointers by group ---
    # Weight: each group owns a (M_g, K_g) block
    weight_ptr = weight_ptr + g * M_g * K_g
    # Input: group g reads channels [g*C_in_g, (g+1)*C_in_g)
    input_ptr = input_ptr + g * C_in_g * stride_ic
    # Output: group g writes to output channels [g*M_g, (g+1)*M_g)
    output_ptr = output_ptr + g * M_g * stride_om

    # --- Program ID → tile (pid_m, pid_n) with grouped ordering ---
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
    weight_ptrs = weight_ptr + (offs_m[:, None] * stride_wk * K_g // K_g + offs_m[:, None] * stride_wm + offs_k[None, :] * stride_wk)
    # Simplified: weight is reshaped to (M_g, K_g) contiguous, so
    # stride_wm = K_g, stride_wk = 1 (set by wrapper).
    # But we keep it general via the strides passed in.
    weight_ptrs = weight_ptr + (offs_m[:, None] * stride_wm + offs_k[None, :] * stride_wk)

    # --- Pre-loop: decompose offs_n into output spatial coords ---
    HW_out = H_out * W_out
    d_out = offs_n // HW_out
    h_out = (offs_n % HW_out) // W_out
    w_out = offs_n % W_out

    # --- K-loop: accumulate tiles ---
    accumulator = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)
    kHW = kH * kW
    kDHW = kD * kHW

    for k_block in range(0, tl.cdiv(K_g, BLOCK_SIZE_K)):
        k_indices = offs_k + k_block * BLOCK_SIZE_K
        k_mask = k_indices < K_g

        # A-side: load weight tile — shape (BLOCK_SIZE_M, BLOCK_SIZE_K)
        a = tl.load(weight_ptrs, mask=k_mask[None, :], other=0.0)

        # B-side: implicit im2col — compute input addresses on-the-fly
        # Decompose k_indices → (ci, kd, kh, kw) — ci is relative to the group
        ci = k_indices // kDHW
        rem = k_indices % kDHW
        kd_val = rem // kHW
        kh_val = (rem % kHW) // kW
        kw_val = rem % kW

        # Input coords — broadcast to (BLOCK_SIZE_K, BLOCK_SIZE_N)
        d_in = d_out[None, :] * stride_d + kd_val[:, None] * dil_d - pad_d
        h_in = h_out[None, :] * stride_h + kh_val[:, None] * dil_h - pad_h
        w_in = w_out[None, :] * stride_w + kw_val[:, None] * dil_w - pad_w

        # Bounds check for padded regions
        valid = ((d_in >= 0) & (d_in < D)
                 & (h_in >= 0) & (h_in < H)
                 & (w_in >= 0) & (w_in < W))

        # Compute flat address into input tensor
        # ci is relative to the group; input_ptr already offset by g * C_in_g * stride_ic
        input_addrs = (ci[:, None] * stride_ic
                       + d_in * stride_id
                       + h_in * stride_ih
                       + w_in * stride_iw)

        b = tl.load(input_ptr + input_addrs, mask=k_mask[:, None] & valid, other=0.0)

        # Accumulate via tl.dot (tensor cores require float16 inputs)
        accumulator = tl.dot(a.to(tl.float16), b.to(tl.float16), accumulator)

        # Advance weight pointers
        weight_ptrs += BLOCK_SIZE_K * stride_wk

    # --- Store output tile ---
    c = accumulator.to(tl.float32)
    offs_cm = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    offs_cn = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    c_ptrs = output_ptr + stride_om * offs_cm[:, None] + stride_on * offs_cn[None, :]
    c_mask = (offs_cm[:, None] < M_g) & (offs_cn[None, :] < N)
    tl.store(c_ptrs, c, mask=c_mask)


# ============================================================
# B) Depthwise Conv3d kernel (groups == C_in == C_out)
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
    # Input tensor strides — layout (1, C, D, H, W)
    stride_ic, stride_id, stride_ih, stride_iw,
    # Input spatial dims
    D, H, W,
    # Output spatial dims
    H_out, W_out,
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
    """Depthwise Conv3d: each channel has its own single filter.

    Grid: (C, cdiv(N_pos, BLOCK_SIZE_N)).
    Each program handles one channel + one tile of output spatial positions.
    M_g=1, so the matmul degenerates to a dot product per position.
    """
    channel = tl.program_id(0)
    pid_n = tl.program_id(1)

    offs_n = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    n_mask = offs_n < N_pos

    # Decompose linear output index → (d_out, h_out, w_out)
    HW_out = H_out * W_out
    d_out = offs_n // HW_out
    h_out = (offs_n % HW_out) // W_out
    w_out = offs_n % W_out

    # Weight base for this channel: weight shape is (C, 1, kD, kH, kW)
    # reshaped to (C, kD*kH*kW), so row = channel
    kHW = kH * kW
    kDHW = kD * kHW
    w_base = channel * kDHW

    # Accumulate over kernel volume (loops unroll because kD/kH/kW are constexpr)
    acc = tl.zeros((BLOCK_SIZE_N,), dtype=tl.float32)

    for kd in range(kD):
        for kh in range(kH):
            for kw in range(kW):
                # Load one weight scalar for this channel's kernel position
                w = tl.load(weight_ptr + w_base + kd * kHW + kh * kW + kw)

                # Compute input spatial coordinates
                d_in = d_out * stride_d + kd * dil_d - pad_d
                h_in = h_out * stride_h + kh * dil_h - pad_h
                w_in = w_out * stride_w + kw * dil_w - pad_w

                # Bounds check
                valid = ((d_in >= 0) & (d_in < D)
                         & (h_in >= 0) & (h_in < H)
                         & (w_in >= 0) & (w_in < W))

                # Load input value for this channel
                inp_addr = (channel * stride_ic
                            + d_in * stride_id
                            + h_in * stride_ih
                            + w_in * stride_iw)
                inp = tl.load(input_ptr + inp_addr, mask=valid & n_mask, other=0.0)

                acc += w * inp

    # Store output
    out_addr = channel * N_pos + offs_n
    tl.store(output_ptr + out_addr, acc, mask=n_mask)


# ============================================================
# Wrapper: general grouped convolution
# ============================================================

def _conv3d_grouped(
    input: torch.Tensor,
    weight: torch.Tensor,
    bias: torch.Tensor | None,
    stride: tuple[int, int, int],
    padding: tuple[int, int, int],
    dilation: tuple[int, int, int],
    groups: int,
) -> torch.Tensor:
    """Grouped Conv3d via implicit im2col. groups >= 1."""
    N_batch, C_in, D, H, W = input.shape
    C_out, C_in_per_group_w, kD, kH, kW = weight.shape
    assert N_batch == 1, "Only batch_size=1 is supported"
    assert C_in % groups == 0, f"C_in ({C_in}) must be divisible by groups ({groups})"
    assert C_out % groups == 0, f"C_out ({C_out}) must be divisible by groups ({groups})"
    C_in_g = C_in // groups
    assert C_in_per_group_w == C_in_g, (
        f"Weight C_in dimension ({C_in_per_group_w}) must equal C_in // groups ({C_in_g})"
    )

    sD, sH, sW = stride
    pD, pH, pW = padding
    dD, dH, dW = dilation

    # Output spatial dimensions
    D_out = (D + 2 * pD - dD * (kD - 1) - 1) // sD + 1
    H_out = (H + 2 * pH - dH * (kH - 1) - 1) // sH + 1
    W_out = (W + 2 * pW - dW * (kW - 1) - 1) // sW + 1

    # Per-group matrix dimensions
    M_g = C_out // groups
    N_pos = D_out * H_out * W_out
    K_g = C_in_g * kD * kH * kW

    # Ensure contiguous float32 on GPU
    input_gpu = input.to(device=DEVICE, dtype=torch.float32).contiguous()

    # Reshape weight to (groups, M_g, K_g) then view as contiguous block
    # PyTorch Conv3d weight shape: (C_out, C_in_g, kD, kH, kW)
    # = (groups * M_g, C_in_g, kD, kH, kW) already laid out group-by-group
    weight_matrix = weight.reshape(C_out, -1).to(device=DEVICE, dtype=torch.float32).contiguous()
    # weight_matrix shape: (C_out, K_g) = (groups * M_g, K_g)
    # Group g's block starts at row g * M_g

    # Allocate output as (C_out, N_pos) = (groups * M_g, N_pos)
    output_flat = torch.empty((C_out, N_pos), device=DEVICE, dtype=torch.float32)

    # Launch kernel — 2D grid: (num_tiles, groups)
    grid = lambda META: (
        triton.cdiv(M_g, META["BLOCK_SIZE_M"]) * triton.cdiv(N_pos, META["BLOCK_SIZE_N"]),
        groups,
    )
    conv3d_grouped_kernel[grid](
        input_gpu, weight_matrix, output_flat,
        M_g, N_pos, K_g,
        weight_matrix.stride(0), weight_matrix.stride(1),
        output_flat.stride(0), output_flat.stride(1),
        input_gpu.stride(1), input_gpu.stride(2), input_gpu.stride(3), input_gpu.stride(4),
        D, H, W,
        kD, kH, kW,
        H_out, W_out,
        sD, sH, sW,
        pD, pH, pW,
        dD, dH, dW,
        C_in_g,
    )

    # Add bias
    if bias is not None:
        output_flat += bias.to(DEVICE, dtype=torch.float32).reshape(C_out, 1)

    return output_flat.reshape(1, C_out, D_out, H_out, W_out)


# ============================================================
# Wrapper: depthwise convolution
# ============================================================

def _conv3d_depthwise(
    input: torch.Tensor,
    weight: torch.Tensor,
    bias: torch.Tensor | None,
    stride: tuple[int, int, int],
    padding: tuple[int, int, int],
    dilation: tuple[int, int, int],
) -> torch.Tensor:
    """Depthwise Conv3d: groups == C_in == C_out."""
    N_batch, C, D, H, W = input.shape
    C_out, one, kD, kH, kW = weight.shape
    assert N_batch == 1, "Only batch_size=1 is supported"
    assert C == C_out, f"Depthwise requires C_in ({C}) == C_out ({C_out})"
    assert one == 1, f"Depthwise weight must have C_in_per_group = 1, got {one}"

    sD, sH, sW = stride
    pD, pH, pW = padding
    dD, dH, dW = dilation

    D_out = (D + 2 * pD - dD * (kD - 1) - 1) // sD + 1
    H_out = (H + 2 * pH - dH * (kH - 1) - 1) // sH + 1
    W_out = (W + 2 * pW - dW * (kW - 1) - 1) // sW + 1
    N_pos = D_out * H_out * W_out

    input_gpu = input.to(device=DEVICE, dtype=torch.float32).contiguous()
    # Flatten weight to (C, kD*kH*kW) for simple linear addressing
    weight_flat = weight.reshape(C, -1).to(device=DEVICE, dtype=torch.float32).contiguous()

    output_flat = torch.empty((C, N_pos), device=DEVICE, dtype=torch.float32)

    # Grid: (channels, output position tiles)
    grid = lambda META: (C, triton.cdiv(N_pos, META["BLOCK_SIZE_N"]))
    conv3d_depthwise_kernel[grid](
        input_gpu, weight_flat, output_flat,
        C, N_pos,
        input_gpu.stride(1), input_gpu.stride(2), input_gpu.stride(3), input_gpu.stride(4),
        D, H, W,
        H_out, W_out,
        sD, sH, sW,
        pD, pH, pW,
        dD, dH, dW,
        kD, kH, kW,
    )

    if bias is not None:
        output_flat += bias.to(DEVICE, dtype=torch.float32).reshape(C, 1)

    return output_flat.reshape(1, C, D_out, H_out, W_out)


# ============================================================
# C) Dispatcher
# ============================================================

def conv3d_groups(
    input: torch.Tensor,
    weight: torch.Tensor,
    bias: torch.Tensor | None = None,
    stride: tuple[int, int, int] = (1, 1, 1),
    padding: tuple[int, int, int] = (0, 0, 0),
    dilation: tuple[int, int, int] = (1, 1, 1),
    groups: int = 1,
) -> torch.Tensor:
    """Dispatch to depthwise or general grouped Conv3d.

    Args:
        input: Input tensor of shape (1, C_in, D, H, W).
        weight: Weight tensor of shape (C_out, C_in // groups, kD, kH, kW).
        bias: Optional bias tensor of shape (C_out,).
        stride: Convolution stride (sD, sH, sW).
        padding: Zero-padding (pD, pH, pW).
        dilation: Kernel dilation (dD, dH, dW).
        groups: Number of groups (default 1).

    Returns:
        Output tensor of shape (1, C_out, D_out, H_out, W_out).
    """
    C_in = input.shape[1]
    C_out = weight.shape[0]

    if groups == C_in and groups == C_out:
        return _conv3d_depthwise(input, weight, bias, stride, padding, dilation)
    else:
        return _conv3d_grouped(input, weight, bias, stride, padding, dilation, groups)
