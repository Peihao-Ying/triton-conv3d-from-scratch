"""
Winograd F(2x2x2, 3x3x3) Conv3D — reduces multiplications for 3x3x3 kernels.

Algorithm overview:
  Direct convolution: 27 multiplies per output element
  Winograd F(2,3) in 3D: 64 multiplies per 8 output elements = 8 per element (3.375x fewer)

The trade-off: we add transform overhead (matrix multiplies to enter/exit the
Winograd domain) but dramatically reduce the core multiply count.  For large
enough C_in/C_out the batched matmul dominates and the savings are real.

Constraints:
  - Kernel size must be exactly 3x3x3
  - Stride must be (1,1,1)
  - Dilation must be (1,1,1)

Implementation approach:
  - PyTorch handles the transforms (small fixed-size matrix ops on tiles)
  - Triton kernel does the heavy batched element-wise matmul in the
    transform domain (64 independent (C_out, C_in) @ (C_in, N*tiles) matmuls)

1D Winograd F(2,3) transform matrices:

  Filter transform G (4x3):
    [[1,    0,    0   ],
     [0.5,  0.5,  0.5 ],
     [0.5, -0.5,  0.5 ],
     [0,    0,    1   ]]

  Input transform B^T (4x4):
    [[1,  0, -1,  0],
     [0,  1,  1,  0],
     [0, -1,  1,  0],
     [0,  1,  0, -1]]

  Output transform A^T (2x4):
    [[1,  1,  1,  0],
     [0,  1, -1, -1]]

3D extension: apply each 1D transform separably along depth, height, width.
"""

import torch
import triton
import triton.language as tl

DEVICE = triton.runtime.driver.active.get_active_torch_device()

# ============================================================
# Fixed Winograd F(2,3) transform matrices (1D)
# ============================================================

# Filter transform: maps 3-element filter to 4-element Winograd domain
G = torch.tensor([
    [1.0,  0.0,  0.0],
    [0.5,  0.5,  0.5],
    [0.5, -0.5,  0.5],
    [0.0,  0.0,  1.0],
], dtype=torch.float32)

# Input transform: maps 4-element input tile to 4-element Winograd domain
BT = torch.tensor([
    [1.0,  0.0, -1.0,  0.0],
    [0.0,  1.0,  1.0,  0.0],
    [0.0, -1.0,  1.0,  0.0],
    [0.0,  1.0,  0.0, -1.0],
], dtype=torch.float32)

# Output transform: maps 4-element Winograd result back to 2-element output
AT = torch.tensor([
    [1.0,  1.0,  1.0,  0.0],
    [0.0,  1.0, -1.0, -1.0],
], dtype=torch.float32)



# ============================================================
# Transform helpers (applied separably along 3 dimensions)
# ============================================================

def _transform_3d(tensor, matrix, _unused=None):
    """Apply a separable transform along the last 3 dimensions of `tensor`.

    For a tensor with shape (..., X, Y, Z), applies the same 1D transform
    matrix along each of the last 3 dims:
        result[..., i, j, k] = sum_{a,b,c} M[i,a] * M[j,b] * M[k,c] * tensor[..., a, b, c]

    This is done sequentially (depth, then height, then width) via einsum.

    For filter transform:  matrix = G    (4x3),  maps (3,3,3) -> (4,4,4)
    For input transform:   matrix = B^T  (4x4),  maps (4,4,4) -> (4,4,4)
    For output transform:  matrix = A^T  (2x4),  maps (4,4,4) -> (2,2,2)
    """
    # Apply along dim -3 (depth): (..., X, Y, Z) -> (..., X', Y, Z)
    t = torch.einsum("ia,...ayz->...iyz", matrix, tensor)
    # Apply along dim -2 (height): (..., X', Y, Z) -> (..., X', Y', Z)
    t = torch.einsum("jb,...xbz->...xjz", matrix, t)
    # Apply along dim -1 (width): (..., X', Y', Z) -> (..., X', Y', Z')
    t = torch.einsum("kc,...xyc->...xyk", matrix, t)
    return t


def _filter_transform(weight, G_dev):
    """Transform filters into Winograd domain.

    Args:
        weight: (C_out, C_in, 3, 3, 3)
        G_dev:  G matrix on device, shape (4, 3)

    Returns:
        U: (C_out, C_in, 4, 4, 4) — filters in Winograd domain
    """
    return _transform_3d(weight, G_dev, G_dev)


def _input_transform(patches, BT_dev):
    """Transform input patches into Winograd domain.

    Args:
        patches: (N, C_in, num_tiles, 4, 4, 4) — overlapping 4x4x4 patches
        BT_dev:  B^T matrix on device, shape (4, 4)

    Returns:
        V: same shape, patches in Winograd domain
    """
    return _transform_3d(patches, BT_dev, BT_dev)


def _output_transform(M_tensor, AT_dev):
    """Transform element-wise product back from Winograd domain.

    Args:
        M_tensor: (N, C_out, num_tiles, 4, 4, 4)
        AT_dev:   A^T matrix on device, shape (2, 4)

    Returns:
        Y: (N, C_out, num_tiles, 2, 2, 2) — output tiles
    """
    return _transform_3d(M_tensor, AT_dev, AT_dev)


# ============================================================
# Tiling: extract overlapping 4x4x4 patches from padded input
# ============================================================

def _extract_tiles(input_padded, tile_d, tile_h, tile_w):
    """Extract overlapping 4x4x4 input tiles with stride 2.

    Args:
        input_padded: (N, C_in, D_pad, H_pad, W_pad) — input with padding applied
        tile_d, tile_h, tile_w: number of tiles along each spatial dimension

    Returns:
        patches: (N, C_in, tile_d * tile_h * tile_w, 4, 4, 4)
    """
    N, C_in, D_pad, H_pad, W_pad = input_padded.shape

    # Use unfold to extract overlapping patches with stride 2 along each dim
    # unfold(dim, size, step): extract windows of `size` with `step` stride
    patches = input_padded.unfold(2, 4, 2)  # -> (N, C_in, tile_d, H_pad, W_pad, 4)
    patches = patches.unfold(3, 4, 2)       # -> (N, C_in, tile_d, tile_h, W_pad, 4, 4)
    patches = patches.unfold(4, 4, 2)       # -> (N, C_in, tile_d, tile_h, tile_w, 4, 4, 4)

    # Merge the tile dimensions into one: (N, C_in, num_tiles, 4, 4, 4)
    num_tiles = tile_d * tile_h * tile_w
    patches = patches.contiguous().reshape(N, C_in, num_tiles, 4, 4, 4)
    return patches


# ============================================================
# Triton kernel: batched matmul in Winograd domain
# ============================================================
# For each of the 64 positions (i,j,k) in the 4x4x4 transform domain,
# we compute: M[n, c_out, tile, i, j, k] = sum_{c_in} U[c_out, c_in, i,j,k] * V[n, c_in, tile, i,j,k]
#
# This is 64 independent matmuls: (C_out, C_in) @ (C_in, N*num_tiles)
# We launch a 2D grid: one dim for the 64 positions, one for tiling M and N.

@triton.autotune(
    configs=[
        triton.Config(
            {"BLOCK_M": 64, "BLOCK_N": 64, "BLOCK_K": 32},
            num_stages=4, num_warps=4,
        ),
        triton.Config(
            {"BLOCK_M": 32, "BLOCK_N": 64, "BLOCK_K": 32},
            num_stages=4, num_warps=4,
        ),
        triton.Config(
            {"BLOCK_M": 64, "BLOCK_N": 32, "BLOCK_K": 32},
            num_stages=4, num_warps=4,
        ),
        triton.Config(
            {"BLOCK_M": 32, "BLOCK_N": 32, "BLOCK_K": 32},
            num_stages=5, num_warps=2,
        ),
    ],
    key=["M_dim", "N_dim", "K_dim"],
)
@triton.jit
def winograd_batched_matmul_kernel(
    # Pointers
    U_ptr,      # Filter in Winograd domain: (C_out, C_in, 64)  [flattened last 3 dims]
    V_ptr,      # Input  in Winograd domain: (N_batch * num_tiles, C_in, 64)
    M_ptr,      # Output in Winograd domain: (N_batch * num_tiles, C_out, 64)
    # Dimensions
    M_dim,      # C_out
    N_dim,      # N_batch * num_tiles
    K_dim,      # C_in
    # Strides for U: (C_out, C_in, 64) — row-major
    stride_u_m, stride_u_k, stride_u_p,
    # Strides for V: (N_batch * num_tiles, C_in, 64) — row-major
    stride_v_n, stride_v_k, stride_v_p,
    # Strides for M: (N_batch * num_tiles, C_out, 64) — row-major
    stride_m_n, stride_m_m, stride_m_p,
    # Constexprs
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    """Batched matmul for 64 Winograd domain positions.

    For each position p in [0, 64):
        M[:, :, p] = V[:, :, p] @ U[:, :, p]^T
    i.e. (N_dim, C_out) = (N_dim, C_in) @ (C_in, C_out) for each p.

    Grid: (num_tile_blocks, 64) where
        axis=0 tiles over (M_dim, N_dim) blocks
        axis=1 indexes the 64 Winograd positions
    """
    # Position in the 4x4x4 Winograd domain
    pid_p = tl.program_id(axis=1)

    # Tile index for the matmul dimensions
    pid_mn = tl.program_id(axis=0)
    num_pid_m = tl.cdiv(M_dim, BLOCK_M)
    num_pid_n = tl.cdiv(N_dim, BLOCK_N)
    pid_m = pid_mn // num_pid_n
    pid_n = pid_mn % num_pid_n

    # Offsets
    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)  # C_out indices
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)  # batch*tile indices
    offs_k = tl.arange(0, BLOCK_K)                     # C_in indices

    # Pointers into U for this position p: U[offs_m, offs_k, p]
    u_ptrs = U_ptr + (offs_m[:, None] * stride_u_m
                      + offs_k[None, :] * stride_u_k
                      + pid_p * stride_u_p)

    # Pointers into V for this position p: V[offs_n, offs_k, p]
    v_ptrs = V_ptr + (offs_n[:, None] * stride_v_n
                      + offs_k[None, :] * stride_v_k
                      + pid_p * stride_v_p)

    # Accumulate: for each K-block
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    for k_start in range(0, tl.cdiv(K_dim, BLOCK_K)):
        k_offs = offs_k + k_start * BLOCK_K
        k_mask = k_offs < K_dim

        # Load U tile: (BLOCK_M, BLOCK_K) — U[c_out, c_in, p]
        u = tl.load(u_ptrs, mask=(offs_m[:, None] < M_dim) & k_mask[None, :], other=0.0)

        # Load V tile: (BLOCK_N, BLOCK_K) — V[batch_tile, c_in, p]
        v = tl.load(v_ptrs, mask=(offs_n[:, None] < N_dim) & k_mask[None, :], other=0.0)

        # Matmul: (BLOCK_M, BLOCK_K) @ (BLOCK_K, BLOCK_N) = (BLOCK_M, BLOCK_N)
        # We want M[c_out, batch_tile] = sum_cin U[c_out, cin] * V[batch_tile, cin]
        # = U @ V^T, so we do dot(u, trans(v))
        acc = tl.dot(u.to(tl.float16), tl.trans(v).to(tl.float16), acc)

        # Advance K pointers
        u_ptrs += BLOCK_K * stride_u_k
        v_ptrs += BLOCK_K * stride_v_k

    # Store result: M[batch_tile, c_out, p] — note we store transposed
    # so that output is (N_dim, M_dim, 64) matching expected layout
    offs_cm = offs_m  # c_out
    offs_cn = offs_n  # batch*tile
    m_ptrs = M_ptr + (offs_cn[None, :] * stride_m_n
                      + offs_cm[:, None] * stride_m_m
                      + pid_p * stride_m_p)
    m_mask = (offs_cm[:, None] < M_dim) & (offs_cn[None, :] < N_dim)
    tl.store(m_ptrs, acc, mask=m_mask)


def _batched_matmul_triton(U_flat, V_flat):
    """Batched matmul across 64 Winograd positions using Triton.

    Args:
        U_flat: (C_out, C_in, 64) — filter in Winograd domain, flattened spatial
        V_flat: (N * num_tiles, C_in, 64) — input in Winograd domain

    Returns:
        M_flat: (N * num_tiles, C_out, 64) — product in Winograd domain
    """
    C_out, C_in, _ = U_flat.shape
    NT, _, _ = V_flat.shape  # NT = N_batch * num_tiles

    M_flat = torch.empty((NT, C_out, 64), device=U_flat.device, dtype=torch.float32)

    M_dim = C_out
    N_dim = NT
    K_dim = C_in

    grid = lambda META: (
        triton.cdiv(M_dim, META["BLOCK_M"]) * triton.cdiv(N_dim, META["BLOCK_N"]),
        64,
    )

    winograd_batched_matmul_kernel[grid](
        U_flat, V_flat, M_flat,
        M_dim, N_dim, K_dim,
        # U strides: (C_out, C_in, 64) contiguous
        U_flat.stride(0), U_flat.stride(1), U_flat.stride(2),
        # V strides: (NT, C_in, 64) contiguous
        V_flat.stride(0), V_flat.stride(1), V_flat.stride(2),
        # M strides: (NT, C_out, 64) contiguous
        M_flat.stride(0), M_flat.stride(1), M_flat.stride(2),
    )

    return M_flat


# ============================================================
# Main entry point
# ============================================================

def conv3d_winograd(input, weight, bias=None, padding=(0, 0, 0)):
    """Winograd F(2x2x2, 3x3x3) Conv3D.

    Only supports 3x3x3 kernels with stride=1, dilation=1.  Reduces
    the number of multiplications from 27 to 8 per output element by
    working in the Winograd transform domain.

    Args:
        input:   (N, C_in, D, H, W) input tensor
        weight:  (C_out, C_in, 3, 3, 3) filter tensor
        bias:    optional (C_out,) bias
        padding: (pad_d, pad_h, pad_w) zero-padding

    Returns:
        (N, C_out, D_out, H_out, W_out) output tensor
    """
    # ------------------------------------------------------------------
    # Validate constraints: Winograd F(2,3) only works for 3x3x3/s1/d1
    # ------------------------------------------------------------------
    N_batch, C_in, D, H, W = input.shape
    C_out, C_in_w, kD, kH, kW = weight.shape
    assert C_in == C_in_w, "Input/weight channel mismatch"
    assert (kD, kH, kW) == (3, 3, 3), \
        f"Winograd F(2,3) requires 3x3x3 kernel, got ({kD},{kH},{kW})"

    pD, pH, pW = padding

    # Output spatial dimensions (stride=1, dilation=1)
    D_out = D + 2 * pD - 2  # D + 2*pad - (3-1)
    H_out = H + 2 * pH - 2
    W_out = W + 2 * pW - 2
    assert D_out > 0 and H_out > 0 and W_out > 0, \
        f"Output dims must be positive, got ({D_out},{H_out},{W_out})"

    # ------------------------------------------------------------------
    # Move to device
    # ------------------------------------------------------------------
    input_dev = input.to(device=DEVICE, dtype=torch.float32).contiguous()
    weight_dev = weight.to(device=DEVICE, dtype=torch.float32).contiguous()

    G_dev = G.to(device=DEVICE)
    BT_dev = BT.to(device=DEVICE)
    AT_dev = AT.to(device=DEVICE)

    # ------------------------------------------------------------------
    # Step 0: Pad input if needed
    # ------------------------------------------------------------------
    if pD > 0 or pH > 0 or pW > 0:
        input_dev = torch.nn.functional.pad(
            input_dev, (pW, pW, pH, pH, pD, pD))  # F.pad order: W, H, D

    D_pad = D + 2 * pD
    H_pad = H + 2 * pH
    W_pad = W + 2 * pW

    # ------------------------------------------------------------------
    # Step 1: Compute tile counts
    # Winograd F(2,3) produces 2 outputs per tile, so we tile with stride 2.
    # Each tile reads a 4-element window (tile_size = output_tile + kernel - 1 = 2+3-1 = 4).
    # We need enough tiles to cover D_out, H_out, W_out, padding the last
    # tile if D_out/H_out/W_out is odd.
    # ------------------------------------------------------------------
    import math
    tile_d = math.ceil(D_out / 2)
    tile_h = math.ceil(H_out / 2)
    tile_w = math.ceil(W_out / 2)
    num_tiles = tile_d * tile_h * tile_w

    # Pad input so that tile extraction doesn't go out of bounds.
    # We need: tile_count * 2 + 2 <= padded_dim  (since window=4, stride=2)
    # i.e. padded_dim >= tile_count * 2 + 2
    need_d = tile_d * 2 + 2
    need_h = tile_h * 2 + 2
    need_w = tile_w * 2 + 2
    extra_d = max(0, need_d - D_pad)
    extra_h = max(0, need_h - H_pad)
    extra_w = max(0, need_w - W_pad)
    if extra_d > 0 or extra_h > 0 or extra_w > 0:
        input_dev = torch.nn.functional.pad(
            input_dev, (0, extra_w, 0, extra_h, 0, extra_d))

    # ------------------------------------------------------------------
    # Step 2: Filter transform — done once per convolution
    # weight: (C_out, C_in, 3, 3, 3) -> U: (C_out, C_in, 4, 4, 4)
    # ------------------------------------------------------------------
    U = _filter_transform(weight_dev, G_dev)  # (C_out, C_in, 4, 4, 4)

    # ------------------------------------------------------------------
    # Step 3: Extract overlapping 4x4x4 input patches and transform them
    # ------------------------------------------------------------------
    patches = _extract_tiles(input_dev, tile_d, tile_h, tile_w)
    # patches: (N, C_in, num_tiles, 4, 4, 4)

    V = _input_transform(patches, BT_dev)  # (N, C_in, num_tiles, 4, 4, 4)

    # ------------------------------------------------------------------
    # Step 4: Batched multiply in Winograd domain using Triton
    # Reshape for the kernel:
    #   U: (C_out, C_in, 4, 4, 4) -> (C_out, C_in, 64)
    #   V: (N, C_in, num_tiles, 4, 4, 4) -> permute to (N, tiles, C_in, 4,4,4)
    #      -> reshape (N*tiles, C_in, 64)
    # For each of 64 positions: M[:, c_out, p] = sum_cin U[c_out, cin, p] * V[:, cin, p]
    # ------------------------------------------------------------------
    U_flat = U.reshape(C_out, C_in, 64).contiguous()
    # V is (N, C_in, tiles, 4, 4, 4) — must move tiles next to N before flatten
    V_perm = V.permute(0, 2, 1, 3, 4, 5).contiguous()  # (N, tiles, C_in, 4, 4, 4)
    V_flat = V_perm.reshape(N_batch * num_tiles, C_in, 64).contiguous()

    M_flat = _batched_matmul_triton(U_flat, V_flat)
    # M_flat: (N * num_tiles, C_out, 64)

    # ------------------------------------------------------------------
    # Step 5: Output transform — back to spatial domain
    # Reshape M to (N, C_out, num_tiles, 4, 4, 4) then apply A^T transform
    # ------------------------------------------------------------------
    M_tensor = M_flat.reshape(N_batch, num_tiles, C_out, 4, 4, 4)
    M_tensor = M_tensor.permute(0, 2, 1, 3, 4, 5).contiguous()
    # M_tensor: (N, C_out, num_tiles, 4, 4, 4)

    Y = _output_transform(M_tensor, AT_dev)
    # Y: (N, C_out, num_tiles, 2, 2, 2)

    # ------------------------------------------------------------------
    # Step 6: Stitch tiles back into output tensor
    # ------------------------------------------------------------------
    Y = Y.reshape(N_batch, C_out, tile_d, tile_h, tile_w, 2, 2, 2)
    # Permute so spatial dims interleave: (N, C_out, tile_d, 2, tile_h, 2, tile_w, 2)
    Y = Y.permute(0, 1, 2, 5, 3, 6, 4, 7).contiguous()
    # Merge tile+local: (N, C_out, tile_d*2, tile_h*2, tile_w*2)
    Y = Y.reshape(N_batch, C_out, tile_d * 2, tile_h * 2, tile_w * 2)

    # Trim to exact output size (tiles may overshoot if D_out/H_out/W_out is odd)
    output = Y[:, :, :D_out, :H_out, :W_out]

    # ------------------------------------------------------------------
    # Add bias
    # ------------------------------------------------------------------
    if bias is not None:
        bias_dev = bias.to(device=DEVICE, dtype=torch.float32)
        output = output + bias_dev.reshape(1, C_out, 1, 1, 1)

    return output.contiguous()
