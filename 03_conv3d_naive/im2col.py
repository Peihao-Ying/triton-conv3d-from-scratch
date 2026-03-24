"""
im2col for 3D convolution

im2col ("image to column") rearranges input patches into columns of a matrix
so that convolution becomes a single matrix multiplication.

For a 5D input (N, C_in, D, H, W) with kernel size (kD, kH, kW):
  - Each output position (d, h, w) corresponds to a patch of shape (C_in, kD, kH, kW)
  - We flatten each patch into a column of length C_in * kD * kH * kW
  - There are D_out * H_out * W_out output positions total

Result: a matrix of shape (C_in * kD * kH * kW, D_out * H_out * W_out)

This layout lets us compute conv3d as:
    output = weight_matrix @ im2col_matrix + bias
where weight_matrix has shape (C_out, C_in * kD * kH * kW).

Constraints: stride=1, padding=0, dilation=1, groups=1, batch_size=1.
"""

import torch


def im2col_3d(input: torch.Tensor, kernel_size: tuple[int, int, int]) -> torch.Tensor:
    """Convert 3D input to column matrix for convolution.

    Args:
        input: Input tensor of shape (N, C_in, D, H, W). Only N=1 supported.
        kernel_size: Tuple of (kD, kH, kW).

    Returns:
        Column matrix of shape (C_in * kD * kH * kW, D_out * H_out * W_out).
    """
    N, C_in, D, H, W = input.shape
    assert N == 1, "Only batch_size=1 is supported"
    kD, kH, kW = kernel_size

    # Output spatial dimensions (stride=1, padding=0)
    D_out = D - kD + 1
    H_out = H - kH + 1
    W_out = W - kW + 1

    K = C_in * kD * kH * kW          # flattened patch length
    N_positions = D_out * H_out * W_out  # total output positions

    # Build the matrix by extracting each patch
    # We iterate over every output position and flatten the corresponding patch
    cols = torch.zeros(K, N_positions, dtype=input.dtype, device=input.device)

    pos = 0
    for d in range(D_out):
        for h in range(H_out):
            for w in range(W_out):
                # Extract patch: shape (C_in, kD, kH, kW)
                patch = input[0, :, d:d + kD, h:h + kH, w:w + kW]
                # Flatten to column of length K
                cols[:, pos] = patch.reshape(-1)
                pos += 1

    return cols
