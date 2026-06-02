import torch
import triton
import triton.language as tl


@triton.jit
def matrix_transpose_kernel(
    input,
    output,
    rows,
    cols,
    stride_ir,
    stride_ic,
    stride_or,
    stride_oc,
    ROW_BLOCK: tl.constexpr,
    COL_BLOCK: tl.constexpr,
):
    row_pid = tl.program_id(0)
    col_pid = tl.program_id(1)

    # get the tile offsets
    row_offsets = row_pid * ROW_BLOCK + tl.arange(0, ROW_BLOCK)
    col_offsets = col_pid * COL_BLOCK + tl.arange(0, COL_BLOCK)

    input_offsets = (
        input + row_offsets[:, None] * stride_ir + col_offsets[None, :] * stride_ic
    )
    in_mask = (row_offsets[:, None] < rows) & (col_offsets[None, :] < cols)
    val = tl.load(input_offsets, mask=in_mask, other=0.0)

    output_offsets = (
        output + col_offsets[:, None] * stride_or + row_offsets[None, :] * stride_oc
    )
    out_mask = tl.trans(in_mask)
    tl.store(output_offsets, tl.trans(val), mask=out_mask)


# input, output are tensors on the GPU
def solve(input: torch.Tensor, output: torch.Tensor, rows: int, cols: int):
    stride_ir, stride_ic = cols, 1
    stride_or, stride_oc = rows, 1
    ROW_BLOCK = 128
    COL_BLOCK = 64
    grid = (triton.cdiv(rows, ROW_BLOCK), triton.cdiv(cols, COL_BLOCK))
    matrix_transpose_kernel[grid](
        input,
        output,
        rows,
        cols,
        stride_ir,
        stride_ic,
        stride_or,
        stride_oc,
        ROW_BLOCK=ROW_BLOCK,
        COL_BLOCK=COL_BLOCK,
    )
