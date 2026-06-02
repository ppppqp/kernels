import torch
import triton
import triton.language as tl


@triton.jit
def matrix_transpose_kernel(
    input, output, rows, cols, stride_ir, stride_ic, stride_or, stride_oc
):
    row_pid = tl.program_id(0)
    col_pid = tl.program_id(1)
    src_offset = input + row_pid * stride_ir + col_pid * stride_ic
    tar_offset = output + col_pid * stride_or + row_pid * stride_oc
    val = tl.load(src_offset)
    tl.store(tar_offset, val)


# input, output are tensors on the GPU
def solve(input: torch.Tensor, output: torch.Tensor, rows: int, cols: int):
    stride_ir, stride_ic = cols, 1
    stride_or, stride_oc = rows, 1

    grid = (rows, cols)
    matrix_transpose_kernel[grid](
        input, output, rows, cols, stride_ir, stride_ic, stride_or, stride_oc
    )
