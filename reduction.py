import torch
import triton
import triton.language as tl


@triton.jit
def reduce_sum_kernel(input, output, N, BLOCK: tl.constexpr):
    pid = tl.program_id(0)
    offsets = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offsets < N
    data = tl.load(input + offsets, mask=mask, other=0.0)
    sum = tl.sum(data)
    tl.atomic_add(output, sum, sem="relaxed")


# input, output are tensors on the GPU
def solve(input: torch.Tensor, output: torch.Tensor, N: int):
    BLOCK = 1024
    grid = (triton.cdiv(N, BLOCK),)
    reduce_sum_kernel[grid](input, output, N, BLOCK)
