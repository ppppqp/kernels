import torch
import triton
import triton.language as tl


@triton.jit
def matrix_multiplication_kernel(
    a, b, c, M, N, K, stride_am, stride_an, stride_bn, stride_bk, stride_cm, stride_ck
):
    BLOCK: tl.constexpr = 8192
    pid_m = tl.program_id(0)
    pid_k = tl.program_id(1)

    offset = tl.arange(0, BLOCK)
    mask = offset < N
    # get (pid_m) row from a
    offsets_m = pid_m * N + offset
    aa = tl.load(a + offsets_m, mask=mask)

    # get (pid_k) column from b
    offsets_k = offset * stride_bn + pid_k * stride_bk
    bb = tl.load(b + offsets_k, mask=mask)
    # perform vector product

    acc = tl.sum(aa * bb)
    # store to c
    offsets_c = c + stride_cm * pid_m + stride_ck * pid_k
    tl.store(offsets_c, acc)


# a, b, c are tensors on the GPU
def solve(a: torch.Tensor, b: torch.Tensor, c: torch.Tensor, M: int, N: int, K: int):
    stride_am, stride_an = N, 1
    stride_bn, stride_bk = K, 1
    stride_cm, stride_ck = K, 1

    grid = (M, K)
    matrix_multiplication_kernel[grid](
        a,
        b,
        c,
        M,
        N,
        K,
        stride_am,
        stride_an,
        stride_bn,
        stride_bk,
        stride_cm,
        stride_ck,
    )
