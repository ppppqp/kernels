import torch
import triton
import triton.language as tl


@triton.jit
def matrix_multiplication_kernel(
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
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    pid_m = tl.program_id(0)
    pid_k = tl.program_id(1)
    m_0 = pid_m * BLOCK_M
    k_0 = pid_k * BLOCK_K
    offsets_m = m_0 + tl.arange(0, BLOCK_M)
    offsets_k = k_0 + tl.arange(0, BLOCK_K)

    # load the initial a_cur and b_cur
    mask_m = offsets_m < M
    mask_k = offsets_k < K
    mask_c = mask_m[:, None] & mask_k[None, :]
    accu = tl.zeros((BLOCK_M, BLOCK_K), dtype=tl.float32)

    a_base = a + offsets_m[:, None] * stride_am
    b_base = b + offsets_k[None, :] * stride_bk
    # iterate through the tiles on k dimension and accumulate
    for n in tl.range(0, tl.cdiv(N, BLOCK_N)):
        offsets_n = n * BLOCK_N + tl.arange(0, BLOCK_N)
        mask_n = offsets_n < N
        # a[i, block_n_start:block_n_end]
        a_cur = tl.load(
            a_base + offsets_n[None, :] * stride_an,
            mask=mask_m[:, None] & mask_n[None, :],
            other=0.0,
        )
        # b[block_n_start:block_n_end,j]
        b_cur = tl.load(
            b_base + offsets_n[:, None] * stride_bn,
            mask=mask_n[:, None] & mask_k[None, :],
            other=0.0,
        )
        # accu += tl.dot(a_cur[:, None], b_cur[None, :])
        accu += tl.dot(a_cur, b_cur)

    # accu is now a[block_start:block_end, :] * b[:, block_start:block_end]
    offsets_c = c + offsets_m[:, None] * stride_cm + offsets_k[None, :] * stride_ck
    tl.store(offsets_c, accu, mask=mask_c)


# a, b, c are tensors on the GPU
def solve(a: torch.Tensor, b: torch.Tensor, c: torch.Tensor, M: int, N: int, K: int):
    stride_am, stride_an = N, 1
    stride_bn, stride_bk = K, 1
    stride_cm, stride_ck = K, 1

    BLOCK_SIZE_M = 64
    BLOCK_SIZE_N = 32
    BLOCK_SIZE_K = 128

    grid = (triton.cdiv(M, BLOCK_SIZE_M), triton.cdiv(K, BLOCK_SIZE_K))
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
        BLOCK_SIZE_M,
        BLOCK_SIZE_N,
        BLOCK_SIZE_K,
    )
