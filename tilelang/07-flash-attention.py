"""
Puzzle 07: Scalar FlashAttention
==============
From softmax to FlashAttention, we just need some computation.

Category: ["official"]
Difficulty: ["medium"]
"""

import tilelang
import tilelang.language as T
import torch

from common.utils import bench_puzzle, test_puzzle

"""
Now we have conquered softmax / online softmax, we can now implement one of the most important
operator in LLMs: FlashAttention.

To ensure a progressive learning experience, we will implement a scalar version of FlashAttention.
And we also remove the multi-head attention part. So in total we only have two dimensions: batch
size B and sequence length S, which are aligned with N, M in the previous puzzle. After such
simplification, you will find we are not so far from the FlashAttention algorithm. And with
TileLang, we can easily extend it to the full FlashAttention.

06-1: Simplified Scalar Flash Attention.

Inputs:
    Q: Tensor([B, S], float32)  # input tensor
    K: Tensor([B, S], float32)  # input tensor
    V: Tensor([B, S], float32)  # input tensor
    B: int   # batch size dimension. 1 <= B <= 256
    S: int   # sequence length dimension. 1 <= S <= 16384

Output:
    O: Tensor([B, S], float32)  # output tensor

Intermediates:
    MAX: float32  # max value of each row
    SUM: float32  # summation of each row
    QK: Tensor([B, S], float32)  # results of q*k
    P:  Tensor([B, S], float32)  # results of softmax(q*k) (not divided by summation).

Definition:
    for i in range(B):
        SUM = 0
        MAX = -inf
        for j in range(S):
            QK[i, j] = Q[i, j] * K[i, j]
            MAX = max(QK[i, j], MAX)
        for j in range(S):
            P[i, j] = exp(QK[i, j] - MAX)
            SUM += P[i, j]
        for j in range(M):
            O[i, j] = P[i, j] / SUM * V[i, j]
"""


def ref_scalar_flash_attn(Q: torch.Tensor, K: torch.Tensor, V: torch.Tensor):
    assert len(Q.shape) == 2
    assert len(K.shape) == 2
    assert len(V.shape) == 2
    assert Q.shape[0] == K.shape[0] == V.shape[0]  # B
    assert Q.shape[1] == K.shape[1] == V.shape[1]  # S
    assert Q.dtype == K.dtype == V.dtype == torch.float32
    return torch.softmax(Q * K, dim=1).mul_(V)


@tilelang.jit(
    pass_configs={
        tilelang.PassConfigKey.TL_DISABLE_WARP_SPECIALIZED: True,
        tilelang.PassConfigKey.TL_DISABLE_TMA_LOWER: True,
    },
)
def tl_scalar_flash_attn(Q, K, V, BLOCK_B: int, BLOCK_S: int):
    log2_e = 1.44269504
    B, S = T.const("B, S")
    dtype = T.float32
    Q: T.Tensor((B, S), dtype)
    K: T.Tensor((B, S), dtype)
    V: T.Tensor((B, S), dtype)
    O = T.empty((B, S), dtype)

    # TODO: Implement this function
    with T.Kernel(B // BLOCK_B, threads=256) as bx, by:
        local_Q = T.alloc_fragment([BLOCK_B, BLOCK_S], dtype)
        local_K = T.alloc_fragment([BLOCK_B, BLOCK_S], dtype)
        local_V = T.alloc_fragment([BLOCK_B, BLOCK_S], dtype)
        local_O = T.alloc_fragment([BLOCK_B, BLOCK_S], dtype)
        local_QK = T.alloc_fragment([BLOCK_B, BLOCK_B], dtype)
        # local_QKV = T.alloc_fragment([BLOCK_B, BLOCK_S], dtype)

        current_max = T.alloc_fragment([BLOCK_B], dtype)
        new_max = T.alloc_fragment([BLOCK_B], dtype)
        current_sum = T.alloc_fragment([BLOCK_B], dtype)
        alpha = T.alloc_fragment([BLOCK_B], dtype)
        incre = T.alloc_fragment([BLOCK_B], dtype)
        weights = T.alloc_fragment([BLOCK_B, BLOCK_B], dtype)

        for m_idx in T.Serial(S // BLOCK_S):
            T.copy(Q[bx * BLOCK_B, m_idx * BLOCK_S], local_Q)
            T.copy(K[bx * BLOCK_B, m_idx * BLOCK_S], local_K)
            T.gemm(local_Q, local_K, local_QK, transpose_B = True, clear_accum=True)
            # softmax
            T.copy(current_max, new_max)
            T.reduce_max(local_QK, new_max, clear=True)
            # alpha = T.exp2(current_max - new_max) * log2_e
            # weights = T.exp2(local_QK - new_max) * log2_e
            # T.reduce_sum(weights, incre, clear=True)
            
            # current_sum = current_sum * alpha + incre
            for nn in T.Parallel(BLOCK_B):
                diff = current_max[nn] - new_max[nn]
                alpha[nn] = T.exp2(diff) * log2_e
                current_sum[nn] = current_sum[nn] * alpha[nn] + incre[nn]
            for nn, mm in T.Parallel(BLOCK_B, BLOCK_S):
                weights[nn, mm] = T.exp2(local_QK[nn, mm] - new_max[nn]) * log2_e

            T.reduce_sum(weights, incre, clear=True)
            current_max = T.copy(new_max, current_max)

            # with V
            # the problem here is the V block computed here is not the final result (sum is not finalized)
            # so we can't directly dump local_O block to O 
            # we are looping through K's S dim and V's S dim concurrently
            # how Triton kernel solves this is that each thread only care about one V block and recompute all things above
            # which we can actually reuse
            # that's why we should compute the block-reusable lse first (like official example did)
            for nn in T.Parallel(BLOCK_B, BLOCK_S):
                local_O[nn, mm] = local_O[nn, mm] * alpha[nn]
            T.gemm(weights, local_V, local_O, clear_accum=False)

            
        for nn, mm in T.Parallel(BLOCK_B, BLOCK_S):
            local_O[nn, mm] = local_O[nn, mm] / current_sum[nn]
        T.copy(local_O, O[bx * BLOCK_B, :])


    return O


def run_scalar_flash_attn():
    print("\n=== Scalar Flash Attention ===\n")
    B = 256
    S = 16384
    BLOCK_B = 16
    BLOCK_S = 128
    test_puzzle(
        tl_scalar_flash_attn,
        ref_scalar_flash_attn,
        {"B": B, "S": S, "BLOCK_B": BLOCK_B, "BLOCK_S": BLOCK_S},
    )
    bench_puzzle(
        tl_scalar_flash_attn,
        ref_scalar_flash_attn,
        {"B": B, "S": S, "BLOCK_B": BLOCK_B, "BLOCK_S": BLOCK_S},
        bench_torch=True,
    )


if __name__ == "__main__":
    run_scalar_flash_attn()