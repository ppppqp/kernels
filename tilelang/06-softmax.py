"""
Puzzle 06: Softmax
==============
Softmax is the first fundermental NN operator we learn in this tutorial.

Category: ["official"]
Difficulty: ["medium"]
"""

import tilelang
import tilelang.language as T
import torch

from common.utils import bench_puzzle, test_puzzle

r"""
Softmax operator goes a little beyond the reduce sum. We also need to use serial loop to
accumulate the summation. And we need to perform an element-wise exp operation on each element
at the same time.

Note that softmax needs to be computed in numerically stable form as in Python. To achieve this,
we need to subtract the maximum value of each row from all elements in that row
before applying the exponential function.

HINT:
1. Use `T.fill` to set the initial value of the buffer. `T.clear` sets all elements to zero by
default, which may not be what you want.

3.We recommend not using `T.exp` but instead using `T.exp2`. You need the identity

.. math::
    \exp(x) = 2^{\log_2(e) x}

The constant log2_e is provided.

BONUS: Use "Online Softmax" algorithm to implement optimized softmax. This is also a core idea of
FlashAttention algorithm. Through this, we can implement softmax with only two passes / loops.

06-1: Softmax.

Inputs:
    A: Tensor([N, M], float32)  # input tensor
    N: int   # size of the tensor. 1 <= N <= 4096
    M: int   # size of the tensor. 1 <= M <= 16384

Output:
    B: Tensor([N, M], float16)  # output tensor

Intermediates:
    MAX: float32  # max value of each row
    SUM: float32  # summation of each row

Definition:
    for i in range(N):
        SUM = 0
        MAX = -inf
        for j in range(M):
            MAX = max(A[i, j], MAX)
        for j in range(M):
            B[i, j] = exp(A[i, j] - MAX)
            SUM += B[i, j]
        for j in range(M):
            B[i, j] /= SUM
"""


def ref_softmax(A: torch.Tensor):
    assert len(A.shape) == 2
    assert A.dtype == torch.float32
    return torch.softmax(A, dim=1)


@tilelang.jit(
    pass_configs={
        tilelang.PassConfigKey.TL_DISABLE_WARP_SPECIALIZED: True,
        tilelang.PassConfigKey.TL_DISABLE_TMA_LOWER: True,
    },
)
def tl_softmax(A, BLOCK_N: int, BLOCK_M: int):
    log2_e = 1.44269504
    N, M = T.const("N, M")
    dtype = T.float32
    A: T.Tensor((N, M), dtype)
    B = T.empty((N, M), dtype)


    # TODO: Implement this function
    with T.Kernel(N // BLOCK_N, threads=256) as bx:

        A_local = T.alloc_fragment((BLOCK_N, BLOCK_M), dtype)
        B_local = T.alloc_fragment((BLOCK_N, BLOCK_M), dtype)
        cur_max = T.alloc_fragment((BLOCK_N,), dtype)
        cur_exp = T.alloc_fragment((BLOCK_N, BLOCK_M), dtype)
        cur_sum = T.alloc_fragment((BLOCK_N,), dtype)

        # log sum exponential
        lse = T.alloc_fragment((BLOCK_N), dtype)
        T.fill(lse, -T.infinity(dtype))

        for m_idx in T.Serial(M // BLOCK_M):
            T.copy(A[bx * BLOCK_N, m_idx * BLOCK_M], A_local)
            T.reduce_max(A_local, cur_max, dim=1, clear=True)
            for i, j in T.Parallel(BLOCK_N, BLOCK_M):
                # minus current max only for numerical stability
                cur_exp[i, j] = T.exp2(A_local[i, j] * log2_e - cur_max[i] * log2_e)

            T.reduce_sum(cur_exp, cur_sum, dim=1, clear=True)
            # assume i being column (dimension m)
            # in previous block, we computed log(sum(exp(x_{0..i}))), now want to add sum(exp(x_{i+1} - cur_max))
            # 1. first do exp(lse) to unwrap: sum(exp(x_{0..i}))
            # 2. divide by cur_max, so it's sum(exp(x_{0..i} - cur_max))
            # 3. now we can do the addition to be sum(exp(x_{0..i+1} - cur_max))
            # 4. applying back the log: log(sum(exp(x_{0..i+1} - cur_max)))
            # 5. offset the cur_max: log(sum(exp(x_{0..i+1} - cur_max)) + cur_max
            #                      = log(sum(exp(x_{0..i+1} - cur_max)) * exp(cur_max))
            #                      = log(sum(exp(x_{0..i+1} - cur_max + cur_max)))
            for nn in T.Parallel(BLOCK_N):
                lse[nn] = T.log2(T.exp2(lse[nn] - cur_max[nn] * log2_e) + cur_sum[nn]) + cur_max[nn] * log2_e
        
        for m_idx in T.Serial(M // BLOCK_M):
            T.copy(A[bx * BLOCK_N, m_idx * BLOCK_M], A_local)
            for i, j in T.Parallel(BLOCK_N, BLOCK_M):
                B_local[i, j] = T.exp2(A_local[i, j] * log2_e - lse[i])
        
            T.copy(B_local, B[bx * BLOCK_N, m_idx * BLOCK_M])

            
            

   
    return B


def run_softmax():
    print("\n=== Softmax ===\n")
    N = 4096
    M = 16384
    BLOCK_N = 16
    BLOCK_M = 256
    test_puzzle(
        tl_softmax,
        ref_softmax,
        {"N": N, "M": M, "BLOCK_N": BLOCK_N, "BLOCK_M": BLOCK_M},
    )
    bench_puzzle(
        tl_softmax,
        ref_softmax,
        {"N": N, "M": M, "BLOCK_N": BLOCK_N, "BLOCK_M": BLOCK_M},
        bench_torch=True,
    )


if __name__ == "__main__":
    run_softmax()