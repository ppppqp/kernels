import torch
import tilelang
import tilelang.language as T


@tilelang.jit
def grouped_attention(Q, K, BLOCK_B: int, BLOCK_S: int):
    QB, B, S = T.const("QB, B, S")
    dtype = T.float32
    Q: T.Tensor((QB, S))
    K: T.Tensor((B,))
    O = T.empty()
    head_num = QB // B
    with T.Kernel(B // BLOCK_B, threads=64) as pid_b:
        Q_local = T.alloc_fragment((BLOCK_B * head_num, BLOCK_S), dtype)
        cur_max_QK = T.alloc_fragment([head_num, BLOCK_B], dtype)
        h = 0
        for h in T.Serial(head_num):
            T.copy(
                Q[h * B + pid_b * BLOCK_B, BLOCK_S],
                Q_local[h * BLOCK_B, :],
            )
        Q_local_reshaped = T.reshape(Q_local, (head_num, BLOCK_B, BLOCK_S))
        for h, i, j in T.Parallel(head_num, BLOCK_B, BLOCK_S):
            Q_local_reshaped[h, i, j] - cur_max_QK[h, i]
        for h, i in T.Parallel(head_num, BLOCK_B):
            cur_max_QK[h, i]
    return O


def main(*args, **kwargs):
    q = torch.empty((256, 256))
    k = torch.empty((128,))
    BLOCK_B = 16
    BLOCK_S = 32
    grouped_attention.compile(q, k, BLOCK_B, BLOCK_S)


main()
