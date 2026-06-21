import torch
import tilelang
import tilelang.language as T


@tilelang.jit
def grouped_attention(Q, K, V, BLOCK_B: int, BLOCK_S: int):
    log2_e = 1.44269504
    QB, B, S = T.const("QB, B, S")
    dtype = T.float32
    # the only diff is that shape of Q is QB, S instead of B, S
    Q: T.Tensor((QB, S), dtype)
    K: T.Tensor((B, S), dtype)
    V: T.Tensor((B, S), dtype)
    O = T.empty((QB, S), dtype)

    head_num = QB // B
    with T.Kernel(B // BLOCK_B, threads=256) as pid_b:
        Q_local = T.alloc_fragment((BLOCK_B * head_num, BLOCK_S), dtype)
        K_local = T.alloc_fragment((BLOCK_B, BLOCK_S), dtype)
        V_local = T.alloc_fragment((BLOCK_B, BLOCK_S), dtype)
        O_local = T.alloc_fragment((BLOCK_B * head_num, BLOCK_S), dtype)

        cur_QK = T.alloc_fragment([head_num, BLOCK_B, BLOCK_S], dtype)
        cur_exp_QK = T.alloc_fragment([head_num, BLOCK_B, BLOCK_S], dtype)
        cur_max_QK = T.alloc_fragment([head_num, BLOCK_B], dtype)
        cur_sum_exp_QK = T.alloc_fragment([head_num, BLOCK_B], dtype)

        lse = T.alloc_fragment([head_num, BLOCK_B], dtype)

        T.fill(lse, -T.infinity(dtype))

        # The first loop use an online algorithm to compute LSE.
        for s_blk_id in T.Serial(S // BLOCK_S):
            # copy with reshape?
            for h in T.Serial(head_num):
                T.copy(
                    Q[h * B + pid_b * BLOCK_B, s_blk_id * BLOCK_S],
                    Q_local[h * BLOCK_B, :],
                )
            Q_local_reshaped = T.reshape(Q_local, (head_num, BLOCK_B, BLOCK_S))
            # T.copy(K[pid_b * BLOCK_B, s_blk_id * BLOCK_S], K_local)

            for h, i, j in T.Parallel(head_num, BLOCK_B, BLOCK_S):
                cur_QK[h, i, j] = Q_local_reshaped[h, i, j] * K_local[i, j]

            T.reduce_max(cur_QK, cur_max_QK, dim=2, clear=True)

            for h, i, j in T.Parallel(head_num, BLOCK_B, BLOCK_S):
                cur_exp_QK[h, i, j] = T.exp2(
                    cur_QK[h, i, j] * log2_e - cur_max_QK[h, i] * log2_e
                )

            T.reduce_sum(cur_exp_QK, cur_sum_exp_QK, dim=2, clear=True)

            for h, i in T.Parallel(head_num, BLOCK_B):
                lse[h, i] = cur_max_QK[h, i] * log2_e + T.log2(
                    T.exp2(lse[h, i] - cur_max_QK[h, i] * log2_e) + cur_sum_exp_QK[h, i]
                )

        # The second loop use LSE to get the final output.
        # TODO: improve the efficiency here. Maybe pipeline it?
        for s_blk_id in T.Serial(S // BLOCK_S):
            for h in T.Serial(head_num):
                T.copy(
                    Q[h * B + pid_b * BLOCK_B, s_blk_id * BLOCK_S],
                    Q_local[h * BLOCK_B, :],
                )
            Q_local_reshaped = T.reshape(Q_local, (head_num, BLOCK_B, BLOCK_S))
            T.copy(K[pid_b * BLOCK_B, s_blk_id * BLOCK_S], K_local)
            T.copy(V[pid_b * BLOCK_B, s_blk_id * BLOCK_S], V_local)

            for h, i, j in T.Parallel(head_num, BLOCK_B, BLOCK_S):
                O_local[h * BLOCK_B + i, j] = (
                    T.exp2(
                        Q_local_reshaped[h, i, j] * K_local[i, j] * log2_e - lse[h, i]
                    )
                    * V_local[i, j]
                )
            for h in T.Serial(head_num):
                T.copy(
                    O_local[h * BLOCK_B, :],
                    O[h * B + pid_b * BLOCK_B, s_blk_id * BLOCK_S],
                )

    return O


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    q = torch.empty((512, 512), device=device, dtype=torch.float32)
    k = torch.empty((256, 512), device=device, dtype=torch.float32)
    v = torch.empty((256, 512), device=device, dtype=torch.float32)
    BLOCK_B = 16
    BLOCK_S = 128
    # The failure occurs during TileLang lowering/layout inference, before a
    # valid CUDA device is required.
    grouped_attention.compile(q, k, v, BLOCK_B, BLOCK_S)


if __name__ == "__main__":
    main()
