import tilelang
import tilelang.language as T


@tilelang.jit
def symbolic_sliced_copy_reshape_kernel(
    Q, K, BLOCK_B: int, BLOCK_S: int, THREADS: int, ANNOTATE: bool
):
    QB, B, S = T.const("QB, B, S")
    dtype = T.float32
    # the only diff is that shape of Q is QB, S instead of B, S
    Q: T.Tensor((QB, S), dtype)
    K: T.Tensor((B, S), dtype)
    O = T.empty((QB, S), dtype)

    head_num = QB // B
    with T.Kernel(B // BLOCK_B, threads=THREADS) as pid_b:
        Q_local = T.alloc_fragment((BLOCK_B * head_num, BLOCK_S), dtype)
        K_local = T.alloc_fragment((BLOCK_B, BLOCK_S), dtype)
        cur_QK = T.alloc_fragment([head_num, BLOCK_B, BLOCK_S], dtype)
        lse = T.alloc_fragment([head_num, BLOCK_B], dtype)

        T.fill(lse, -T.infinity(dtype))

        for s_blk_id in T.Serial(S // BLOCK_S):
            for h in T.Serial(head_num):
                T.copy(
                    Q[h * B + pid_b * BLOCK_B, s_blk_id * BLOCK_S],
                    Q_local[h * BLOCK_B, :],
                )
            Q_local_reshaped = T.reshape(Q_local, (head_num, BLOCK_B, BLOCK_S))

            for h, i, j in T.Parallel(head_num, BLOCK_B, BLOCK_S):
                cur_QK[h, i, j] = Q_local_reshaped[h, i, j] * K_local[i, j]
    return O


@tilelang.jit
def symbolic_sliced_copy_reshape_kernel_2(
    Q,
    K,
    BLOCK_B: int,
    BLOCK_S: int,
    THREADS: int,
    ANNOTATE: bool,
):
    dtype = T.float32
    QB, B, S = T.const("QB, B, S")
    Q: T.Tensor((QB, S), dtype)
    K: T.Tensor((B, S), dtype)
    Y = T.empty((QB, S), dtype)

    head_num = QB // B
    rows = BLOCK_B * head_num

    with T.Kernel(B // BLOCK_B, threads=THREADS) as pid_b:
        local = T.alloc_fragment((rows, BLOCK_S), dtype)
        if ANNOTATE:
            T.annotate_layout(
                {
                    local: T.Fragment(
                        (rows, BLOCK_S),
                        lambda i, j: (i % THREADS, (i // THREADS) * BLOCK_S + j),
                    )
                }
            )

        for s_blk_id in T.Serial(S // BLOCK_S):
            for h in T.Serial(head_num):
                T.copy(
                    Q[h * B + pid_b * BLOCK_B, s_blk_id * BLOCK_S],
                    local[h * BLOCK_B, :],
                )

            local_reshaped = T.reshape(local, (head_num, BLOCK_B, BLOCK_S))

            for h, i, j in T.Parallel(head_num, BLOCK_B, BLOCK_S):
                local_reshaped[h, i, j] = (
                    local_reshaped[h, i, j]
                    + K[pid_b * BLOCK_B + i, s_blk_id * BLOCK_S + j]
                )

            for h in T.Serial(head_num):
                T.copy(
                    local[h * BLOCK_B, :],
                    Y[h * B + pid_b * BLOCK_B, s_blk_id * BLOCK_S],
                )

    return Y


@tilelang.jit
def sliced_copy_reshape_kernel(
    X, BLOCK_B: int, BLOCK_S: int, HEADS: int, THREADS: int, ANNOTATE: bool
):
    dtype = T.float32
    rows = BLOCK_B * HEADS
    X: T.Tensor((rows, BLOCK_S), dtype)
    Y = T.empty((rows, BLOCK_S), dtype)

    with T.Kernel(1, threads=THREADS):
        local = T.alloc_fragment((rows, BLOCK_S), dtype)
        if ANNOTATE:
            T.annotate_layout(
                {
                    local: T.Fragment(
                        (rows, BLOCK_S),
                        lambda i, j: (i % THREADS, (i // THREADS) * BLOCK_S + j),
                    )
                }
            )

        for h in T.Serial(HEADS):
            T.copy(
                X[h * BLOCK_B, 0],
                local[h * BLOCK_B, :],
            )

        local_reshaped = T.reshape(local, (HEADS, BLOCK_B, BLOCK_S))

        for h, i, j in T.Parallel(HEADS, BLOCK_B, BLOCK_S):
            local_reshaped[h, i, j] = local_reshaped[h, i, j] + T.float32(1.0)

        for h in T.Serial(HEADS):
            T.copy(
                local[h * BLOCK_B, :],
                Y[h * BLOCK_B, 0],
            )

    return Y


@tilelang.jit
def copy_reshape_kernel(
    X, BLOCK_B: int, BLOCK_S: int, HEADS: int, THREADS: int, ANNOTATE: bool
):
    dtype = T.float32
    rows = BLOCK_B * HEADS
    X: T.Tensor((rows, BLOCK_S), dtype)
    Y = T.empty((rows, BLOCK_S), dtype)

    with T.Kernel(1, threads=THREADS):
        local = T.alloc_fragment((rows, BLOCK_S), dtype)
        if ANNOTATE:
            T.annotate_layout(
                {
                    local: T.Fragment(
                        (rows, BLOCK_S),
                        lambda i, j: (i % THREADS, (i // THREADS) * BLOCK_S + j),
                    )
                }
            )

        T.copy(X[0, 0], local)
        local_reshaped = T.reshape(local, (HEADS, BLOCK_B, BLOCK_S))

        for h, i, j in T.Parallel(HEADS, BLOCK_B, BLOCK_S):
            local_reshaped[h, i, j] = local_reshaped[h, i, j] + T.float32(1.0)

        T.copy(local, Y[0, 0])

    return Y


@tilelang.jit
def copy_flat_kernel(X, BLOCK_B: int, BLOCK_S: int, HEADS: int, THREADS: int):
    dtype = T.float32
    rows = BLOCK_B * HEADS
    X: T.Tensor((rows, BLOCK_S), dtype)
    Y = T.empty((rows, BLOCK_S), dtype)

    with T.Kernel(1, threads=THREADS):
        local = T.alloc_fragment((rows, BLOCK_S), dtype)
        T.annotate_layout(
            {
                local: T.Fragment(
                    (rows, BLOCK_S),
                    lambda i, j: (i % THREADS, (i // THREADS) * BLOCK_S + j),
                )
            }
        )

        T.copy(X[0, 0], local)
        for i, j in T.Parallel(rows, BLOCK_S):
            local[i, j] = local[i, j] + T.float32(1.0)
        T.copy(local, Y[0, 0])

    return Y


@tilelang.jit
def elementwise_affine_kernel(X, BLOCK_B: int, BLOCK_S: int, HEADS: int, THREADS: int):
    dtype = T.float32
    rows = BLOCK_B * HEADS
    X: T.Tensor((rows, BLOCK_S), dtype)
    Y = T.empty((rows, BLOCK_S), dtype)

    with T.Kernel(1, threads=THREADS):
        for h, i, j in T.Parallel(HEADS, BLOCK_B, BLOCK_S):
            row = h * BLOCK_B + i
            Y[row, j] = X[row, j] + T.float32(1.0)

    return Y
