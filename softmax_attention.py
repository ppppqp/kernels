import torch
import triton
import triton.language as tl


@triton.jit
def softmax_attention(
    Q,
    K,
    V,
    M,
    N,
    d,
    out,
    Q_stride_M,
    Q_stride_d,
    K_stride_N,
    K_stride_d,
    V_stride_N,
    V_stride_d,
    out_stride_M,
    out_stride_d,
    M_BLOCK: tl.constexpr,
    N_BLOCK: tl.constexpr,
    d_BLOCK: tl.constexpr,
):
    # tiled gemm
    row_pid = tl.program_id(0)
    col_pid = tl.program_id(1)
    m_indexes = row_pid * M_BLOCK + tl.arange(0, M_BLOCK)

    # this col_indexes is for final V multiplication, not the initial gemm
    v_indexes = col_pid * d_BLOCK + tl.arange(0, d_BLOCK)
    m_mask = m_indexes < M
    v_mask = v_indexes < d

    running_sum = tl.zeros([M_BLOCK], dtype=tl.float32)
    running_max = tl.full([M_BLOCK], float("-inf"), dtype=tl.float32)
    out_acc = tl.zeros([M_BLOCK, d_BLOCK], dtype=tl.float32)
    scale = 1 / tl.sqrt(d + 0.0)

    # we need to iterate through all columns (compute one complete row) to get the softmax
    # but this can be unrolled when compiled
    for nn in tl.range(0, tl.cdiv(N, N_BLOCK)):
        n_indexes = nn * N_BLOCK + tl.arange(0, N_BLOCK)
        n_mask = n_indexes < N
        # block gemm accumulator
        acc = tl.zeros((M_BLOCK, N_BLOCK), dtype=tl.float32)
        # this dd is for QK inner dimension tiling
        for dd in tl.range(0, tl.cdiv(d, d_BLOCK)):
            d_indexes = dd * d_BLOCK + tl.arange(0, d_BLOCK)
            d_mask = d_indexes < d
            M_offsets = (
                Q_stride_M * m_indexes[:, None] + Q_stride_d * d_indexes[None, :]
            )
            # do a transpose on the fly
            # this might happen to be efficient for since both are row major right not lol
            K_offsets = (
                K_stride_N * n_indexes[:, None] + K_stride_d * d_indexes[None, :]
            )

            M_mask = m_mask[:, None] & d_mask[None, :]
            K_mask = n_mask[:, None] & d_mask[None, :]
            # load
            cur_q = tl.load(Q + M_offsets, mask=M_mask, other=0.0)
            cur_k = tl.load(K + K_offsets, mask=K_mask, other=0.0)

            # partial sum, normalization on the fly to avoid overflow
            acc += tl.dot(cur_q, tl.trans(cur_k)) * scale
        # acc is an M_BLOCK * N_BLOCK gemm result
        # mask the overflow to -inf so that the alpha can be computed correctly
        acc = tl.where(m_mask[:, None] & n_mask[None, :], acc, float("-inf"))
        # run max on column, getting M_BLOCK * 1 maximum per row
        current_max = tl.max(acc, axis=-1)
        # compare with the running max and update the running_max
        new_running_max = tl.maximum(current_max, running_max)
        # alpha on updating the running_sum given the update on running_max
        alpha = tl.exp(running_max - new_running_max)  # the diff

        # compute increment to running_sum on this N_BLOCK
        weights = tl.exp(acc - new_running_max[:, None])
        # directly sum because all the denomiators are new_running_max
        incre = tl.sum(weights, axis=1)

        # running_sum = running_sum * alpha + incre
        running_sum = tl.fma(running_sum, alpha, incre)
        running_max = new_running_max

        # gemm with V.
        # N is now the inner dimension (M_BLOCK, N_BLOCK) * (N_BLOCK * d_BLOCK)
        v_offsets = n_indexes[:, None] * V_stride_N + v_indexes[None, :] * V_stride_d
        V_mask = n_mask[:, None] & v_mask[None, :]
        cur_v = tl.load(V + v_offsets, mask=V_mask)

        # dot the weights now, will normalize with running_sum later
        weighted_v = tl.dot(weights, cur_v)
        # the accumulator for gemm with V, also need to correct the error
        out_acc = tl.fma(out_acc, alpha[:, None], weighted_v)

    # the running sum is correct now
    out_acc /= running_sum[:, None]

    out_offsets = m_indexes[:, None] * out_stride_M + v_indexes[None, :] * out_stride_d
    out_mask = m_mask[:, None] & v_mask[None, :]
    tl.store(out + out_offsets, out_acc, mask=out_mask)


# Q, K, V, output are tensors on the GPU
def solve(
    Q: torch.Tensor,
    K: torch.Tensor,
    V: torch.Tensor,
    output: torch.Tensor,
    M: int,
    N: int,
    d: int,
):
    M_BLOCK = 16
    d_BLOCK = 128
    N_BLOCK = 64

    grid = (triton.cdiv(M, M_BLOCK), triton.cdiv(d, d_BLOCK))
    Q_stride_M, Q_stride_d = Q.stride()
    K_stride_N, K_stride_d = K.stride()
    V_stride_N, V_stride_d = V.stride()
    out_stride_M, out_stride_d = output.stride()

    softmax_attention[grid](
        Q=Q,
        K=K,
        V=V,
        out=output,
        M=M,
        N=N,
        d=d,
        Q_stride_M=Q_stride_M,
        Q_stride_d=Q_stride_d,
        K_stride_N=K_stride_N,
        K_stride_d=K_stride_d,
        V_stride_N=V_stride_N,
        V_stride_d=V_stride_d,
        out_stride_M=out_stride_M,
        out_stride_d=out_stride_d,
        M_BLOCK=M_BLOCK,
        d_BLOCK=d_BLOCK,
        N_BLOCK=N_BLOCK,
    )
