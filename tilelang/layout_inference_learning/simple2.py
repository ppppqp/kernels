import torch
import tilelang
import tilelang.language as T
from tilelang import tvm
from pathlib import Path


def dump_lowered_artifact(artifact, dump_dir: str, name: str) -> None:
    if dump_dir is None:
        return
    root = Path(dump_dir)
    root.mkdir(parents=True, exist_ok=True)
    kernel_path = root / f"{name}.cu"
    host_path = root / f"{name}_host_ir.txt"
    kernel_path.write_text(artifact.kernel_source)
    host_path.write_text(str(artifact.host_mod))
    print(f"dumped source-only kernel: {kernel_path}")
    print(f"dumped source-only host IR: {host_path}")


@tilelang.jit
def grouped_attention(Q, K, BLOCK_B: int, BLOCK_S: int, THREADS: int):
    QB, B, S = T.const("QB, B, S")
    dtype = T.float32
    Q: T.Tensor((QB, S))
    K: T.Tensor((B,))
    O = T.empty()
    head_num = QB // B
    with T.Kernel(B // BLOCK_B, threads=THREADS) as pid_b:
        Q_local = T.alloc_fragment((BLOCK_B * head_num, BLOCK_S), dtype)
        # T.annotate_layout(
        #     {
        #         Q_local: T.Fragment(
        #             (BLOCK_B * head_num, BLOCK_S),
        #             lambda i, j: (i % THREADS, (i // THREADS) * BLOCK_S + j),
        #         )
        #     }
        # )
        cur_max_QK = T.alloc_fragment([head_num, BLOCK_B], dtype)
        h = 0
        for h in T.Serial(head_num):
            T.copy(
                Q[h * B + pid_b * BLOCK_B, BLOCK_S],
                Q_local[h * BLOCK_B, :],
            )
        Q_local_reshaped = T.reshape(Q_local, (head_num, BLOCK_B, BLOCK_S))

        for h, i, j in T.Parallel(head_num, BLOCK_B, BLOCK_S):
            Q_local_reshaped[h, i, j] = cur_max_QK[h, i]
        for h, i in T.Parallel(head_num, BLOCK_B):
            cur_max_QK[h, i] = Q_local_reshaped[h, i, 0]
    return O


def main(*args, **kwargs):
    q = torch.empty((256, 256))
    k = torch.empty((128,))
    BLOCK_B = 16
    BLOCK_S = 32
    THREADS = 64  # if head size * BLOCK_B, happened to work
    # grouped_attention.compile(q, k, BLOCK_B, BLOCK_S, THREADS)
    # tir = grouped_attention.get_tir(q, k, BLOCK_B, BLOCK_S, THREADS)
    target_obj = tvm.target.Target("cuda")
    with target_obj:
        tir = grouped_attention.get_tir(q, k, BLOCK_B, BLOCK_S, THREADS)
        artifact = tilelang.lower(tir, target=target_obj)
        print(tir)
        dump_lowered_artifact(artifact, dump_dir="./dump", name="simple2")


main()
