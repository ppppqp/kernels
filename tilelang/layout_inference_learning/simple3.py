import tilelang
import tilelang.language as T
from tilelang import tvm


@tilelang.jit
def plan_then_buffer_layout():
    @T.prim_func
    def main(
        A: T.Tensor((4, 16), T.float32),
        B: T.Tensor((4, 16), T.float32),
    ):
        with T.Kernel(1, threads=64):
            fragment = T.alloc_fragment((4, 16), T.float32)

            # No fragment has a known layout yet, so this loop uses
            # ComputePlanCandidate. With 64 iterations and 64 threads, the
            # expected mapping is:
            #
            #   flat(row, col) = row * 16 + col
            #   thread         = flat
            #   local index    = 0
            for row, col in T.Parallel(4, 16):
                fragment[row, col] = A[row, col]

            # The fragment layout inferred above is now known. This loop reads
            # it through a non-identity, three-variable access:
            #
            #   fragment[group * 2 + row_in_group, col]
            #
            # ComputeLoopLayoutFromBuffer substitutes that access into the
            # fragment's thread expression:
            #
            #   thread = (group * 2 + row_in_group) * 16 + col
            for group, row_in_group, col in T.Parallel(2, 2, 16):
                row = group * 2 + row_in_group
                B[row, col] = fragment[row, col] + T.float32(1)

    return main


if __name__ == "__main__":
    target = tvm.target.Target({"kind": "cuda", "arch": "sm_80"})
    tir = plan_then_buffer_layout.get_tir()
    with target:
        artifact = tilelang.lower(tir, target=target, enable_device_compile=False)

    print("Input TIR:")
    print(tir)
    print("\nGenerated CUDA:")
    print(artifact.kernel_source)
