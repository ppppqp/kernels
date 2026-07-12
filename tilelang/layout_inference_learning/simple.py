import tilelang
import tilelang.language as T
import tvm


@tilelang.jit
def inner_serial_var_in_thread_mapping(A):
    A: T.Tensor((8, 4), T.float32)
    O = T.empty((8, 4), T.float32)

    with T.Kernel(1, threads=32):
        frag = T.alloc_fragment((8, 4), T.float32)

        def frag_layout(i, k):
            # Force each logical element onto its own thread.
            #
            #   frag[i, k] -> thread = i * 4 + k
            #                 local  = 0
            #
            # This is the key difference from the previous example.
            return i * 4 + k, 0

        T.annotate_layout(
            {frag: T.Fragment((8, 4), forward_fn=lambda i, k: (i * 4 + k, 0))}
        )

        # Establish/populate frag with the annotated layout.
        for i, k in T.Parallel(8, 4):
            frag[i, k] = A[i, k]

        # Problematic loop.
        #
        # The parallel loop variables are only:
        #   i
        #
        # But the source fragment layout says:
        #   thread = i * 4 + k
        #
        # After substitution, ComputeLoopLayoutFromBuffer gets:
        #   loop_var_to_thread = i * 4 + k
        #
        # k is from the inner T.Serial loop, so PostOrderVisit should find k in
        # inner_vars_ and throw LayoutConflictException.
        for i in T.Parallel(8):
            for k in T.Serial(4):
                O[i, k] = frag[i, k]

    return O


def main():
    target = tvm.target.Target("cuda")

    with target:
        tir = inner_serial_var_in_thread_mapping.get_tir()
        tilelang.lower(tir, target=target)
        print("compiled successfully")


if __name__ == "__main__":
    main()
