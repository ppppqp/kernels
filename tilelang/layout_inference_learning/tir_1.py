# from tvm.script import tirx as T


@T.prim_func
def grouped_attention(Q: T.handle, K: T.handle, O: T.Buffer((), "float32")):
    T.func_attr({"tilelang_out_idx": [-1]})
    Q_1 = T.match_buffer(Q, (256, 256), strides=(256, 1))
    K_1 = T.match_buffer(K, (128,), strides=(1,))
    # with T.sblock("root"):
    bx = T.launch_thread("blockIdx.x", 8)
    tx = T.launch_thread("threadIdx.x", 64)
    ty = T.launch_thread("threadIdx.y", 1)
    tz = T.launch_thread("threadIdx.z", 1)
    with T.sblock("tilelang_root"):
        T.reads()
        T.writes()
        Q_local_reshaped = T.handle("float32", "local.fragment")
        T.sblock_attr({"layout_map": {Q_local_reshaped: metadata["tl.Fragment"][0]}})
        Q_local = T.sblock_alloc_buffer(
            (32, 32), data=Q_local_reshaped, scope="local.fragment"
        )
        cur_max_QK = T.sblock_alloc_buffer((2, 16), scope="local.fragment")
        for h in range(2):
            T.copy(
                T.region(Q_1[h * 128 + bx * 16, 32], 1, 1, 32),
                T.region(Q_local[h * 16, 0], 2, 1, 32),
            )
        for h in T.parallel(2):
            for i in T.parallel(16):
                for j in T.parallel(32):
                    Q_local_reshaped_1 = T.Buffer(
                        (2, 16, 32),
                        data=Q_local_reshaped,
                        strides=(512, 32, 1),
                        scope="local.fragment",
                    )
                    T.evaluate(Q_local_reshaped_1[h, i, j] - cur_max_QK[h, i])
        for h in T.parallel(2):
            for i in T.parallel(16):
                T.evaluate(cur_max_QK[h, i])
