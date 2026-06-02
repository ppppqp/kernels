import argparse
import json
from pathlib import Path

import torch
import tilelang
import tilelang.language as T
from tilelang.profiler import do_bench


TORCH_DTYPES = {
    "float16": torch.float16,
    "float32": torch.float32,
    "float64": torch.float64,
}


@tilelang.jit
def vector_add_v1(n: int, block_size: int = 256, dtype: str = "float32"):
    @T.prim_func
    def kernel(
        a: T.Tensor((n,), dtype),
        b: T.Tensor((n,), dtype),
        c: T.Tensor((n,), dtype),
    ):
        with T.Kernel(T.ceildiv(n, block_size), threads=block_size) as block_idx:
            for thread_idx in T.Parallel(block_size):
                i = block_idx * block_size + thread_idx
                c[i] = a[i] + b[i]

    return kernel


@tilelang.jit
def vector_add_v2(
    n: int,
    block_size: int = 256,
    num_per_thread: int = 8,
    dtype: str = "float32",
):
    @T.prim_func
    def kernel(
        a: T.Tensor((n,), dtype),
        b: T.Tensor((n,), dtype),
        c: T.Tensor((n,), dtype),
    ):
        with T.Kernel(T.ceildiv(n, block_size * num_per_thread), threads=block_size) as block_idx:
            for thread_idx, j in T.Parallel(block_size, num_per_thread):
                i = (block_idx * block_size + thread_idx) * num_per_thread
                c[i + j] = a[i + j] + b[i + j]

    return kernel


@tilelang.jit
def vector_add_v3(
    n: int,
    block_size: int = 256,
    num_per_thread: int = 8,
    dtype: str = "float32",
):
    @T.prim_func
    def kernel(
        a: T.Tensor((n,), dtype),
        b: T.Tensor((n,), dtype),
        c: T.Tensor((n,), dtype),
    ):
        with T.Kernel(T.ceildiv(n, block_size * num_per_thread), threads=block_size) as block_idx:
            A_register = T.alloc_fragment((block_size * num_per_thread), dtype)
            B_register = T.alloc_fragment((block_size * num_per_thread), dtype)
            C_register = T.alloc_fragment((block_size * num_per_thread), dtype)
            s_start = block_idx * block_size * num_per_thread
            s_end = (block_idx + 1) * block_size * num_per_thread
            T.copy(a[s_start:s_end], A_register)
            T.copy(b[s_start:s_end], B_register)

            for thread_idx, j in T.Parallel(block_size, num_per_thread):
                i = (thread_idx * num_per_thread) + j
                C_register[i] = A_register[i] + B_register[i]

            T.copy(C_register, c[s_start:s_end])

    return kernel


KERNELS = {
    "v1": vector_add_v1,
    "v2": vector_add_v2,
    "v3": vector_add_v3,
}


def parse_args():
    parser = argparse.ArgumentParser(description="Profile TileLang vector-add kernel variants.")
    parser.add_argument(
        "--version",
        choices=KERNELS.keys(),
        default="v3",
        help="Kernel variant to compile and profile.",
    )
    parser.add_argument("--n", type=int, default=1 << 20, help="Number of elements.")
    parser.add_argument("--block-size", type=int, default=256, help="Threads per block.")
    parser.add_argument(
        "--num-per-thread",
        type=int,
        default=8,
        help="Elements per thread for v2 and v3.",
    )
    parser.add_argument(
        "--dtype",
        choices=TORCH_DTYPES.keys(),
        default="float32",
        help="Element dtype.",
    )
    parser.add_argument("--warmup-ms", type=int, default=25, help="Benchmark warmup time.")
    parser.add_argument("--profile-ms", type=int, default=100, help="Benchmark profile time.")
    parser.add_argument(
        "--output-path",
        type=Path,
        default=Path("./vector-add"),
        help="Directory where emitted IR, source, PTX, and SASS artifacts are written.",
    )
    return parser.parse_args()


def compile_kernel(args):
    kernel_factory = KERNELS[args.version]
    kwargs = {
        "n": args.n,
        "block_size": args.block_size,
        "dtype": args.dtype,
    }
    if args.version in {"v2", "v3"}:
        kwargs["num_per_thread"] = args.num_per_thread
    return kernel_factory(**kwargs)


def dump_artifacts(kernel, args):
    output_path = args.output_path
    output_path.mkdir(parents=True, exist_ok=True)

    kernel_name = f"vector_add_{args.version}"
    artifact = getattr(kernel, "artifact", None)
    adapter = getattr(kernel, "adapter", None)
    metadata = {
        "kernel": kernel_name,
        "n": args.n,
        "block_size": args.block_size,
        "num_per_thread": args.num_per_thread if args.version in {"v2", "v3"} else None,
        "dtype": args.dtype,
        "warmup_ms": args.warmup_ms,
        "profile_ms": args.profile_ms,
        "execution_backend": getattr(kernel, "execution_backend", None),
        "has_artifact": artifact is not None,
    }

    (output_path / "metadata.json").write_text(json.dumps(metadata, indent=2) + "\n")
    (output_path / f"{kernel_name}.prim_func.txt").write_text(str(kernel.prim_func))

    host_mod = getattr(artifact, "host_mod", None) or getattr(adapter, "host_mod", None)
    device_mod = getattr(artifact, "device_mod", None) or getattr(adapter, "device_mod", None)
    if host_mod is not None:
        (output_path / f"{kernel_name}.host_ir.py").write_text(host_mod.script())
    else:
        (output_path / f"{kernel_name}.host_ir.not_available.txt").write_text(
            "host_mod was not available for this TileLang execution backend.\n"
        )
    if device_mod is not None:
        (output_path / f"{kernel_name}.device_ir.py").write_text(device_mod.script())
    else:
        (output_path / f"{kernel_name}.device_ir.not_available.txt").write_text(
            "device_mod was not available for this TileLang execution backend.\n"
        )

    (output_path / f"{kernel_name}.cu").write_text(kernel.get_kernel_source(kernel_only=True))
    (output_path / f"{kernel_name}.host.cc").write_text(kernel.get_host_source())
    kernel.export_ptx(str(output_path / f"{kernel_name}.ptx"))

    try:
        kernel.export_sass(str(output_path / f"{kernel_name}.sass"))
    except Exception as exc:
        (output_path / f"{kernel_name}.sass.error.txt").write_text(f"{type(exc).__name__}: {exc}\n")

    print(f"Artifacts written to {output_path}")


def main():
    args = parse_args()
    torch_dtype = TORCH_DTYPES[args.dtype]

    a = torch.randn(args.n, device="cuda", dtype=torch_dtype)
    b = torch.randn(args.n, device="cuda", dtype=torch_dtype)
    c = torch.empty_like(a)

    kernel = compile_kernel(args)
    kernel(a, b, c)
    dump_artifacts(kernel, args)

    torch.testing.assert_close(c, a + b)
    print(f"vector add {args.version} passed")

    tilelang_latency_ms = do_bench(
        lambda: kernel(a, b, c),
        warmup=args.warmup_ms,
        rep=args.profile_ms,
        backend="event",
        return_mode="mean",
    )
    torch_latency_ms = do_bench(
        lambda: torch.add(a, b, out=c),
        warmup=args.warmup_ms,
        rep=args.profile_ms,
        backend="event",
        return_mode="mean",
    )
    bytes_moved = 3 * a.numel() * a.element_size()
    tilelang_bandwidth_gbps = bytes_moved / (tilelang_latency_ms * 1e-3) / 1e9
    torch_bandwidth_gbps = bytes_moved / (torch_latency_ms * 1e-3) / 1e9

    print(f"Kernel: vector_add_{args.version}")
    print(f"TileLang latency: {tilelang_latency_ms:.4f} ms")
    print(f"TileLang bandwidth: {tilelang_bandwidth_gbps:.2f} GB/s")
    print(f"PyTorch latency: {torch_latency_ms:.4f} ms")
    print(f"PyTorch bandwidth: {torch_bandwidth_gbps:.2f} GB/s")
    print(f"Speedup: {torch_latency_ms / tilelang_latency_ms:.2f}x")


if __name__ == "__main__":
    if not torch.cuda.is_available():
        raise SystemExit("CUDA is required to run this TileLang example.")

    main()
