import argparse
import os
import sys
from pathlib import Path

import torch
import torch.nn.functional as F

import tilelang
import tilelang.language as T
from tilelang import tvm
from tilelang.profiler import do_bench

# Allow this file to be run directly while still using package-relative imports.
current_dir = Path(__file__).resolve().parent
if __package__ in (None, ""):
    sys.path.insert(0, str(current_dir.parent))
    __package__ = current_dir.name

from .utils import dump_lowered_artifact, dump_sources
from .kernels import (
    symbolic_sliced_copy_reshape_kernel,
    sliced_copy_reshape_kernel,
    copy_reshape_kernel,
    copy_flat_kernel,
    elementwise_affine_kernel,
)


def compile_layout_kernel(
    pattern: str,
    x: torch.Tensor,
    k: torch.Tensor | None,
    block_b: int,
    block_s: int,
    heads: int,
    threads: int,
):
    if pattern.startswith("symbolic_") and k is None:
        raise ValueError(f"{pattern} requires k")

    if pattern == "symbolic_sliced_copy_reshape":
        return symbolic_sliced_copy_reshape_kernel.compile(
            QB=x.shape[0],
            B=k.shape[0],
            S=x.shape[1],
            BLOCK_B=block_b,
            BLOCK_S=block_s,
            THREADS=threads,
            ANNOTATE=True,
        )
    if pattern == "symbolic_sliced_copy_reshape_no_annotate":
        return symbolic_sliced_copy_reshape_kernel.compile(
            QB=x.shape[0],
            B=k.shape[0],
            S=x.shape[1],
            BLOCK_B=block_b,
            BLOCK_S=block_s,
            THREADS=threads,
            ANNOTATE=False,
        )
    if pattern == "sliced_copy_reshape":
        return sliced_copy_reshape_kernel.compile(
            BLOCK_B=block_b,
            BLOCK_S=block_s,
            HEADS=heads,
            THREADS=threads,
            ANNOTATE=True,
        )
    if pattern == "sliced_copy_reshape_no_annotate":
        return sliced_copy_reshape_kernel.compile(
            BLOCK_B=block_b,
            BLOCK_S=block_s,
            HEADS=heads,
            THREADS=threads,
            ANNOTATE=False,
        )
    if pattern == "copy_reshape":
        return copy_reshape_kernel.compile(
            BLOCK_B=block_b,
            BLOCK_S=block_s,
            HEADS=heads,
            THREADS=threads,
            ANNOTATE=True,
        )
    if pattern == "copy_reshape_no_annotate":
        return copy_reshape_kernel.compile(
            BLOCK_B=block_b,
            BLOCK_S=block_s,
            HEADS=heads,
            THREADS=threads,
            ANNOTATE=False,
        )
    if pattern == "copy_flat":
        return copy_flat_kernel.compile(
            BLOCK_B=block_b,
            BLOCK_S=block_s,
            HEADS=heads,
            THREADS=threads,
        )
    if pattern == "elementwise_affine":
        return elementwise_affine_kernel.compile(
            BLOCK_B=block_b,
            BLOCK_S=block_s,
            HEADS=heads,
            THREADS=threads,
        )
    raise ValueError(f"unknown pattern: {pattern}")


def lower_layout_kernel_source(
    pattern: str,
    x: torch.Tensor,
    k: torch.Tensor | None,
    block_b: int,
    block_s: int,
    heads: int,
    threads: int,
    target: str = "cuda",
):
    if pattern.startswith("symbolic_") and k is None:
        raise ValueError(f"{pattern} requires k")

    target_obj = tvm.target.Target(target)
    with target_obj:
        if pattern == "symbolic_sliced_copy_reshape":
            prim_func = symbolic_sliced_copy_reshape_kernel.get_tir(
                QB=x.shape[0],
                B=k.shape[0],
                S=x.shape[1],
                BLOCK_B=block_b,
                BLOCK_S=block_s,
                THREADS=threads,
                ANNOTATE=True,
            )
        elif pattern == "symbolic_sliced_copy_reshape_no_annotate":
            prim_func = symbolic_sliced_copy_reshape_kernel.get_tir(
                QB=x.shape[0],
                B=k.shape[0],
                S=x.shape[1],
                BLOCK_B=block_b,
                BLOCK_S=block_s,
                THREADS=threads,
                ANNOTATE=False,
            )
        elif pattern == "sliced_copy_reshape":
            prim_func = sliced_copy_reshape_kernel.get_tir(
                BLOCK_B=block_b,
                BLOCK_S=block_s,
                HEADS=heads,
                THREADS=threads,
                ANNOTATE=True,
            )
        elif pattern == "sliced_copy_reshape_no_annotate":
            prim_func = sliced_copy_reshape_kernel.get_tir(
                BLOCK_B=block_b,
                BLOCK_S=block_s,
                HEADS=heads,
                THREADS=threads,
                ANNOTATE=False,
            )
        elif pattern == "copy_reshape":
            prim_func = copy_reshape_kernel.get_tir(
                BLOCK_B=block_b,
                BLOCK_S=block_s,
                HEADS=heads,
                THREADS=threads,
                ANNOTATE=True,
            )
        elif pattern == "copy_reshape_no_annotate":
            prim_func = copy_reshape_kernel.get_tir(
                BLOCK_B=block_b,
                BLOCK_S=block_s,
                HEADS=heads,
                THREADS=threads,
                ANNOTATE=False,
            )
        elif pattern == "copy_flat":
            prim_func = copy_flat_kernel.get_tir(
                BLOCK_B=block_b,
                BLOCK_S=block_s,
                HEADS=heads,
                THREADS=threads,
            )
        elif pattern == "elementwise_affine":
            prim_func = elementwise_affine_kernel.get_tir(
                BLOCK_B=block_b,
                BLOCK_S=block_s,
                HEADS=heads,
                THREADS=threads,
            )
        else:
            raise ValueError(f"unknown pattern: {pattern}")
        return tilelang.lower(prim_func, target=target_obj)


def run_layout_case(
    pattern: str,
    block_b: int,
    block_s: int,
    heads: int,
    threads: int,
    source_only: bool = False,
    dump_dir: str | os.PathLike[str] | None = None,
) -> None:
    rows = block_b * heads
    x = torch.randn((rows, block_s), dtype=torch.float32, device="cuda")
    case_name = f"{pattern}_b{block_b}_s{block_s}_h{heads}_t{threads}"
    if pattern.startswith("symbolic_"):
        k = torch.randn((block_b, block_s), dtype=torch.float32, device="cuda")
    else:
        k = None
    print(f"\ncase: {case_name}")
    if source_only:
        artifact = lower_layout_kernel_source(
            pattern,
            x,
            k,
            block_b,
            block_s,
            heads,
            threads,
        )
        dump_lowered_artifact(artifact, dump_dir, case_name)
        return
    kernel = compile_layout_kernel(pattern, x, k, block_b, block_s, heads, threads)
    dump_sources(kernel, dump_dir, case_name)

    run = (lambda: kernel(x, k)) if k is not None else (lambda: kernel(x))
    ms = do_bench(run, warmup=25, rep=100)
    bytes_moved = x.numel() * x.element_size() * 2
    gbps = bytes_moved / (ms * 1.0e-3) / 1.0e9
    print(f"latency: {ms:.4f} ms, effective bandwidth: {gbps:.2f} GB/s")


def main():
    run_layout_case(
        pattern="symbolic_sliced_copy_reshape_no_annotate",
        block_b=128,
        block_s=64,
        heads=16,
        threads=64,
        source_only=False,
        dump_dir="./dump",
    )


if __name__ == "__main__":
    main()
