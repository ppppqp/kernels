import argparse
import re
import shutil
from pathlib import Path

import torch
import tilelang
import tilelang.language as T


M, N, K = 1024, 1024, 1024
block_M, block_N, block_K = 128, 128, 32


def make_kernel(disable_ws: bool, dump_dir: Path):
    pass_configs = {
        tilelang.PassConfigKey.TL_ENABLE_FAST_MATH: False,
        tilelang.PassConfigKey.TL_ENABLE_DUMP_IR: True,
        tilelang.PassConfigKey.TL_DUMP_IR_DIR: str(dump_dir),
    }
    if disable_ws:
        pass_configs[tilelang.PassConfigKey.TL_DISABLE_WARP_SPECIALIZED] = True

    @tilelang.jit(out_idx=[-1], pass_configs=pass_configs)
    def gemm_bias_relu(
        M, N, K, block_M, block_N, block_K, dtype="float16", accum_dtype="float32"
    ):
        @T.prim_func
        def main(
            A: T.Tensor((M, K), dtype),
            B: T.Tensor((K, N), dtype),
            bias: T.Tensor((N,), dtype),
            C: T.Tensor((M, N), dtype),
        ):
            with T.Kernel(
                T.ceildiv(N, block_N), T.ceildiv(M, block_M), threads=128
            ) as (bx, by):
                A_shared = T.alloc_shared((block_M, block_K), dtype)
                B_shared = T.alloc_shared((block_K, block_N), dtype)
                bias_shared = T.alloc_shared((block_N,), dtype)
                C_local = T.alloc_fragment((block_M, block_N), accum_dtype)

                # Issue 2443: this prologue copy is outside T.Pipelined, but
                # its destination is consumed by the post-loop consumer code.
                T.copy(bias[bx * block_N : (bx + 1) * block_N], bias_shared)

                T.clear(C_local)
                for k in T.Pipelined(T.ceildiv(K, block_K), num_stages=3):
                    T.copy(A[by * block_M, k * block_K], A_shared)
                    T.copy(B[k * block_K, bx * block_N], B_shared)
                    T.gemm(A_shared, B_shared, C_local)

                for i, j in T.Parallel(block_M, block_N):
                    C_local[i, j] = T.max(C_local[i, j] + bias_shared[j], 0)

                T.copy(C_local, C[by * block_M, bx * block_N])

        return main

    tilelang.disable_cache()
    try:
        return gemm_bias_relu(M, N, K, block_M, block_N, block_K)
    finally:
        tilelang.enable_cache()


def read_text(path: Path) -> str:
    return path.read_text(errors="replace")


def snippet(text: str, pos: int, radius: int = 900) -> str:
    if pos < 0:
        return "<not found>"
    start = max(0, pos - radius)
    end = min(len(text), pos + radius)
    return text[start:end]


def find_dump_files(dump_dir: Path):
    return sorted(path for path in dump_dir.rglob("*") if path.is_file())


def choose_ws_dump(dump_dir: Path) -> tuple[Path | None, str]:
    candidates = []
    for path in find_dump_files(dump_dir):
        text = read_text(path)
        score = 0
        if "ProducerConsumerWarpSpecialized" in path.name:
            score += 5
        if "warp_special" in text or "warp_specialization" in text:
            score += 4
        if "bias_shared" in text:
            score += 2
        if "mbarrier" in text:
            score += 1
        if score:
            candidates.append((score, path, text))
    if not candidates:
        return None, ""
    candidates.sort(key=lambda item: (item[0], str(item[1])), reverse=True)
    _, path, text = candidates[0]
    return path, text


def find_bias_copy_evidence(text: str) -> dict[str, object]:
    bias_positions = [m.start() for m in re.finditer(r"bias_shared", text)]
    branch_positions = [
        m.start()
        for m in re.finditer(
            r"warp_special|threadIdx\.x.*<|threadIdx_x.*<|T\.attr.*warp",
            text,
        )
    ]
    consumer_remap_positions = [
        m.start()
        for m in re.finditer(
            r"threadIdx\.x.*-\s*128|threadIdx_x.*-\s*128|threadIdx.*128",
            text,
        )
    ]
    return {
        "bias_occurrences": len(bias_positions),
        "first_bias_pos": bias_positions[0] if bias_positions else -1,
        "last_bias_pos": bias_positions[-1] if bias_positions else -1,
        "first_branch_pos": branch_positions[0] if branch_positions else -1,
        "consumer_remap_occurrences": len(consumer_remap_positions),
    }


def write_evidence_report(label: str, dump_dir: Path, cuda_source: str, out_dir: Path):
    source_path = out_dir / f"{label}.cu"
    source_path.write_text(cuda_source)

    ws_dump_path, ws_dump_text = choose_ws_dump(dump_dir)
    report_path = out_dir / f"{label}_evidence.txt"

    source_evidence = {
        "launch_bounds_128": "__launch_bounds__(128, 1)" in cuda_source,
        "launch_bounds_256": "__launch_bounds__(256, 1)" in cuda_source,
        "producer_branch": "threadIdx.x) < 128" in cuda_source
        or "threadIdx.x < 128" in cuda_source,
        "bias_shared_mentions": cuda_source.count("bias_shared"),
        "tma_load_mentions": cuda_source.count("tma_load"),
    }

    lines = [
        f"label: {label}",
        f"dump_dir: {dump_dir}",
        f"cuda_source: {source_path}",
        f"ws_dump: {ws_dump_path if ws_dump_path else '<not found>'}",
        "",
        "CUDA evidence:",
    ]
    for key, value in source_evidence.items():
        lines.append(f"  {key}: {value}")

    if ws_dump_text:
        ir_evidence = find_bias_copy_evidence(ws_dump_text)
        lines.extend(["", "IR evidence:"])
        for key, value in ir_evidence.items():
            lines.append(f"  {key}: {value}")

        first_bias_pos = int(ir_evidence["first_bias_pos"])
        first_branch_pos = int(ir_evidence["first_branch_pos"])
        lines.extend(
            [
                "",
                "Snippet around first warp-specialization marker:",
                snippet(ws_dump_text, first_branch_pos),
                "",
                "Snippet around first bias_shared occurrence:",
                snippet(ws_dump_text, first_bias_pos),
            ]
        )
    else:
        lines.extend(["", "IR evidence: no WS dump candidate found"])

    report_path.write_text("\n".join(lines))
    print(f"[{label}] wrote {report_path}")
    print(f"[{label}] wrote {source_path}")


def run_numeric_check(kernel, label: str, a, b, bias, ref):
    c = kernel(a, b, bias)
    ok = torch.allclose(c, ref, rtol=1e-2, atol=1e-2)
    mism = (c.float() - ref.float()).abs().gt(1e-2).sum().item()
    print(
        f"{label:16s}: allclose={ok} mismatched={mism}/{c.numel()} ({100 * mism / c.numel():.1f}%)"
    )
    if not ok:
        raise AssertionError(f"{label} failed numeric check")


def main():
    parser = argparse.ArgumentParser(
        description="Compile issue 2443 reproduction and emit CUDA/TIR evidence for WS prelude placement."
    )
    parser.add_argument("--out-dir", default="verify_issue_2443_artifacts")
    parser.add_argument(
        "--skip-run",
        action="store_true",
        help="Only compile and emit evidence; skip CUDA execution.",
    )
    parser.add_argument(
        "--keep",
        action="store_true",
        help="Keep any existing output directory contents.",
    )
    args = parser.parse_args()

    out_dir = Path(args.out_dir).resolve()
    if out_dir.exists() and not args.keep:
        shutil.rmtree(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    kernels = {}
    for label, disable_ws in (("ws_on", False), ("ws_off", True)):
        dump_dir = out_dir / f"{label}_dump_ir"
        dump_dir.mkdir(parents=True, exist_ok=True)
        print(f"[{label}] compiling with IR dump at {dump_dir}")
        kernel = make_kernel(disable_ws=disable_ws, dump_dir=dump_dir)
        kernels[label] = kernel
        write_evidence_report(label, dump_dir, kernel.get_kernel_source(), out_dir)

    if not args.skip_run:
        torch.manual_seed(0)
        a = torch.randn(M, K, dtype=torch.float16, device="cuda")
        b = torch.randn(K, N, dtype=torch.float16, device="cuda")
        bias = torch.randn(N, dtype=torch.float16, device="cuda")
        ref = torch.relu(a.float() @ b.float() + bias.float()).half()
        run_numeric_check(kernels["ws_on"], "WS ON", a, b, bias, ref)
        run_numeric_check(kernels["ws_off"], "WS OFF", a, b, bias, ref)
        torch.cuda.synchronize()

    print(f"Artifacts written to: {out_dir}")
    print("Inspect *_evidence.txt first, then the matching *.cu and *_dump_ir files.")


if __name__ == "__main__":
    main()
