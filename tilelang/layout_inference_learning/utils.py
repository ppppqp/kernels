import os
from pathlib import Path

import torch


def dump_sources(kernel, dump_dir: str | os.PathLike[str] | None, name: str) -> None:
    if dump_dir is None:
        return
    root = Path(dump_dir)
    root.mkdir(parents=True, exist_ok=True)
    kernel_path = root / f"{name}.cu"
    host_path = root / f"{name}_host.cc"
    kernel.export_sources(kernel_path=str(kernel_path), host_path=str(host_path))
    print(f"dumped kernel source: {kernel_path}")
    print(f"dumped host source:   {host_path}")


def dump_lowered_artifact(
    artifact, dump_dir: str | os.PathLike[str] | None, name: str
) -> None:
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
