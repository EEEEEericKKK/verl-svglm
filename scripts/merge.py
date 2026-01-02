#!/usr/bin/env python3

from __future__ import annotations

import argparse
import os
import re
import subprocess
import sys
from pathlib import Path


_GLOBAL_STEP_RE = re.compile(r"^global_step_(\d+)$")


def _sorted_checkpoint_dirs(checkpoint_dirs: list[Path]) -> list[Path]:
    def key(p: Path) -> tuple[int, str]:
        m = _GLOBAL_STEP_RE.match(p.name)
        if m:
            return (int(m.group(1)), p.name)
        return (10**18, p.name)

    return sorted(checkpoint_dirs, key=key)


def _discover_checkpoints(input_dir: Path) -> tuple[Path, list[Path]]:
    """Return (run_dir, checkpoint_dirs).

    - If input_dir contains subdirs named global_step_*, treat those as checkpoints and run_dir=input_dir.
    - Else if input_dir itself is named global_step_*, treat it as a single checkpoint and run_dir=input_dir.parent.
    - Otherwise treat input_dir as a single checkpoint and run_dir=input_dir.
    """
    if not input_dir.is_dir():
        raise NotADirectoryError(f"Input path is not a directory: {input_dir}")

    candidates = [p for p in input_dir.iterdir() if p.is_dir() and _GLOBAL_STEP_RE.match(p.name)]
    if candidates:
        return input_dir, _sorted_checkpoint_dirs(candidates)

    if _GLOBAL_STEP_RE.match(input_dir.name):
        return input_dir.parent, [input_dir]

    return input_dir, [input_dir]


def _output_paths(run_dir: Path, checkpoint_dir: Path, multiple: bool) -> tuple[Path, Path]:
    """Return (output_root, target_dir)."""
    output_root = run_dir.with_name(run_dir.name + "_merged")

    # When merging a single checkpoint that isn't part of a run_dir/global_step_* set,
    # write directly into <checkpoint>_merged/ (not nested).
    if not multiple and checkpoint_dir == run_dir and not _GLOBAL_STEP_RE.match(checkpoint_dir.name):
        return output_root, output_root

    return output_root, output_root / checkpoint_dir.name


def _run_merge(local_dir: Path, target_dir: Path) -> None:
    target_dir.parent.mkdir(parents=True, exist_ok=True)

    cmd = [
        sys.executable,
        "-m",
        "verl.model_merger",
        "merge",
        "--local_dir",
        str(local_dir),
        "--target_dir",
        str(target_dir),
        "--backend",
        "fsdp",
    ]

    print("+", " ".join(cmd), flush=True)
    subprocess.run(cmd, check=True)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Merge VERL checkpoints under an input directory. "
            "If the input contains global_step_* subdirectories, each is merged into <input>_merged/global_step_*/. "
            "Otherwise the input directory itself is treated as a single checkpoint."
        )
    )
    parser.add_argument(
        "input_dir",
        help="Path to a run directory (containing global_step_*) or a single checkpoint directory.",
    )

    args = parser.parse_args(argv)

    input_dir = Path(os.path.expanduser(args.input_dir)).resolve()
    run_dir, checkpoint_dirs = _discover_checkpoints(input_dir)

    multiple = len(checkpoint_dirs) > 1 or (checkpoint_dirs and checkpoint_dirs[0].parent == run_dir and _GLOBAL_STEP_RE.match(checkpoint_dirs[0].name) is not None)

    print(f"Input:  {input_dir}")
    print(f"Run:    {run_dir}")

    for ckpt in checkpoint_dirs:
        output_root, target_dir = _output_paths(run_dir, ckpt, multiple=multiple)
        print(f"Merge:  {ckpt} -> {target_dir}")

        if target_dir.exists():
            print(f"Skip:   target already exists: {target_dir}")
            continue

        _run_merge(ckpt, target_dir)

    print("Done.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
