#!/usr/bin/env python3
"""
Gaussian Splatting pipeline for rover capture sessions.

This script expects a session directory created by colmap_capture.py or Rowfollow.py:

  session/
    images/

It can:
  1. run COLMAP sparse reconstruction using reconstruct.py helpers,
  2. optionally export berries.json for farm_twin.html,
  3. optionally run a local Graphdeco Gaussian Splatting trainer checkout,
  4. write pipeline_manifest.json pointing the website to the output files.

The Hugging Face overview describes the high-level flow as:
images -> Structure from Motion/COLMAP -> gaussians -> training -> renderable splat.
"""

import argparse
import json
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Dict, Optional

import reconstruct


def latest_point_cloud(model_dir: Path) -> Optional[Path]:
    point_root = model_dir / "point_cloud"
    if not point_root.is_dir():
        return None

    candidates = []
    for ply in point_root.glob("iteration_*/point_cloud.ply"):
        iteration_name = ply.parent.name.replace("iteration_", "")
        try:
            iteration = int(iteration_name)
        except ValueError:
            iteration = -1
        candidates.append((iteration, ply))

    if not candidates:
        return None

    return sorted(candidates, key=lambda item: item[0])[-1][1]


def run_berry_export(session_dir: Path, skip_colmap: bool) -> Path:
    args = [sys.executable, "reconstruct.py", str(session_dir)]
    if skip_colmap:
        args.append("--skip-colmap")
    subprocess.run(args, check=True)
    return session_dir / "berries.json"


def run_gaussian_training(
    session_dir: Path,
    gs_repo: Path,
    iterations: int,
    output_dir: Path,
    extra_args: list[str],
) -> Path:
    train_py = gs_repo / "train.py"
    if not train_py.is_file():
        raise FileNotFoundError(
            f"No train.py found at {train_py}. Pass --gs-repo pointing to a local gaussian-splatting checkout."
        )

    output_dir.mkdir(parents=True, exist_ok=True)
    cmd = [
        sys.executable,
        str(train_py),
        "-s", str(session_dir),
        "-m", str(output_dir),
        "--iterations", str(iterations),
        *extra_args,
    ]
    print("[GS] Running trainer:")
    print("     " + " ".join(cmd))
    subprocess.run(cmd, cwd=gs_repo, check=True)

    ply = latest_point_cloud(output_dir)
    if ply is None:
        raise RuntimeError(f"Training completed, but no point_cloud.ply was found in {output_dir}")

    exported = session_dir / "gaussian_splat.ply"
    shutil.copy2(ply, exported)
    print(f"[GS] Exported latest splat point cloud: {exported}")
    return exported


def write_manifest(session_dir: Path, values: Dict[str, Optional[str]]) -> Path:
    manifest = {
        "session": str(session_dir),
        "images": str(session_dir / "images"),
        **values,
    }
    out = session_dir / "pipeline_manifest.json"
    with out.open("w") as f:
        json.dump(manifest, f, indent=2)
    return out


def main() -> int:
    parser = argparse.ArgumentParser(description="Build rover Gaussian Splatting outputs for farm_twin.html.")
    parser.add_argument("session", help="Capture session directory containing images/.")
    parser.add_argument("--skip-colmap", action="store_true", help="Use an existing sparse/ reconstruction.")
    parser.add_argument("--skip-berries", action="store_true", help="Do not run reconstruct.py berry detection/export.")
    parser.add_argument("--gs-repo", type=str, default=None, help="Path to a local graphdeco-inria/gaussian-splatting checkout.")
    parser.add_argument("--iterations", type=int, default=7000, help="Gaussian Splatting training iterations.")
    parser.add_argument("--gs-output", type=str, default=None, help="Trainer output directory. Default: <session>/gaussian_model")
    parser.add_argument(
        "--trainer-arg",
        action="append",
        default=[],
        help="Extra argument passed through to train.py. Repeat for multiple args.",
    )
    args = parser.parse_args()

    session_dir = Path(args.session).resolve()
    image_dir = session_dir / "images"
    if not image_dir.is_dir():
        print(f"[ERROR] Missing images folder: {image_dir}")
        return 2

    image_count = len(list(image_dir.glob("*.jpg"))) + len(list(image_dir.glob("*.png")))
    if image_count < 12:
        print(f"[WARN] Only {image_count} images found. Gaussian Splatting usually needs many overlapping views.")

    print("[PIPELINE] Step 1: COLMAP sparse reconstruction")
    if args.skip_colmap:
        model_dirs = sorted((session_dir / "sparse").iterdir()) if (session_dir / "sparse").is_dir() else []
        if not model_dirs:
            print("[ERROR] --skip-colmap was set, but no sparse/ model exists.")
            return 2
        model_path = model_dirs[0]
        print(f"[COLMAP] Using existing model: {model_path}")
    else:
        model_path = reconstruct.run_colmap(session_dir)

    berries_path = None
    if not args.skip_berries:
        print("[PIPELINE] Step 2: berry detection export")
        berries_path = run_berry_export(session_dir, skip_colmap=True)

    splat_path = session_dir / "gaussian_splat.ply"
    if args.gs_repo:
        print("[PIPELINE] Step 3: Gaussian Splatting training")
        gs_output = Path(args.gs_output).resolve() if args.gs_output else session_dir / "gaussian_model"
        splat_path = run_gaussian_training(
            session_dir=session_dir,
            gs_repo=Path(args.gs_repo).resolve(),
            iterations=args.iterations,
            output_dir=gs_output,
            extra_args=args.trainer_arg,
        )
    else:
        print("[PIPELINE] Step 3 skipped: pass --gs-repo to run Gaussian Splatting training.")
        print("           The manifest will still be written for berries.json and future splat output.")
        splat_path = splat_path if splat_path.exists() else None

    manifest_path = write_manifest(
        session_dir,
        {
            "colmap_model": str(model_path),
            "berries_json": str(berries_path) if berries_path else None,
            "gaussian_splat_ply": str(splat_path) if splat_path else None,
        },
    )

    print("\n[PIPELINE] Done")
    print(f"  Manifest: {manifest_path}")
    if berries_path:
        print(f"  Berries:  {berries_path}")
    if splat_path:
        print(f"  Splat:    {splat_path}")
    print("  Open farm_twin.html and load berries.json plus gaussian_splat.ply.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
