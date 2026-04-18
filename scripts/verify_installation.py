#!/usr/bin/env python3
"""
Titan-GeoAlign — Installation Verification Script

Checks that all required components (Python packages, GPU, model files)
are installed and functional, then prints a colour-coded status report.

Usage:
    python scripts/verify_installation.py [--models-dir ./models] [--verbose]
"""
from __future__ import annotations

import argparse
import importlib
import os
import platform
import subprocess
import sys
from pathlib import Path
from typing import Callable

# ── ANSI colour helpers ───────────────────────────────────────────────────────
_USE_COLOUR = sys.stdout.isatty() and os.name != "nt" or os.environ.get("FORCE_COLOR")

def _c(code: str, text: str) -> str:
    return f"\033[{code}m{text}\033[0m" if _USE_COLOUR else text

def green(t: str)  -> str: return _c("32", t)
def red(t: str)    -> str: return _c("31", t)
def yellow(t: str) -> str: return _c("33", t)
def cyan(t: str)   -> str: return _c("36", t)
def bold(t: str)   -> str: return _c("1",  t)

PASS = green("  PASS")
FAIL = red("  FAIL")
WARN = yellow("  WARN")
SKIP = yellow("  SKIP")

# ── Result tracking ───────────────────────────────────────────────────────────
results: list[tuple[str, str, str]] = []  # (category, item, status_line)

def record(category: str, item: str, ok: bool | None, detail: str = "") -> None:
    if ok is True:
        symbol = PASS
    elif ok is False:
        symbol = FAIL
    else:
        symbol = WARN
    line = f"{symbol}  {item}"
    if detail:
        line += f"  {cyan('·')}  {detail}"
    results.append((category, item, line))
    print(line)

# ── Check helpers ─────────────────────────────────────────────────────────────
def check_import(pkg: str, extra: str = "") -> bool:
    try:
        mod = importlib.import_module(pkg)
        version = getattr(mod, "__version__", "?")
        record("packages", pkg, True, f"v{version}" + (f" — {extra}" if extra else ""))
        return True
    except ImportError as exc:
        record("packages", pkg, False, str(exc))
        return False

def check_command(cmd: list[str], label: str) -> bool:
    try:
        out = subprocess.check_output(cmd, stderr=subprocess.STDOUT, timeout=10)
        record("tools", label, True, out.decode().strip().splitlines()[0])
        return True
    except (subprocess.CalledProcessError, FileNotFoundError, subprocess.TimeoutExpired) as e:
        record("tools", label, False, str(e))
        return False

def check_file(path: Path, label: str) -> bool:
    if path.exists():
        size_mb = path.stat().st_size / 1_048_576
        record("models", label, True, f"{size_mb:.1f} MB")
        return True
    else:
        record("models", label, None, f"not found: {path}")
        return False

# ── Section: Python version ───────────────────────────────────────────────────
def check_python() -> None:
    print(bold("\n── Python Runtime ───────────────────────────────────────────────────"))
    vi = sys.version_info
    ok = vi >= (3, 10)
    record("python", "Python version", ok,
           f"{vi.major}.{vi.minor}.{vi.micro}  (≥3.10 required)")
    record("python", "Platform", True, platform.platform())
    record("python", "Architecture", True, platform.machine())

# ── Section: Required packages ───────────────────────────────────────────────
REQUIRED_PACKAGES = [
    ("numpy",        ""),
    ("torch",        ""),
    ("torchvision",  ""),
    ("rasterio",     ""),
    ("pyproj",       ""),
    ("shapely",      ""),
    ("cv2",          "opencv-python"),
    ("PIL",          "Pillow"),
    ("hydra",        "hydra-core"),
    ("omegaconf",    ""),
    ("einops",       ""),
    ("timm",         ""),
    ("transformers", ""),
    ("scipy",        ""),
    ("sklearn",      "scikit-learn"),
    ("tqdm",         ""),
    ("rich",         ""),
    ("fastapi",      ""),
    ("uvicorn",      ""),
]

OPTIONAL_PACKAGES = [
    ("cupy",         "GPU acceleration"),
    ("kornia",       ""),
    ("open3d",       "3-D point cloud support"),
    ("h5py",         "HDF5 support"),
    ("netCDF4",      "NetCDF support"),
    ("boto3",        "AWS S3 support"),
]

def check_packages() -> None:
    print(bold("\n── Required Python Packages ─────────────────────────────────────────"))
    for pkg, note in REQUIRED_PACKAGES:
        check_import(pkg, note)

    print(bold("\n── Optional Python Packages ─────────────────────────────────────────"))
    for pkg, note in OPTIONAL_PACKAGES:
        try:
            mod = importlib.import_module(pkg)
            version = getattr(mod, "__version__", "?")
            record("optional", pkg, True, f"v{version}" + (f" — {note}" if note else ""))
        except ImportError:
            record("optional", pkg, None, f"not installed (optional: {note})")

# ── Section: GPU availability ─────────────────────────────────────────────────
def check_gpu() -> None:
    print(bold("\n── GPU / Compute ─────────────────────────────────────────────────────"))
    try:
        import torch
        cuda_available = torch.cuda.is_available()
        if cuda_available:
            gpu_name  = torch.cuda.get_device_name(0)
            gpu_mem   = torch.cuda.get_device_properties(0).total_memory // 1_048_576
            cuda_ver  = torch.version.cuda
            record("gpu", "CUDA available", True,
                   f"{gpu_name} — {gpu_mem} MB VRAM — CUDA {cuda_ver}")
            n = torch.cuda.device_count()
            if n > 1:
                record("gpu", f"Multi-GPU", True, f"{n} devices detected")
        else:
            record("gpu", "CUDA available", None, "no GPU found — will run on CPU")

        mps = getattr(torch.backends, "mps", None)
        if mps and mps.is_available():
            record("gpu", "Apple MPS (Metal)", True, "available")
    except ImportError:
        record("gpu", "PyTorch", False, "not installed — cannot check GPU")

# ── Section: System tools ─────────────────────────────────────────────────────
def check_tools() -> None:
    print(bold("\n── System Tools ─────────────────────────────────────────────────────"))
    check_command(["gdal-config", "--version"], "GDAL")
    check_command(["proj",  "--version"],        "PROJ")
    check_command(["git",   "--version"],        "Git")
    check_command(["git", "lfs", "version"],     "Git LFS")

# ── Section: Model weights ────────────────────────────────────────────────────
def check_models(models_dir: Path) -> None:
    print(bold("\n── Model Weights ─────────────────────────────────────────────────────"))
    model_files = [
        (models_dir / "foundation" / "prithvi_eo_2" / "Prithvi_EO_V2_300M.pt",
         "Prithvi-EO-2.0"),
        (models_dir / "foundation" / "clay_v1" / "clay-v1-5.ckpt",
         "Clay v1.5"),
        (models_dir / "foundation" / "dofa" / "DOFA_ViT_base_e030.pth",
         "DOFA"),
        (models_dir / "matchers" / "xfeat" / "xfeat.pt",
         "XFeat"),
        (models_dir / "matchers" / "lightglue" / "superpoint_lightglue.pth",
         "LightGlue (SuperPoint)"),
        (models_dir / "matchers" / "roma" / "roma_outdoor.pth",
         "RoMa v2 (outdoor)"),
    ]
    for path, label in model_files:
        check_file(path, label)

# ── Section: Quick sanity test ────────────────────────────────────────────────
def check_sanity() -> None:
    print(bold("\n── Quick Sanity Tests ───────────────────────────────────────────────"))
    # NumPy array ops
    try:
        import numpy as np
        a = np.random.rand(64, 64).astype(np.float32)
        b = np.linalg.svd(a, full_matrices=False)
        assert b[1].shape == (64,)
        record("sanity", "NumPy SVD", True, "64×64 matrix decomposition OK")
    except Exception as exc:
        record("sanity", "NumPy SVD", False, str(exc))

    # PyTorch forward pass
    try:
        import torch
        x = torch.randn(2, 3, 64, 64)
        conv = torch.nn.Conv2d(3, 16, 3, padding=1)
        y = conv(x)
        assert y.shape == (2, 16, 64, 64)
        record("sanity", "PyTorch Conv2d forward pass", True, f"output {list(y.shape)}")
    except Exception as exc:
        record("sanity", "PyTorch Conv2d forward pass", False, str(exc))

    # Rasterio CRS round-trip
    try:
        from pyproj import CRS, Transformer
        crs = CRS.from_epsg(4326)
        transformer = Transformer.from_crs(crs, CRS.from_epsg(32633), always_xy=True)
        x2, y2 = transformer.transform(13.4, 52.5)  # Berlin
        assert abs(x2 - 392802) < 1000
        record("sanity", "pyproj CRS transform", True,
               f"EPSG:4326 → EPSG:32633  ({x2:.0f}, {y2:.0f})")
    except Exception as exc:
        record("sanity", "pyproj CRS transform", False, str(exc))

# ── Summary ───────────────────────────────────────────────────────────────────
def print_summary() -> int:
    passed = sum(1 for _, _, l in results if PASS in l)
    failed = sum(1 for _, _, l in results if FAIL in l)
    warned = sum(1 for _, _, l in results if WARN in l or SKIP in l)
    total  = len(results)

    print(bold("\n════════════════════════════════════════════════════════════════════"))
    print(bold("  Verification Summary"))
    print(f"  Total:   {total}")
    print(green(f"  Passed:  {passed}"))
    if warned:
        print(yellow(f"  Warnings:{warned}"))
    if failed:
        print(red(f"  Failed:  {failed}"))
    print(bold("════════════════════════════════════════════════════════════════════"))

    if failed:
        print(red("\n✗  Some checks FAILED. Review the output above."))
    elif warned:
        print(yellow("\n⚠  Installation complete with warnings."))
    else:
        print(green("\n✓  All checks passed. Titan-GeoAlign is ready."))

    return 1 if failed else 0

# ── Entry point ───────────────────────────────────────────────────────────────
def main() -> int:
    parser = argparse.ArgumentParser(
        description="Verify Titan-GeoAlign installation",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--models-dir", default="./models",
                        help="Directory containing model weights")
    parser.add_argument("--verbose", action="store_true",
                        help="Show extra detail")
    args = parser.parse_args()

    print(bold(cyan(
        "\n╔══════════════════════════════════════════════════════════════════╗\n"
        "║          Titan-GeoAlign  ·  Installation Verifier                ║\n"
        "╚══════════════════════════════════════════════════════════════════╝"
    )))

    models_dir = Path(args.models_dir)

    check_python()
    check_packages()
    check_gpu()
    check_tools()
    check_models(models_dir)
    check_sanity()
    return print_summary()


if __name__ == "__main__":
    sys.exit(main())
