"""Read-only Task E preflight. Does not launch Isaac Sim or run GPU kernels."""
from __future__ import annotations

import argparse
import importlib
import importlib.metadata as metadata
import importlib.util
import json
import os
from pathlib import Path
import site
import subprocess
import sys


ROOT = Path(os.environ.get("ATEC_TASK_ROOT", Path(__file__).resolve().parents[2])).resolve()


def package_version(name: str) -> str | None:
    try:
        return metadata.version(name)
    except metadata.PackageNotFoundError:
        return None


def module_path(name: str) -> Path | None:
    spec = importlib.util.find_spec(name)
    if spec is None:
        return None
    if spec.origin:
        return Path(spec.origin).resolve()
    if spec.submodule_search_locations:
        return Path(next(iter(spec.submodule_search_locations))).resolve()
    return None


def runtime_version() -> str | None:
    version = package_version("isaacsim-kernel") or package_version("isaacsim")
    if version:
        return version
    path = module_path("isaacsim")
    if path and (path.parent / "VERSION").is_file():
        return (path.parent / "VERSION").read_text().strip()
    return None


def compatible_experience(headless: bool = False) -> Path | None:
    if not (runtime_version() or "").startswith("4.5"):
        return None
    path = module_path("isaaclab")
    if path is None:
        return None
    name = "isaaclab.python.headless.rendering.kit" if headless else "isaaclab.python.rendering.kit"
    for parent in path.parents:
        candidate = parent / "apps" / "isaacsim_4_5" / name
        if candidate.is_file():
            return candidate
    return None


def check_environment(training: bool = False) -> dict:
    errors, warnings = [], []
    report = {
        "python": sys.executable,
        "python_version": sys.version.split()[0],
        "user_site_enabled": site.ENABLE_USER_SITE,
        "modules": {},
        "isaacsim_runtime": runtime_version(),
        "omniverse_kit": package_version("omniverse-kit"),
        "checks_scope": "imports, installed versions, local assets and nvidia-smi; no simulator startup",
    }
    if site.ENABLE_USER_SITE:
        errors.append("User site is enabled; launch with PYTHONNOUSERSITE=1 to avoid torch/torchvision conflicts.")
    for name in ("isaaclab", "isaacsim", "atec_rl_lab"):
        path = module_path(name)
        if path is None:
            errors.append(f"Missing module: {name}")
        else:
            report["modules"][name] = {"path": str(path)}
    # Do not import isaaclab.tasks or pxr before SimulationApp startup.
    for name in ("torch", "torchvision", "numpy", "scipy", "PIL.Image", "gymnasium"):
        try:
            module = importlib.import_module(name)
            entry = {"path": getattr(module, "__file__", None), "version": getattr(module, "__version__", None)}
            report["modules"][name] = entry
            if name == "torch":
                entry["cuda_build"] = module.version.cuda
                if module.version.cuda is None:
                    errors.append("Installed torch is a CPU-only build; Task E evaluation requires CUDA.")
        except Exception as exc:
            errors.append(f"Import {name} failed: {type(exc).__name__}: {exc}")

    try:
        gpu = subprocess.run(
            ["nvidia-smi", "--query-gpu=name,driver_version,memory.total,memory.used", "--format=csv,noheader"],
            check=True, capture_output=True, text=True, timeout=15,
        )
        report["gpu"] = gpu.stdout.strip().splitlines()
        if not report["gpu"]:
            errors.append("nvidia-smi returned no GPUs.")
    except (OSError, subprocess.SubprocessError) as exc:
        errors.append(f"NVIDIA GPU check failed: {exc}")

    asset = ROOT / "atec_robot_model" / "robot" / "piper" / "piper.usd"
    report["piper_asset"] = str(asset)
    if not asset.is_file():
        errors.append(f"Missing Piper asset: {asset}")

    report["training_missing"] = [name for name in ("h5py", "diffusers", "tyro", "tensorboard") if importlib.util.find_spec(name) is None]
    if training and report["training_missing"]:
        message = "ACT training dependencies missing: " + ", ".join(report["training_missing"])
        errors.append(message)
    experience = compatible_experience(headless=True)
    report["isaacsim_45_headless_experience"] = str(experience) if experience else None
    if (report["isaacsim_runtime"] or "").startswith("4.5"):
        if experience is None:
            warnings.append("Isaac Sim 4.5 detected but no bundled IsaacLab 4.5 camera experience was found.")
        elif package_version("isaacsim") is None:
            warnings.append("Split Isaac Sim 4.5 packages have no isaacsim metapackage; use scripts/evaluate.sh to select the matching experience.")
    if training:
        warnings.append("This check does not validate the ACT training import chain, dataset or simulator startup.")
    report.update(errors=errors, warnings=warnings, ok=not errors)
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--training", action="store_true", help="Treat missing ACT training dependencies as errors.")
    parser.add_argument("--json", action="store_true", help="Print a machine-readable report to stdout.")
    parser.add_argument("--experience-path", action="store_true", help="Print the bundled Isaac Sim 4.5 camera experience, if needed.")
    parser.add_argument("--headless", action="store_true", help="Select the headless camera experience with --experience-path.")
    args = parser.parse_args()
    if args.experience_path:
        path = compatible_experience(args.headless)
        if path:
            print(path)
        return 0
    report = check_environment(args.training)
    if args.json:
        print(json.dumps(report, indent=2))
    else:
        print(f"[Task E] Python {report['python_version']}: {report['python']}")
        print(f"[Task E] Isaac Sim runtime: {report['isaacsim_runtime']}; Kit: {report['omniverse_kit']}")
        for name in ("torch", "torchvision"):
            entry = report["modules"].get(name)
            if entry:
                print(f"[Task E] {name} {entry['version']}: {entry['path']}")
        for gpu in report.get("gpu", []):
            print(f"[Task E] GPU: {gpu}")
        for warning in report["warnings"]:
            print(f"[Task E] WARNING: {warning}")
        for error in report["errors"]:
            print(f"[Task E] ERROR: {error}", file=sys.stderr)
        print("[Task E] Preflight " + ("passed (simulator startup still requires verification)." if report["ok"] else "failed."))
    return 0 if report["ok"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
