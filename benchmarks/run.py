"""Run benchmark problems and save results to benchmarks/results/<problem>_<label>.json.

Usage:
    python benchmarks/run.py [-problem cube|thermal|hyperelasticity|all] [-N 10 20 ...]
                             [-device cpu|cuda] [--label NAME] [--hardware DESC]
"""

import argparse
import json
import platform
import sys
from pathlib import Path

import scipy
import torch

import cubes
import hyperelasticity
import thermal
from utils import profile_and_capture_cpu, profile_and_capture_gpu

# name -> problem module; each module defines a `PROBLEM` descriptor and is
# runnable as the child process that solves one (N, device) case.
PROBLEMS = {"cube": cubes, "thermal": thermal, "hyperelasticity": hyperelasticity}


def run_problem(name: str, N_values: list[int], device: str) -> list[dict]:
    use_cuda = device == "cuda"
    script = PROBLEMS[name].__file__
    header_mem = " Peak VRAM" if use_cuda else "  Peak RAM"
    rows = []
    print(f"Running {name} benchmark on device={device}")
    print(f"|    N |     DOFs |     Setup | FWD Solve | BWD Solve | {header_mem} |")
    print("| ---- | -------- | --------- | --------- | --------- | ---------- |")

    for N in N_values:
        cmd = [sys.executable, script, "-N", str(N), "-device", device]
        profiler = profile_and_capture_gpu if use_cuda else profile_and_capture_cpu
        mem, tags = profiler(cmd)
        dofs = int(tags["DOFS"])
        setup_t = tags["SETUP_DONE"] - tags["START"]
        fwd_t = tags["FWD_DONE"] - tags["SETUP_DONE"]
        bwd_t = tags["BWD_DONE"] - tags["FWD_DONE"]
        peak_mem = mem["peak_vram_mb"] if use_cuda else mem["peak_ram_mb"]
        mem_str = f"{peak_mem:8.1f}MB"
        print(
            f"| {N:4d} | {dofs:8d} | {setup_t:8.2f}s | {fwd_t:8.2f}s "
            f"| {bwd_t:8.2f}s | {mem_str} |"
        )
        rows.append(
            {
                "N": N,
                "dofs": dofs,
                "setup_s": round(setup_t, 4),
                "fwd_s": round(fwd_t, 4),
                "bwd_s": round(bwd_t, 4),
                "peak_ram_mb": round(peak_mem, 1) if not use_cuda else None,
                "peak_vram_mb": round(peak_mem, 1) if use_cuda else None,
            }
        )
    return rows


def main():
    parser = argparse.ArgumentParser(
        description="Run benchmark problems and save JSON results."
    )
    parser.add_argument("-problem", type=str, default="cube", choices=[*PROBLEMS, "all"])
    parser.add_argument("-N", type=int, nargs="+", default=None)
    parser.add_argument("-device", type=str, default="cpu")
    parser.add_argument("--label", type=str, default=None)
    parser.add_argument("--hardware", type=str, default=None)
    args = parser.parse_args()

    if args.problem == "all" and args.N is not None:
        parser.error("-N cannot be combined with '-problem all'.")
    names = list(PROBLEMS) if args.problem == "all" else [args.problem]

    label = args.label or args.device
    hardware = args.hardware or platform.processor() or platform.machine()

    use_cuda = args.device == "cuda"
    if use_cuda:
        import cupy

        cuda_ver = cupy.cuda.runtime.runtimeGetVersion()
        cuda_str = f"{cuda_ver // 1000}.{cuda_ver % 1000 // 10}"
        libraries = f"CuPy {cupy.__version__}, CUDA {cuda_str}"
    else:
        libraries = f"SciPy {scipy.__version__}, PyTorch {torch.__version__}"

    results_dir = Path(__file__).parent / "results"
    results_dir.mkdir(exist_ok=True)

    for name in names:
        problem = PROBLEMS[name].PROBLEM
        N_values = args.N or problem.default_N
        out_path = results_dir / f"{name}_{label}.json"
        rows = run_problem(name, N_values, args.device)
        payload = {
            "hardware": hardware,
            "device": args.device,
            "python": platform.python_version(),
            "libraries": libraries,
            "dtype": "float64",
            "problem": problem.id,
            "rows": rows,
        }
        with open(out_path, "w") as f:
            json.dump(payload, f, indent=2)
        print(f"\nResults written to {out_path}\n")


if __name__ == "__main__":
    main()
