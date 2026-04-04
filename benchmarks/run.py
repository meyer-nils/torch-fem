"""Run the cube benchmark and save results to benchmarks/results/<label>.json.

Usage:
    python benchmarks/run.py [-N 10 20 ...] [-device cpu|cuda] [-order 1|2]
                             [--label NAME] [--hardware DESC]
"""

import argparse
import json
import platform
import sys
from pathlib import Path

import scipy
import torch
from utils import profile_and_capture


def main():
    parser = argparse.ArgumentParser(
        description="Run cube benchmark and save JSON results."
    )
    parser.add_argument(
        "-N", type=int, nargs="+", default=[10, 20, 30, 40, 50, 60, 70, 80]
    )
    parser.add_argument("-device", type=str, default="cpu")
    parser.add_argument("-order", type=int, default=1)
    parser.add_argument("--label", type=str, default=None)
    parser.add_argument("--hardware", type=str, default=None)
    args = parser.parse_args()

    label = args.label or args.device

    results_dir = Path(__file__).parent / "results"
    results_dir.mkdir(exist_ok=True)
    out_path = results_dir / f"{label}.json"

    hardware = args.hardware or platform.processor() or platform.machine()
    libraries = f"SciPy {scipy.__version__}, PyTorch {torch.__version__}"

    use_cuda = args.device == "cuda"
    header_mem = " Peak VRAM" if use_cuda else "  Peak RAM"
    rows = []
    print(f"Running benchmark on device={args.device}, order={args.order}")
    print(f"|  N  |     DOFs |     Setup | FWD Solve | BWD Solve | {header_mem} |")
    print("| --- | -------- | --------- | --------- | --------- | ---------- |")

    cubes = Path(__file__).parent / "cubes.py"
    for N in args.N:
        cmd = [
            sys.executable,
            str(cubes),
            "-N",
            str(N),
            "-device",
            args.device,
            "-order",
            str(args.order),
        ]
        mem_data, clock = profile_and_capture(cmd, args.device)
        ram_vals = [r for _, r, _ in mem_data]
        vram_vals = [v for _, _, v in mem_data if v is not None]
        setup_t = clock["SETUP_DONE"] - clock["START"]
        fwd_t = clock["FWD_DONE"] - clock["SETUP_DONE"]
        bwd_t = clock["BWD_DONE"] - clock["FWD_DONE"]
        peak_ram = max(ram_vals) if ram_vals else 0.0
        peak_vram = max(vram_vals) if vram_vals else None
        dofs = 3 * N**3
        mem_str = (
            f"{peak_vram:8.1f}MB"
            if (use_cuda and peak_vram is not None)
            else "     n/aMB" if use_cuda else f"{peak_ram:8.1f}MB"
        )
        print(
            f"| {N:3d} | {dofs:8d} | {setup_t:8.2f}s | {fwd_t:8.2f}s "
            f"| {bwd_t:8.2f}s | {mem_str} |"
        )
        rows.append(
            {
                "N": N,
                "dofs": dofs,
                "setup_s": round(setup_t, 4),
                "fwd_s": round(fwd_t, 4),
                "bwd_s": round(bwd_t, 4),
                "peak_ram_mb": round(peak_ram, 1),
                "peak_vram_mb": round(peak_vram, 1) if peak_vram is not None else None,
            }
        )

    payload = {
        "hardware": hardware,
        "device": args.device,
        "python": platform.python_version(),
        "libraries": libraries,
        "dtype": "float64",
        "problem": "cube_hexa_extension",
        "order": args.order,
        "rows": rows,
    }

    with open(out_path, "w") as f:
        json.dump(payload, f, indent=2)

    print(f"\nResults written to {out_path}")


if __name__ == "__main__":
    main()
