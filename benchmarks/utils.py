"""Shared utilities for benchmarking."""

import re
import subprocess
import time

import psutil


def profile_and_capture_cpu(
    cmd: list[str],
) -> tuple[list[tuple[float, float]], dict[str, float]]:

    # Start process
    proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, text=True)
    ps_proc = psutil.Process(proc.pid)

    mem_records: list[tuple[float, float]] = []

    # Monitoring loop
    while proc.poll() is None:
        try:
            now = time.time()
            ram = ps_proc.memory_info().rss / (1024 * 1024)
            for child in ps_proc.children(recursive=True):
                ram += child.memory_info().rss / (1024 * 1024)
            mem_records.append((now, ram))
        except psutil.NoSuchProcess:
            break
        time.sleep(0.05)

    # Capture all output after process ends
    stdout_data, _ = proc.communicate()

    # Parse timestamps using Regex
    clock = {tag: float(ts) for tag, ts in re.findall(r"(\w+):(\d+\.\d+)", stdout_data)}
    return mem_records, clock

def profile_and_capture_gpu(
    cmd: list[str],
) -> tuple[list[tuple[float, float]], dict[str, float]]:

    # Start process
    proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, text=True)

    vram_records: list[tuple[float, float]] = []

    # Monitoring loop
    while proc.poll() is None:
        try:
            now = time.time()

            try:
                out = subprocess.check_output(
                    [
                        "nvidia-smi",
                        "--query-gpu=memory.used",
                        "--format=csv,noheader,nounits",
                    ],
                    text=True,
                    stderr=subprocess.DEVNULL,
                    creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
                )
                vals = [float(line.strip().split(",", 1)[0]) for line in out.splitlines() if line.strip()]
                vram_records.append((now, sum(vals) if vals else 0.0))
            except (subprocess.SubprocessError, FileNotFoundError, OSError, ValueError):
                pass
        except psutil.NoSuchProcess:
            break
        time.sleep(0.05)

    # Capture all output after process ends
    stdout_data, _ = proc.communicate()

    if vram_records:
        baseline = vram_records[0][1]
        vram_records = [(t, max(v - baseline, 0.0)) for t, v in vram_records]

    # Parse timestamps using Regex
    clock = {tag: float(ts) for tag, ts in re.findall(r"(\w+):(\d+\.\d+)", stdout_data)}
    return vram_records, clock
