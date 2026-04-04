"""Shared utilities for benchmarking."""

import re
import subprocess
import time

import psutil


def _sample_vram(pid: int) -> float | None:
    """Query VRAM used by pid in MB via nvidia-smi, or None if unavailable."""
    try:
        out = subprocess.check_output(
            [
                "nvidia-smi",
                "--query-compute-apps=pid,used_memory",
                "--format=csv,noheader,nounits",
            ],
            text=True,
            stderr=subprocess.DEVNULL,
        )
        for line in out.splitlines():
            parts = line.strip().split(", ")
            if len(parts) == 2 and int(parts[0]) == pid:
                return float(parts[1])
    except (FileNotFoundError, subprocess.CalledProcessError, ValueError):
        pass
    return None


def profile_and_capture(
    cmd: list[str], device: str = "cpu"
) -> tuple[list[tuple[float, float, float | None]], dict[str, float]]:
    # Start process
    proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, text=True)
    ps_proc = psutil.Process(proc.pid)
    track_vram = device == "cuda"

    mem_records: list[tuple[float, float, float | None]] = []

    # Monitoring loop
    while proc.poll() is None:
        try:
            now = time.time()
            ram = ps_proc.memory_info().rss / (1024 * 1024)
            for child in ps_proc.children(recursive=True):
                ram += child.memory_info().rss / (1024 * 1024)
            vram = _sample_vram(proc.pid) if track_vram else None
            mem_records.append((now, ram, vram))
        except psutil.NoSuchProcess:
            break
        time.sleep(0.05)

    # Capture all output after process ends
    stdout_data, _ = proc.communicate()

    # Parse timestamps using Regex
    clock = {tag: float(ts) for tag, ts in re.findall(r"(\w+):(\d+\.\d+)", stdout_data)}
    return mem_records, clock
