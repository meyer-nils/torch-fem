"""Shared utilities for benchmarking."""

import re
import subprocess
import time

import psutil


def profile_and_capture(
    cmd: list[str], device: str = "cpu"
) -> tuple[list[tuple[float, float]], dict[str, float]]:
    _ = device

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
