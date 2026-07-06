"""Benchmark measurement helpers.

``profile_and_capture_*`` run a child process and return ``(mem, clock)``: peak
memory in MB and phase timestamps (START/SETUP_DONE/FWD_DONE/BWD_DONE). CPU RAM
is sampled from the parent via psutil; GPU VRAM is sampled inside the child by
``VramMonitor`` and returned as tagged stdout lines parsed here.
"""

import re
import subprocess
import threading
import time

import psutil
import torch

_TAG_RE = re.compile(r"(\w+):(\d+\.\d+)")
_CLOCK_TAGS = ("START", "SETUP_DONE", "FWD_DONE", "BWD_DONE")


class VramMonitor:
    """Sample peak GPU memory at the driver level via ``torch.cuda.mem_get_info``.

    Runs in the benchmark child process. Unlike PyTorch's allocator stats, this
    also covers the CuPy pool and cuSOLVER/cuSPARSE workspaces (see
    torchfem.sparse). Emits MB figures as ``TAG:value`` stdout lines:
    VRAM_BASELINE_MB (init/context overhead) and PEAK_VRAM_DRIVER_MB (run peak).
    """

    def __init__(self, interval: float = 0.005):
        self._interval = interval
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self._peak_used = 0.0
        self._baseline = 0.0

    @staticmethod
    def _used_mb() -> float:
        free, total = torch.cuda.mem_get_info()
        return (total - free) / (1024**2)

    def start(self):
        torch.cuda.init()
        # Force CuPy init so its one-time overhead lands in the baseline.
        try:
            import cupy

            cupy.zeros(1)
            cupy.get_default_memory_pool().free_all_blocks()
        except Exception:
            pass
        torch.cuda.synchronize()
        self._baseline = self._used_mb()
        self._peak_used = self._baseline
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    def _run(self):
        while not self._stop.is_set():
            self._peak_used = max(self._peak_used, self._used_mb())
            self._stop.wait(self._interval)

    def stop(self):
        # Sync pending work and take a final sample before stopping.
        torch.cuda.synchronize()
        self._peak_used = max(self._peak_used, self._used_mb())
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=2)

    def report(self) -> dict[str, float]:
        return {
            "VRAM_BASELINE_MB": round(self._baseline, 3),
            "PEAK_VRAM_DRIVER_MB": round(self._peak_used, 3),
        }


def _parse_tags(stdout_data: str) -> dict[str, float]:
    return {tag: float(val) for tag, val in _TAG_RE.findall(stdout_data)}


def profile_and_capture_cpu(
    cmd: list[str],
) -> tuple[dict[str, float], dict[str, float]]:
    proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, text=True)
    ps_proc = psutil.Process(proc.pid)

    # Sample the process tree's resident set until the child exits.
    peak_ram = 0.0
    while proc.poll() is None:
        try:
            ram = ps_proc.memory_info().rss / (1024 * 1024)
            for child in ps_proc.children(recursive=True):
                ram += child.memory_info().rss / (1024 * 1024)
            peak_ram = max(peak_ram, ram)
        except psutil.NoSuchProcess:
            break
        time.sleep(0.05)

    stdout_data, _ = proc.communicate()
    tags = _parse_tags(stdout_data)
    clock = {k: tags[k] for k in _CLOCK_TAGS if k in tags}
    return {"peak_ram_mb": peak_ram}, clock


def profile_and_capture_gpu(
    cmd: list[str],
) -> tuple[dict[str, float], dict[str, float]]:
    proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, text=True)
    stdout_data, _ = proc.communicate()

    tags = _parse_tags(stdout_data)
    clock = {k: tags[k] for k in _CLOCK_TAGS if k in tags}
    baseline = tags.get("VRAM_BASELINE_MB", 0.0)
    driver = tags.get("PEAK_VRAM_DRIVER_MB", 0.0)
    # Problem's peak device usage: driver peak minus the fixed init baseline.
    mem = {"peak_vram_mb": max(driver - baseline, 0.0)}
    return mem, clock
