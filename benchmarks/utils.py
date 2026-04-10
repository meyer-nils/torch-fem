"""Shared utilities for benchmarking."""

import json
import os
import re
import subprocess
import tempfile
import textwrap
import time

import psutil

# Wrapper script injected into the child process. Samples
# torch.cuda.memory_allocated() via a daemon thread and writes
# the time-series to a JSON file.
_GPU_WRAPPER = textwrap.dedent("""\
    import json, sys, threading, time
    import torch

    _mem_file = sys.argv[1]
    _records = []
    _stop = threading.Event()

    def _monitor():
        while not _stop.is_set():
            try:
                _records.append((time.time(),
                                 torch.cuda.memory_allocated() / (1024**2)))
            except Exception:
                _records.append((time.time(), 0.0))
            _stop.wait(0.02)

    _t = threading.Thread(target=_monitor, daemon=True)
    _t.start()

    sys.argv[:] = sys.argv[2:]
    _script = sys.argv[0]
    with open(_script) as _f:
        exec(compile(_f.read(), _script, "exec"), {"__name__": "__main__"})

    _stop.set()
    _t.join(timeout=2)
    with open(_mem_file, "w") as _f:
        json.dump(_records, _f)
""")


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

    # Write the wrapper script to a temp file
    wrapper_fd, wrapper_path = tempfile.mkstemp(suffix=".py")
    mem_fd, mem_path = tempfile.mkstemp(suffix=".json")
    os.close(wrapper_fd)
    os.close(mem_fd)
    try:
        with open(wrapper_path, "w") as f:
            f.write(_GPU_WRAPPER)

        new_cmd = [cmd[0], wrapper_path, mem_path] + cmd[1:]

        proc = subprocess.Popen(new_cmd, stdout=subprocess.PIPE, text=True)
        stdout_data, _ = proc.communicate()

        # Read memory records written by the child
        with open(mem_path) as f:
            vram_records: list[tuple[float, float]] = [
                (t, m) for t, m in json.load(f)
            ]
    finally:
        os.unlink(wrapper_path)
        try:
            os.unlink(mem_path)
        except OSError:
            pass

    # Parse timestamps using Regex
    clock = {tag: float(ts) for tag, ts in re.findall(r"(\w+):(\d+\.\d+)", stdout_data)}
    return vram_records, clock
