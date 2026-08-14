import os
import platform
import subprocess
import time
from functools import cache

import torch

WIDTH = 88
# The columns leave a margin at the right edge for the substep flags.
COLUMNS = "  {:>10}  {:>12}  {:>9}  {:>10}  {:>15}  {:>10}"
# Longest pause between notebook redraws, which bounds their message rate.
REFRESH = 0.05


def _rule(title: str = "") -> str:
    """Horizontal rule of `WIDTH` characters, optionally carrying a title."""
    if not title:
        return "-" * WIDTH
    return f"--- {title} " + "-" * (WIDTH - len(title) - 5)


def _plural(n: int, unit: str) -> str:
    """Count of `unit`, dropping its trailing `s` for a single item."""
    return f"{n} {unit if n != 1 else unit.removesuffix('s')}"


@cache
def _host() -> tuple[str, str]:
    """CPU name and memory of the host, queried once per session.

    The memory is empty if the platform does not expose it.
    """
    cpu = platform.processor() or platform.machine()
    try:
        if platform.system() == "Darwin":
            cpu = subprocess.check_output(
                ["sysctl", "-n", "machdep.cpu.brand_string"], text=True
            ).strip()
        elif platform.system() == "Linux":
            with open("/proc/cpuinfo") as cpuinfo:
                for line in cpuinfo:
                    if line.startswith("model name"):
                        cpu = line.split(":", 1)[1].strip()
                        break
    except (OSError, subprocess.SubprocessError):
        pass

    try:
        memory = os.sysconf("SC_PAGE_SIZE") * os.sysconf("SC_PHYS_PAGES")
        return cpu, f"{round(memory / 1024**3)} GB RAM"
    except (AttributeError, ValueError, OSError):
        return cpu, ""


def machine(device: str = "cpu") -> str:
    """Host CPU, torch thread count, memory, and the GPU of a CUDA solve.

    Threads are read per call, because `torch.set_num_threads` may change them.
    """
    cpu, memory = _host()
    parts = [cpu, _plural(torch.get_num_threads(), "threads")]
    if memory:
        parts.append(memory)
    if device == "cuda" and torch.cuda.is_available():
        parts.append(torch.cuda.get_device_name())
    return " | ".join(parts)


def _display_handle():
    """Return an IPython display handle, or None outside a notebook kernel."""
    try:
        from IPython.core.getipython import get_ipython
        from IPython.display import display
    except ImportError:
        return None
    shell = get_ipython()
    if shell is None or shell.__class__.__name__ != "ZMQInteractiveShell":
        return None
    return display({"text/plain": ""}, raw=True, display_id=True)


class SolveReport:
    """Table of solver progress with one row per increment.

    A notebook redraws the whole block in place through an IPython display
    handle, so a running increment shows its residual live and only the final
    table is stored. Everywhere else rows are streamed with `print`.

    Args:
        title: Title placed in the top rule.
        header: Label and text of the lines above the table.
        label: Name of the first column.
        value: Name of the second column.
        unit: Plural noun for one row, used in the summary.
    """

    def __init__(
        self,
        title: str,
        header: dict[str, str],
        label: str = "Increment",
        value: str = "Load factor",
        unit: str = "increments",
    ):
        self.unit = unit
        self.handle = _display_handle()
        self.rows: list[str] = []
        self.foot: list[str] = []
        self.total = 0
        self.t0 = self.drawn = time.perf_counter()
        self.running = False

        self.head = [
            _rule(title),
            *(f" {key:<8} {text}" for key, text in header.items()),
            _rule(),
            COLUMNS.format(
                label, value, "Steps", "Iterations", "Residual", "Wall time"
            ),
        ]
        self._emit(self.head)

    def begin(self, index: int, value: float) -> None:
        """Open the row for an increment."""
        self.index = index
        self.value = value
        self.steps = 0
        self.iters = 0
        self.res: float | None = None
        self.cutbacks = 0
        self.growths = 0
        self.t = time.perf_counter()
        self.running = True
        self._draw()

    def iteration(self, i: int, res: float) -> None:
        """Record the residual of Newton iteration `i`, which opens a substep at
        `i == 0`. The count reports linear solves, so a linear problem needs one.
        """
        if i == 0:
            self.steps += 1
        else:
            self.iters += 1
        self.res = float(res)
        self._draw()

    def cutback(self) -> None:
        """Record that the substep of the open increment was cut back."""
        self.cutbacks += 1
        self._draw()

    def growth(self) -> None:
        """Record that the substep of the open increment was grown again."""
        self.growths += 1
        self._draw()

    def end(self) -> None:
        """Close the row of the open increment."""
        self.running = False
        self.total += self.iters
        self.rows.append(self._row())
        self._emit(self.rows[-1:])

    def close(self) -> None:
        """Write the summary below the table."""
        self.foot = [
            _rule(),
            f" converged | {_plural(len(self.rows), self.unit)}"
            f" | {_plural(self.total, 'iterations')}"
            f" | {time.perf_counter() - self.t0:.2f} s",
        ]
        self._emit(self.foot)

    def _row(self) -> str:
        """One table row, with the substep flags in the right margin."""
        flags = ["..."] if self.running else []
        flags += [f"v{self.cutbacks}"] if self.cutbacks else []
        flags += [f"^{self.growths}"] if self.growths else []
        row = COLUMNS.format(
            self.index,
            f"{self.value:.4g}",
            self.steps,
            self.iters,
            f"{self.res:.2e}" if self.res is not None else "-",
            f"{time.perf_counter() - self.t:.2f} s",
        )
        return f"{row}  {' '.join(flags)}".rstrip()

    def _lines(self) -> list[str]:
        """Render the whole block, including the increment being solved."""
        running = [self._row()] if self.running else []
        return self.head + self.rows + running + self.foot

    def _emit(self, lines: list[str]) -> None:
        """Print `lines`, or redraw the whole notebook block unconditionally."""
        if self.handle is None:
            print("\n".join(lines))
        else:
            self._draw(force=True)

    def _draw(self, force: bool = False) -> None:
        """Redraw the notebook block, at most every `REFRESH` seconds, since
        Jupyter drops output above its message rate limit.
        """
        now = time.perf_counter()
        if self.handle is None or (not force and now - self.drawn < REFRESH):
            return
        self.drawn = now
        self.handle.update({"text/plain": "\n".join(self._lines())}, raw=True)
