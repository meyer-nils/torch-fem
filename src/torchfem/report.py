import os
import platform
import sys
import time

import torch

from .sparse import describe_method

WIDTH = 88
# The columns leave a margin at the right edge for the substep flags.
COLUMNS = "  {:>9}  {:>11}  {:>5}  {:>6}  {:>6}  {:>8}  {:>9}"
# Rule, separator and substep flags, in ASCII where a console cannot encode them
RICH, ASCII = ("\u2500", " \u00b7 ", "\u2193", "\u2191"), ("-", " | ", "v", "^")


def _glyphs() -> tuple[str, str, str, str]:
    try:
        "".join(RICH).encode(getattr(sys.stdout, "encoding", None) or "ascii")
        return RICH
    except (UnicodeEncodeError, LookupError):
        return ASCII


def _plural(n: int, unit: str) -> str:
    """Count of `unit`, dropping its trailing `s` for a single item."""
    return f"{n} {unit if n != 1 else unit.removesuffix('s')}"


def machine(device: str = "cpu") -> str:
    """Host processor, torch thread count and memory, and the GPU of a CUDA solve.

    Threads are read per call, because `torch.set_num_threads` may change them.
    """
    cpu = platform.processor() or platform.machine()
    parts = [cpu, _plural(torch.get_num_threads(), "threads")]
    if hasattr(os, "sysconf"):  # not on Windows
        ram = os.sysconf("SC_PAGE_SIZE") * os.sysconf("SC_PHYS_PAGES")
        parts.append(f"{ram / 1024**3:.0f} GB RAM")
    if device == "cuda" and torch.cuda.is_available():
        _, vram = torch.cuda.mem_get_info()
        parts.append(f"{torch.cuda.get_device_name()} | {vram / 1024**3:.0f} GB VRAM")
    return " | ".join(parts)


class SolveReport:
    """Table of solver progress, printed one row per increment as it completes.

    It is drawn with arrows and rules where the console can encode them, and in
    ASCII otherwise.

    Args:
        header: Label and text of the lines above the table. None for a report
            that records progress but writes nothing.
        label: Name of the first column.
        value: Name of the second column.
        unit: Plural noun for one row, used in the summary.
    """

    def __init__(
        self,
        header: dict[str, str] | None,
        label: str = "Increment",
        value: str = "Load factor",
        unit: str = "increments",
    ):
        self.silent = header is None
        self.unit = unit
        self.rule, self.sep, self.down, self.up = _glyphs()
        self.rule *= WIDTH
        self.rows = 0
        self.total = 0
        self.total_krylov = 0
        self.t0 = time.perf_counter()
        self._print(
            self.rule,
            *(f" {key:<8} {text}" for key, text in (header or {}).items()),
            self.rule,
            COLUMNS.format(
                label, value, "Steps", "Newton", "Krylov", "Residual", "Wall time"
            ),
        )

    def begin(self, index: int, value: float) -> None:
        """Open the row for an increment."""
        self.index = index
        self.value = value
        self.steps = 0
        self.iters = 0
        self.res: float | None = None
        self.krylov = 0
        self.cutbacks = 0
        self.growths = 0
        self.t = time.perf_counter()

    def iteration(self, i: int, res: float) -> None:
        """Record the residual of Newton iteration `i`, which opens a substep at
        `i == 0`. The count reports linear solves, so a linear problem needs one.
        """
        if self.silent:
            return
        if i == 0:
            self.steps += 1
        else:
            self.iters += 1
        self.res = float(res)

    def linear(self, iterations: int) -> None:
        """Record the iterations of a linear solve, which a direct solve has none of."""
        self.krylov += iterations

    def cutback(self) -> None:
        """Record that the substep of the open increment was cut back."""
        self.cutbacks += 1

    def growth(self) -> None:
        """Record that the substep of the open increment was grown again."""
        self.growths += 1

    def end(self) -> None:
        """Print the row of the open increment, with the substep flags in the margin."""
        self.rows += 1
        self.total += self.iters
        self.total_krylov += self.krylov
        flags = [f"{self.down}{self.cutbacks}"] if self.cutbacks else []
        flags += [f"{self.up}{self.growths}"] if self.growths else []
        row = COLUMNS.format(
            self.index,
            f"{self.value:.4g}",
            self.steps,
            self.iters,
            self.krylov or "-",
            f"{self.res:.2e}" if self.res is not None else "-",
            f"{time.perf_counter() - self.t:.2f} s",
        )
        self._print(f"{row}  {' '.join(flags)}".rstrip())

    def close(self) -> None:
        """Print the summary below the table."""
        counts = _plural(self.total, "Newton iterations")
        if self.total_krylov:
            counts += f" | {_plural(self.total_krylov, 'Krylov iterations')}"
        self._print(
            self.rule,
            f" converged | {_plural(self.rows, self.unit)}"
            f" | {counts}"
            f" | {time.perf_counter() - self.t0:.2f} s",
        )

    def _print(self, *lines: str) -> None:
        """Print the lines, with the separator the console can encode."""
        if not self.silent:
            print("\n".join(lines).replace(" | ", self.sep))


def solve_report(
    verbose: bool,
    model: str,
    method: str,
    preconditioner: str | None,
    device: str,
    newton: str,
    **kwargs: str,
) -> SolveReport:
    """Open a report on a solve, which writes nothing unless verbose.

    It is headed by `model`, the machine, the linear solver named from `method`
    and `preconditioner`, and `newton`.
    """
    if not verbose:
        return SolveReport(None)
    dtype = str(torch.get_default_dtype()).removeprefix("torch.")
    header = {
        "model": f"{model} | {dtype}",
        "machine": machine(device),
        "solver": describe_method(method, device, preconditioner),
        "newton": newton,
    }
    if dtype != "float64":
        header["warning"] = (
            "single precision, prefer torch.set_default_dtype(torch.float64)"
        )
    return SolveReport(header, **kwargs)
