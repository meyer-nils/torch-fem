"""Generate benchmark plots. Usage: python benchmarks/plot.py"""

import json
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from run import PROBLEMS

RESULTS_DIR = Path(__file__).parent / "results"
IMAGES_DIR = Path(__file__).parent.parent / "docs" / "images" / "benchmark"

# problem id -> (plot title, image filename prefix)
PROBLEM_META = {
    mod.PROBLEM.id: (mod.PROBLEM.title, mod.PROBLEM.plot_prefix)
    for mod in PROBLEMS.values()
}


# Phase breakdown of a run, in the order they are stacked.
PHASES = (("setup_s", "Setup"), ("fwd_s", "Forward"), ("bwd_s", "Backward"))

# Common device memory sizes (GB), marked in the memory plots for orientation.
MEMORY_GUIDES = (8, 16, 24, 32)

# Shared y range (GB) of the memory plots, so that suites compare directly.
MEMORY_YLIM = (0.1, 36)

# Shared degrees-of-freedom range of every plot.
DOFS_XLIM = (9_000, 11_000_000)

# Shared time range (s) of the timing plots, spanning every suite.
TIMING_YLIM = (0.1, 100)


def load_results() -> dict[str, list[dict]]:
    """Load all result JSONs, grouped by problem id."""
    files = sorted(RESULTS_DIR.glob("*.json"))
    if not files:
        raise FileNotFoundError(f"No JSON files found in {RESULTS_DIR}")
    grouped: dict[str, list[dict]] = {}
    for f in files:
        ds = json.loads(f.read_text())
        for row in ds["rows"]:
            row["total_s"] = sum(row[metric] for metric, _ in PHASES)
        grouped.setdefault(ds["problem"], []).append(ds)
    return grouped


def _label(ds: dict) -> str:
    return f"{ds['hardware']} ({ds['device'].upper()})"


def plot_timing(
    datasets: list[dict], out_path: Path, metric: str, title: str, suite: str
) -> None:
    cpu_ds = [ds for ds in datasets if ds["device"] != "cuda"]
    gpu_ds = [ds for ds in datasets if ds["device"] == "cuda"]
    colors = {_label(ds): f"C{i}" for i, ds in enumerate(datasets)}

    ncols = int(bool(cpu_ds)) + int(bool(gpu_ds))
    fig, axes = plt.subplots(1, ncols, figsize=(5 * ncols, 4), squeeze=False)
    axes = axes[0]

    if cpu_ds:
        ax = axes[0]
        for ds in cpu_ds:
            lbl = _label(ds)
            ax.loglog(
                [r["dofs"] for r in ds["rows"]],
                [r[metric] for r in ds["rows"]],
                marker="o",
                label=lbl,
                color=colors[lbl],
            )
        ax.set_ylim(*TIMING_YLIM)
        _setup_ax(ax, f"{title} (CPU)", "Time (s)")

    if gpu_ds:
        ax = axes[-1]
        for ds in gpu_ds:
            lbl = _label(ds)
            ax.loglog(
                [r["dofs"] for r in ds["rows"]],
                [r[metric] for r in ds["rows"]],
                marker="o",
                label=lbl,
                color=colors[lbl],
            )
        ax.set_ylim(*TIMING_YLIM)
        _setup_ax(ax, f"{title} (GPU)", "Time (s)")

    fig.suptitle(f"{suite} — {title.lower()}", fontweight="bold")
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight", transparent=True)
    plt.close(fig)
    print(f"Saved {out_path}")


class PlainLogFormatter(ticker.LogFormatterSciNotation):
    """Log tick labels as plain numbers: 100 instead of 10^2, 0.1 instead of 10^-1."""

    def __call__(self, x: float, pos=None) -> str:
        # The parent decides which ticks carry a label; only the text differs.
        if not super().__call__(x, pos):
            return ""
        return f"{x:,.0f}" if x >= 1 else f"{x:g}"


def _setup_ax(ax, title: str, ylabel: str) -> None:
    ax.set_title(title)
    ax.set_xlabel("Degrees of freedom")
    ax.set_ylabel(ylabel)
    ax.set_xlim(*DOFS_XLIM)
    ax.xaxis.set_major_formatter(PlainLogFormatter())
    ax.yaxis.set_major_formatter(PlainLogFormatter())
    ax.grid(True, which="major", linestyle="-", alpha=0.25)
    ax.grid(True, which="minor", linestyle="--", alpha=0.25)
    handles, labels = ax.get_legend_handles_labels()
    if handles:
        ax.legend(fontsize=7)


def _memory_guides(ax) -> None:
    """Mark common device memory sizes and fix the y range to MEMORY_YLIM."""
    ax.set_ylim(*MEMORY_YLIM)
    for gb in MEMORY_GUIDES:
        ax.axhline(gb, color="gray", linewidth=1.8, alpha=0.7, zorder=1)
        ax.annotate(
            f"{gb} GB",
            xy=(0.99, gb),
            xycoords=("axes fraction", "data"),
            xytext=(0, -1.5),
            textcoords="offset points",
            ha="right",
            va="top",
            fontsize=6,
            color="gray",
        )


def plot_ram(datasets: list[dict], out_path: Path, suite: str) -> None:
    cpu_ds = [ds for ds in datasets if ds["device"] != "cuda"]
    gpu_ds = [ds for ds in datasets if ds["device"] == "cuda"]
    colors = {_label(ds): f"C{i}" for i, ds in enumerate(datasets)}

    ncols = int(bool(cpu_ds)) + int(bool(gpu_ds))
    fig, axes = plt.subplots(1, ncols, figsize=(5 * ncols, 4), squeeze=False)
    axes = axes[0]

    if cpu_ds:
        ax = axes[0]
        for ds in cpu_ds:
            lbl = _label(ds)
            ax.loglog(
                [r["dofs"] for r in ds["rows"]],
                [r["peak_ram_mb"] / 1024 for r in ds["rows"]],
                marker="o",
                label=lbl,
                color=colors[lbl],
            )
        _memory_guides(ax)
        _setup_ax(ax, "Peak RAM (CPU)", "Peak RAM (GB)")

    if gpu_ds:
        ax = axes[-1]
        for ds in gpu_ds:
            dofs = [r["dofs"] for r in ds["rows"]]
            vram = [r.get("peak_vram_mb") for r in ds["rows"]]
            pairs = [(d, v / 1024) for d, v in zip(dofs, vram) if v is not None]
            if not pairs:
                continue
            x, y = zip(*pairs)
            lbl = _label(ds)
            ax.loglog(x, y, marker="o", label=lbl, color=colors[lbl])
        _memory_guides(ax)
        _setup_ax(ax, "Peak VRAM (GPU)", "Peak VRAM (GB)")

    fig.suptitle(f"{suite} — memory", fontweight="bold")
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight", transparent=True)
    plt.close(fig)
    print(f"Saved {out_path}")


def main():
    IMAGES_DIR.mkdir(parents=True, exist_ok=True)
    for problem, datasets in load_results().items():
        suite, prefix = PROBLEM_META.get(problem, (problem, problem))
        # One variant per docs color scheme, selected via #only-light / #only-dark.
        for theme, style in (("light", "default"), ("dark", "dark_background")):
            with plt.style.context(style):
                plot_timing(
                    datasets,
                    IMAGES_DIR / f"{prefix}_timing_{theme}.png",
                    "total_s",
                    "Total time",
                    suite,
                )
                plot_ram(datasets, IMAGES_DIR / f"{prefix}_ram_{theme}.png", suite)


if __name__ == "__main__":
    main()
