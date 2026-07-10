"""Generate benchmark plots. Usage: python benchmarks/plot.py"""

import json
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

RESULTS_DIR = Path(__file__).parent / "results"
IMAGES_DIR = Path(__file__).parent.parent / "docs" / "images"

# problem id -> (plot title, image filename prefix)
PROBLEM_META = {
    "cube_hexa_extension": ("Cube extension benchmark", "benchmark"),
    "thermal_slab_simp": ("Thermal SIMP slab benchmark", "benchmark_thermal"),
}


def load_results() -> dict[str, list[dict]]:
    """Load all result JSONs, grouped by problem id."""
    files = sorted(RESULTS_DIR.glob("*.json"))
    if not files:
        raise FileNotFoundError(f"No JSON files found in {RESULTS_DIR}")
    grouped: dict[str, list[dict]] = {}
    for f in files:
        ds = json.loads(f.read_text())
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
        _setup_ax(ax, f"{title} (GPU)", "Time (s)")

    fig.suptitle(f"{suite} — {title.lower()}", fontweight="bold")
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {out_path}")


def _setup_ax(ax, title: str, ylabel: str) -> None:
    ax.set_title(title)
    ax.set_xlabel("Degrees of freedom")
    ax.set_ylabel(ylabel)
    ax.xaxis.set_major_formatter(ticker.FuncFormatter(lambda x, _: f"{x:,.0f}"))
    ax.grid(True, which="both", linestyle="--", alpha=0.4)
    handles, labels = ax.get_legend_handles_labels()
    if handles:
        ax.legend(fontsize=7)


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
                [r["peak_ram_mb"] for r in ds["rows"]],
                marker="o",
                label=lbl,
                color=colors[lbl],
            )
        _setup_ax(ax, "Peak RAM (CPU)", "Peak RAM (MB)")

    if gpu_ds:
        ax = axes[-1]
        for ds in gpu_ds:
            dofs = [r["dofs"] for r in ds["rows"]]
            vram = [r.get("peak_vram_mb") for r in ds["rows"]]
            pairs = [(d, v) for d, v in zip(dofs, vram) if v is not None]
            if not pairs:
                continue
            x, y = zip(*pairs)
            lbl = _label(ds)
            ax.loglog(x, y, marker="o", label=lbl, color=colors[lbl])
        _setup_ax(ax, "Peak VRAM (GPU)", "Peak VRAM (MB)")

    fig.suptitle(f"{suite} — memory", fontweight="bold")
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {out_path}")


def main():
    IMAGES_DIR.mkdir(parents=True, exist_ok=True)
    for problem, datasets in load_results().items():
        suite, prefix = PROBLEM_META.get(problem, (problem, f"benchmark_{problem}"))
        plot_timing(
            datasets,
            IMAGES_DIR / f"{prefix}_timing.png",
            "fwd_s",
            "Forward solve",
            suite,
        )
        plot_timing(
            datasets,
            IMAGES_DIR / f"{prefix}_backward.png",
            "bwd_s",
            "Backward solve",
            suite,
        )
        plot_ram(datasets, IMAGES_DIR / f"{prefix}_ram.png", suite)


if __name__ == "__main__":
    main()
