"""Generate benchmark plots. Usage: python benchmarks/plot.py"""

import json
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

RESULTS_DIR = Path(__file__).parent / "results"
IMAGES_DIR = Path(__file__).parent.parent / "docs" / "images"


def load_results() -> list[dict]:
    files = sorted(RESULTS_DIR.glob("*.json"))
    if not files:
        raise FileNotFoundError(f"No JSON files found in {RESULTS_DIR}")
    return [json.loads(f.read_text()) for f in files]


def _label(ds: dict) -> str:
    return f"{ds['hardware']} ({ds['device'].upper()})"


def plot_timing(datasets: list[dict], out_path: Path) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(10, 4), sharex=True)

    for ds in datasets:
        dofs = [r["dofs"] for r in ds["rows"]]
        lbl = _label(ds)
        axes[0].loglog(dofs, [r["fwd_s"] for r in ds["rows"]], marker="o", label=lbl)
        axes[1].loglog(dofs, [r["bwd_s"] for r in ds["rows"]], marker="o", label=lbl)

    for ax, title in zip(axes, ["Forward solve", "Backward solve"]):
        _setup_ax(ax, title, "Time (s)")

    fig.suptitle("Cube extension benchmark — solve times", fontweight="bold")
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


def plot_ram(datasets: list[dict], out_path: Path) -> None:
    cpu_ds = [ds for ds in datasets if ds["device"] != "cuda"]
    gpu_ds = [ds for ds in datasets if ds["device"] == "cuda"]

    fig, axes = plt.subplots(1, 2, figsize=(10, 4), squeeze=False)
    axes = axes[0]

    if cpu_ds:
        ax = axes[0]
        for ds in cpu_ds:
            ax.loglog(
                [r["dofs"] for r in ds["rows"]],
                [r["peak_ram_mb"] for r in ds["rows"]],
                marker="o",
                label=_label(ds),
            )
        _setup_ax(ax, "Peak RAM — CPU runs", "Peak RAM (MB)")

    if gpu_ds:
        ax = axes[-1]
        for ds in gpu_ds:
            dofs = [r["dofs"] for r in ds["rows"]]
            vram = [r.get("peak_vram_mb") for r in ds["rows"]]
            pairs = [(d, v) for d, v in zip(dofs, vram) if v is not None]
            if not pairs:
                continue
            x, y = zip(*pairs)
            ax.loglog(x, y, marker="o", label=_label(ds))
        _setup_ax(ax, "Peak VRAM — GPU runs", "Peak VRAM (MB)")

    fig.suptitle("Cube extension benchmark — memory", fontweight="bold")
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {out_path}")


def main():
    IMAGES_DIR.mkdir(parents=True, exist_ok=True)
    datasets = load_results()
    plot_timing(datasets, IMAGES_DIR / "benchmark_timing.png")
    plot_ram(datasets, IMAGES_DIR / "benchmark_ram.png")


if __name__ == "__main__":
    main()
