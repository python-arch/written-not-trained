"""Figure: Tower of Hanoi accuracy against disks. The transformer Claude wrote (100% at every size from 1 to 20 disks,
checked only up to 6 while building) against two reasoning models from Shojaee et al. 2025, Figure 8, default prompts
(hanoi_reasoning_models.json, read from the paper's vector figures). Saves hanoi.svg/.png and the two layers."""
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import make_figures as mf  # noqa: E402


def main() -> None:
    plt = mf.setup_matplotlib()
    F = mf.F
    rm = json.loads((HERE / "hanoi_reasoning_models.json").read_text(encoding="utf-8"))["models"]
    fs = F * 8.3

    fig = plt.figure(figsize=(5.8, 3.75))
    ax = fig.add_axes([0.125, 0.17, 0.83, 0.66])
    ax.set_xlim(0.5, 20.5)
    ax.set_ylim(-5, 110)
    ax.set_xticks([1, 5, 10, 15, 20])
    ax.set_yticks([0, 25, 50, 75, 100])
    ax.grid(axis="y", color=mf.GRID, lw=0.8)
    ax.set_axisbelow(True)
    for s in ("top", "right", "left"):
        ax.spines[s].set_visible(False)
    ax.spines["bottom"].set_color(mf.GRID)
    ax.tick_params(length=0, labelsize=fs, pad=5)
    ax.set_xlabel("Disks", fontsize=fs, labelpad=6)
    ax.set_ylabel("Accuracy (%)", fontsize=fs, labelpad=8)

    # The sizes Claude checked while building, and the largest size the paper tests.
    ax.axvspan(0.5, 6.5, color=mf.BAND, lw=0, zorder=0)
    ax.text(3.5, 8, "Checked while\nbuilding", fontsize=fs * 0.9, color=mf.MUTED, ha="center", va="center",
            linespacing=1.3)
    ax.axvline(12.5, color=mf.MUTED, lw=0.9, ls=(0, (3, 2.5)), zorder=1)
    ax.text(12.75, 8, "Largest size\nthe paper tests", fontsize=fs * 0.9, color=mf.MUTED, ha="left", va="center",
            linespacing=1.3)

    # Reasoning models (default prompts).
    styles = {"Claude 3.7 Sonnet (thinking)": (mf.YELLOW, mf.YELLOW_TEXT, "claude", (8.25, 76)),
              "DeepSeek-R1": (mf.TRAINED, mf.TRAINED_TEXT, "deepseek", (8.55, 41))}
    labels = []
    for name, pts in rm.items():
        col, tcol, logo, (lx, ly) = styles[name]
        xs, ys = zip(*pts)
        ax.plot(xs, ys, color=col, lw=2.1, zorder=3, solid_joinstyle="round", solid_capstyle="round", gid="data")
        ax.scatter(xs, ys, s=22, color=col, edgecolor="white", linewidth=0.9, zorder=4, gid="data")
        t = ax.text(lx, ly, name, fontsize=fs, color=tcol, fontweight="semibold", ha="left", va="center",
                    gid="data")
        labels.append((t, logo))

    # The transformer Claude wrote: 100% at every size, 1 to 20 disks.
    ns = list(range(1, 21))
    ax.plot(ns, [100] * len(ns), color=mf.WRITTEN, lw=3.0, zorder=6, solid_capstyle="round", gid="data")
    ax.scatter(ns, [100] * len(ns), s=34, color=mf.WRITTEN, edgecolor="white", linewidth=1.1, zorder=7,
               gid="data")
    t_ours = ax.text(20.45, 105, "Written by Claude", fontsize=fs * 1.04, color=mf.WRITTEN_TEXT,
                     fontweight="semibold", ha="right", va="bottom", gid="data")
    ax.annotate("1,048,575 moves at 20 disks,\nzero mistakes", xy=(20, 100), xytext=(20.45, 47),
                fontsize=fs * 0.95, color=mf.WRITTEN_TEXT, ha="right", va="center", linespacing=1.3, gid="data",
                arrowprops=dict(arrowstyle="-", color=mf.WRITTEN, lw=0.8, shrinkA=2, shrinkB=5))

    for t, logo in labels:
        mf.add_logos(fig, ax, t, [logo], shift=True)
    mf.add_logos(fig, ax, t_ours, ["claude"])

    mf.save(fig, "hanoi", "Tower of Hanoi accuracy against disks")
    plt.close(fig)


if __name__ == "__main__":
    main()
