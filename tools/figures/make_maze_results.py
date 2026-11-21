"""Maze-Hard results as horizontal bars on a cut scale (was Table 1). The x-axis jumps from 30% to 80%, so the LLMs
(26%, 26%, 12%) and the close scores near the top (84.6% to 100%) are both readable. Any shortest path counts. Trained
scores are from Davis et al. (NeurIPS 2026), Table 3; LLM scores from their Table 25.
Also draws maze_size: test mazes solved against weights (log scale), with logos.
Saves maze_results.* and maze_size.* (svg, png and the two layers)."""
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import make_figures as mf  # noqa: E402

HRM_TEXT = "#11805a"
ATT_TEXT = "#4a3bb0"      # Attractor Models (Fein-Ashley and Rashidinejad, 2026)
ROWS = [  # kind, label, score, colour, text colour, logo
    ("bar", "Written by Claude, 19,936 weights", 100.0, mf.WRITTEN, mf.WRITTEN_TEXT, "claude"),
    ("head", "Trained on Maze-Hard"),
    ("bar", "LDT, 1.8M weights", 99.8, mf.YELLOW, mf.YELLOW_TEXT, None),
    ("bar", "Attractor Models, 27M weights", 93.1, mf.VIOLET, ATT_TEXT, None),
    ("bar", "TRM, 7M weights", 91.9, mf.TRAINED, mf.TRAINED_TEXT, "samsung"),
    ("bar", "HRM, 27M weights", 84.6, mf.AQUA, HRM_TEXT, "sapient"),
    ("head", "Frontier LLMs, no maze training"),
    ("bar", "Claude Opus 5.5", 26, mf.YELLOW, mf.YELLOW_TEXT, "claude"),
    ("bar", "GPT-6 Astra", 26, mf.YELLOW, mf.YELLOW_TEXT, "openai"),
    ("bar", "GPT-6.1 Sol", 12, mf.YELLOW, mf.YELLOW_TEXT, "openai"),
]
CUT_LO, CUT_HI = 30, 80          # the scale jumps from 30% to 80%


def results(plt) -> None:
    fs = mf.F * 8.3
    W, H = 6.7, 3.75
    fig = plt.figure(figsize=(W, H))
    left, right, bottom, top = 0.425, 0.975, 0.15, 0.985
    gap = 0.018
    wl = (right - left - gap) * 0.36
    axl = fig.add_axes([left, bottom, wl, top - bottom])
    axr = fig.add_axes([left + wl + gap, bottom, right - left - gap - wl, top - bottom], sharey=axl)
    n = len(ROWS)
    for ax, lo, hi, ticks in ((axl, 0, CUT_LO, [0, 10, 20]), (axr, CUT_HI, 100, [80, 90, 100])):
        ax.set_xlim(lo, hi)
        ax.set_xticks(ticks)
        ax.grid(axis="x", color=mf.GRID, lw=0.8)
        ax.set_axisbelow(True)
        for s in ("top", "right", "left", "bottom"):
            ax.spines[s].set_visible(False)
        ax.tick_params(length=0, labelsize=fs, pad=5, labelleft=False)
    axl.set_ylim(n - 0.45, -0.6)
    axl.set_yticks([])
    fig.text(left + (right - left) / 2, 0.015, "Test mazes solved (%)",
             fontsize=fs, color=mf.MUTED, ha="center", va="bottom")
    # Break marks between the two parts of the scale.
    for ax, x in ((axl, 1.0), (axr, 0.0)):
        ax.plot([x - 0.02 * (1 if ax is axl else 0.55), x + 0.02 * (1 if ax is axl else 0.55)], [-0.02, 0.02],
                transform=ax.transAxes, color=mf.MUTED, lw=1.0, clip_on=False)

    h = 0.62
    for i, row in enumerate(ROWS):
        if row[0] == "head":
            fig.text(0.015, axl.transData.transform((0, i))[1] / fig.dpi / H, row[1], fontsize=fs * 0.86,
                     color=mf.MUTED, ha="left", va="center", fontweight="semibold")
            continue
        _, label, v, col, tcol, logo = row
        for ax in (axl, axr):
            ax.barh(i, v, height=h, color=col, edgecolor="none", zorder=3, gid="data")
        end_ax = axl if v <= CUT_LO else axr
        inside = v >= 84 or 20 <= v <= CUT_LO
        end_ax.text(v - 0.6 if inside else v + 0.6, i, f"{v:g}%", fontsize=fs * 0.95, va="center",
                    ha="right" if inside else "left", fontweight="semibold",
                    color=(mf.INK if col == mf.YELLOW else "white") if inside else tcol, zorder=4, gid="data")
        bold = label.startswith("Written")
        t = fig.text(left - 0.012, axl.transData.transform((0, i))[1] / fig.dpi / H, label, fontsize=fs,
                     ha="right", va="center", color=tcol if bold else mf.INK,
                     fontweight="semibold" if bold else "normal")
        if logo:
            mf.add_logos(fig, axl, t, [logo])
    mf.save(fig, "maze_results", "Maze-Hard results")
    plt.close(fig)


def size(plt) -> None:
    fs = mf.F * 8.3
    fig = plt.figure(figsize=(5.6, 3.75))
    ax = fig.add_axes([0.14, 0.17, 0.82, 0.78])
    ax.set_xscale("log")
    ax.set_xlim(5e2, 1.2e8)
    ax.set_ylim(79, 103.5)
    ax.set_yticks([80, 85, 90, 95, 100])
    ax.set_xticks([1e3, 1e4, 1e5, 1e6, 1e7, 1e8])
    ax.set_xticklabels(["1K", "10K", "100K", "1M", "10M", "100M"])
    ax.minorticks_off()
    ax.grid(axis="y", color=mf.GRID, lw=0.8)
    ax.set_axisbelow(True)
    for s in ("top", "right", "left"):
        ax.spines[s].set_visible(False)
    ax.spines["bottom"].set_color(mf.GRID)
    ax.tick_params(length=0, labelsize=fs, pad=5)
    ax.set_xlabel("Weights (log scale)", fontsize=fs, labelpad=6)
    ax.set_ylabel("Test mazes solved (%)", fontsize=fs, labelpad=8)

    pts = [  # x, y, colour, text colour, label, logo, label offset (points), alignment
        (19936, 100.0, mf.WRITTEN, mf.WRITTEN_TEXT, "Written by Claude, 100%", "claude", (0, 12), "center"),
        (1.8e6, 99.8, mf.YELLOW, mf.YELLOW_TEXT, "LDT, 99.8%", None, (0, 12), "center"),
        (7.0e6, 91.9, mf.TRAINED, mf.TRAINED_TEXT, "TRM, 91.9%", "samsung", (-10, 0), "right"),
        (2.7e7, 84.6, mf.AQUA, HRM_TEXT, "HRM, 84.6%", "sapient", (-10, 0), "right"),
        (2.7e7, 93.1, mf.VIOLET, ATT_TEXT, "Attractor Models, 93.1%", None, (-10, 0), "right"),
    ]
    for x, y, col, tcol, lab, logo, off, ha in pts:
        big = lab.startswith("Written")
        ax.scatter([x], [y], s=150 if big else 80, color=col, edgecolor="white", linewidth=1.6, zorder=5,
                   gid="data")
        t = ax.annotate(lab, (x, y), xytext=off, textcoords="offset points", ha=ha,
                        va="bottom" if off[1] > 0 else "center", fontsize=fs * (1.02 if big else 0.95),
                        color=tcol, fontweight="semibold" if big else "normal", gid="data")
        if logo:
            mf.add_logos(fig, ax, t, [logo], shift=(ha == "center"))
    # Five more transformers Claude wrote under a cap on the number of weights (sk_exps/10_maze_size_limit).
    capped = [928, 4359, 4380, 7682, 8876]
    ax.scatter(capped, [100.0] * len(capped), s=58, color=mf.WRITTEN, edgecolor="white", linewidth=1.3, zorder=5, gid="data")
    ax.annotate("written under a weight cap", (700, 100.0), xytext=(0, -9), textcoords="offset points",
                ha="left", va="top", fontsize=fs * 0.84, color=mf.WRITTEN_TEXT, gid="data")
    # How much smaller the smallest written transformer (928 weights, under a weight cap) is than LDT.
    ax.annotate("", xy=(1.8e6, 95.4), xytext=(928, 95.4),
                arrowprops=dict(arrowstyle="<->", color=mf.INK, lw=1.4, shrinkA=0, shrinkB=0), gid="data")
    ax.text((928 * 1.8e6) ** 0.5, 96.0, "1,900× fewer weights than LDT", fontsize=fs * 1.22, color=mf.INK,
            fontweight="semibold", ha="center", va="bottom", gid="data")
    mf.save(fig, "maze_size", "Accuracy against size")
    plt.close(fig)


if __name__ == "__main__":
    p = mf.setup_matplotlib()
    results(p)
    size(p)
