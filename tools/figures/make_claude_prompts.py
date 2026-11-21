"""Claude solving Maze-Hard itself, with better prompts, against the transformer Claude wrote (was Table 1).
Bars are the share of the same test mazes solved (any shortest path counts), one try per maze; error bars are 95%
Wilson intervals (sk_exps/8_table1_intervals). The first row is Davis et al.'s published result for Claude Opus 5.5;
the next four are our runs with the same model (prompts in loopws/prompts/08_*). Saves claude_prompts.* and layers."""
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import make_figures as mf  # noqa: E402

GREY_FILL, GREY_EDGE, GREY_TEXT, ICON = "#ecebe7", "#a9a8a1", "#6f6e69", "#7a7a76"
ROWS = [  # kind, label, value, low, high, icon
    ("head", "Claude solves the maze itself"),
    ("bar", "The paper's prompt", 26, 15.9, 39.6, "doc"),
    ("bar", "Told how to search", 48, 34.8, 61.5, "wave"),
    ("bar", "+ one cell per token", 44, 31.2, 57.7, "cells"),
    ("bar", "+ answer with the path only", 50, 36.6, 63.4, "path"),
    ("bar", "+ path only, no grid", 38, 25.9, 51.8, "list"),
    ("head", "Claude writes a transformer"),
    ("bar", "The transformer in this post", 100, 92.9, 100, "claude"),
]


def icon(fig, kind, cx, cy, size):
    """A small pictogram centred at figure coordinates (cx, cy), size = height in figure-height units."""
    from matplotlib.patches import Circle, FancyBboxPatch, Polygon, Rectangle
    w_in, h_in = fig.get_size_inches()
    ia = fig.add_axes([cx - size * h_in / w_in / 2, cy - size / 2, size * h_in / w_in, size])
    ia.set_xlim(0, 1); ia.set_ylim(0, 1); ia.axis("off"); ia.set_aspect("equal")
    lw = 1.25
    if kind == "doc":
        ia.add_patch(Polygon([(0.2, 0.05), (0.2, 0.95), (0.62, 0.95), (0.82, 0.75), (0.82, 0.05)], closed=True,
                             fill=False, ec=ICON, lw=lw))
        ia.plot([0.62, 0.62, 0.82], [0.95, 0.75, 0.75], color=ICON, lw=lw)
        for y in (0.55, 0.38, 0.21):
            ia.plot([0.33, 0.69], [y, y], color=ICON, lw=lw)
    elif kind == "wave":
        for r, a in ((0.42, 0.45), (0.27, 0.75)):
            ia.add_patch(Circle((0.5, 0.5), r, fill=False, ec=ICON, lw=lw, alpha=a))
        ia.add_patch(Circle((0.5, 0.5), 0.1, color=ICON))
    elif kind == "cells":
        for i in range(3):
            for j in range(3):
                ia.add_patch(Rectangle((0.1 + i * 0.29, 0.1 + j * 0.29), 0.22, 0.22, fill=(i + j) % 2 == 0,
                                       color=ICON, lw=lw * 0.8))
    elif kind == "path":
        ia.plot([0.12, 0.12, 0.5, 0.5, 0.88], [0.12, 0.6, 0.6, 0.88, 0.88], color=ICON, lw=lw * 1.3,
                solid_joinstyle="round")
        ia.add_patch(Circle((0.12, 0.12), 0.09, color=ICON)); ia.add_patch(Circle((0.88, 0.88), 0.09, color=ICON))
    elif kind == "list":
        for y in (0.78, 0.5, 0.22):
            ia.add_patch(Circle((0.17, y), 0.07, color=ICON))
            ia.plot([0.34, 0.88], [y, y], color=ICON, lw=lw)
    elif kind == "claude":
        from matplotlib.offsetbox import AnnotationBbox, OffsetImage
        im = mf.logo_img("claude")
        ab = AnnotationBbox(OffsetImage(im, zoom=size * h_in * 72 * 0.9 / im.shape[0] * (fig.dpi / 72)),
                            (0.5, 0.5), xycoords="data", frameon=False, pad=0)
        ia.add_artist(ab)
    del FancyBboxPatch
    return ia


def main() -> None:
    plt = mf.setup_matplotlib()
    fs = mf.F * 8.3
    W, H = 5.6, 4.1
    fig = plt.figure(figsize=(W, H))
    left, right, bottom, top = 0.43, 0.965, 0.14, 0.985
    ax = fig.add_axes([left, bottom, right - left, top - bottom])
    n = len(ROWS)
    ax.set_xlim(0, 113)
    ax.set_ylim(n - 0.45, -0.6)
    ax.set_xticks([0, 25, 50, 75, 100])
    ax.set_yticks([])
    ax.grid(axis="x", color=mf.GRID, lw=0.8)
    ax.set_axisbelow(True)
    for s in ("top", "right", "left", "bottom"):
        ax.spines[s].set_visible(False)
    ax.tick_params(length=0, labelsize=fs, pad=5)
    fig.text(left + (right - left) * 100 / 113 / 2, 0.015, "Test mazes solved (%)", fontsize=fs, color=mf.MUTED,
             ha="center", va="bottom")
    to_fig = lambda i: ax.transData.transform((0, i))[1] / fig.dpi / H
    h = 0.6
    claude_rows = [i for i, r in enumerate(ROWS) if r[0] == "bar" and r[5] != "claude"]
    # nothing Claude tried on its own gets past 50%
    ax.plot([50, 50], [claude_rows[0] - 0.5, claude_rows[-1] + 0.5], color=mf.INK, lw=1.0, ls=(0, (3, 2.5)),
            zorder=4, gid="data")
    ax.text(51, claude_rows[0] - 0.62, "none passes 50%", fontsize=fs * 0.82, color=mf.INK, ha="left", va="center",
            gid="data")
    for i, row in enumerate(ROWS):
        y = to_fig(i)
        if row[0] == "head":
            fig.text(0.02, y, row[1], fontsize=fs * 0.86, color=mf.MUTED, ha="left", va="center",
                     fontweight="semibold")
            continue
        _, label, v, lo, hi, ic = row
        written = ic == "claude"
        if written:
            ax.barh(i, v, height=h, color=mf.WRITTEN, edgecolor="none", zorder=3, gid="data")
        else:
            ax.barh(i, v, height=h, color=GREY_FILL, edgecolor=GREY_EDGE, hatch="////", lw=0.8, zorder=3, gid="data")
        ax.errorbar(v, i, xerr=[[v - lo], [hi - v]], fmt="none", ecolor=mf.INK if written else "#4d4c48",
                    elinewidth=1.3, capsize=3.5, capthick=1.3, zorder=5, gid="data")
        xl = hi + 1.6
        if xl < 50 < xl + 9:                      # keep the value label off the 50% line
            xl = 51.5
        ax.text(xl, i, f"{v:g}%", fontsize=fs * (1.02 if written else 0.95), va="center", ha="left",
                color=mf.WRITTEN_TEXT if written else GREY_TEXT, fontweight="bold" if written else "semibold",
                zorder=6, gid="data")
        icon(fig, ic, 0.045, y, 0.062)
        fig.text(0.085, y, label, fontsize=fs, ha="left", va="center",
                 color=mf.WRITTEN_TEXT if written else mf.INK, fontweight="semibold" if written else "normal")
    mf.save(fig, "claude_prompts", "Claude with better prompts")
    plt.close(fig)


if __name__ == "__main__":
    main()
