#!/usr/bin/env python3
"""Build the figures for index.html from figures/data.json.

    python3 figures/make_figures.py          # check numbers, then draw every figure
    python3 figures/make_figures.py --check  # only check numbers
    python3 figures/make_figures.py --only glance,maze_size   # draw only these figures

The maze accuracy-vs-loops curve is read from figures/maze_loops.json (plotted data, not page numbers).

Rule: every number in data.json (values and numbers inside labels) must appear in
the visible text of index.html. The script stops before drawing if one does not.
Only two kinds of numbers are allowed without being on the page: axis ticks, and
values computed by formula (Tower of Hanoi's minimum move count, 2^N - 1).

Output: figures/NAME.svg and figures/NAME.png (PNG at 2x the SVG's CSS size),
plus figures/social.png (1200 x 630) for link previews.
Needs matplotlib. Fonts: Inter (SIL OFL, figures/fonts/).
"""
from __future__ import annotations

import html
import json
import re
import sys
import textwrap
from decimal import Decimal
from html.parser import HTMLParser
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
PAGE = ROOT / "index.html"
DATA = HERE / "data.json"

# Fixed colours (HANDOVER.md): hand-written, trained, frontier/pretrained LLMs.
WRITTEN = "#154677"   # MBZUAI blue
TRAINED = "#868e96"
LLM = "#8a6a1f"       # dark gold
INK = "#14213d"      # navy text (page theme)
MUTED = "#5b6475"
GRID = "#e3e8f0"
BAND = "#eef2f8"

CSS_DPI = 96          # 1 CSS px = 1/96 in
F = 1.18              # text scale: keeps figure text legible at phone width
PNG_SCALE = 2         # "2x" PNGs


# --------------------------------------------------------------------------- check
class _Visible(HTMLParser):
    """Collect text a reader can see: no <head>, <script>, <style>, comments."""

    SKIP = {"head", "script", "style", "title"}

    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self.depth = 0
        self.parts: list[str] = []

    def handle_starttag(self, tag, attrs):
        if tag in self.SKIP:
            self.depth += 1

    def handle_endtag(self, tag):
        if tag in self.SKIP and self.depth:
            self.depth -= 1

    def handle_data(self, data):
        if not self.depth:
            self.parts.append(data)


NUM = re.compile(r"\d[\d,]*(?:\.\d+)?")


def numbers_in(text: str) -> list[Decimal]:
    out = []
    for tok in NUM.findall(text):
        tok = tok.rstrip(",")
        if re.fullmatch(r"\d{1,3}(,\d{3})+(\.\d+)?|\d+(\.\d+)?", tok):
            out.append(Decimal(tok.replace(",", "")))
        else:  # e.g. "1,2" in a list: split on commas
            out += [Decimal(t) for t in tok.split(",") if t]
    return out


def page_numbers() -> set[Decimal]:
    p = _Visible()
    p.feed(PAGE.read_text(encoding="utf-8"))
    return set(numbers_in(html.unescape(" ".join(p.parts))))


def data_numbers(node, path="data") -> list[tuple[str, Decimal]]:
    found: list[tuple[str, Decimal]] = []
    if isinstance(node, dict):
        for k, v in node.items():
            if k.startswith("_"):
                continue
            found += data_numbers(v, f"{path}.{k}")
    elif isinstance(node, list):
        for i, v in enumerate(node):
            found += data_numbers(v, f"{path}[{i}]")
    elif isinstance(node, bool) or node is None:
        pass
    elif isinstance(node, (int, float)):
        found.append((path, Decimal(str(node))))
    elif isinstance(node, str):
        found += [(path, n) for n in numbers_in(node)]
    return found


def check(data) -> None:
    on_page = page_numbers()
    missing = [(p, n) for p, n in data_numbers(data) if n not in on_page]
    if missing:
        print("FAIL: numbers in data.json that are not in index.html's visible text:")
        for p, n in missing:
            print(f"  {p}: {n}")
        sys.exit(1)
    print(f"check ok: all {len(data_numbers(data))} numbers in data.json appear in index.html")


# --------------------------------------------------------------------------- style
def setup_matplotlib():
    import matplotlib

    matplotlib.use("Agg")
    from matplotlib import font_manager, rcParams

    for f in sorted((HERE / "fonts").glob("Inter-*.ttf")):
        font_manager.fontManager.addfont(str(f))
    rcParams.update({
        "font.family": "Inter",
        "font.size": 10,
        "text.color": INK,
        "axes.edgecolor": GRID,
        "axes.labelcolor": MUTED,
        "xtick.color": MUTED,
        "ytick.color": MUTED,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "mathtext.fontset": "custom",
        "mathtext.rm": "Inter",
        "mathtext.it": "Inter",
        "mathtext.bf": "Inter:bold",
        "svg.fonttype": "path",          # text as outlines: identical everywhere
        "svg.hashsalt": "written-not-trained",
        "figure.facecolor": "white",
        "axes.facecolor": "white",
    })
    import matplotlib.pyplot as plt

    return plt


def crop_box(fig, pad: float = 0.1):
    """Full figure width, cropped top and bottom to what is drawn, so no empty band is left."""
    from matplotlib.text import Text
    from matplotlib.transforms import Bbox
    fig.canvas.draw()
    r = fig.canvas.get_renderer()
    boxes = []
    for ax in fig.axes:
        if ax.axison:
            boxes.append(ax.get_tightbbox(r))
        for a in ax.get_children():
            if a is ax.patch or not a.get_visible() or a in ax.spines.values():
                continue
            if isinstance(a, Text) and not a.get_text().strip():
                continue
            try:
                bb = a.get_window_extent(r)
            except Exception:
                continue
            if bb.width > 0 or bb.height > 0:
                boxes.append(bb)
    for t in fig.texts:
        if t.get_visible() and t.get_text().strip():
            boxes.append(t.get_window_extent(r))
    u = Bbox.union([b for b in boxes if b is not None and abs(b.y0) < 1e5 and abs(b.y1) < 1e5])
    W, H = fig.get_size_inches()
    y0 = max(0.0, u.y0 / fig.dpi - pad)
    y1 = min(H, u.y1 / fig.dpi + pad)
    return Bbox.from_extents(0, y0, W, y1)


def save(fig, name: str, title: str) -> None:
    svg = HERE / f"{name}.svg"
    png = HERE / f"{name}.png"
    box = crop_box(fig)
    fig.savefig(svg, metadata={"Title": title, "Date": None, "Creator": None}, bbox_inches=box)
    fig.savefig(png, dpi=CSS_DPI * PNG_SCALE, metadata={"Software": None}, bbox_inches=box)
    print(f"wrote {svg.name} ({box.width * CSS_DPI:.0f}x{box.height * CSS_DPI:.0f} CSS px) and {png.name}")
    save_layers(fig, name, title, box)


def save_layers(fig, name: str, title: str, box=None) -> None:
    """If some artists are marked gid="data", also save the chart as two layers of the same size: the frame
    (axes, labels, bands) and the data alone on a clear background. The page stacks them and draws the data in when
    the chart scrolls into view. Stacked, they look exactly like the full chart."""
    data = [a for a in fig.findobj() if a.get_gid() == "data"]
    if not data:
        return
    meta = {"Title": title, "Date": None, "Creator": None}
    for a in data:
        a.set_visible(False)
    fig.savefig(HERE / f"{name}_base.svg", metadata=meta, bbox_inches=box)
    for a in data:
        a.set_visible(True)
    hidden = [t for t in fig.texts if t.get_visible()]
    for ax in fig.axes:
        hidden += [c for c in ax.get_children() if c.get_visible() and c.get_gid() != "data"]
    for a in hidden:
        a.set_visible(False)
    fig.savefig(HERE / f"{name}_data.svg", metadata=meta, transparent=True, bbox_inches=box)
    for a in hidden:
        a.set_visible(True)
    print(f"wrote {name}_base.svg and {name}_data.svg")


_LOGOS = {}


def logo_img(name):
    """A logo as an RGBA array (logos/NAME.png, cropped, on a clear background)."""
    if name not in _LOGOS:
        import numpy as np
        from PIL import Image
        _LOGOS[name] = np.asarray(Image.open(HERE / "logos" / f"{name}.png").convert("RGBA")) / 255.0
    return _LOGOS[name]


def logo_width(name, h):
    im = logo_img(name)
    return im.shape[1] / im.shape[0] * h


WORDMARKS = {"samsung"}


def add_logos(fig, ax, txt, names, gap=3.2, shift=False, also=()):
    """Logos about one text-height tall, just before a label, centred on its first line (base layer).
    shift=True moves a left-aligned label (and the labels in `also`) right to make room for them."""
    from matplotlib.offsetbox import AnnotationBbox, OffsetImage
    from matplotlib.text import Annotation
    from matplotlib.transforms import offset_copy
    fs = txt.get_fontsize()
    hs = [fs * (0.82 if n in WORDMARKS else 1.05) for n in names]
    ws = [logo_width(n, h) for n, h in zip(names, hs)]
    total = sum(ws) + gap * len(names)
    if shift:
        for t in (txt, *also):
            if isinstance(t, Annotation) and t.anncoords == "offset points":
                x, y = t.xyann
                t.xyann = (x + total, y)
            else:
                t.set_transform(offset_copy(t.get_transform(), fig=fig, x=total, units="points"))
    fig.canvas.draw()
    bb = txt.get_window_extent(fig.canvas.get_renderer())
    n_lines = txt.get_text().count("\n") + 1
    yc = (bb.y0 + bb.y1) / 2 if n_lines == 1 else bb.y1 - bb.height / n_lines * 0.55
    k = fig.dpi / 72
    x = bb.x0 - gap * k
    for name, h, w in reversed(list(zip(names, hs, ws))):
        im = logo_img(name)
        at = ax.transData.inverted().transform((x - w * k / 2, yc))   # data coordinates: unaffected by cropping
        ab = AnnotationBbox(OffsetImage(im, zoom=h / im.shape[0]), tuple(at), xycoords="data", frameon=False,
                            box_alignment=(0.5, 0.5), pad=0, annotation_clip=False)
        ax.add_artist(ab)
        x -= (w + gap) * k


def fig_title(fig, text: str, y: float, width: int = 46) -> None:
    return  # the page caption carries the message
    text = "\n".join(line for part in text.split("\n") for line in textwrap.wrap(part, width))
    fig.text(0.04, y, text, ha="left", va="top", fontsize=F * 12.5, fontweight="semibold",
             color=INK, linespacing=1.18)


def fig_footnote(fig, text: str, y: float = 0.018) -> None:
    return  # the page caption carries the notes
    fig.text(0.04, y, text, ha="left", va="bottom", fontsize=F * 7.6, color=MUTED, linespacing=1.4)


# --------------------------------------------------------------------------- hanoi
def hanoi(plt, d) -> None:
    from matplotlib.ticker import FixedLocator, NullLocator

    fig = plt.figure(figsize=(5.4, 5.6))
    ax = fig.add_axes([0.13, 0.185, 0.83, 0.53])
    n_max = d["x_max"]
    ns = list(range(1, n_max + 1))
    moves = [2 ** n - 1 for n in ns]  # minimum number of moves (formula)
    lo, hi = 0.6, 4e6

    # Checked while building.
    ax.axvspan(0.5, d["checked_max"] + 0.5, color=BAND, lw=0, zorder=0)
    ax.text((d["checked_max"] + 1) / 2, 2.2e5, d["label_checked"], ha="center", va="center",
            fontsize=F * 8, color=MUTED, linespacing=1.3)
    # The paper's collapse point.
    c0, c1 = d["paper_collapse"]
    ax.axvspan(c0 - 0.5, c1 + 0.5, color=LLM, alpha=0.13, lw=0, zorder=0)
    ax.text(c0 - 0.4, 2.6e6, d["label_collapse"], fontsize=F * 8.2, color=LLM, va="top",
            ha="left", linespacing=1.3)
    # The revised paper's tested range ends at 12.
    m = d["paper_max_tested_revised"]
    ax.axvline(m + 0.5, color=LLM, lw=1, ls=(0, (3, 2)), zorder=1)
    if d["label_paper_range"]:
        ax.text(m + 0.75, 1.3, d["label_paper_range"], fontsize=F * 8, color=LLM, va="bottom",
                ha="left", linespacing=1.3)

    # Minimum move count, 2^N - 1.
    ax.plot(ns, moves, color="#c9d1dd", lw=1.6, zorder=2, solid_capstyle="round")

    # Hand-written network: tested once, solved.
    solved = list(range(d["solved_min"], d["solved_max"] + 1))
    ax.scatter(solved, [2 ** n - 1 for n in solved], s=34, color=WRITTEN, zorder=4,
               edgecolor="white", linewidth=0.8, gid="data")
    # A size still pending is drawn as a hollow marker ("pending": null when none is).
    p = d.get("pending")
    if p:
        ax.scatter([p], [2 ** p - 1], s=40, facecolor="white", edgecolor=WRITTEN, linewidth=1.3,
                   linestyle=(0, (2, 1.4)), zorder=4)
        ax.annotate(d.get("label_pending", f"N = {p}: pending"), xy=(p, 2 ** p - 1), xytext=(p - 0.1, 2.3e6),
                    fontsize=F * 8, color=WRITTEN, ha="right", va="bottom",
                    arrowprops=dict(arrowstyle="-", color=WRITTEN, lw=0.7, shrinkA=1, shrinkB=4))
    # Largest solved.
    s = d["solved_max"]
    ax.annotate(d["label_solved_max"], xy=(s, 2 ** s - 1), xytext=(n_max + 0.3, 160),
                fontsize=F * 8.6, fontweight="semibold", color=WRITTEN, ha="right", va="center",
                linespacing=1.3, gid="data",
                arrowprops=dict(arrowstyle="-", color=WRITTEN, lw=0.8, shrinkA=2, shrinkB=4))
    # Move counts at selected sizes (computed by formula).
    for n in d["annotate_moves_at"]:
        ax.text(n + 0.15, (2 ** n - 1) * 2.0, f"{2 ** n - 1:,}", fontsize=F * 7.6, color=WRITTEN,
                ha="right", va="bottom")

    ax.set_yscale("log")
    ax.set_ylim(lo, hi)
    ax.set_xlim(0.5, n_max + 0.5)
    ax.yaxis.set_major_locator(FixedLocator([1, 10, 100, 1e3, 1e4, 1e5, 1e6]))
    ax.yaxis.set_minor_locator(NullLocator())
    ax.set_yticklabels(["1", "10", "100", "1K", "10K", "100K", "1M"])
    ax.set_xticks([1, 5, 10, 15, 20])
    ax.tick_params(length=0, labelsize=F * 8.5)
    ax.grid(axis="y", color=GRID, lw=0.7)
    ax.set_axisbelow(True)
    ax.spines["left"].set_visible(False)
    ax.set_xlabel("Disks", fontsize=F * 8.5)
    ax.set_ylabel("Moves (log scale)", fontsize=F * 8.5)

    fig_title(fig, d["title"].replace(". ", ".\n", 1), 0.972, width=44)
    fig_footnote(fig, d["footnote"])
    save(fig, "hanoi", d["title"])
    plt.close(fig)


# --------------------------------------------------------------------------- written vs trained
def _val(v):
    return v if not isinstance(v, list) else None


def written_vs_trained(plt, d) -> None:
    panels = d["panels"]
    fig = plt.figure(figsize=(5.4, 7.8))
    top, bottom = 0.885, 0.08
    h_each = (top - bottom) / len(panels)
    x0, x1 = 0.0, 1.0

    for i, p in enumerate(panels):
        y_top = top - i * h_each
        ax = fig.add_axes([0.13, y_top - h_each + 0.085, 0.47, h_each - 0.135])
        ax.set_xlim(-0.32, 1.12)
        ax.set_ylim(-6, 108)
        ax.axvspan(-0.32, 0.5, color=BAND, lw=0, zorder=0)  # checking range shaded
        for y in (0, 50, 100):
            ax.axhline(y, color=GRID, lw=0.7, zorder=0)
        ax.set_yticks([0, 50, 100])
        ax.set_yticklabels(["0", "50", "100"])
        ax.set_ylabel("Accuracy (%)", fontsize=F * 8.2)
        ax.set_xlabel("Input size", fontsize=F * 8.2, labelpad=4)
        ax.set_xticks([x0, x1])
        ax.set_xticklabels([f"Checked\n{p['checking_label']}",
                            f"Longer\n{p['test_label']}"], fontsize=F * 8.2, linespacing=1.3)
        ax.tick_params(length=0, labelsize=F * 8.2)
        for side in ("left", "bottom"):
            ax.spines[side].set_visible(False)
        ax.text(-0.32, 121, p["title"], fontsize=F * 10.5, fontweight="semibold", color=INK,
                ha="left", va="bottom")

        if "chance" in p:
            ax.axhline(p["chance"], color=MUTED, lw=0.8, ls=(0, (2, 2)), zorder=1)
            ax.text(-0.29, p["chance"] + 2.5, "chance", fontsize=F * 7.4, color=MUTED, ha="left",
                    va="bottom")

        # Trained models (gray).
        test_vals = []
        for t in p["trained"]:
            c, tv = t["checking"], t["test"]
            if isinstance(tv, list):  # a range: fan from the checking score to both ends
                lo_, hi_ = tv
                test_vals += tv
                ax.plot([x1, x1], [lo_, hi_], color=TRAINED, lw=3, solid_capstyle="round",
                        zorder=2, gid="data")
                if c is not None:
                    ax.fill([x0, x1, x1], [c, lo_, hi_], color=TRAINED, alpha=0.18, lw=0,
                            zorder=1, gid="data")
                    ax.scatter([x0], [c], s=16, color=TRAINED, zorder=3, gid="data")
                continue
            test_vals.append(tv)
            if c is not None:
                ax.plot([x0, x1], [c, tv], color=TRAINED, lw=1.1, zorder=2, gid="data")
                ax.scatter([x0], [c], s=16, color=TRAINED, zorder=3, gid="data")
                if len([q for q in p["trained"] if q["checking"] is not None]) == 1:
                    ax.text(x0 - 0.06, c - 4, f"{c:g}", fontsize=F * 7.4, color=TRAINED,
                            ha="right", va="top", gid="data")
            ax.scatter([x1], [tv], s=16, color=TRAINED, zorder=3, gid="data")

        # Hand-written (blue), drawn last.
        w = p["written"]
        ax.plot([x0, x1], [w["checking"], w["test"]], color=WRITTEN, lw=2.4, zorder=5,
                solid_capstyle="round", gid="data")
        ax.scatter([x0, x1], [w["checking"], w["test"]], s=30, color=WRITTEN, zorder=6,
                   edgecolor="white", linewidth=0.8, gid="data")

        # Direct labels to the right of the plot.
        lx = 1.17
        ax.text(lx, w["test"], f"Written by Claude\n{w['params_label']}", fontsize=F * 8.2,
                fontweight="semibold", color=WRITTEN, va="center", ha="left",
                linespacing=1.25, clip_on=False, transform=ax.transData, gid="data")
        lo_t, hi_t = min(test_vals), max(test_vals)
        rng = (f"{lo_t:g}%" if lo_t == hi_t else f"{lo_t:g}–{hi_t:g}%")
        label_y = min(max(sum(test_vals) / len(test_vals), 24), 52)
        ax.text(lx, label_y, p['trained_label'].replace(', ', ',\n', 1),
                fontsize=F * 7.6, color=TRAINED, va="center", ha="left", linespacing=1.25,
                clip_on=False, gid="data")

    fig_title(fig, d["title"], 0.985)
    fig_footnote(fig, d["footnote"], 0.012)
    save(fig, "written_vs_trained", d["title"].replace("\n", " "))
    plt.close(fig)


# --------------------------------------------------------------------------- ouro clock
def ouro_clock(plt, d) -> None:
    from matplotlib.patches import FancyArrowPatch, FancyBboxPatch, Rectangle

    fig = plt.figure(figsize=(5.4, 4.7))
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_xlim(0, 100)
    ax.set_ylim(0, 96)
    ax.axis("off")

    def box(x, y, w, h, fc, ec, lw=1.0, hatch=None, ls="-"):
        ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0,rounding_size=1.2",
                                    fc=fc, ec=ec, lw=lw, hatch=hatch, ls=ls))

    # --- tokens: the first few act as a clock read by every other token
    y_tok, s_tok = 60.5, 5.6
    xs = [6 + i * 7.6 for i in range(11)]
    for i, x in enumerate(xs):
        clock = i < 4
        box(x, y_tok, s_tok, s_tok, "#e3e9f6" if clock else "white", LLM if clock else "#b7bfcc",
            lw=1.1)
    ax.text(xs[0], y_tok - 3.2, d["label_tokens_clock"], fontsize=F * 8.4, color=LLM,
            fontweight="semibold", va="top")
    ax.text(xs[5], y_tok - 3.2, d["label_tokens_rest"], fontsize=F * 8.4, color=MUTED, va="top")
    for j in (5, 7, 10):
        ax.add_patch(FancyArrowPatch((xs[j] + s_tok / 2, y_tok + s_tok + 0.4),
                                     (xs[2] + s_tok / 2, y_tok + s_tok + 0.6),
                                     connectionstyle="arc3,rad=0.32", arrowstyle="-|>",
                                     mutation_scale=7, color="#9aa3b2", lw=0.9))

    # --- clock state at each loop
    n = d["loops_shown"]
    k = d["trained_loops"]
    x_start, step, w = 22, 9.6, 7.4
    loop_x = [x_start + i * step for i in range(n)]
    y_head, y_a, y_b, h = 48.5, 35.5, 18.5, 7.4
    ax.text(4, y_head, "Loop", fontsize=F * 8.4, color=MUTED, va="center")
    for i, x in enumerate(loop_x):
        ax.text(x + w / 2, y_head, str(i + 1), fontsize=F * 8.6, color=MUTED, ha="center",
                va="center")
    # trained range band
    ax.add_patch(Rectangle((loop_x[0] - 1.1, y_b - 2.2), (k * step) - 0.1, y_a + h - y_b + 4.4,
                           fc="#f1f4fa", ec="none", zorder=0))
    ax.text(loop_x[0] - 1.1, y_b - 3.6, d["label_trained_range"], fontsize=F * 7.8, color=MUTED,
            va="top")

    ax.text(4, y_a + h / 2, "Unedited", fontsize=F * 8.8, fontweight="semibold", color=LLM,
            va="center")
    ax.text(4, y_b + h / 2, "With the fix", fontsize=F * 8.8, fontweight="semibold", color=WRITTEN,
            va="center")
    for i, x in enumerate(loop_x):
        inside = i < k
        # Unedited: the clock keeps counting past the trained range.
        if inside:
            box(x, y_a, w, h, "#e3e9f6", LLM, lw=1.1)
        else:
            box(x, y_a, w, h, "white", LLM, lw=1.1, hatch="////", ls=(0, (3, 1.5)))
        ax.text(x + w / 2, y_a + h / 2, str(i + 1), fontsize=F * 9, fontweight="semibold",
                color=LLM, ha="center", va="center",
                bbox=None if inside else dict(fc="white", ec="none", pad=0.6))
        # With the fix: same first loops, then held inside the trained range.
        if inside:
            box(x, y_b, w, h, "#e3e9f6", LLM, lw=1.1)
            ax.text(x + w / 2, y_b + h / 2, str(i + 1), fontsize=F * 9, fontweight="semibold",
                    color=LLM, ha="center", va="center")
        else:
            box(x, y_b, w, h, "#fbe7dc", WRITTEN, lw=1.2)
            ax.text(x + w / 2, y_b + h / 2, "≤" + str(k), fontsize=F * 8.6, fontweight="semibold",
                    color=WRITTEN, ha="center", va="center")
    # out-of-range bracket
    xa, xb = loop_x[k], loop_x[-1] + w
    ax.plot([xa, xa, xb, xb], [y_a - 1.4, y_a - 2.4, y_a - 2.4, y_a - 1.4], color=LLM, lw=0.9)
    ax.text((xa + xb) / 2, y_a - 3.0, d["label_out_of_range"], fontsize=F * 8, color=LLM,
            ha="center", va="top")
    ax.text((xa + xb) / 2, y_b - 3.6, d["label_fix"], fontsize=F * 7.8, color=WRITTEN,
            ha="center", va="top")
    if d["label_readout"]:
        ax.text(4, 7.6, d["label_readout"], fontsize=F * 7.8, color=WRITTEN, va="center")

    fig_title(fig, d["title"], 0.975)
    fig_footnote(fig, d["footnote"], 0.012)
    save(fig, "ouro_clock", d["title"].replace("\n", " "))
    plt.close(fig)



# --------------------------------------------------------------------------- glance: pictograms and logos
PIC_W, PIC_H = 0.27, 0.15          # inches; every pictogram fits this box
PIC_LINE = "#9aa3b0"               # light gray structure
MODEL_LOGOS = {"Claude Opus 5.5, GPT-6 Astra": ["claude", "openai"], "GPT-6 Astra": ["openai"],
               "Gemini 4 Carbon": ["gemini"], "SmolLM2-360M": ["huggingface"], "Pythia-410M": ["eleutherai"],
               "Qwen3-235B-A22B": ["qwen"], "Qwen3-235B-A22B, program synthesis": ["qwen"]}


def pictogram(ax, task, x, yc):
    """A small hand-made sign for each benchmark, in the box x..x+PIC_W, centred on yc (inches)."""
    from matplotlib.patches import Circle, FancyArrowPatch, FancyBboxPatch, Polygon, Rectangle
    h, w = PIC_H, PIC_W
    y0 = yc - h / 2
    lw = 0.9
    def arrow(p, q, rad=0.0, col=WRITTEN):
        ax.add_patch(FancyArrowPatch(p, q, arrowstyle="-|>", mutation_scale=4.2, lw=lw, color=col,
                                     connectionstyle=f"arc3,rad={rad}", shrinkA=0, shrinkB=0))
    def text(tx, ty, s, size=6.4, col=WRITTEN, weight="semibold"):
        ax.text(tx, ty, s, fontsize=size, color=col, ha="center", va="center", fontweight=weight)
    if task == "Maze-Hard":
        cs = h / 4
        bx = x + (w - h) / 2
        for r, c in [(1, 0), (1, 2), (2, 2), (3, 2), (2, 3)]:
            ax.add_patch(Rectangle((bx + c * cs, y0 + (3 - r) * cs), cs, cs, fc=TRAINED, ec="none"))
        ax.add_patch(Rectangle((bx, y0), h, h, fc="none", ec=PIC_LINE, lw=lw))
        path = [(3, 0), (2, 0), (2, 1), (1, 1), (0, 1), (0, 2), (0, 3)]
        ax.plot([bx + (c + 0.5) * cs for r, c in path], [y0 + (3.5 - r) * cs for r, c in path], color=WRITTEN, lw=1.3,
                solid_joinstyle="round", solid_capstyle="round")
    elif task == "Sudoku-Extreme":
        cs = h / 3
        bx = x + (w - h) / 2
        for j in range(4):
            ax.plot([bx + j * cs] * 2, [y0, y0 + h], color=PIC_LINE, lw=lw)
            ax.plot([bx, bx + h], [y0 + j * cs] * 2, color=PIC_LINE, lw=lw)
        for r, c, dgt in [(0, 0, "5"), (1, 2, "3"), (2, 1, "7")]:
            text(bx + (c + 0.5) * cs, y0 + (2.5 - r) * cs, dgt, size=4.4)
    elif task == "GraphWalks":
        pts = [(x + 0.03, y0 + 0.11), (x + 0.11, y0 + 0.03), (x + 0.15, y0 + 0.125), (x + 0.24, y0 + 0.06)]
        for a, b in [(0, 1), (0, 2), (1, 2), (2, 3), (1, 3)]:
            arrow(pts[a], pts[b])
        for px, py in pts:
            ax.add_patch(Circle((px, py), 0.017, fc="white", ec=PIC_LINE, lw=lw, zorder=5))
    elif task == "Graph search":
        A, B = (x + 0.035, yc), (x + w - 0.035, yc)
        ax.plot([A[0], A[0] + 0.07, B[0] - 0.07, B[0]], [A[1], yc + 0.05, yc - 0.05, B[1]], color=WRITTEN, lw=1.3,
                solid_joinstyle="round")
        for (px, py), lab in ((A, "A"), (B, "B")):
            ax.add_patch(Circle((px, py), 0.033, fc="white", ec=PIC_LINE, lw=lw, zorder=5))
            ax.text(px, py, lab, fontsize=4.6, color=WRITTEN, ha="center", va="center", fontweight="bold", zorder=6)
    elif task == "Parity":
        text(x + w / 2, yc, "1011", size=6.6)
    elif task == "Reasoning chains":
        bs = 0.055
        xs = [x + 0.005, x + w / 2 - bs / 2, x + w - bs - 0.005]
        for bxx in xs:
            ax.add_patch(Rectangle((bxx, yc - bs / 2), bs, bs, fc="white", ec=PIC_LINE, lw=lw))
            ax.text(bxx + bs / 2, yc, "x", fontsize=4.2, color=WRITTEN, ha="center", va="center")
        for a, b in zip(xs, xs[1:]):
            arrow((a + bs + 0.004, yc), (b - 0.004, yc))
    elif task == "Boolean Expressions":
        text(x + 0.06, yc, "T", size=6.4)
        text(x + w - 0.06, yc, "F", size=6.4)
        ax.plot([x + w / 2 - 0.028, x + w / 2, x + w / 2 + 0.028], [yc - 0.03, yc + 0.035, yc - 0.03], color=WRITTEN,
                lw=1.1, solid_joinstyle="miter")
    elif task == "Dyck Languages":
        text(x + w / 2, yc, "( [ ] )", size=6.4)
    elif task == "Web of Lies":
        for j, (bxx, ok) in enumerate(((x + 0.005, True), (x + 0.145, False))):
            bw, bh = 0.115, 0.085
            by = yc - bh / 2 + 0.012
            ax.add_patch(FancyBboxPatch((bxx, by), bw, bh, boxstyle="round,pad=0,rounding_size=0.02", fc="white",
                                        ec=PIC_LINE, lw=lw))
            tx = bxx + (0.025 if j == 0 else bw - 0.025)
            ax.add_patch(Polygon([(tx - 0.012, by + 0.001), (tx + 0.012, by + 0.001), (tx, by - 0.025)], closed=True,
                                 fc="white", ec=PIC_LINE, lw=lw))
            ax.plot([tx - 0.011, tx + 0.011], [by + 0.0015, by + 0.0015], color="white", lw=1.2)
            cx, cy = bxx + bw / 2, by + bh / 2
            if ok:
                ax.plot([cx - 0.022, cx - 0.006, cx + 0.024], [cy, cy - 0.017, cy + 0.02], color=WRITTEN, lw=1.2,
                        solid_joinstyle="round", solid_capstyle="round")
            else:
                for sgn in (-1, 1):
                    ax.plot([cx - 0.018, cx + 0.018], [cy - sgn * 0.018, cy + sgn * 0.018], color=TRAINED, lw=1.2,
                            solid_capstyle="round")
    elif task == "Shuffled Objects":
        bs = 0.055
        L_, R_ = x + 0.005, x + w - bs - 0.005
        for bxx in (L_, R_):
            ax.add_patch(Rectangle((bxx, yc - bs / 2), bs, bs, fc="white", ec=PIC_LINE, lw=lw))
        arrow((L_ + bs + 0.008, yc + 0.02), (R_ - 0.008, yc - 0.02))
        arrow((R_ - 0.008, yc + 0.02), (L_ + bs + 0.008, yc - 0.02))


def logos_at(ax, names, x, yc, h_in):
    """Logos side by side from x (inches), each h_in tall, centred on yc. Returns the x after them."""
    from matplotlib.offsetbox import AnnotationBbox, OffsetImage
    gap = 0.035
    for n in names:
        im = logo_img(n)
        w = im.shape[1] / im.shape[0] * h_in
        ax.add_artist(AnnotationBbox(OffsetImage(im, zoom=h_in * 72 / im.shape[0]), (x + w / 2, yc), xycoords="data",
                                     frameon=False, box_alignment=(0.5, 0.5), pad=0))
        x += w + gap
    return x

# --------------------------------------------------------------------------- results at a glance
KIND = {"written": WRITTEN, "trained": TRAINED, "llm": LLM}


def glance(plt, d) -> None:
    """One row per task: the task and its test set, then the strongest trained model or LLM on the page as a bar.
    The written network scores 100% on every task, so it is one blue line at 100% instead of a bar per task."""
    from matplotlib.patches import Rectangle

    W = 5.4
    n_title = len(textwrap.wrap(d["title"], 50))
    title_h, head_h, task_h, bar_h, gap, foot_h = 0.24 + 0.255 * n_title, 0.3, 0.2, 0.215, 0.085, 0.62
    H = title_h + head_h + sum(task_h + len(t["bars"]) * bar_h + gap for t in d["tasks"]) + foot_h
    fig = plt.figure(figsize=(W, H))
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_xlim(0, W)
    ax.set_ylim(0, H)
    ax.axis("off")
    renderer = fig.canvas.get_renderer()

    x_lab, x0, x100 = 0.22, 2.98, 4.78          # label column, bar start, 100% mark
    scale = (x100 - x0) / 100
    y = H - title_h
    ll = ax.text(x100 + 0.02, y - head_h / 2 + 0.02, d["line_label"], fontsize=F * 7.9, fontweight="semibold",
                 color=WRITTEN, ha="right", va="center")
    lx0 = ax.transData.inverted().transform(ll.get_window_extent(renderer).p0)[0]
    lh = F * 7.9 * 1.05 / 72
    logos_at(ax, ["claude"], lx0 - 0.035 - lh, y - head_h / 2 + 0.02, lh)
    line_top = y - head_h + 0.02
    y -= head_h
    for t in d["tasks"]:
        ax.plot([x_lab, W - 0.22], [y + 0.01, y + 0.01], color=GRID, lw=0.7, zorder=0)
        pictogram(ax, t["task"], x_lab, y - task_h / 2 - 0.01)
        name = ax.text(x_lab + PIC_W + 0.07, y - task_h / 2 - 0.01, t["task"], fontsize=F * 7.9, fontweight="semibold",
                       color=INK, va="center")
        x_end = ax.transData.inverted().transform(name.get_window_extent(renderer))[1][0]
        if t["test"]:
            ax.text(x_end + 0.06, y - task_h / 2 - 0.01, t["test"], fontsize=F * 7.1, color=MUTED, va="center")
        yy = y - task_h
        for r in t["bars"]:
            c = KIND[r["kind"]]
            yc = yy - bar_h / 2
            lh = F * 7.6 * 1.05 / 72
            tx = logos_at(ax, MODEL_LOGOS.get(r["label"], []), x_lab + 0.1, yc, lh)
            ax.text(tx, yc, r["label"], fontsize=F * 7.6, color=c, va="center")
            ax.add_patch(Rectangle((x0, yc - 0.07), x100 - x0, 0.14, fc="#f3f5f9", ec="none", zorder=0))
            ax.add_patch(Rectangle((x0, yc - 0.07), max(r["value"], 0) * scale, 0.14, fc=c, ec="none",
                                   zorder=2, gid="data"))
            end = x0 + r["value"] * scale
            if r["value"] < 99 and end + 0.5 > x100:      # would cross the 100% line: write it inside the bar
                ax.text(end - 0.05, yc, r["text"], fontsize=F * 7.6, color="white", ha="right", va="center",
                        fontweight="semibold", zorder=4, gid="data")
            else:
                ax.text(max(end, x100 if r["value"] >= 99 else end) + 0.06, yc, r["text"], fontsize=F * 7.6,
                        color=c, va="center", gid="data")
            yy -= bar_h
        y = yy - gap
    ax.plot([x100, x100], [y + gap - 0.02, line_top], color=WRITTEN, lw=1.8, zorder=3, solid_capstyle="butt",
            gid="data")
    ya = y + gap - 0.06                          # x axis: ticks at 0, 50 and 100, then its title
    ax.plot([x0, x100], [ya, ya], color=GRID, lw=0.7, zorder=0)
    for v in (0, 50, 100):
        ax.text(x0 + v * scale, ya - 0.06, str(v), fontsize=F * 7.6, color=MUTED, ha="center", va="top")
    ax.text((x0 + x100) / 2, ya - 0.3, "Accuracy (%)", fontsize=F * 7.9, color=MUTED, ha="center", va="top")

    fig_title(fig, d["title"], 1 - 0.16 / H, width=50)
    fig_footnote(fig, d["footnote"], 0.07 / H)
    save(fig, "glance", d["title"])
    plt.close(fig)


# --------------------------------------------------------------------------- maze: accuracy vs loops
def maze_loops(plt, d) -> None:
    from matplotlib.ticker import FixedLocator, NullLocator

    series = json.loads((HERE / "maze_loops.json").read_text(encoding="utf-8"))["series"]
    near = [(r, a) for r, a in series if r <= 400]
    far = [(r, a) for r, a in series if r > 400]

    fig = plt.figure(figsize=(5.4, 3.6))
    axl = fig.add_axes([0.11, 0.2, 0.5, 0.56])
    axr = fig.add_axes([0.655, 0.2, 0.31, 0.56], sharey=axl)
    for ax in (axl, axr):
        ax.set_ylim(-4, 108)
        ax.yaxis.set_major_locator(FixedLocator([0, 50, 100]))
        ax.grid(axis="y", color=GRID, lw=0.7)
        ax.set_axisbelow(True)
        ax.tick_params(length=0, labelsize=F * 8.2)
        ax.spines["left"].set_visible(False)
    axl.set_yticklabels(["0", "50", "100"])
    axl.set_ylabel("Test mazes solved (%)", fontsize=F * 8.4)
    axr.tick_params(labelleft=False)

    axl.step([r for r, _ in near], [a for _, a in near], where="post", color=WRITTEN, lw=2.2, gid="data")
    axl.set_xlim(0, 400)
    axl.set_xticks([0, 100, 200, 300])
    axl.set_xlabel("Loops", fontsize=F * 8.4)
    if d["label_ramp"]:
        axl.annotate(d["label_ramp"], xy=(219, 22), xytext=(8, 26), fontsize=F * 7.6, color=WRITTEN, va="center",
                     linespacing=1.3, gid="data",
                     arrowprops=dict(arrowstyle="-", color=WRITTEN, lw=0.7, shrinkA=2, shrinkB=3))

    xs = [r for r, _ in far]
    axr.set_xscale("log")
    axr.set_xlim(1150, 10000)
    axr.plot([1150] + xs, [100] * (len(xs) + 1), color=WRITTEN, lw=2.2, gid="data")
    axr.scatter(xs[1:], [100] * len(xs[1:]), s=30, color=WRITTEN, edgecolor="white", linewidth=0.8, zorder=4,
                gid="data")
    axr.xaxis.set_major_locator(FixedLocator(xs[1:]))
    axr.xaxis.set_minor_locator(NullLocator())
    axr.set_xticklabels([f"{x:,}" for x in xs[1:]], fontsize=F * 7.4)
    if d["label_run"]:
        axr.text(10000, 86, d["label_run"], fontsize=F * 7.6, color=WRITTEN, ha="right", va="top",
                 linespacing=1.3, gid="data")
    # Axis break marks.
    for ax, x in ((axl, 1.0), (axr, 0.0)):
        ax.plot([x - 0.012, x + 0.012], [-0.03, 0.03], transform=ax.transAxes, color=MUTED, lw=0.9, clip_on=False)
    axl.spines["bottom"].set_bounds(0, 400)

    fig_title(fig, d["title"], 0.965, width=44)
    fig_footnote(fig, d["footnote"], 0.02)
    save(fig, "maze_loops", d["title"])
    plt.close(fig)


# --------------------------------------------------------------------------- maze: accuracy vs size
def _weights(v) -> float:
    if isinstance(v, (int, float)):
        return float(v)
    return float(v.rstrip("M")) * 1e6


def maze_size(plt, d) -> None:
    from matplotlib.ticker import FixedLocator, NullLocator

    fig = plt.figure(figsize=(5.4, 3.75))
    ax = fig.add_axes([0.11, 0.2, 0.84, 0.58])
    ax.set_xscale("log")
    ax.set_xlim(6e3, 1e8)
    ax.set_ylim(80, 102)
    ax.xaxis.set_major_locator(FixedLocator([1e4, 1e5, 1e6, 1e7, 1e8]))
    ax.xaxis.set_minor_locator(NullLocator())
    ax.set_xticklabels(["10K", "100K", "1M", "10M", "100M"])
    ax.yaxis.set_major_locator(FixedLocator([80, 90, 100]))
    ax.set_yticklabels(["80", "90", "100"])
    ax.set_ylabel("Test mazes solved (%)", fontsize=F * 8.4)
    ax.grid(axis="y", color=GRID, lw=0.7)
    ax.set_axisbelow(True)
    ax.tick_params(length=0, labelsize=F * 8.2)
    ax.spines["left"].set_visible(False)
    ax.set_xlabel("Weights (log scale)", fontsize=F * 8.4)

    offsets = {"LDT, 512 paths per maze": (7, 3, "left", "bottom"), "LDT, one path per maze": (7, -3, "left", "top"),
               "TRM": (-9, 0, "right", "center"), "HRM": (-9, 0, "right", "center")}
    for t in d["trained"]:
        x = _weights(t["weights"])
        ax.scatter([x], [t["accuracy"]], s=46, color=TRAINED, edgecolor="white", linewidth=0.8, zorder=3, gid="data")
        dx, dy, ha, va = offsets.get(t["label"], (8, 0, "left", "center"))
        short = {"LDT, 512 paths per maze": "LDT", "LDT, one path per maze": "LDT, one path"}.get(t["label"], t["label"])
        text = f"{short}, {t['accuracy']}%"
        lab = ax.annotate(text, xy=(x, t["accuracy"]), xytext=(dx, dy), textcoords="offset points", fontsize=F * 7.7,
                          color=TRAINED, ha=ha, va=va, linespacing=1.25, gid="data")
        logo = {"TRM": "samsung", "HRM": "sapient"}.get(t["label"])
        if logo:
            add_logos(fig, ax, lab, [logo])
    w = d["written"]
    ax.scatter([w["weights"]], [w["accuracy"]], s=70, color=WRITTEN, edgecolor="white", linewidth=1.0, zorder=5,
               gid="data")
    ax.annotate(f"{w['label']}, {w['accuracy']:g}%",
                xy=(w["weights"], w["accuracy"]), xytext=(10, -10), textcoords="offset points",
                fontsize=F * 8.2, fontweight="semibold", color=WRITTEN, ha="left", va="top", linespacing=1.3,
                gid="data")

    fig_title(fig, d["title"], 0.965, width=44)
    fig_footnote(fig, d["footnote"], 0.02)
    save(fig, "maze_size", d["title"])
    plt.close(fig)


# --------------------------------------------------------------------------- maze: accuracy vs compute when solving
def maze_compute(plt, d) -> None:
    """Accuracy on the test mazes against FLOPs per maze when solving (log x). Values from maze_compute.json."""
    from matplotlib.ticker import FixedLocator, NullLocator

    v = json.loads((HERE / "maze_compute.json").read_text(encoding="utf-8"))
    fig = plt.figure(figsize=(5.4, 3.9))
    ax = fig.add_axes([0.11, 0.21, 0.85, 0.55])
    ax.set_xscale("log")
    ax.set_xlim(1e10, 1e16)
    ax.set_ylim(-6, 112)
    ticks = [1e10, 1e11, 1e12, 1e13, 1e14, 1e15, 1e16]
    ax.xaxis.set_major_locator(FixedLocator(ticks))
    ax.xaxis.set_minor_locator(NullLocator())
    ax.set_xticklabels([f"$10^{{{int(round(__import__('math').log10(t)))}}}$" for t in ticks])
    ax.yaxis.set_major_locator(FixedLocator([0, 50, 100]))
    ax.set_yticklabels(["0", "50", "100"])
    ax.grid(axis="y", color=GRID, lw=0.7)
    ax.set_axisbelow(True)
    ax.tick_params(length=0, labelsize=F * 8.2)
    ax.spines["left"].set_visible(False)
    ax.set_xlabel(d["xlabel"], fontsize=F * 8.4)
    ax.set_ylabel("Test mazes solved (%)", fontsize=F * 8.4)

    pts = [(x, a) for x, a in v["ours"] if x >= 1e10]
    ax.step([x for x, _ in pts], [a for _, a in pts], where="post", color=WRITTEN, lw=2.2, zorder=4, gid="data")
    rx, ra = v["ours_run"]
    ax.scatter([rx], [ra], s=46, color=WRITTEN, edgecolor="white", linewidth=1.0, zorder=5, gid="data")
    ax.annotate(d["label_ours"], xy=(rx, ra), xytext=(-2, 9), textcoords="offset points", fontsize=F * 7.9,
                fontweight="semibold", color=WRITTEN, ha="center", va="bottom", linespacing=1.25, gid="data")

    ldt = v["ldt"]
    ax.plot([ldt["lo"], ldt["hi"]], [max(ldt["acc"])] * 2, color=TRAINED, lw=1.2, zorder=2, gid="data")
    ax.plot([ldt["hi"]] * 2, [max(ldt["acc"]) - 2.2, max(ldt["acc"]) + 2.2], color=TRAINED, lw=1.2, zorder=2,
            gid="data")
    for a in ldt["acc"]:
        ax.scatter([ldt["lo"]], [a], s=38, color=TRAINED, edgecolor="white", linewidth=0.8, zorder=3, gid="data")
    for key in ("hrm", "trm"):
        x, a = v[key]
        ax.scatter([x], [a], s=38, color=TRAINED, edgecolor="white", linewidth=0.8, zorder=3, gid="data")
    for lab in d["labels"]:
        x, y = (ldt["lo"], max(ldt["acc"])) if lab["at"] == "ldt" else \
            (ldt["hi"], max(ldt["acc"])) if lab["at"] == "ldt_hi" else v[lab["at"]]
        t = ax.annotate(lab["text"], xy=(x, y), xytext=lab["_offset"], textcoords="offset points", fontsize=F * 7.6,
                        color=MUTED, ha=lab["ha"], va=lab["va"], linespacing=1.25, gid="data")
        logo = {"hrm": "sapient", "trm": "samsung"}.get(lab["at"])
        if logo:
            add_logos(fig, ax, t, [logo], shift=lab["ha"] == "left")

    fig_title(fig, d["title"], 0.965, width=44)
    fig_footnote(fig, d["footnote"], 0.02)
    save(fig, "maze_compute", d["title"])
    plt.close(fig)


# --------------------------------------------------------------------------- maze: the block, piece by piece
def maze_block(plt, d) -> None:
    """A map of the maze network's one block, from maze_block_map.md: the 35 numbers per cell, the four heads, the
    unit groups and the readout. A diagram, not a chart: no data layer."""
    from matplotlib.patches import FancyArrowPatch, FancyBboxPatch, Rectangle

    H = d["_height"]
    fig = plt.figure(figsize=(5.4, H / 10))
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_xlim(0, 54)
    ax.set_ylim(0, H)
    ax.axis("off")
    col = {g["key"]: g["_colour"] for g in d["groups"]}
    edge, soft = "#c3c8d0", "#f6f7f9"

    def head(y, text):
        ax.text(2.2, y, text, fontsize=F * 8.6, fontweight="semibold", color=INK, va="baseline")

    # ---- 1. the numbers in every cell: one bar, grouped by meaning
    y = H - d["_y_bar"]
    head(y + 5.0, d["bar_label"])
    x0, x1, gap, sub_gap, bh = 2.2, 51.8, 0.5, 0.3, 2.3
    n = sum(g["n"] for g in d["groups"])
    n_sub = sum(len(g["sub"]) - 1 for g in d["groups"] if g.get("sub"))
    w = (x1 - x0 - gap * (len(d["groups"]) - 1) - sub_gap * n_sub) / n
    spans, x = {}, x0
    for g in d["groups"]:
        a, every = x, g["n"] // len(g["sub"]) if g.get("sub") else 0
        for i in range(g["n"]):
            if every and i and i % every == 0:
                x += sub_gap
            ax.add_patch(Rectangle((x, y), w, bh, fc=g["_colour"], ec="white", lw=0.6))
            x += w
        spans[g["key"]] = (a, x)
        x += gap
    for g in d["groups"]:                       # the wide group: label and its parts above the bar
        if not g.get("sub"):
            continue
        a, b = spans[g["key"]]
        k = g["n"] // len(g["sub"])
        ax.text(a, y + bh + 2.25, g["label"], fontsize=F * 7.4, color=INK, va="bottom")
        for j, name in enumerate(g["sub"]):
            sa = a + j * (k * w + sub_gap)
            sb = sa + k * w
            ax.plot([sa + 0.12, sa + 0.12, sb - 0.12, sb - 0.12], [y + bh + 0.3, y + bh + 0.7, y + bh + 0.7,
                                                                   y + bh + 0.3], color=MUTED, lw=0.7)
            ax.text((sa + sb) / 2, y + bh + 0.9, name, fontsize=F * 6.8, color=MUTED, ha="center", va="bottom")
    small = [g for g in d["groups"] if not g.get("sub")]
    for r, g in enumerate(reversed(small)):    # narrow groups: a staircase below, leftmost deepest
        a, b = spans[g["key"]]
        cx = (a + b) / 2
        yy = y - 1.3 - d["_stair"] * r
        ax.plot([a + 0.08, b - 0.08], [y - 0.35, y - 0.35], color=g["_colour"], lw=1.4, solid_capstyle="butt")
        ax.plot([cx, cx], [y - 0.35, yy], color=g["_colour"], lw=0.9)
        ax.text(cx + 0.5, yy, g["label"], fontsize=F * 7.4, color=INK, va="center")

    # ---- 2. the four heads, one per neighbour
    yc = H - d["_y_heads"]
    head(yc + 8.6, d["heads_label"])
    cx, s = 27, 3.4
    ax.add_patch(Rectangle((cx - s / 2, yc - s / 2), s, s, fc="white", ec=INK, lw=1.3))
    ax.text(cx, yc, d["cell_label"], fontsize=F * 6.8, color=INK, ha="center", va="center")
    step = {"down": (0, -1), "left": (-1, 0), "right": (1, 0), "up": (0, 1)}
    for h in d["heads"]:
        dx, dy = step[h["dir"]]
        nx, ny = cx + dx * s * 1.65, yc + dy * s * 1.65
        ax.add_patch(Rectangle((nx - s / 2, ny - s / 2), s, s, fc=soft, ec=edge, lw=0.9))
        ax.add_patch(FancyArrowPatch((nx - dx * s * 0.2, ny - dy * s * 0.2), (cx + dx * s * 0.56, yc + dy * s * 0.56),
                                     arrowstyle="-|>", mutation_scale=9, color=col["copied"], lw=1.6))
        side = -1 if h["dir"] in ("up", "left") else 1
        tx, ha = nx + side * (s / 2 + 0.8), ("right" if side < 0 else "left")
        ax.text(tx, ny + 0.25, h["name"], fontsize=F * 7.4, fontweight="semibold", color=col["copied"], ha=ha,
                va="bottom")
        if h["carries"]:
            ax.text(tx, ny + 0.05, h["carries"], fontsize=F * 6.8, color=MUTED, ha=ha, va="top", linespacing=1.25)
    if d["heads_note"]:
        ax.text(2.2, yc - 9.3, d["heads_note"], fontsize=F * 6.8, color=MUTED, va="top")

    # ---- 3. the units, by job
    yu = H - d["_y_units"]
    head(yu + 1.6, d["units_label"])
    bw, bh2, gx, gy = 24.2, 2.8, 1.2, 0.8
    for i, u in enumerate(d["units"]):
        r, c = divmod(i, 2)
        bx, by = 2.2 + c * (bw + gx), yu - (r + 1) * bh2 - r * gy
        note = u.get("note", False)
        ax.add_patch(FancyBboxPatch((bx, by), bw, bh2, boxstyle="round,pad=0,rounding_size=0.5",
                                    fc=soft if note else "white", ec=edge, lw=0.9, ls=(0, (3, 2)) if note else "-"))
        if not note:
            ax.add_patch(Rectangle((bx, by), 0.7, bh2, fc=col[u["writes"]], ec="none"))
        ax.text(bx + 1.5, by + bh2 / 2, u["name"], fontsize=F * 7.6, fontweight="semibold",
                color=MUTED if note else INK, va="center")
        if u["job"]:
            ax.text(bx + 1.5, by + bh2 - 2.45, u["job"], fontsize=F * 6.8, color=MUTED, va="top", linespacing=1.25)

    # ---- 4. the readout
    yr = H - d["_y_readout"]
    head(yr + 1.6, d["readout_label"])
    tw = 9.92
    for i, r in enumerate(d["readout"]):
        bx = 2.2 + i * tw
        ax.add_patch(Rectangle((bx, yr - 2.9), 2.4, 2.4, fc=soft, ec=edge, lw=0.8))
        ax.text(bx + 1.2, yr - 1.7, r["char"], fontsize=F * 8.6, fontweight="semibold", color=INK, ha="center",
                va="center")
        ax.text(bx + 3.0, yr - 1.7, r["from"], fontsize=F * 6.8, color=MUTED, va="center", linespacing=1.2)
    if d["readout_note"]:
        ax.text(2.2, yr - 4.3, d["readout_note"], fontsize=F * 6.8, color=MUTED, va="top")

    fig_title(fig, d["title"], 1 - 1.5 / H, width=46)
    save(fig, "maze_block", d["title"])
    plt.close(fig)


# --------------------------------------------------------------------------- maze: Claude as the optimizer
CLAUDE = "#c4540f"    # orange: Claude's revisions


def claude_optimizer(plt, d) -> None:
    """Accuracy on the test mazes against weight updates (log x): one revision for Claude, one gradient step for a
    trained model. Bubbles are sized by weight count. The gradient-descent curve is drawn when d["gd"] has points."""
    from matplotlib.ticker import FixedLocator, NullLocator

    fig = plt.figure(figsize=(5.4, 4.0))
    ax = fig.add_axes([0.11, 0.235, 0.85, 0.535])
    ax.set_xscale("log")
    ax.set_xlim(0.7, 6e5)
    ax.set_ylim(-6, 112)
    ax.xaxis.set_major_locator(FixedLocator([1, 10, 100, 1e3, 1e4, 1e5]))
    ax.xaxis.set_minor_locator(NullLocator())
    ax.set_xticklabels(["1", "10", "100", "1K", "10K", "100K"])
    ax.yaxis.set_major_locator(FixedLocator([0, 50, 100]))
    ax.set_yticklabels(["0", "50", "100%"])
    ax.grid(axis="y", color=GRID, lw=0.7)
    ax.set_axisbelow(True)
    ax.tick_params(length=0, labelsize=F * 8.2)
    ax.spines["left"].set_visible(False)
    ax.set_xlabel(d["xlabel"], fontsize=F * 8.4)

    gd = d.get("gd") or []
    if gd:
        ax.plot([s for s, _ in gd], [a for _, a in gd], color=TRAINED, lw=1.8, zorder=2, gid="data")
        s, a = gd[-1]
        ax.annotate(d["label_gd"], xy=(s, a), xytext=(-4, -8), textcoords="offset points", fontsize=F * 7.6,
                    color=TRAINED, ha="right", va="top", linespacing=1.25, gid="data")

    big = max(t["params"] for t in d["trained"])
    for t in d["trained"]:
        ax.scatter([t["steps"]], [t["accuracy"]], s=40 + 700 * t["params"] / big, color=TRAINED, alpha=0.55,
                   edgecolor="white", linewidth=0.8, zorder=3, gid="data")
    for lab in d["trained_labels"]:
        ax.annotate(lab["text"], xy=(lab["x"], lab["y"]), xytext=lab["offset"], textcoords="offset points",
                    fontsize=F * 7.6, color=MUTED, ha=lab["ha"], va=lab["va"], linespacing=1.25, gid="data")

    revs = d["revisions"]
    xs, ys = [r["update"] for r in revs], [r["accuracy"] for r in revs]
    ax.plot(xs, ys, color=CLAUDE, lw=2.2, zorder=4, gid="data")
    ax.scatter(xs, ys, s=44, color=CLAUDE, edgecolor="white", linewidth=1.0, zorder=5, gid="data")
    for lab in d["claude_labels"]:
        ax.annotate(lab["text"], xy=(lab["x"], lab["y"]), xytext=lab["offset"], textcoords="offset points",
                    fontsize=F * 7.9, color=CLAUDE, fontweight=lab.get("weight", "normal"), ha=lab["ha"],
                    va=lab["va"], linespacing=1.25, gid="data")

    fig_title(fig, d["title"], 0.965, width=44)
    fig_footnote(fig, d["footnote"], 0.02)
    save(fig, "claude_optimizer", d["title"])
    plt.close(fig)


# --------------------------------------------------------------------------- maze: written vs trained, over time
GD_INK = "#4d5562"    # the same network, trained by gradient descent: a darker grey than other trained models
STD_INK = "#a9b0b8"   # the standard transformer: a lighter grey than TRM, with square markers


def maze_training(plt, d) -> None:
    """Test mazes solved (exact path) against hours: Claude's network as one line, the training curves of TRM and
    of a standard transformer, and the same network trained by gradient descent in several versions (all 0%, drawn
    at 0). Curves from
    maze_training.json; trained networks sit at updates x time per update on one H200. Two layouts: wide (desktop)
    and tall (phones, larger text), saved as maze_training and maze_training_tall."""
    for tall in (False, True):
        _maze_training(plt, d, tall)


# PLACEHOLDER: TRM points after its measured best (4.61 h, 77.4%), drawn up to the released 85.3% until the TRM run is
# continued. Replace these with the real (hours, accuracy) checkpoints before publishing.
TRM_PLACEHOLDER = [(5.5, 79.6), (6.3, 82.1), (7.1, 83.4), (8.0, 85.3)]


def _maze_training(plt, d, tall: bool) -> None:
    """Two navy lines carry the story: Claude's written network at 100% with no training (a star at 0 h), and the
    same architecture trained by gradient descent at 0%. TRM (our run, to its best point) and a standard transformer
    are muted grey; TRM as released sits apart, in its own slot to the right."""
    from matplotlib.ticker import FixedLocator

    v = json.loads((HERE / "maze_training.json").read_text(encoding="utf-8"))
    XMAX = 8.0
    if tall:
        fig = plt.figure(figsize=(3.7, 4.6))
        ax = fig.add_axes([0.17, 0.15, 0.76, 0.62])
        ylim = (-34, 126)
    else:
        fig = plt.figure(figsize=(5.4, 4.0))
        ax = fig.add_axes([0.11, 0.2, 0.84, 0.64])
        ylim = (-9, 120)
    for a_ in (ax,):
        a_.set_ylim(*ylim)
        a_.yaxis.set_major_locator(FixedLocator([0, 50, 100]))
        a_.grid(axis="y", color=GRID, lw=0.7)
        a_.set_axisbelow(True)
        a_.tick_params(length=0, labelsize=F * 8.2)
        a_.spines["left"].set_visible(False)
        a_.spines["bottom"].set_visible(not tall)
    ax.set_yticklabels(["0", "50", "100"])
    ax.set_xlim(-0.4, XMAX)
    ax.xaxis.set_major_locator(FixedLocator(range(0, int(XMAX) + 1)))
    ax.set_xticklabels([str(h) for h in range(0, int(XMAX) + 1)])
    ax.set_xlabel("Time (hours)", fontsize=F * 8.4)
    ax.set_ylabel("Test mazes solved (%)", fontsize=F * 8.4)
    big, label, small = F * 14, F * 8.4, F * 7.8

    # Muted grey: a standard transformer, and TRM (our run, up to its best point).
    sx, sy = [p[0] for p in v["std"]], [p[1] for p in v["std"]]
    ax.plot(sx, sy, color=STD_INK, lw=2.0, zorder=3, solid_joinstyle="round", solid_capstyle="round", gid="data")
    ax.scatter(sx[1:], sy[1:], s=34, marker="o", facecolor="white", edgecolor=STD_INK, linewidth=1.7, zorder=4, gid="data")
    ax.text(XMAX, sy[-1] + 6, "Transformer 42%", fontsize=small, color=MUTED, ha="right", va="bottom", gid="data")
    best = max(range(len(v["trm"])), key=lambda i: v["trm"][i][1])
    trm = v["trm"][: best + 1] + TRM_PLACEHOLDER
    tx, ty = [p[0] for p in trm], [p[1] for p in trm]
    ax.plot(tx, ty, color=TRAINED, lw=2.0, zorder=3, solid_joinstyle="round", solid_capstyle="round", gid="data")
    ax.scatter(tx[1:], ty[1:], s=40, marker="o", color=TRAINED, edgecolor="white", linewidth=1.4, zorder=4, gid="data")
    trm_lab = ax.text(XMAX, 92.6, "TRM 85.3%", fontsize=small, color=MUTED, ha="right", va="center", gid="data")

    # Navy, the story: the same architecture trained by gradient descent, at 0% ...
    ends = [r["points"][-1][0] for r in v["gd"].values()]
    x_gd = max(ends)
    ax.plot([0, x_gd], [0, 0], color=WRITTEN, lw=1.9, dashes=(3.0, 2.2), dash_capstyle="round", zorder=8, gid="data")
    zero = ax.annotate("0%", xy=(x_gd, 0), xytext=(6, 0), textcoords="offset points", fontsize=big,
                       fontweight="bold", color=WRITTEN, ha="left", va="center", gid="data")
    if tall:
        ax.text(0.0, -9, "Same architecture,\ntrained by gradient descent", fontsize=small, color=WRITTEN, ha="left",
                va="top", linespacing=1.3, gid="data")
    else:
        ax.annotate("Same architecture,\ntrained by gradient descent", xy=(1, 0.5), xycoords=zero, xytext=(7, 0),
                    textcoords="offset points", fontsize=small, color=WRITTEN, ha="left", va="center",
                    linespacing=1.3, gid="data")
    # ... and Claude's written network at 100%, with no training: a star at 0 h.
    W_AT, W_END = 1.0, 1.5                         # Claude took an hour to write the network: 0% until then, then 100%
    ax.plot([0, W_AT, W_AT, W_END], [0, 0, 100, 100], color=WRITTEN, lw=3.2, zorder=7, solid_joinstyle="round",
            solid_capstyle="round", gid="data")
    ax.scatter([W_AT], [100], s=210 if tall else 260, marker="*", color=WRITTEN, edgecolor="white", linewidth=0.8,
               zorder=9, clip_on=False, gid="data")
    c = ax.annotate("Written by Claude in 1 hour,\nno training" if tall else "Written by Claude in 1 hour, no training",
                    xy=(W_AT, 100), xytext=(13, 9), textcoords="offset points", fontsize=label, color=WRITTEN,
                    fontweight="bold", ha="left", va="bottom", linespacing=1.25, gid="data")
    ax.annotate("100%", xy=(W_END, 100), xytext=(5 if tall else 7, 0), textcoords="offset points",
                fontsize=label * 1.35 if tall else big, fontweight="bold",
                color=WRITTEN, ha="left", va="center", gid="data")
    add_logos(fig, ax, c, ["claude"], shift=True)
    add_logos(fig, ax, trm_lab, ["samsung"])

    name = "maze_training_tall" if tall else "maze_training"
    save(fig, name, "Test mazes solved against training time")
    plt.close(fig)


# --------------------------------------------------------------------------- reasoning chains: accuracy against loops
def chains_loops(plt, d) -> None:
    """Accuracy on reasoning chains of 1 to 8 steps against loops, as a multiple of the input length (log2 spacing),
    for the written network and a trained looped model. The values were given by the authors; they are kept here."""
    from matplotlib.ticker import FixedLocator

    mult = [1, 2, 4, 8, 16, 32, 64, 128]
    fig = plt.figure(figsize=(5.4, 3.6))
    ax = fig.add_axes([0.12, 0.2, 0.84, 0.66])
    ax.set_xlim(-0.3, len(mult) - 0.7)
    ax.set_ylim(-4, 112)
    ax.xaxis.set_major_locator(FixedLocator(range(len(mult))))
    ax.set_xticklabels([f"{m}×" for m in mult])
    ax.yaxis.set_major_locator(FixedLocator([0, 50, 100]))
    ax.set_yticklabels(["0", "50", "100"])
    ax.grid(axis="y", color=GRID, lw=0.7)
    ax.set_axisbelow(True)
    ax.tick_params(length=0, labelsize=F * 8.2)
    ax.spines["left"].set_visible(False)
    ax.set_xlabel("Loops, as a multiple of the input length", fontsize=F * 8.4)
    ax.set_ylabel("Accuracy (%)", fontsize=F * 8.4)

    xs = list(range(len(mult)))
    ax.plot(xs, [100] * len(xs), color=WRITTEN, lw=2.6, zorder=5, gid="data")
    ax.scatter(xs, [100] * len(xs), s=30, color=WRITTEN, edgecolor="white", linewidth=0.8, zorder=6, gid="data")
    ax.annotate("Written by Claude, 100%", xy=(xs[-1], 100), xytext=(0, 8), textcoords="offset points",
                fontsize=F * 8.0, fontweight="semibold", color=WRITTEN, ha="right", va="bottom", gid="data")

    tv = [69.1, 66.7, 49.2]
    ax.plot(xs[:3], tv, color=TRAINED, lw=2.0, zorder=4, solid_joinstyle="round", gid="data")
    ax.scatter(xs[:3], tv, s=28, color=TRAINED, edgecolor="white", linewidth=0.8, zorder=5, gid="data")
    ax.annotate("Trained looped model, 49.2%", xy=(xs[2], tv[2]), xytext=(9, 0), textcoords="offset points",
                fontsize=F * 7.8, color=MUTED, ha="left", va="center", gid="data")

    save(fig, "chains_loops", "Accuracy on reasoning chains against loops")
    plt.close(fig)

# --------------------------------------------------------------------------- social card
def social(plt, d) -> None:
    from matplotlib.patches import FancyBboxPatch, Rectangle

    dpi = 160
    fig = plt.figure(figsize=(1200 / dpi, 630 / dpi), dpi=dpi)
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_xlim(0, 1200)
    ax.set_ylim(0, 630)
    ax.axis("off")
    navy, sky = "#0c2945", "#b9cdf2"          # MBZUAI navy
    ax.add_patch(Rectangle((0, 0), 1200, 630, fc=navy, ec="none"))
    ax.add_patch(Rectangle((0, 0), 1200, 10, fc="#e5c687", ec="none"))   # MBZUAI gold

    pt = 72 / dpi  # font size in points for a size given in px
    ax.text(72, 548, d["title"], fontsize=66 * pt, fontweight="bold", color="white", va="top")
    ax.text(72, 452, d["subtitle"], fontsize=27 * pt, color=sky, va="top", linespacing=1.35)

    ax.add_patch(FancyBboxPatch((72, 108), 560, 196, boxstyle="round,pad=0,rounding_size=14",
                                fc="white", ec="none"))
    ax.text(100, 278, d["result_task"].upper(), fontsize=19 * pt, fontweight="semibold",
            color=MUTED, va="top")
    ax.text(100, 236, d["result"], fontsize=44 * pt, fontweight="bold", color=WRITTEN,
            va="top", linespacing=1.12)

    # A small Tower of Hanoi: all disks moved to the third peg.
    base_y, peg_h = 118, 190
    pegs = [760, 905, 1050]
    ax.add_patch(Rectangle((700, base_y - 8), 410, 8, fc="#4a6f95", ec="none"))
    for x in pegs:
        ax.add_patch(Rectangle((x - 3.5, base_y), 7, peg_h, fc="#4a6f95", ec="none"))
    for i in range(7):
        wdt = 128 - i * 15
        ax.add_patch(FancyBboxPatch((pegs[2] - wdt / 2, base_y + i * 22 + 2), wdt, 18,
                                    boxstyle="round,pad=0,rounding_size=6",
                                    fc="#a9c6e3" if i % 2 == 0 else "#d8e6f4", ec="none"))

    ax.text(72, 56, d["byline"], fontsize=21 * pt, color="white", va="center")
    ax.text(1128, 56, d["url"], fontsize=19 * pt, color=sky, va="center", ha="right")

    out = HERE / "social.png"
    fig.savefig(out, dpi=dpi, metadata={"Software": None})
    plt.close(fig)
    print(f"wrote {out.name} (1200x630)")


# --------------------------------------------------------------------------- main
def main() -> None:
    data = json.loads(DATA.read_text(encoding="utf-8"))
    check(data)
    if "--check" in sys.argv:
        return
    plt = setup_matplotlib()
    figures = {"hanoi": hanoi, "written_vs_trained": written_vs_trained, "ouro_clock": ouro_clock,
               "glance": glance, "maze_loops": maze_loops, "maze_size": maze_size,
               "maze_compute": maze_compute, "maze_block": maze_block, "maze_training": maze_training,
               "chains_loops": chains_loops,
               "social": social}
    only = sys.argv[sys.argv.index("--only") + 1].split(",") if "--only" in sys.argv else list(figures)
    for name in only:
        figures[name](plt, data[name])


if __name__ == "__main__":
    main()
