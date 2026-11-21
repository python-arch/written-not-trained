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
# Palette (Observable-10-like, checked with the dataviz validator; adjacent pairs pass in this order).
BLUE, ORANGE, AQUA, YELLOW = "#2a78d6", "#eb6834", "#1baf7a", "#eda100"
MAGENTA, GREEN, VIOLET, RED = "#e87ba4", "#008300", "#6250d6", "#e34948"
RAMP = ["#cde2fb", "#9ec5f4", "#6da7ec", "#3987e5", "#2a78d6", "#1c5cab", "#104281", "#0d366b"]
# Roles, the same in every chart. Marks use the base colour; a coloured label uses the darker text shade.
WRITTEN, WRITTEN_TEXT = ORANGE, "#c4471a"      # the transformer Claude wrote: the hero series
TRAINED, TRAINED_TEXT = BLUE, "#1c5cab"        # TRM and trained looped models
HRM_C, AQUA_TEXT = AQUA, "#0f7d56"             # HRM and the standard transformer
LLM, YELLOW_TEXT = YELLOW, "#a87200"           # gradient descent on the written architecture, LDT, LLMs
DT_C = VIOLET                                  # DeepThinking (Bansal et al.)
INK = "#14213d"
MUTED = "#5b6475"
GRID = "#e1e0d9"
BAND = "#f3f2ec"
NEUTRAL = "#b8b6ae"                            # structure that is not a series

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
    for lg in fig.legends:                       # figure-level legends count too
        if lg.get_visible():
            boxes.append(lg.get_window_extent(r))
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
    ax.text(c0 - 0.4, 2.6e6, d["label_collapse"], fontsize=F * 8.2, color=YELLOW_TEXT, va="top",
            ha="left", linespacing=1.3)
    # The revised paper's tested range ends at 12.
    m = d["paper_max_tested_revised"]
    ax.axvline(m + 0.5, color=LLM, lw=1, ls=(0, (3, 2)), zorder=1)
    if d["label_paper_range"]:
        ax.text(m + 0.75, 1.3, d["label_paper_range"], fontsize=F * 8, color=YELLOW_TEXT, va="bottom",
                ha="left", linespacing=1.3)

    # Minimum move count, 2^N - 1.
    ax.plot(ns, moves, color="#d6d4cc", lw=1.6, zorder=2, solid_capstyle="round")

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
                    fontsize=F * 8, color=WRITTEN_TEXT, ha="right", va="bottom",
                    arrowprops=dict(arrowstyle="-", color=WRITTEN, lw=0.7, shrinkA=1, shrinkB=4))
    # Largest solved.
    s = d["solved_max"]
    ax.annotate(d["label_solved_max"], xy=(s, 2 ** s - 1), xytext=(n_max + 0.3, 160),
                fontsize=F * 8.6, fontweight="semibold", color=WRITTEN_TEXT, ha="right", va="center",
                linespacing=1.3, gid="data",
                arrowprops=dict(arrowstyle="-", color=WRITTEN, lw=0.8, shrinkA=2, shrinkB=4))
    # Move counts at selected sizes (computed by formula).
    for n in d["annotate_moves_at"]:
        ax.text(n + 0.15, (2 ** n - 1) * 2.0, f"{2 ** n - 1:,}", fontsize=F * 7.6, color=WRITTEN_TEXT,
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
    """Figure 8, beyond the checking range. Three slope panels (graph search, parity, reasoning chains), each joining
    a model's accuracy on the sizes Claude checked to its accuracy on longer inputs. Two layouts: wide (desktop, the
    panels side by side with one y-axis) and tall (phones, stacked), saved as written_vs_trained and
    written_vs_trained_tall."""
    for tall in (False, True):
        _written_vs_trained(plt, d, tall)


def _wvt_panel(ax, p, show_y: bool, fs: float) -> None:
    from matplotlib.patches import FancyBboxPatch

    x0, x1 = 0.0, 1.0
    ax.set_xlim(-0.36, 1.62)
    ax.set_ylim(-9, 109)
    # The sizes Claude checked: a soft rounded band behind the left column.
    ax.add_patch(FancyBboxPatch((-0.28, -7), 0.56, 114, boxstyle="round,pad=0,rounding_size=0.09",
                                mutation_aspect=60, facecolor=BAND, edgecolor="none", zorder=0))
    for y in (0, 50, 100):
        ax.axhline(y, color=GRID, lw=0.8, zorder=0.5)
    ax.set_yticks([0, 50, 100])
    ax.tick_params(length=0, labelsize=fs, pad=5)
    if show_y:
        ax.set_ylabel("Accuracy (%)", fontsize=fs, labelpad=6)
    else:
        ax.tick_params(labelleft=False)
    for s in ax.spines.values():
        s.set_visible(False)
    ax.set_xticks([x0, x1])
    ax.set_xticklabels([f"Checked\n{p['checking_label']}", f"Longer\n{p['test_label']}"], fontsize=fs * 0.95,
                       linespacing=1.35)
    for lab, col, w in zip(ax.get_xticklabels(), (INK, INK), ("semibold", "semibold")):
        lab.set_color(MUTED)

    if "chance" in p:
        ax.axhline(p["chance"], color=MUTED, lw=0.9, ls=(0, (2.5, 2.5)), zorder=1)
        ax.text(1.62, p["chance"] - 3, "chance", fontsize=fs * 0.86, color=MUTED, ha="right", va="top")

    # Trained models: thin blue slopes with a dot at each end.
    tests = []
    for t in p["trained"]:
        c, tv = t["checking"], t["test"]
        if isinstance(tv, list):
            lo_, hi_ = tv
            tests += tv
            mid = (lo_ + hi_) / 2
            ax.plot([x0, x1], [c, mid], color=TRAINED, lw=1.7, alpha=0.9, zorder=2, solid_capstyle="round",
                    gid="data")
            ax.plot([x1, x1], [lo_, hi_], color=TRAINED, lw=5, alpha=0.9, zorder=3, solid_capstyle="round",
                    gid="data")
        else:
            tests.append(tv)
            ax.plot([x0, x1], [c, tv], color=TRAINED, lw=1.7, alpha=0.9, zorder=2, solid_capstyle="round",
                    gid="data")
            ax.scatter([x1], [tv], s=46, marker="X", color=TRAINED, edgecolor="white", linewidth=0.8, zorder=4,
                       gid="data")
        ax.scatter([x0], [c], s=30, facecolor="white", edgecolor=TRAINED, linewidth=1.5, zorder=4, gid="data")

    # Written by Claude: one thick orange line on top.
    w = p["written"]
    ax.plot([x0, x1], [w["checking"], w["test"]], color=WRITTEN, lw=3.4, zorder=6, solid_capstyle="round",
            gid="data")
    from matplotlib.path import Path
    check = Path([(-0.55, 0.02), (-0.15, -0.38), (0.6, 0.42)], [Path.MOVETO, Path.LINETO, Path.LINETO])
    ax.scatter([x0, x1], [w["checking"], w["test"]], s=150, color=WRITTEN, edgecolor="white", linewidth=1.4,
               zorder=7, gid="data")
    ax.plot([x0, x1], [w["checking"], w["test"]], ls="none", marker=check, ms=7.5, mfc="none", mec="white",
            mew=2.0, zorder=8, gid="data")

    # End values, right of the "Longer" column.
    lx = x1 + 0.13
    ax.text(lx, w["test"], f"{w['test']:g}%", fontsize=fs * 1.12, fontweight="bold", color=WRITTEN_TEXT,
            ha="left", va="center", gid="data")
    lo_t, hi_t = min(tests), max(tests)
    rng = f"{lo_t:g}%" if lo_t == hi_t else f"{lo_t:g}–{hi_t:g}%"
    ty = (lo_t + hi_t) / 2
    if "chance" in p:
        ty = hi_t + 9          # above the chance line and its label
    else:
        ty = max(ty, 0) + 9
    ax.text(lx, ty, rng, fontsize=fs * 0.98, fontweight="semibold", color=TRAINED_TEXT, ha="left",
            va="center", gid="data")


def _written_vs_trained(plt, d, tall: bool) -> None:
    panels = d["panels"]
    fs = F * (9.0 if tall else 8.4)

    if tall:
        W, H = 4.7, 10.9
        fig = plt.figure(figsize=(W, H))
        ph, step = 1.75 / H, 3.55 / H
        boxes = [[0.16, 1 - 1.0 / H - ph - i * step, 0.8, ph] for i in range(3)]
    else:
        W, H = 9.6, 4.3
        fig = plt.figure(figsize=(W, H))
        boxes = [[0.075 + i * 0.31, 0.2, 0.275, 0.44] for i in range(3)]

    signs = []
    for i, (p, box) in enumerate(zip(panels, boxes)):
        ax = fig.add_axes(box)
        _wvt_panel(ax, p, show_y=(tall or i == 0), fs=fs)
        line = 0.125 if not tall else 0.14
        top_t = max(t["params"] for t in p["trained"])
        pic_s = 1.6                                   # task sign scale
        tx = (PIC_W * pic_s + 0.08) / (box[2] * W)    # title starts after the sign (axes fraction)
        ax.text(tx, 1 + 3.15 * line, p["title"], transform=ax.transAxes, fontsize=fs * 1.22,
                fontweight="semibold", color=INK, ha="left", va="bottom")
        signs.append((p["title"], box[0] * W, (box[1] + box[3] * (1 + 3.42 * line)) * H, pic_s))
        ax.text(0, 1 + 1.95 * line, f"Claude, {p['written']['params_label']}", transform=ax.transAxes,
                fontsize=fs * 0.88, color=WRITTEN_TEXT, ha="left", va="bottom")
        ax.text(0, 1 + 0.95 * line, f"Trained, up to {top_t:,} weights", transform=ax.transAxes,
                fontsize=fs * 0.88, color=TRAINED_TEXT, ha="left", va="bottom")

    # The task signs (the same ones as in the benchmark overview), drawn on a clear overlay in inches.
    ov = fig.add_axes([0, 0, 1, 1], zorder=-1)
    ov.set_xlim(0, W)
    ov.set_ylim(0, H)
    ov.axis("off")
    for task, x_in, yc_in, s in signs:
        sub = ov.inset_axes([x_in, yc_in - PIC_H * s / 2, PIC_W * s, PIC_H * s], transform=ov.transData)
        sub.set_xlim(0, PIC_W)
        sub.set_ylim(0, PIC_H)
        sub.axis("off")
        pictogram(sub, task, 0, PIC_H / 2)
        # The sign was drawn for its small size: scale its text and strokes with it.
        for a in sub.texts:
            a.set_fontsize(a.get_fontsize() * s)
        for a in sub.lines:
            a.set_linewidth(a.get_linewidth() * s)
        for a in sub.patches:
            a.set_linewidth(a.get_linewidth() * s)
            if hasattr(a, "set_mutation_scale"):
                a.set_mutation_scale(a.get_mutation_scale() * s)

    # One x-axis title for the whole figure.
    if tall:
        fig.text(0.56, 0.1 / H, "Input size", ha="center", va="bottom", fontsize=fs, color=MUTED)
    else:
        fig.text(0.075 + 0.31 + 0.1375, 0.02, "Input size", ha="center", va="bottom", fontsize=fs, color=MUTED)
    name = "written_vs_trained_tall" if tall else "written_vs_trained"
    save(fig, name, "Beyond the checking range")
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
        box(x, y_tok, s_tok, s_tok, "#fdf1d2" if clock else "white", LLM if clock else "#c9c7bf",
            lw=1.1)
    ax.text(xs[0], y_tok - 3.2, d["label_tokens_clock"], fontsize=F * 8.4, color=YELLOW_TEXT,
            fontweight="semibold", va="top")
    ax.text(xs[5], y_tok - 3.2, d["label_tokens_rest"], fontsize=F * 8.4, color=MUTED, va="top")
    for j in (5, 7, 10):
        ax.add_patch(FancyArrowPatch((xs[j] + s_tok / 2, y_tok + s_tok + 0.4),
                                     (xs[2] + s_tok / 2, y_tok + s_tok + 0.6),
                                     connectionstyle="arc3,rad=0.32", arrowstyle="-|>",
                                     mutation_scale=7, color=NEUTRAL, lw=0.9))

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
                           fc=BAND, ec="none", zorder=0))
    ax.text(loop_x[0] - 1.1, y_b - 3.6, d["label_trained_range"], fontsize=F * 7.8, color=MUTED,
            va="top")

    ax.text(4, y_a + h / 2, "Unedited", fontsize=F * 8.8, fontweight="semibold", color=YELLOW_TEXT,
            va="center")
    ax.text(4, y_b + h / 2, "With the fix", fontsize=F * 8.8, fontweight="semibold", color=WRITTEN_TEXT,
            va="center")
    for i, x in enumerate(loop_x):
        inside = i < k
        # Unedited: the clock keeps counting past the trained range.
        if inside:
            box(x, y_a, w, h, "#fdf1d2", LLM, lw=1.1)
        else:
            box(x, y_a, w, h, "white", LLM, lw=1.1, hatch="////", ls=(0, (3, 1.5)))
        ax.text(x + w / 2, y_a + h / 2, str(i + 1), fontsize=F * 9, fontweight="semibold",
                color=YELLOW_TEXT, ha="center", va="center",
                bbox=None if inside else dict(fc="white", ec="none", pad=0.6))
        # With the fix: same first loops, then held inside the trained range.
        if inside:
            box(x, y_b, w, h, "#fdf1d2", LLM, lw=1.1)
            ax.text(x + w / 2, y_b + h / 2, str(i + 1), fontsize=F * 9, fontweight="semibold",
                    color=YELLOW_TEXT, ha="center", va="center")
        else:
            box(x, y_b, w, h, "#fbe7dc", WRITTEN, lw=1.2)
            ax.text(x + w / 2, y_b + h / 2, "≤" + str(k), fontsize=F * 8.6, fontweight="semibold",
                    color=WRITTEN_TEXT, ha="center", va="center")
    # out-of-range bracket
    xa, xb = loop_x[k], loop_x[-1] + w
    ax.plot([xa, xa, xb, xb], [y_a - 1.4, y_a - 2.4, y_a - 2.4, y_a - 1.4], color=YELLOW_TEXT, lw=0.9)
    ax.text((xa + xb) / 2, y_a - 3.0, d["label_out_of_range"], fontsize=F * 8, color=YELLOW_TEXT,
            ha="center", va="top")
    ax.text((xa + xb) / 2, y_b - 3.6, d["label_fix"], fontsize=F * 7.8, color=WRITTEN_TEXT,
            ha="center", va="top")
    if d["label_readout"]:
        ax.text(4, 7.6, d["label_readout"], fontsize=F * 7.8, color=WRITTEN_TEXT, va="center")

    fig_title(fig, d["title"], 0.975)
    fig_footnote(fig, d["footnote"], 0.012)
    save(fig, "ouro_clock", d["title"].replace("\n", " "))
    plt.close(fig)



# --------------------------------------------------------------------------- glance: pictograms and logos
PIC_W, PIC_H = 0.27, 0.15          # inches; every pictogram fits this box
PIC_LINE = "#a9a79f"               # light warm gray structure
PIC_INK = INK                      # the strokes that carry the meaning
MODEL_LOGOS = {"Claude Opus 5.5, GPT-6 Astra": ["claude", "openai"], "GPT-6 Astra": ["openai"],
               "Gemini 4 Carbon": ["gemini"], "SmolLM2-360M": ["huggingface"], "Pythia-410M": ["eleutherai"],
               "Qwen3-235B-A22B": ["qwen"], "Qwen3-235B-A22B, program synthesis": ["qwen"]}


def pictogram(ax, task, x, yc):
    """A small hand-made sign for each benchmark, in the box x..x+PIC_W, centred on yc (inches)."""
    from matplotlib.patches import Circle, FancyArrowPatch, FancyBboxPatch, Polygon, Rectangle
    h, w = PIC_H, PIC_W
    y0 = yc - h / 2
    lw = 0.9
    def arrow(p, q, rad=0.0, col=PIC_INK):
        ax.add_patch(FancyArrowPatch(p, q, arrowstyle="-|>", mutation_scale=4.2, lw=lw, color=col,
                                     connectionstyle=f"arc3,rad={rad}", shrinkA=0, shrinkB=0))
    def text(tx, ty, s, size=6.4, col=PIC_INK, weight="semibold"):
        ax.text(tx, ty, s, fontsize=size, color=col, ha="center", va="center", fontweight=weight)
    if task == "Maze-Hard":
        cs = h / 4
        bx = x + (w - h) / 2
        for r, c in [(1, 0), (1, 2), (2, 2), (3, 2), (2, 3)]:
            ax.add_patch(Rectangle((bx + c * cs, y0 + (3 - r) * cs), cs, cs, fc="#8f8d85", ec="none"))
        ax.add_patch(Rectangle((bx, y0), h, h, fc="none", ec=PIC_LINE, lw=lw))
        path = [(3, 0), (2, 0), (2, 1), (1, 1), (0, 1), (0, 2), (0, 3)]
        ax.plot([bx + (c + 0.5) * cs for r, c in path], [y0 + (3.5 - r) * cs for r, c in path], color=PIC_INK, lw=1.3,
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
        ax.plot([A[0], A[0] + 0.07, B[0] - 0.07, B[0]], [A[1], yc + 0.05, yc - 0.05, B[1]], color=PIC_INK, lw=1.3,
                solid_joinstyle="round")
        for (px, py), lab in ((A, "A"), (B, "B")):
            ax.add_patch(Circle((px, py), 0.033, fc="white", ec=PIC_LINE, lw=lw, zorder=5))
            ax.text(px, py, lab, fontsize=4.6, color=PIC_INK, ha="center", va="center", fontweight="bold", zorder=6)
    elif task == "Parity":
        text(x + w / 2, yc, "1011", size=6.6)
    elif task == "Reasoning chains":
        bs = 0.055
        xs = [x + 0.005, x + w / 2 - bs / 2, x + w - bs - 0.005]
        for bxx in xs:
            ax.add_patch(Rectangle((bxx, yc - bs / 2), bs, bs, fc="white", ec=PIC_LINE, lw=lw))
            ax.text(bxx + bs / 2, yc, "x", fontsize=4.2, color=PIC_INK, ha="center", va="center")
        for a, b in zip(xs, xs[1:]):
            arrow((a + bs + 0.004, yc), (b - 0.004, yc))
    elif task == "Boolean Expressions":
        text(x + 0.06, yc, "T", size=6.4)
        text(x + w - 0.06, yc, "F", size=6.4)
        ax.plot([x + w / 2 - 0.028, x + w / 2, x + w / 2 + 0.028], [yc - 0.03, yc + 0.035, yc - 0.03], color=PIC_INK,
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
                ax.plot([cx - 0.022, cx - 0.006, cx + 0.024], [cy, cy - 0.017, cy + 0.02], color=PIC_INK, lw=1.2,
                        solid_joinstyle="round", solid_capstyle="round")
            else:
                for sgn in (-1, 1):
                    ax.plot([cx - 0.018, cx + 0.018], [cy - sgn * 0.018, cy + sgn * 0.018], color=MUTED, lw=1.2,
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
KIND_TEXT = {"written": WRITTEN_TEXT, "trained": TRAINED_TEXT, "llm": YELLOW_TEXT}


def bar_role(r):
    """The colour role of a glance bar: LDT is yellow by the shared mapping, other trained models blue."""
    return "llm" if r["label"].startswith("LDT") else r["kind"]


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
                 color=WRITTEN_TEXT, ha="right", va="center")
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
            role = bar_role(r)
            c, ct = KIND[role], KIND_TEXT[role]
            yc = yy - bar_h / 2
            lh = F * 7.6 * 1.05 / 72
            tx = logos_at(ax, MODEL_LOGOS.get(r["label"], []), x_lab + 0.1, yc, lh)
            ax.text(tx, yc, r["label"], fontsize=F * 7.6, color=ct, va="center")
            ax.add_patch(Rectangle((x0, yc - 0.07), x100 - x0, 0.14, fc="#f3f2ec", ec="none", zorder=0))
            ax.add_patch(Rectangle((x0, yc - 0.07), max(r["value"], 0) * scale, 0.14, fc=c, ec="none",
                                   zorder=2, gid="data"))
            end = x0 + r["value"] * scale
            if r["value"] < 99 and end + 0.5 > x100:      # would cross the 100% line: write it inside the bar
                ax.text(end - 0.05, yc, r["text"], fontsize=F * 7.6, color=INK if role == "llm" else "white",
                        ha="right", va="center",
                        fontweight="semibold", zorder=4, gid="data")
            else:
                ax.text(max(end, x100 if r["value"] >= 99 else end) + 0.06, yc, r["text"], fontsize=F * 7.6,
                        color=ct, va="center", gid="data")
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
def _maze_icon(fig, rect):
    """A small maze sign (walls, start, goal and an orange path) in its own little axes at `rect` (figure coords)."""
    from matplotlib.patches import Rectangle
    ia = fig.add_axes(rect)
    ia.set_xlim(0, 5)
    ia.set_ylim(0, 5)
    ia.set_aspect("equal")
    ia.axis("off")
    walls = [(0, 1), (1, 1), (1, 3), (2, 3), (3, 1), (3, 2), (3, 3), (4, 3)]
    ia.add_patch(Rectangle((0, 0), 5, 5, fc="white", ec=MUTED, lw=1.0))
    for c, r in walls:
        ia.add_patch(Rectangle((c, 4 - r), 1, 1, fc="#3b4048", ec="white", lw=0.6))
    path = [(0, 0), (1, 0), (2, 0), (2, 1), (2, 2), (1, 2), (0, 2), (0, 3), (0, 4), (1, 4), (2, 4), (3, 4), (4, 4)]
    ia.plot([c + 0.5 for c, r in path], [4.5 - r for c, r in path], color=WRITTEN, lw=1.6,
            solid_joinstyle="round", solid_capstyle="round")
    for (c, r), col in (((0, 0), BLUE), ((4, 4), WRITTEN)):
        ia.add_patch(Rectangle((c + 0.22, 4 - r + 0.22), 0.56, 0.56, fc=col, ec="none"))
    return ia


def maze_loops(plt, d) -> None:
    """Figure 6. Mazes solved against loops, on one log axis, for four looped models. The transformer Claude wrote
    (Maze-Hard, any shortest path, maze_loops.json), TRM and HRM (Maze-Hard, our runs of the released checkpoints at
    1-64 reasoning steps, maze_loops_baselines.json) and DeepThinking (13x13 mazes, read from the vector data of
    Figure 6 in Bansal et al. 2022, maze_dt_overthink.json). An open circle marks the most loops each trained model
    saw in training."""
    from matplotlib.ticker import FixedLocator, NullLocator

    ours = json.loads((HERE / "maze_loops.json").read_text(encoding="utf-8"))["series"]
    base = json.loads((HERE / "maze_loops_baselines.json").read_text(encoding="utf-8"))
    dt = json.loads((HERE / "maze_dt_overthink.json").read_text(encoding="utf-8"))["series"]

    HRM_C, HRM_TEXT = AQUA, "#11805a"
    DT_C, DT_TEXT = YELLOW, "#9a6a00"
    fs = F * 8.3

    W, H = 6.3, 3.75
    fig = plt.figure(figsize=(W, H))
    ax = fig.add_axes([0.125, 0.17, 0.7, 0.66])
    ax.set_xscale("log")
    ax.set_xlim(1, 10000)
    ax.set_ylim(-5, 108)
    ax.yaxis.set_major_locator(FixedLocator([0, 25, 50, 75, 100]))
    ax.xaxis.set_major_locator(FixedLocator([1, 10, 100, 1000, 10000]))
    ax.xaxis.set_minor_locator(NullLocator())
    ax.set_xticklabels(["1", "10", "100", "1,000", "10,000"])
    ax.grid(axis="y", color=GRID, lw=0.8)
    ax.grid(axis="x", color=GRID, lw=0.6, ls=(0, (1.5, 2.5)))
    ax.set_axisbelow(True)
    for s in ("top", "right", "left"):
        ax.spines[s].set_visible(False)
    ax.spines["bottom"].set_color(GRID)
    ax.tick_params(length=0, labelsize=fs, pad=5)
    ax.set_xlabel("Loops (log scale)", fontsize=fs, labelpad=6)
    ax.set_ylabel("Mazes solved (%)", fontsize=fs, labelpad=8)

    # The maze sign sits at the top of the y-axis, above its title.
    yl = 0.125 - 0.095
    _maze_icon(fig, [yl - 0.003, 0.835, 0.06, 0.06 * W / H])

    def line(xs, ys, col, lw, z):
        ax.plot(xs, ys, color=col, lw=lw, zorder=z, solid_joinstyle="round", solid_capstyle="round", gid="data")

    # DeepThinking (13x13 mazes): rises, then overthinks.
    dx, dy = zip(*dt)
    line(dx, dy, DT_C, 2.2, 3)
    # TRM and HRM on Maze-Hard: reasoning steps x passes of the repeated block.
    for key, col in (("trm", TRAINED), ("hrm", HRM_C)):
        xs, ys = zip(*base[key]["series"])
        line(xs, ys, col, 2.2, 4)
        ax.scatter([xs[0]], [ys[0]], s=22, color=col, edgecolor="white", linewidth=1.0, zorder=5, gid="data")
    # The transformer Claude wrote: 0% until each maze's path is complete, then 100% to 7,200 loops.
    ox, oy = zip(*[(r, a) for r, a in ours if r >= 1])
    ax.step(ox, oy, where="post", color=WRITTEN, lw=3.0, zorder=6, gid="data")
    far = [r for r, a in ours if r >= 1800]
    ax.scatter(far, [100] * len(far), s=40, color=WRITTEN, edgecolor="white", linewidth=1.2, zorder=7, gid="data")

    # Open circles: the most loops each trained model saw in training.
    marks = [(30, dict(dt)[30], DT_C),
             (base["trm"]["trained_steps"] * base["trm"]["loops_per_step"],
              dict(map(tuple, base["trm"]["series"]))[base["trm"]["trained_steps"] * base["trm"]["loops_per_step"]], TRAINED),
             (base["hrm"]["trained_steps"] * base["hrm"]["loops_per_step"],
              dict(map(tuple, base["hrm"]["series"]))[base["hrm"]["trained_steps"] * base["hrm"]["loops_per_step"]], HRM_C)]
    for x, y, col in marks:
        ax.scatter([x], [y], s=46, facecolor="white", edgecolor=col, linewidth=1.8, zorder=8, gid="data")

    # Direct labels at the ends of the lines.
    trm_end, hrm_end = base["trm"]["series"][-1], base["hrm"]["series"][-1]
    t_trm = ax.text(trm_end[0] * 1.18, trm_end[1] - 6.5, f"TRM {trm_end[1]:g}%", fontsize=fs, color=TRAINED_TEXT,
                    fontweight="semibold", ha="left", va="center", gid="data")
    t_hrm = ax.text(hrm_end[0] * 1.18, hrm_end[1] - 6, f"HRM {hrm_end[1]:g}%", fontsize=fs, color=HRM_TEXT,
                    fontweight="semibold", ha="left", va="center", gid="data")
    t_dt = ax.text(1.12, 42, "DeepThinking\n13×13 mazes", fontsize=fs * 0.95, color=DT_TEXT, fontweight="semibold",
                   ha="left", va="center", linespacing=1.3, gid="data")
    t_ours = ax.text(10000, 104.5, "Written by Claude", fontsize=fs * 1.04, color=WRITTEN_TEXT,
                     fontweight="semibold", ha="right", va="bottom", gid="data")
    ax.text(10000, 4, "most loops used in training", fontsize=fs * 0.86, color=MUTED, ha="right", va="center")
    key = ax.text(10000, 4, "most loops used in training", fontsize=fs * 0.86, alpha=0, ha="right", va="center")
    fig.canvas.draw()
    bb = key.get_window_extent(fig.canvas.get_renderer())
    kx, ky = ax.transData.inverted().transform((bb.x0 - 9 * fig.dpi / 72, (bb.y0 + bb.y1) / 2))
    ax.scatter([kx], [ky], s=40, facecolor="white", edgecolor=MUTED, linewidth=1.5, zorder=8, clip_on=False)
    add_logos(fig, ax, t_trm, ["samsung"], shift=True)
    add_logos(fig, ax, t_hrm, ["sapient"], shift=True)
    add_logos(fig, ax, t_ours, ["claude"])

    save(fig, "maze_loops", "Mazes solved against loops")
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
        mc = LLM if t["label"].startswith("LDT") else HRM_C if t["label"] == "HRM" else TRAINED
        ax.scatter([x], [t["accuracy"]], s=46, color=mc, edgecolor="white", linewidth=0.8, zorder=3, gid="data")
        dx, dy, ha, va = offsets.get(t["label"], (8, 0, "left", "center"))
        short = {"LDT, 512 paths per maze": "LDT", "LDT, one path per maze": "LDT, one path"}.get(t["label"], t["label"])
        text = f"{short}, {t['accuracy']}%"
        lab = ax.annotate(text, xy=(x, t["accuracy"]), xytext=(dx, dy), textcoords="offset points", fontsize=F * 7.7,
                          color=MUTED, ha=ha, va=va, linespacing=1.25, gid="data")
        logo = {"TRM": "samsung", "HRM": "sapient"}.get(t["label"])
        if logo:
            add_logos(fig, ax, lab, [logo])
    w = d["written"]
    ax.scatter([w["weights"]], [w["accuracy"]], s=70, color=WRITTEN, edgecolor="white", linewidth=1.0, zorder=5,
               gid="data")
    ax.annotate(f"{w['label']}, {w['accuracy']:g}%",
                xy=(w["weights"], w["accuracy"]), xytext=(10, -10), textcoords="offset points",
                fontsize=F * 8.2, fontweight="semibold", color=WRITTEN_TEXT, ha="left", va="top", linespacing=1.3,
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
                fontweight="semibold", color=WRITTEN_TEXT, ha="center", va="bottom", linespacing=1.25, gid="data")

    ldt = v["ldt"]
    ax.plot([ldt["lo"], ldt["hi"]], [max(ldt["acc"])] * 2, color=LLM, lw=1.4, zorder=2, gid="data")
    ax.plot([ldt["hi"]] * 2, [max(ldt["acc"]) - 2.2, max(ldt["acc"]) + 2.2], color=LLM, lw=1.4, zorder=2,
            gid="data")
    for a in ldt["acc"]:
        ax.scatter([ldt["lo"]], [a], s=38, color=LLM, edgecolor="white", linewidth=0.8, zorder=3, gid="data")
    for key in ("hrm", "trm"):
        x, a = v[key]
        ax.scatter([x], [a], s=38, color=HRM_C if key == "hrm" else TRAINED, edgecolor="white", linewidth=0.8,
                   zorder=3, gid="data")
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
    """The maze transformer's one block as a left-to-right model figure, from maze_block_map.md: a cell's 35 numbers
    (the input token), attention with four heads (one per neighbour), the MLP's 86 neurons by job, the character each
    cell writes, and the loop back to the input. A diagram, not a chart: no data layer."""
    from matplotlib.patches import FancyArrowPatch, FancyBboxPatch, Rectangle

    W_CSS, XU = 920, 96.0                              # 920 CSS px wide, 96 drawing units across
    H = 38.0
    u_in = W_CSS / CSS_DPI / XU                        # inches per unit
    fig = plt.figure(figsize=(XU * u_in, H * u_in))
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_xlim(0, XU)
    ax.set_ylim(0, H)
    ax.axis("off")
    col = {g["key"]: g["_colour"] for g in d["groups"]}
    edge, soft, block_fc = "#cfcdc5", "#f6f5f1", "#fdf1eb"
    f_title, f_lab, f_small = F * 9.2, F * 7.8, F * 7.2
    y_title = 36.4

    def box(x0, y0, x1, y1, fc, ec, lw=1.0, r=0.9, z=1):
        ax.add_patch(FancyBboxPatch((x0, y0), x1 - x0, y1 - y0, boxstyle=f"round,pad=0,rounding_size={r}",
                                    fc=fc, ec=ec, lw=lw, zorder=z))

    def arrow(p0, p1, c=MUTED, lw=1.3, ms=10, z=3):
        ax.add_patch(FancyArrowPatch(p0, p1, arrowstyle="-|>", mutation_scale=ms, color=c, lw=lw,
                                     shrinkA=0, shrinkB=0, zorder=z))

    # ---- the input token: a cell's 35 numbers, top to bottom, grouped by meaning
    ax.text(1.0, y_title, d["bar_label"], fontsize=f_title, fontweight="semibold", color=INK, va="baseline")
    bx0, bx1, ch, gap, sub_gap = 15.6, 17.6, 0.68, 0.5, 0.25
    y = 34.4
    spans = {}
    for g in d["groups"]:
        top, every = y, g["n"] // len(g["sub"]) if g.get("sub") else 0
        subs = []
        for i in range(g["n"]):
            if every and i % every == 0:
                if i:
                    y -= sub_gap
                subs.append(y)
            ax.add_patch(Rectangle((bx0, y - ch), bx1 - bx0, ch, fc=g["_colour"], ec="white", lw=0.6, zorder=2))
            y -= ch
        spans[g["key"]] = (top, y, subs, every)
        y -= gap
    bar_bottom = y + gap
    for g in d["groups"]:
        top, bot, subs, every = spans[g["key"]]
        if not g.get("sub"):
            ax.text(14.9, (top + bot) / 2, g["label"], fontsize=f_small, color=INK, ha="right", va="center")
            continue
        for name, st in zip(g["sub"], subs):
            ax.text(14.9, st - every * ch / 2, name, fontsize=f_small, color=MUTED, ha="right", va="center")
        xb = 10.9
        ax.plot([xb + 0.45, xb, xb, xb + 0.45], [top - 0.1, top - 0.1, bot + 0.1, bot + 0.1], color=MUTED, lw=0.8)
        ax.text(xb - 0.6, (top + bot) / 2, g["label"].replace(" from ", " from\n"), fontsize=f_small, color=INK,
                ha="right", va="center", linespacing=1.25)

    # ---- the block: attention, then the MLP
    y_mid = 20.6
    blk = (21.0, 6.4, 77.0, 34.9)
    box(*blk, fc=block_fc, ec=WRITTEN, lw=1.5, r=1.4, z=0)
    ax.text(blk[0], y_title, d["block_label"], fontsize=f_title, fontweight="semibold", color=WRITTEN_TEXT,
            va="baseline")
    arrow((bx1 + 0.3, y_mid), (23.0, y_mid))

    # attention: four heads, each a 3x3 patch with the neighbour it reads
    ax0, ax1 = 23.0, 44.0
    box(ax0, 8.0, ax1, 33.4, "white", edge)
    ax.text(ax0 + 1.2, 32.3, d["attn_label"], fontsize=f_lab, fontweight="semibold", color=INK, va="top")
    s = 2.0
    step = {"up": (0, 1), "left": (-1, 0), "right": (1, 0), "down": (0, -1)}
    centres = [(28.4, 25.0), (38.6, 25.0), (28.4, 14.4), (38.6, 14.4)]
    for h, (cx, cy) in zip(d["heads"], centres):
        dx, dy = step[h["dir"]]
        for i in (-1, 0, 1):
            for j in (-1, 0, 1):
                x, yy = cx + i * s - s / 2, cy + j * s - s / 2
                if (i, j) == (0, 0):
                    fc_, ec_, lw_ = "white", INK, 1.2
                elif (i, j) == (dx, dy):
                    fc_, ec_, lw_ = "#c6eadb", col["copied"], 1.0
                else:
                    fc_, ec_, lw_ = soft, "#dcdad2", 0.6
                ax.add_patch(Rectangle((x, yy), s, s, fc=fc_, ec=ec_, lw=lw_, zorder=2 + (lw_ > 0.7)))
        arrow((cx + dx * s * 1.3, cy + dy * s * 1.3), (cx + dx * s * 0.56, cy + dy * s * 0.56), c=AQUA_TEXT,
              lw=1.5, ms=8, z=5)
        ax.text(cx, cy - 1.5 * s - 0.55, h["name"], fontsize=f_small, fontweight="semibold", color=AQUA_TEXT,
                ha="center", va="top")
    arrow((ax1 + 0.2, y_mid), (46.3, y_mid))

    # MLP: 86 neurons as chips, one row block per job, labelled directly
    mx0, mx1 = 46.5, 75.0
    box(mx0, 8.0, mx1, 33.4, "white", edge)
    n_mlp = sum(u["n"] for u in d["units"])           # the total is computed from the jobs, not stored
    ax.text(mx0 + 1.2, 32.3, f'{d["mlp_label"]}, {n_mlp} neurons', fontsize=f_lab, fontweight="semibold", color=INK,
            va="top")
    chip, pitch, job_gap, per_row = 0.95, 1.22, 0.62, 10
    y = 29.4
    for u in d["units"]:
        c = col[u["writes"]]
        if u.get("_tint"):
            c = mix_hex(c, "#ffffff", u["_tint"])
        rows = -(-u["n"] // per_row)
        for i in range(u["n"]):
            r, k = divmod(i, per_row)
            ax.add_patch(FancyBboxPatch((48.0 + k * pitch, y - r * pitch - chip), chip, chip,
                                        boxstyle="round,pad=0,rounding_size=0.18", fc=c, ec="none", zorder=2))
        yc = y - ((rows - 1) * pitch + chip) / 2
        ax.text(48.0 + (per_row - 1) * pitch + chip + 1.2, yc, f'{u["name"]}, {u["n"]}', fontsize=f_small,
                color=INK, va="center")
        y -= rows * pitch + job_gap
    ax.text(48.0, y - 0.2, d["pairs_note"], fontsize=f_small, color=MUTED, va="top", linespacing=1.3)
    arrow((mx1 + 0.2, y_mid), (79.0, y_mid))

    # ---- what each cell writes
    ox0, ox1 = 79.2, 95.4
    ax.text(ox0, y_title, d["readout_label"], fontsize=f_title, fontweight="semibold", color=INK, va="baseline")
    box(ox0, 8.0, ox1, 33.4, "white", edge)
    for i, r in enumerate(d["readout"]):
        cy = 29.6 - i * 4.4
        path = r["from"] == "path"
        ax.add_patch(Rectangle((ox0 + 1.4, cy - 1.25), 2.5, 2.5, fc="#fbe1d6" if path else soft,
                               ec=WRITTEN if path else edge, lw=0.9, zorder=2))
        ax.text(ox0 + 2.65, cy, r["char"], fontsize=f_lab, fontweight="semibold", color=INK, ha="center",
                va="center", zorder=3)
        ax.text(ox0 + 5.0, cy, r["from"], fontsize=f_lab, color=INK, va="center")

    # ---- the loop: the block's output is the next loop's input
    yl = 3.0
    lx0, lx1 = (bx0 + bx1) / 2, (ox0 + ox1) / 2
    ax.plot([lx1, lx1, lx0], [8.0, yl, yl], color=WRITTEN, lw=1.7, solid_joinstyle="round", zorder=1)
    arrow((lx0, yl), (lx0, bar_bottom - 0.35), c=WRITTEN, lw=1.7, ms=11, z=1)
    ax.text((blk[0] + blk[2]) / 2, yl, d["loop_label"], fontsize=f_lab, fontweight="semibold", color=WRITTEN_TEXT,
            ha="center", va="center", zorder=4,
            bbox=dict(boxstyle="round,pad=0.35,rounding_size=0.6", fc="white", ec="none"))

    fig_title(fig, d["title"], 1 - 1.5 / H, width=46)
    save(fig, "maze_block", d["title"])
    plt.close(fig)


def mix_hex(a: str, b: str, t: float) -> str:
    ca = [int(a[i:i + 2], 16) for i in (1, 3, 5)]
    cb = [int(b[i:i + 2], 16) for i in (1, 3, 5)]
    return "#" + "".join(f"{round(p + (q - p) * t):02x}" for p, q in zip(ca, cb))



# --------------------------------------------------------------------------- maze: Claude as the optimizer
CLAUDE = ORANGE       # orange: Claude's revisions


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
        ax.plot([s for s, _ in gd], [a for _, a in gd], color=GD_INK, lw=1.8, zorder=2, gid="data")
        s, a = gd[-1]
        ax.annotate(d["label_gd"], xy=(s, a), xytext=(-4, -8), textcoords="offset points", fontsize=F * 7.6,
                    color=YELLOW_TEXT, ha="right", va="top", linespacing=1.25, gid="data")

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
                    fontsize=F * 7.9, color=WRITTEN_TEXT, fontweight=lab.get("weight", "normal"), ha=lab["ha"],
                    va=lab["va"], linespacing=1.25, gid="data")

    fig_title(fig, d["title"], 0.965, width=44)
    fig_footnote(fig, d["footnote"], 0.02)
    save(fig, "claude_optimizer", d["title"])
    plt.close(fig)


# --------------------------------------------------------------------------- maze: written vs trained, over time
GD_INK = YELLOW       # the written architecture, trained by gradient descent
STD_INK = AQUA        # the standard transformer


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
    """Test mazes solved against hours. The left of the chart (light orange, "Written weights") holds Claude's
    transformer as one big star at 39 minutes (its final weights) and 100%, with no gradient training. The right (light grey, "Gradient-based methods")
    holds the training runs: TRM, a standard transformer, and the same architecture trained by gradient descent at 0%.
    A legend below the chart names the four series."""
    from matplotlib.lines import Line2D
    from matplotlib.ticker import FixedLocator

    v = json.loads((HERE / "maze_training.json").read_text(encoding="utf-8"))
    XMAX, SPLIT = 8.0, 1.7
    if tall:
        fig = plt.figure(figsize=(3.7, 5.9))
        ax = fig.add_axes([0.16, 0.36, 0.71, 0.56])
    else:
        fig = plt.figure(figsize=(5.4, 4.7))
        ax = fig.add_axes([0.11, 0.3, 0.84, 0.6])
    ax.set_ylim(-9, 122)
    ax.yaxis.set_major_locator(FixedLocator([0, 50, 100]))
    ax.grid(axis="y", color="white", lw=1.0, zorder=1)   # white grid shows on both shaded regions
    ax.set_axisbelow(True)
    ax.tick_params(length=0, labelsize=F * 8.2)
    for s in ("left", "top", "right"):
        ax.spines[s].set_visible(False)
    ax.spines["bottom"].set_color(GRID)
    ax.set_yticklabels(["0", "50", "100"])
    ax.set_xlim(-0.4, XMAX + 0.6)
    ax.xaxis.set_major_locator(FixedLocator(range(0, int(XMAX) + 1)))
    ax.set_xticklabels([str(h) for h in range(0, int(XMAX) + 1)])
    ax.set_xlabel("Time (hours)", fontsize=F * 8.4)
    ax.set_ylabel("Test mazes solved (%)", fontsize=F * 8.4)
    big, label, small = F * 14, F * 8.4, F * 7.8

    # Two regions: written weights on the left, gradient-based methods on the right.
    ax.axvspan(-0.4, SPLIT, color="#fbdac6", lw=0, zorder=0)
    ax.axvspan(SPLIT, XMAX + 0.6, color="#e4e4de", lw=0, zorder=0)
    ax.text((-0.4 + SPLIT) / 2, 114, "Written\nweights" if tall else "Written weights", fontsize=small * 1.05,
            color="#a63c14", fontweight="semibold", ha="center", va="center", linespacing=1.15)
    ax.text((SPLIT + XMAX + 0.6) / 2, 114, "Gradient-based methods", fontsize=small * 1.05, color="#4a5262",
            fontweight="semibold", ha="center", va="center")

    # A standard transformer and TRM (our run up to its best point, then the placeholder points).
    sx, sy = [p[0] for p in v["std"]], [p[1] for p in v["std"]]
    ax.plot(sx, sy, color=STD_INK, lw=2.0, zorder=3, solid_joinstyle="round", solid_capstyle="round", gid="data")
    ax.scatter(sx[1:], sy[1:], s=34, marker="o", facecolor="white", edgecolor=STD_INK, linewidth=1.7, zorder=4, gid="data")
    ax.text(sx[-1] + (0.32 if tall else 0.15), sy[-1], f"{max(sy):g}%", fontsize=small, color=MUTED, ha="left", va="center", gid="data")
    best = max(range(len(v["trm"])), key=lambda i: v["trm"][i][1])
    trm = v["trm"][: best + 1] + TRM_PLACEHOLDER
    tx, ty = [p[0] for p in trm], [p[1] for p in trm]
    ax.plot(tx, ty, color=TRAINED, lw=2.0, zorder=3, solid_joinstyle="round", solid_capstyle="round", gid="data")
    ax.scatter(tx[1:], ty[1:], s=40, marker="o", color=TRAINED, edgecolor="white", linewidth=1.4, zorder=4, gid="data")
    ax.text(tx[-1] + (0.32 if tall else 0.15), ty[-1], f"{ty[-1]:g}%", fontsize=small, color=TRAINED_TEXT, ha="left", va="center",
            gid="data")

    # The same architecture trained by gradient descent, at 0%.
    ends = [r["points"][-1][0] for r in v["gd"].values()]
    x_gd = max(ends)
    ax.plot([0, x_gd], [0, 0], color=GD_INK, lw=2.4, dashes=(3.0, 2.2), dash_capstyle="round", zorder=8, gid="data")
    ax.annotate("0%", xy=(x_gd, 0), xytext=(6, 0), textcoords="offset points", fontsize=label * 1.15,
                fontweight="bold", color=YELLOW_TEXT, ha="left", va="center", gid="data")

    # The transformer Claude wrote: one big star at 39 minutes, when its final weights were ready, at 100%.
    W_AT = 39 / 60
    ax.scatter([W_AT], [100], s=520 if tall else 700, marker="*", color=WRITTEN, edgecolor="white", linewidth=1.0,
               zorder=9, clip_on=False, gid="data")
    ax.annotate("100%", xy=(W_AT, 100), xytext=(14, -1), textcoords="offset points", fontsize=big * (0.85 if tall else 1),
                fontweight="bold", color=WRITTEN_TEXT, ha="left", va="center", gid="data")

    # What each costs: Claude's tokens for the written weights, and TRM's GPU time (8 RTX 6000 GPUs for 8 hours,
    # 64 GPU-hours at a median on-demand price of $1.09 per GPU-hour, November 2026).
    cs = small * (0.92 if tall else 1.0)
    ax.text(XMAX + 0.45, 17, "Claude ≈ $19 in tokens" if tall else "Claude: 196,771 output tokens ≈ $19", fontsize=cs, color=WRITTEN_TEXT, fontweight="semibold",
            ha="right", va="center", gid="data")
    ax.text(XMAX + 0.45, 8, "TRM ≈ $70 in GPU time" if tall else "TRM: 8 RTX 6000 GPUs × 8 h ≈ $70", fontsize=cs, color=TRAINED_TEXT, fontweight="semibold",
            ha="right", va="center", gid="data")

    # Legend below the chart.
    handles = [Line2D([0], [0], marker="*", ls="none", ms=15, color=WRITTEN, mec="white", mew=0.8),
               Line2D([0], [0], color=TRAINED, lw=2.0, marker="o", ms=6, mfc=TRAINED, mec="white", mew=1.2),
               Line2D([0], [0], color=STD_INK, lw=2.0, marker="o", ms=6, mfc="white", mec=STD_INK, mew=1.5),
               Line2D([0], [0], color=GD_INK, lw=2.4, dashes=(3.0, 2.2))]
    labels = ["Written by Claude, no gradient training", "TRM", "Standard transformer",
              "Same architecture, gradient descent"]
    if tall:
        fig.legend(handles, labels, loc="upper left", bbox_to_anchor=(0.04, 0.27), ncol=1, frameon=False,
                   fontsize=small, handlelength=2.4, labelspacing=0.55, handletextpad=0.7)
    else:
        fig.legend([handles[i] for i in (0, 3, 1, 2)], [labels[i] for i in (0, 3, 1, 2)], loc="upper center",
                   bbox_to_anchor=(0.5, 0.175), ncol=2, frameon=False, fontsize=small, handlelength=2.2,
                   columnspacing=1.4, labelspacing=0.6, handletextpad=0.6)

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
                fontsize=F * 8.0, fontweight="semibold", color=WRITTEN_TEXT, ha="right", va="bottom", gid="data")

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
    navy, sky = INK, "#cde2fb"                 # ink background, light blue text
    ax.add_patch(Rectangle((0, 0), 1200, 630, fc=navy, ec="none"))
    ax.add_patch(Rectangle((0, 0), 1200, 10, fc=ORANGE, ec="none"))

    pt = 72 / dpi  # font size in points for a size given in px
    for k, line in enumerate(d["title"].split("\n")):          # one text per line, for even spacing
        ax.text(72, 574 - 54 * k, line, fontsize=44 * pt, fontweight="bold", color="white", va="top")
    for k, line in enumerate(d["subtitle"].split("\n")):
        ax.text(72, 392 - 34 * k, line, fontsize=23 * pt, color=sky, va="top")

    ax.add_patch(FancyBboxPatch((72, 96), 560, 186, boxstyle="round,pad=0,rounding_size=14",
                                fc="white", ec="none"))
    ax.text(100, 258, d["result_task"].upper(), fontsize=19 * pt, fontweight="semibold",
            color=MUTED, va="top")
    ax.text(100, 218, d["result"], fontsize=42 * pt, fontweight="bold", color=WRITTEN_TEXT,
            va="top", linespacing=1.12)

    # A small Tower of Hanoi: all disks moved to the third peg.
    base_y, peg_h = 104, 176
    pegs = [760, 905, 1050]
    ax.add_patch(Rectangle((700, base_y - 8), 410, 8, fc=RAMP[5], ec="none"))
    for x in pegs:
        ax.add_patch(Rectangle((x - 3.5, base_y), 7, peg_h, fc=RAMP[5], ec="none"))
    for i in range(7):
        wdt = 128 - i * 15
        ax.add_patch(FancyBboxPatch((pegs[2] - wdt / 2, base_y + i * 22 + 2), wdt, 18,
                                    boxstyle="round,pad=0,rounding_size=6",
                                    fc=RAMP[1] if i % 2 == 0 else RAMP[0], ec="none"))

    ax.text(72, 50, d["byline"], fontsize=18 * pt, color="white", va="center")
    ax.text(1128, 50, d["url"], fontsize=17 * pt, color=sky, va="center", ha="right")

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
