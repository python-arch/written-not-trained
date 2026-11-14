"""Build the blog's charts as inline SVG "instrument" cards from the run data.

Usage: python3 -I make_fx.py <draft/figures dir> <out dir>
Writes one HTML snippet per chart (<name>.html) into <out dir>.
"""
import json, math, os, sys
from html import escape

SRC, OUT = sys.argv[1], sys.argv[2]
os.makedirs(OUT, exist_ok=True)
D = json.load(open(os.path.join(SRC, "data.json")))
TRAIN = json.load(open(os.path.join(SRC, "maze_training.json")))
LOOPS = json.load(open(os.path.join(SRC, "maze_loops.json")))
COMP = json.load(open(os.path.join(SRC, "maze_compute.json")))

W = 480  # viewBox width; the card scales it to the column
CUR = {'id': 'x', 'h': 300}


def f(x):
    return f"{x:.1f}".rstrip("0").rstrip(".")


def pts(seq):
    return " ".join(f"{f(x)},{f(y)}" for x, y in seq)


def text(x, y, s, cls="t-lbl", anchor="start", extra=""):
    return f'<text x="{f(x)}" y="{f(y)}" class="{cls}" text-anchor="{anchor}"{extra}>{escape(s)}</text>'


def tip(s):
    return f' data-tip="{escape(s, quote=True)}"'


def glow_line(points, cls="w", delay=0.0):
    p = pts(points)
    return (f'<polyline class="halo draw" pathLength="1" points="{p}" style="--d:{delay}s" filter="url(#glow-{CUR["id"]})"/>'
            f'<polyline class="ln {cls} draw" pathLength="1" points="{p}" style="--d:{delay}s"/>')


def line(points, cls="t", delay=0.0, width=None):
    sw = f' stroke-width="{width}"' if width else ""
    return f'<polyline class="ln {cls} draw" pathLength="1" points="{pts(points)}" style="--d:{delay}s"{sw}/>'


def dot(x, y, cls="w", r=4.5, delay=0.0, t=None, hollow=False):
    k = f"dot {cls}{' hollow' if hollow else ''} pop"
    hit = f'<circle cx="{f(x)}" cy="{f(y)}" r="12" class="hit"{tip(t)}/>' if t else ""
    halo = (f'<circle cx="{f(x)}" cy="{f(y)}" r="{r + 3}" class="dhalo pop" style="--d:{delay}s" filter="url(#glow-{CUR["id"]})"/>'
            if cls == "w" and not hollow else "")
    return halo + f'<circle cx="{f(x)}" cy="{f(y)}" r="{r}" class="{k}" style="--d:{delay}s"/>' + hit


def grid_h(y, x0, x1):
    return f'<line class="grid" x1="{f(x0)}" x2="{f(x1)}" y1="{f(y)}" y2="{f(y)}"/>'


def defs(uid, h):
    return (f'<defs><filter id="glow-{uid}" filterUnits="userSpaceOnUse" x="-20" y="-20" width="{W + 40}" height="{h + 40}">'
            f'<feGaussianBlur stdDeviation="3.6"/></filter><linearGradient id="wash-{uid}" x1="0" x2="0" y1="0" y2="1">'
            f'<stop offset="0" stop-color="#3fc4ff" stop-opacity=".28"/>'
            f'<stop offset="1" stop-color="#3fc4ff" stop-opacity="0"/></linearGradient></defs>')


def card(name, h, body, label, caption, fig_id=None, extra_cls=""):
    idattr = f' id="{fig_id}"' if fig_id else ""
    svg = (f'<svg viewBox="0 0 {W} {h}" role="img" aria-label="{escape(label, quote=True)}">'
           f'{defs(name, h)}{body}</svg>')
    html = (f'<figure class="fx reveal{extra_cls}"{idattr}>\n  <div class="fx-card">{svg}</div>\n'
            f'  <figcaption>{caption}</figcaption>\n</figure>')
    open(os.path.join(OUT, f"{name}.html"), "w", encoding="utf-8").write(html)
    return html


def legend(items, x, y):
    out, cx = [], x
    for cls, lab in items:
        out.append(f'<line class="key {cls}" x1="{f(cx)}" x2="{f(cx + 16)}" y1="{f(y - 4)}" y2="{f(y - 4)}"/>')
        out.append(text(cx + 22, y, lab, "t-key2"))
        cx += 22 + 7.2 * len(lab) + 22
    return "".join(out)


# ------------------------------------------------------------------ 1. training time
def maze_training():
    h = 300
    CUR['id'] = 'maze_training'
    x0, x1, y0, y1 = 44, 462, 40, 250
    X = lambda hr: x0 + (x1 - x0) * hr / 8
    Y = lambda a: y1 - (y1 - y0) * a / 100
    b = []
    for a in (0, 50, 100):
        b.append(grid_h(Y(a), x0, x1))
        b.append(text(x0 - 8, Y(a) + 4, f"{a}%" if a == 100 else str(a), "t-tick", "end"))
    for hr in (0, 2, 4, 6, 8):
        b.append(text(X(hr), y1 + 22, f"{hr} h" if hr == 8 else str(hr), "t-tick", "middle"))
    # TRM as released (reference)
    b.append(f'<line class="ref" x1="{f(x0)}" x2="{f(x1)}" y1="{f(Y(85.3))}" y2="{f(Y(85.3))}"/>')
    b.append(text(X(1.12), Y(85.3) - 7, "TRM, authors' release, 85.3%", "t-mut", "start"))
    # gradient descent inside Claude's network (8 runs, all at 0)
    for i, (k, v) in enumerate(TRAIN["gd"].items()):
        seq = [(X(p[0]), Y(p[1])) for p in v["points"]]
        b.append(line(seq, "t gd", 0.2 + 0.05 * i))
    b.append(text(X(2.85), Y(0) - 10, "Trained from scratch, 0%", "t-lbl"))
    # standard transformer
    std = [(X(p[0]), Y(p[1])) for p in TRAIN["std"]]
    b.append(line(std, "t", 0.3))
    for p in TRAIN["std"][1:]:
        b.append(dot(X(p[0]), Y(p[1]), "t", 3, 1.2, f"Standard transformer, {p[0]:g} h, {p[1]}%"))
    b.append(text(x1, Y(41.5) + 22, "Standard transformer, 42%", "t-lbl", "end"))
    # TRM, our run
    trm = [(X(p[0]), Y(p[1])) for p in TRAIN["trm"]]
    b.append(line(trm, "t", 0.15))
    for p in TRAIN["trm"][1:]:
        b.append(dot(X(p[0]), Y(p[1]), "t", 3, 1.2, f"TRM, our run, {p[0]:.1f} h, {p[1]}%"))
    b.append(text(X(5.53) + 10, Y(66.1) + 4, "TRM, best 77.4%", "t-lbl", "start"))
    # Claude's network
    cx = X(TRAIN["claude_at_hours"])
    b.append(f'<path class="wash" d="M{f(cx)},{f(Y(100))} L{f(x1)},{f(Y(100))} L{f(x1)},{f(y1)} L{f(cx)},{f(y1)} Z" fill="url(#wash-maze_training)"/>')
    b.append(glow_line([(cx, Y(100)), (x1, Y(100))], "w", 0))
    b.append(dot(cx, Y(100), "w", 5, 0.1, "Written by Claude, 100% of test mazes, no training"))
    b.append(text(cx, Y(100) - 14, "Written by Claude, 100%", "t-hi"))
    label = ("Line chart of Maze-Hard test mazes solved against training hours. The network Claude wrote sits at 100% "
             "with no training. TRM, which we trained, peaks at 77.4%, and the authors' release scores 85.3%. A standard "
             "transformer peaks at 42%. Claude's network trained from scratch stays at 0% in all runs.")
    cap = ("Test mazes solved against training time. Claude's network needed no training, so it is placed at 1 hour. "
           "When the same network is trained from scratch, it stays at 0% in all six versions. TRM and the standard "
           "transformer, both of which we trained ourselves, are over 300 times bigger than Claude's network. The dotted "
           "line is TRM as released by its authors.")
    return card("maze_training", h, "".join(b), label, cap, fig_id="written-vs-trained-maze")


# ------------------------------------------------------------------ 2. loops
def maze_loops():
    h = 230
    CUR['id'] = 'maze_loops'
    x0, xb, x1, y0, y1 = 44, 330, 462, 30, 186
    X = lambda r: x0 + (xb - 18 - x0) * r / 420
    tail = {1800: xb + 28, 3600: xb + 78, 7200: x1 - 8}
    Y = lambda a: y1 - (y1 - y0) * a / 100
    b = []
    for a in (0, 50, 100):
        b.append(grid_h(Y(a), x0, x1))
        b.append(text(x0 - 8, Y(a) + 4, f"{a}%" if a == 100 else str(a), "t-tick", "end"))
    ser = [(r, a) for r, a in LOOPS["series"] if r <= 420]
    seq = [(X(r), Y(a)) for r, a in ser]
    area = "M" + " L".join(f"{f(x)},{f(y)}" for x, y in seq) + f" L{f(seq[-1][0])},{f(y1)} L{f(seq[0][0])},{f(y1)} Z"
    b.append(f'<path class="wash" d="{area}" fill="url(#wash-maze_loops)"/>')
    b.append(glow_line(seq, "w", 0))
    # axis break
    b.append(f'<path class="brk" d="M{f(xb - 6)},{f(y1 + 6)} l6,-12 M{f(xb)},{f(y1 + 6)} l6,-12"/>')
    b.append(f'<line class="ref" x1="{f(xb + 10)}" x2="{f(x1)}" y1="{f(Y(100))}" y2="{f(Y(100))}"/>')
    for i, (r, x) in enumerate(tail.items()):
        b.append(dot(x, Y(100), "w", 5, 1.3 + 0.15 * i, f"{r:,} loops, 100% of test mazes"))
        b.append(text(x, y1 + 22, f"{r:,}", "t-tick", "middle"))
    for r in (0, 200, 400):
        b.append(text(X(r), y1 + 22, str(r), "t-tick", "middle"))
    b.append(text(x1, y1 + 40, "loops", "t-mut", "end"))
    first = next(p for p in ser if p[1] > 0)
    b.append(text(X(first[0]) - 10, Y(72), "each maze locks in", "t-mut", "end"))
    b.append(text(X(first[0]) - 10, Y(72) + 19, "at twice its path length", "t-mut", "end"))
    label = ("Accuracy on the 1,000 test mazes against the number of loops. It is 0% until loop 218, rises steeply to 100% "
             "as each maze's path completes, and stays at 100% at 1,800, 3,600 and 7,200 loops.")
    cap = "Test mazes solved against the number of loops. A solved maze stays solved."
    return card("maze_loops", h, "".join(b), label, cap)


# ------------------------------------------------------------------ 3. written vs trained (slopes)
def written_vs_trained():
    h = 308
    CUR['id'] = 'written_vs_trained'
    panels = D["written_vs_trained"]["panels"]
    y0, y1 = 74, 240
    Y = lambda a: y1 - (y1 - y0) * a / 100
    pw, gap, xs = 124, 18, 50
    b = [legend([("w", "Written by Claude"), ("t", "Trained")], 50, 22)]
    names = {"graph": ("≤ 4", "8–16"), "parity": ("≤ 20", "21–128"), "chains": ("≤ 8", "12–32")}
    for a in (0, 50, 100):
        b.append(text(xs - 10, Y(a) + 4, f"{a}%" if a == 100 else str(a), "t-tick", "end"))
    for i, pn in enumerate(panels):
        px = xs + i * (pw + gap)
        xa, xb_ = px + 14, px + pw - 14
        b.append(f'<rect class="panel" x="{f(px)}" y="{f(y0 - 14)}" width="{pw}" height="{f(y1 - y0 + 28)}" rx="10"/>')
        b.append(text(px + pw / 2, y0 - 22, pn["title"], "t-key", "middle"))
        for a in (0, 50, 100):
            b.append(grid_h(Y(a), px + 4, px + pw - 4))
        if pn.get("chance"):
            b.append(f'<line class="ref" x1="{f(px + 4)}" x2="{f(px + pw - 4)}" y1="{f(Y(50))}" y2="{f(Y(50))}"/>')
        for j, t in enumerate(pn["trained"]):
            tv = t["test"]
            tv_mid = sum(tv) / 2 if isinstance(tv, list) else tv
            tv_txt = f"{tv[0]}–{tv[1]}%" if isinstance(tv, list) else f"{tv}%"
            d = 0.25 + 0.08 * j
            b.append(line([(xa, Y(t["checking"])), (xb_, Y(tv_mid))], "t thin", d))
            if isinstance(tv, list):
                b.append(f'<line class="ln t thin" x1="{f(xb_)}" x2="{f(xb_)}" y1="{f(Y(tv[0]))}" y2="{f(Y(tv[1]))}"/>')
            tipt = f"{t['name']}, {t['params']:,} weights, {t['checking']}% then {tv_txt}"
            b.append(dot(xa, Y(t["checking"]), "t", 3.2, d + 0.6, tipt))
            b.append(dot(xb_, Y(tv_mid), "t", 3.2, d + 0.9, tipt))
        wt = f"Written by Claude, {pn['written']['params_label']}, 100% then 100%"
        b.append(glow_line([(xa, Y(100)), (xb_, Y(100))], "w", 0))
        b.append(dot(xa, Y(100), "w", 4.5, 0.2, wt))
        b.append(dot(xb_, Y(100), "w", 4.5, 0.6, wt))
        lo, hi = names[pn["key"]]
        b.append(text(xa, y1 + 30, lo, "t-tick", "middle"))
        b.append(text(xb_, y1 + 30, hi, "t-tick", "middle"))
    label = ("Three slope charts comparing the checking range with longer test inputs. In graph search, the written network "
             "scores 100% on both, while three trained models score 0% on paths of 8 to 16 steps. In parity, the written "
             "network scores 100% on both, while five trained models fall from 100% to 45.5–53.8%, near chance. In reasoning "
             "chains, the written network scores 100% on both, while five trained models score 2.2–9.2% on 12 to 32 steps.")
    cap = ("Each line joins a network's accuracy on the sizes it was checked or trained on (left point) to its accuracy on "
           "much longer inputs (right point). The numbers under each panel are path length in steps, string length in bits, "
           "and chain length in steps. In the parity panel, the dotted line marks chance.")
    return card("written_vs_trained", h, "".join(b), label, cap)


# ------------------------------------------------------------------ 4. glance
def glance():
    tasks = D["glance"]["tasks"]
    CUR['id'] = 'glance'
    row, top = 31, 58
    h = top + row * len(tasks) + 22
    x0, x1 = 160, 414
    X = lambda a: x0 + (x1 - x0) * a / 100
    b = [legend([("t", "Best trained model"), ("l", "Best LLM")], 8, 22)]
    for a in (0, 50):
        b.append(text(X(a), h - 2, str(a), "t-tick", "middle"))
    for a in (0, 50):
        b.append(f'<line class="grid" x1="{f(X(a))}" x2="{f(X(a))}" y1="{top - 8}" y2="{f(h - 10)}"/>')
    b.append(f'<rect class="w-col halo-r pop" x="{f(X(100) - 3)}" y="{top - 10}" width="6" height="{f(h - top)}" rx="3" style="--d:1.4s" filter="url(#glow-{CUR["id"]})"/>')
    b.append(f'<rect class="w-col pop" x="{f(X(100) - 3)}" y="{top - 10}" width="6" height="{f(h - top)}" rx="3" style="--d:1.4s"/>')
    b.append(text(X(100), top - 18, "Written by Claude, 100%", "t-hi", "end"))
    for i, tk in enumerate(tasks):
        yc = top + row * i + row / 2
        b.append(text(x0 - 12, yc + 4.5, tk["task"], "t-row", "end"))
        bars = tk["bars"]
        th = 7 if len(bars) > 1 else 9
        for j, br in enumerate(bars):
            yb = yc - (len(bars) * th + (len(bars) - 1) * 3) / 2 + j * (th + 3)
            cls = "t" if br["kind"] == "trained" else "l"
            wv = max(X(br["value"]) - x0, 2.5)
            d = 0.15 + 0.06 * i
            b.append(f'<rect class="bar {cls} grow" x="{x0}" y="{f(yb)}" width="{f(wv)}" height="{th}" rx="3.5" style="--d:{d}s"'
                     f'{tip(f"{tk["task"]}, {br["label"]}, {br["text"]}")}/>')
            vx = x0 + wv + 7 if br["value"] < 88 else X(100) + 10
            b.append(text(vx, yb + th - 0.5, br["text"], "t-val pop", "start", f' style="--d:{d + 0.8}s"'))
    label = ("Bar chart, one row per task. Written by Claude, every network scores 100%. The strongest other results are LDT "
             "at 99.8% on Maze-Hard, where Claude Opus 5.5 and GPT-6 Astra score 26%, Sotaku v2 at 98.96% on Sudoku-Extreme, "
             "where GPT-6 Astra scores 68%, Gemini 4 Carbon at 96.2% F1 on GraphWalks, a trained model at 0% on graph search, "
             "53.8% on parity where chance is 50%, 3.5% on reasoning chains, SmolLM2-360M at 69.6% on Boolean Expressions, "
             "Pythia-410M at 33.2% on Dyck Languages, Qwen3-235B-A22B at 81.2% on Web of Lies, and 56.9% with program "
             "synthesis on Shuffled Objects.")
    cap = ("The written networks score 100% on every task here. Each bar is the strongest trained model or LLM we found for "
           "that task, and hovering over a bar names it. Tower of Hanoi and Checker Jumping are scored in moves rather than "
           "accuracy, so they are covered separately below.")
    return card("glance", h, "".join(b), label, cap)


# ------------------------------------------------------------------ 5. hanoi
def hanoi():
    h = 290
    CUR['id'] = 'hanoi'
    x0, x1, y0, y1 = 52, 452, 50, 240
    X = lambda n: x0 + (x1 - x0) * (n - 1) / 19
    Y = lambda m: y1 - (y1 - y0) * math.log10(m) / math.log10(2 ** 20)
    b = []
    b.append(f'<rect class="band" x="{f(X(6.5))}" y="{y0 - 18}" width="{f(X(8.5) - X(6.5))}" height="{f(y1 - y0 + 18)}" rx="6"/>')
    b.append(text(X(7.5), y0 - 24, "reasoning models collapse", "t-mut", "middle"))
    for m, lab in ((1, "1"), (1000, "1K"), (10 ** 6, "1M")):
        b.append(grid_h(Y(m), x0, x1))
        b.append(text(x0 - 8, Y(m) + 4, lab, "t-tick", "end"))
    for n in (1, 5, 10, 15, 20):
        b.append(text(X(n), y1 + 22, str(n), "t-tick", "middle"))
    b.append(text(x1, y1 + 40, "disks", "t-mut", "end"))
    curve = [(X(n), Y(2 ** n - 1)) for n in range(1, 21)]
    b.append(f'<polyline class="ln w faint draw" pathLength="1" points="{pts(curve)}" style="--d:0s"/>')
    for n in range(1, 21):
        mv = 2 ** n - 1
        if n <= 6:
            b.append(dot(X(n), Y(mv), "w", 3.8, 0.1 + 0.04 * n, f"{n} disks, {mv:,} moves, checked while building", hollow=True))
        else:
            b.append(dot(X(n), Y(mv), "w", 4.6, 0.3 + 0.06 * n, f"{n} disks, {mv:,} moves, 0 mistakes"))
    b.append(text(X(20) - 16, Y(2 ** 20) - 20, "20 disks, 0 mistakes", "t-lbl", "end"))
    b.append(text(X(20) - 16, Y(2 ** 20) + 6, "1,048,575 moves", "t-hero", "end"))
    label = ("Moves needed against number of disks, on a log scale. The network Claude wrote solves every size from 7 to 20 "
             "disks, up to 1,048,575 moves, with no mistakes. Sizes 1 to 6 were checked while building it. Reasoning models "
             "began to collapse at 7 to 8 disks in 2025 tests.")
    cap = ("Each glowing dot is one size the network solved without a single mistake. Hollow dots are the sizes Claude "
           "checked while building it. The amber band marks where the reasoning models in Shojaee et al. (2025) begin to "
           "collapse.")
    return card("hanoi", h, "".join(b), label, cap)


# ------------------------------------------------------------------ 6. accuracy vs size
def maze_size():
    h = 250
    CUR['id'] = 'maze_size'
    x0, x1, y0, y1 = 44, 452, 30, 200
    lx = lambda w: (math.log10(w) - 4) / 4
    X = lambda w: x0 + (x1 - x0) * lx(w)
    Y = lambda a: y1 - (y1 - y0) * (a - 80) / 20
    b = []
    for a in (80, 90, 100):
        b.append(grid_h(Y(a), x0, x1))
        b.append(text(x0 - 8, Y(a) + 4, f"{a}%" if a == 100 else str(a), "t-tick", "end"))
    for w, lab in ((1e4, "10K"), (1e5, "100K"), (1e6, "1M"), (1e7, "10M"), (1e8, "100M")):
        b.append(text(X(w), y1 + 22, lab, "t-tick", "middle"))
    b.append(text(x1, y1 + 40, "weights", "t-mut", "end"))
    pts_ = [(1.8e6, 99.8, "LDT", "start", 10, 8), (1.8e6, 99.0, "", "start", 10, 14),
            (7e6, 91.9, "TRM", "start", 10, 4), (2.7e7, 84.6, "HRM", "end", -10, 4)]
    for i, (w, a, lab, anc, dx, dy) in enumerate(pts_):
        b.append(dot(X(w), Y(a), "t", 5, 0.3 + 0.12 * i, f"{lab or 'LDT, one path per maze'}, {w/1e6:g}M weights, {a}%"))
        if lab:
            b.append(text(X(w) + dx, Y(a) + dy, lab, "t-lbl", anc))
    b.append(dot(X(19936), Y(100), "w", 6, 0.1, "Written by Claude, 19,936 weights, 100%"))
    b.append(text(X(19936) + 12, Y(100) + 18, "Written by Claude", "t-hi", "start"))
    label = ("Accuracy against number of weights, on a log scale. The written network has 19,936 weights and scores 100%. "
             "LDT has 1.8M weights and scores 99.8%, or 99.0% trained on one path per maze. TRM has 7M and scores 91.9%, "
             "while HRM has 27M and scores 84.6%.")
    cap = ("The vertical axis starts at 80%. LDT appears twice, trained on 512 shortest paths per maze and trained on one. "
           "LLM sizes aren't public, so they are left out.")
    return card("maze_size", h, "".join(b), label, cap)


# ------------------------------------------------------------------ 7. compute
def maze_compute():
    h = 250
    CUR['id'] = 'maze_compute'
    x0, x1, y0, y1 = 44, 456, 30, 196
    X = lambda ops: x0 + (x1 - x0) * (math.log10(ops) - 10) / 6.2
    Y = lambda a: y1 - (y1 - y0) * a / 100
    b = []
    for a in (0, 50, 100):
        b.append(grid_h(Y(a), x0, x1))
        b.append(text(x0 - 8, Y(a) + 4, f"{a}%" if a == 100 else str(a), "t-tick", "end"))
    for e in (10, 11, 12, 13, 14, 15, 16):
        b.append(text(X(10 ** e), y1 + 22, f"10^{e}".replace("10^", "10"), "t-tick", "middle",
                      "") if False else f'<text x="{f(X(10 ** e))}" y="{f(y1 + 22)}" class="t-tick" text-anchor="middle">10<tspan dy="-6" class="sup">{e}</tspan></text>')
    b.append(text(x1, y1 + 42, "operations per maze", "t-mut", "end"))
    ours = [(o, a) for o, a in COMP["ours"] if o >= 1.5e10]
    seq = [(X(o), Y(a)) for o, a in ours]
    area = "M" + " L".join(f"{f(x)},{f(y)}" for x, y in seq) + f" L{f(seq[-1][0])},{f(y1)} L{f(seq[0][0])},{f(y1)} Z"
    b.append(f'<path class="wash" d="{area}" fill="url(#wash-maze_compute)"/>')
    b.append(glow_line(seq, "w", 0))
    ro, ra = COMP["ours_run"]
    b.append(dot(X(ro), Y(ra), "w", 5.5, 1.0, "Written by Claude, 1,800 loops, 100%"))
    b.append(text(X(ro), Y(ra) - 14, "Written by Claude", "t-hi", "middle"))
    for key, lab in (("hrm", "HRM"), ("trm", "TRM")):
        o, a = COMP[key]
        b.append(dot(X(o), Y(a), "t", 5, 1.2, f"{lab}, {a}%"))
        b.append(text(X(o) + (10 if key == "trm" else -10), Y(a) + (4 if key == "trm" else 4), lab, "t-lbl",
                      "start" if key == "trm" else "end"))
    L = COMP["ldt"]
    for acc in L["acc"]:
        b.append(f'<line class="ln t" x1="{f(X(L["lo"]))}" x2="{f(X(L["hi"]))}" y1="{f(Y(acc))}" y2="{f(Y(acc))}"/>')
    b.append(dot(X(L["lo"]), Y(99.8), "t", 5, 1.3, "LDT, one search round up to 1,000 rounds, 99.0% to 99.8%"))
    b.append(text(X(L["hi"]), Y(99.8) + 20, "LDT, 1 to 1,000 rounds", "t-lbl", "end"))
    label = ("Accuracy on the 1,000 test mazes against the arithmetic it takes to solve one maze, on a log scale. The written "
             "network reaches 100% far to the left of every trained model. HRM scores 84.6% and TRM 91.9%, both with far more "
             "compute, while LDT scores 99.0 to 99.8% with more still.")
    cap = "The cyan curve is the written network after each number of loops, with a dot at the 1,800 loops we run."
    return card("maze_compute", h, "".join(b), label, cap)


for fn in (maze_training, maze_loops, written_vs_trained, glance, hanoi, maze_size, maze_compute):
    fn()
print("wrote", sorted(os.listdir(OUT)))
