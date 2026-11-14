#!/usr/bin/env python3
"""Render the Maze-Hard hero animation from the real network's per-loop state (build tool, not public).

    python3 figures/make_maze_hero.py [--square-out PATH.mp4] [--only wide|tall|square] [--frames FOLDER]

Input: figures/maze_trace.json (from make_maze_trace.py): for every cell of one real test maze, the loop at which the
network's distance from the goal reached it and the loop at which it joined the path from the start, both read from the
network's own state after each loop. Every colour in the grid at loop t is that state. Only the timing is staged:
the loops are played at a varying speed, and the stretch where nothing changes is played fast.

Outputs: figures/maze_hero(.mp4|.webm|_poster.png) (wide), figures/maze_hero_tall(...) (stacked, for phones), and
the square version (--square-out). Needs Pillow, numpy and ffmpeg.
"""
from __future__ import annotations

import argparse
import json
import math
import random
import subprocess
import sys
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

sys.path.insert(0, str(Path(__file__).resolve().parent))
from make_hero import (BG, MP4, SS, WEBM, clamp, font, mix, smooth, text_fixed)  # noqa: E402

# The blog palette (as in make_figures.py): the transformer Claude wrote and its path in orange, the wave in the blue ramp.
ACCENT = (235, 104, 52)                              # orange #eb6834
FG = (20, 33, 61)                                    # ink #14213d
MUTED = (91, 100, 117)                               # #5b6475
LINE = (214, 212, 204)
TAG = (243, 242, 236)
GRIDLINE = (237, 236, 231)

HERE = Path(__file__).resolve().parent
FPS = 60
WALL = (52, 56, 63)
WAVE_OLD = (205, 226, 251)                           # blue ramp #cde2fb: cells the wave reached a while ago
WAVE_NEW = (109, 167, 236)                           # blue ramp #6da7ec: cells it just reached
FORMATS = {"wide": (1280, 720), "tall": (720, 960)}

# timeline (seconds). The opening is slow so each stage can be read; then the loops speed up.
T_LOOP = 0.45
W_STAGE = 2.4                                        # the timing unit for the opening and the banner stages
T_PASS1 = 2 * W_STAGE                                # the first pass starts here (during banner stage 1)
T_WAVE0 = 5 * W_STAGE                                # the wave starts after the first pass (end of banner stage 2)


class Timeline:
    """Loop shown at each time, from the real trace: the first pass is slow (one stage per part of the block), then
    the wave to loop D (the start), the path to loop `done`, and the remaining loops to R, played fast."""

    def __init__(self, tr):
        self.D = tr["start_reached_at"]
        self.done = tr["readout_last_change"]
        self.R = tr["loops"]
        self.t1 = T_PASS1                            # first loop starts
        self.t2 = T_WAVE0                            # first loop done, wave runs
        self.t3 = self.t2 + 4.0                      # wave reaches the start (loop D)
        self.t3b = self.t3 + 0.3                     # path starts
        self.t4 = self.t3b + 4.5                     # path done (loop `done`)
        self.t5 = self.t4 + 0.4
        self.t6 = self.t5 + 1.8                      # loop R
        self.length = self.t6 + 2.7 + T_LOOP

    def first_pass(self, t):
        """Progress of the first pass in stations: 0 in, 0.25 attention, 0.5 MLP, 0.62 out of the block, 1 back in."""
        keys = [(0.0, 0.0), (0.3, 0.25), (2.2, 0.27), (2.6, 0.5), (4.6, 0.52), (4.95, 0.62), (7.2, 1.0)]
        x = t - self.t1
        for (t0, p0), (t1, p1) in zip(keys, keys[1:]):
            if x <= t1:
                return p0 + (p1 - p0) * smooth((x - t0) / (t1 - t0)) if x >= t0 else 0.0
        return 1.0

    def feed(self, t):
        """Before the first loop, the input rises to embed (0 to 0.5), waits there, then enters the block (to 1)."""
        if t < 0.6:
            return 0.0
        if t < 1.9:
            return 0.5 * smooth((t - 0.6) / 1.3)
        if t < 4.2:
            return 0.5
        return 0.5 + 0.5 * smooth((t - 4.2) / (self.t1 - 4.2))

    def loop(self, t):
        if t < self.t1:
            return 0.0
        if t < self.t2:
            return self.first_pass(t)
        if t < self.t3:                              # the wave: eases in, so the first rings are visible
            x = (t - self.t2) / (self.t3 - self.t2)
            return 1 + (self.D - 1) * (0.55 * x * x + 0.45 * smooth(x))
        if t < self.t3b:
            return float(self.D)
        if t < self.t4:
            return self.D + (self.done - self.D) * smooth((t - self.t3b) / (self.t4 - self.t3b))
        if t < self.t5:
            return float(self.done)
        if t < self.t6:                              # nothing changes: play the rest fast
            x = (t - self.t5) / (self.t6 - self.t5)
            return self.done + (self.R - self.done) * (x ** 2.2)
        return float(self.R)

    def stages(self):
        """(text, part it describes, start, end). One at a time in the banner: the five most important steps."""
        w = W_STAGE
        t2, t3, t4 = self.t2, self.t3, self.t4
        return [
            ("Attention lets each cell read its four neighbours", "attn", 0.0, 2.5 * w),
            ("The same block runs again, loop after loop", "loop", 2.5 * w, 5 * w),
            ("A wave of distances spreads out from the goal", "maze_goal", t2, t3 + w),
            ("The path grows from the start back to the goal", "maze", t3 + w, t4 + w),
            (f"Extra loops leave the answer unchanged, up to loop {self.R:,}", "count", t4 + w, self.length - T_LOOP - 0.05),
        ]


GLOW = (250, 211, 194)                              # the block lit by a pass (light orange)
ACCENT_SOFT = (253, 241, 235)
CAP = 1.0                                           # passes per second shown at most: the pulse never strobes
GROUPS = [6, 24, 8, 8, 40]                          # MLP neurons by job: wave, pointer, pointer clear, chain, clean-up
# One colour per banner stage, from the palette's text shades: blue #1c5cab, aqua #0f7d56, yellow #a87200, orange #c4471a.
# Orange (the path) is kept for the stage where the path grows, blue for the wave; neighbouring stages differ, and the
# loop stage avoids yellow, which would not stand out on the orange loop arrow.
_B, _A, _Y, _O = (28, 92, 171), (15, 125, 86), (168, 114, 0), (196, 71, 26)
STAGE_COLORS = [_Y, _A, _B, _O, _A]
BANNER = {"wide": dict(x=40, y=26, r=21, size=26, lead=35, maxw=520),
          "tall": dict(x=24, y=18, r=20, size=24, lead=32, maxw=620)}

# Every position is in output pixels. "flow" is the direction data moves through the diagram.
LAYOUT = {
    "wide": dict(
        flow="down", cx=270, fs=dict(lab=19, small=16, box=19, count=22, of=16),
        maze=(640, 52, 20.5),
        cells=(135, 190, 18, 5), cells_label=(417, 205),
        embed=(190, 262, 350, 300), embed_note=(362, 287),
        block=(70, 400, 470, 570), tag=(88, 426),
        attn_label=430, heads=(440, 40, 14), mlp_label=518, dots=(534, 548, 8),
        readout=(195, 600, 345, 638),
        io=(382, 585), loop_x=494, count=(506, 495),
        out=[(345, 619), (626, 619)],
        # the thought bubble (stages 3 and 4): zone it covers, cloud (cx, cy, R, bump), close-up (x, y, cell),
        # trail circles (x, y, r), note (x, y, size, lines, tilt), swirl (from, to, loops)
        zone=(0, 120, 636, 376), bubble=(245, 258, 100, 10), zoom=(153, 166, 23),
        trail=[(114, 392, 4.5), (129, 378, 7.5)],
        note=(512, 342, 38, ["loop imagination"], -4), swirl=((470, 310), (364, 240), 1),
        digit=13, arrow=(2.3, 6.8, 2.9), corner=8,
    ),
    "tall": dict(
        flow="down", cx=340, fs=dict(lab=20, small=17, box=20, count=24, of=17),
        maze=(160, 546, 400 / 30),
        cells=(216, 170, 16, 5), cells_label=(476, 184),
        embed=(265, 222, 415, 254), embed_note=(428, 244),
        block=(140, 336, 540, 470), tag=(156, 360),
        attn_label=360, heads=(368, 32, 12), mlp_label=424, dots=(440, 452, 8),
        readout=(275, 494, 405, 522),
        io=(320, 482), loop_x=562, count=(580, 420),
        out=[(340, 522), (340, 542)],
        zone=(0, 76, 720, 316), bubble=(236, 204, 104, 10), zoom=(140, 108, 24),
        trail=[(146, 330, 4.5), (157, 318, 7)],
        note=(524, 284, 36, ["loop imagination"], -4), swirl=((474, 254), (362, 192), 1),
        digit=13.5, arrow=(2.4, 7, 3), corner=8.5,
    ),
}

# ---- the thought bubble: what the block works out near the goal, loop by loop (from maze_inside_trace.json)
ROWS, COLS = range(10, 18), range(3, 11)            # the close-up: 8 x 8 cells around the goal
DIRS = {"D": (1, 0), "L": (0, -1), "R": (0, 1), "U": (-1, 0)}
MARK = (237, 161, 0)                                # yellow #eda100: the close-up frame and the focus box on the maze
NOTE_INK = (196, 71, 26)                            # dark orange #c4471a: the handwritten note and its arrow
BUBBLE_FILL = (253, 241, 235)                       # the block's orange tint
BUBBLE_LINE = (241, 160, 128)                       # a light orange pen
CLOSE_GRID = (228, 227, 221)
CAVEAT = HERE / "fonts" / "Caveat-Variable.ttf"     # SIL Open Font License, fonts/Caveat-OFL.txt


def swirl_points(p0, p1, loops, N=260):
    """A loop-de-loop from p0 to p1: a prolate cycloid laid along the line between them."""
    (x0, y0), (x1, y1) = p0, p1
    L = math.hypot(x1 - x0, y1 - y0)
    ux, uy = (x1 - x0) / L, (y1 - y0) / L
    nx, ny = -uy, ux
    a = L / (2 * math.pi * loops)
    b = 1.75 * a
    t0, t1 = math.pi, math.pi + 2 * math.pi * loops
    pts = []
    for i in range(N + 1):
        t = t0 + (t1 - t0) * i / N
        along = a * (t - t0) - b * (math.sin(t) - math.sin(t0))
        side = -b * (math.cos(t) - math.cos(t0))
        pts.append((x0 + ux * along + nx * side, y0 + uy * along + ny * side))
    return pts


def jitter(pts, amp, seed, closed=True):
    """Move each point along its normal by a smooth random amount (fixed seed: the same wobble every frame)."""
    rnd = random.Random(seed)
    n = len(pts)
    K = max(3, n // 22)
    ctrl = [rnd.uniform(-amp, amp) for _ in range(n // K + 3)]
    out = []
    for i, (x, y) in enumerate(pts):
        j = i / K
        j0, f = int(j), j - int(j)
        f = f * f * (3 - 2 * f)
        off = ctrl[j0] * (1 - f) + ctrl[j0 + 1] * f
        xa, ya = pts[i - 1] if (i > 0 or closed) else pts[i]
        xb, yb = pts[(i + 1) % n] if (closed or i + 1 < n) else pts[i]
        dx, dy = xb - xa, yb - ya
        L = math.hypot(dx, dy) or 1.0
        out.append((x - dy / L * off, y + dx / L * off))
    return out


def sketch(dr, pts, col, width, amp, seed, closed=True):
    """Two pen strokes, each with its own wobble; closed shapes overshoot their start a little."""
    for pass_ in range(2):
        q = jitter(pts, amp, seed * 7 + pass_, closed)
        if closed:
            q = q + q[: max(2, len(q) // 30)]
        dr.line(q, fill=col, width=width, joint="curve")


def cloud_points(cx, cy, R, A, N=480):
    """A scalloped cloud around a squircle, so it hugs the square close-up."""
    pts = []
    for i in range(N):
        th = 2 * math.pi * i / N
        c, s_ = math.cos(th), math.sin(th)
        r = R * (abs(c) ** 8 + abs(s_) ** 8) ** (-1 / 8) + A * abs(math.sin(6 * th))
        pts.append((cx + r * c, cy + r * s_))
    return pts


def circle_points(cx, cy, r, N=72):
    return [(cx + r * math.cos(2 * math.pi * i / N), cy + r * math.sin(2 * math.pi * i / N)) for i in range(N)]


def wrap(dr, text, f, maxw):
    lines, cur = [], ""
    for w in text.split():
        t = (cur + " " + w).strip()
        if cur and dr.textlength(t, font=f) > maxw:
            lines.append(cur); cur = w
        else:
            cur = t
    return lines + [cur]


def fixed_width(dr, s, f):
    cell = dr.textlength("0", font=f)
    return sum(cell if c.isdigit() else dr.textlength(c, font=f) for c in s)


def bump(x, c, w):
    """1 at c, falling smoothly to 0 at c +- w."""
    return max(0.0, 1 - abs(x - c) / w) ** 2


class Polyline:
    def __init__(self, pts):
        self.pts = pts
        self.seg = [math.dist(a, b) for a, b in zip(pts, pts[1:])]
        self.length = sum(self.seg)

    def at(self, u):
        """The point at fraction u (0..1) of the way along."""
        d = (u % 1.0) * self.length
        last = len(self.seg) - 1
        for j, ((a, b), L) in enumerate(zip(zip(self.pts, self.pts[1:]), self.seg)):
            if d <= L or j == last:
                x = min(1.0, d / L) if L else 0
                return a[0] + (b[0] - a[0]) * x, a[1] + (b[1] - a[1]) * x
            d -= L
        return self.pts[-1]


class Scene:
    """Left (top in the tall format): the architecture, one block looped, with a pulse going round the loop and a
    loop counter from the real timeline. Right (below): the whole maze, its real state at that loop."""

    def __init__(self, kind, tr, tl):
        self.kind, self.tr, self.tl = kind, tr, tl
        W, H = FORMATS[kind]
        self.W, self.H = W * SS, H * SS
        self.g = g = LAYOUT[kind]
        k = SS
        self.side = tr["side"]
        self.gx, self.gy, self.cell = (v * k for v in g["maze"])
        self.q = tr["question"]
        self.reach = tr["reach_loop"]
        self.on = tr["on_loop"]
        self.S, self.G = tr["start"], tr["goal"]
        fs = g["fs"]
        self.f = dict(lab=font("Regular", fs["lab"] * k), small=font("Regular", fs["small"] * k),
                      box=font("SemiBold", fs["box"] * k), count=font("SemiBold", fs["count"] * k),
                      of=font("Regular", fs["of"] * k), letter=font("Bold", 12 * k),
                      digit=font("SemiBold", g["digit"] * k), corner=font("Bold", g["corner"] * k))
        # the thought bubble: the real per-loop state of the cells near the goal
        self.ins = json.loads((HERE / "maze_inside_trace.json").read_text())
        assert self.ins["question"] == tr["question"] and self.ins["reach_loop"] == tr["reach_loop"]
        self.cloud = self.make_cloud()
        self.note = self.make_note()
        sw = g["swirl"]
        base = [(x * k, y * k) for x, y in swirl_points(sw[0], sw[1], sw[2])]
        self.swirl = [jitter(base, 1.1 * k, seed=70 + j, closed=False) for j in range(2)]
        # the loop: the block's input, through the block, out, and back round to the input
        cx = g["cx"]
        bx0, by0, bx1, by1 = g["block"]
        i_y, o_y = g["io"]                                 # where the loop meets the input, and leaves the output
        lx = g["loop_x"]
        if g["flow"] == "up":   # in at the bottom, out at the top
            pts = [(cx, i_y), (cx, by1), (cx, by0), (cx, o_y), (lx, o_y), (lx, i_y), (cx, i_y)]
        else:                   # in at the top, out at the bottom
            pts = [(cx, i_y), (cx, by0), (cx, by1), (cx, o_y), (lx, o_y), (lx, i_y), (cx, i_y)]
        self.cycle = Polyline([(x * k, y * k) for x, y in pts])
        d = self.cycle.seg
        self.u_in = d[0] / self.cycle.length                       # the pulse enters the block
        self.u_out = (d[0] + d[1] + d[2]) / self.cycle.length      # and leaves it
        self.loop_pts = [(x * k, y * k) for x, y in pts[3:]]       # the loop arrow itself
        # pulse phase after the first pass: real loops, but at most CAP passes a second
        n = int(tl.length * 240) + 2
        self.dt = 1 / 240
        phase, acc, prev = [], 0.0, tl.loop(tl.t2)
        for i in range(n):
            t = i * self.dt
            if t > tl.t2:
                L = tl.loop(t)
                acc += min(max(L - prev, 0.0), CAP * self.dt)
                prev = L
            phase.append(acc)
        self.phase = phase
        # the first pass: stations (0 in, 0.25 attention, 0.5 MLP, 0.62 out, 1 back in) to places on the loop
        attn_c, mlp_c = (0.27, 0.75) if g["flow"] == "up" else (0.25, 0.78)
        span = self.u_out - self.u_in
        self.stations = [(0.0, 0.0), (0.25, self.u_in + attn_c * span), (0.5, self.u_in + mlp_c * span),
                         (0.62, self.u_out), (1.0, 1.0)]
        self.attn_c, self.mlp_c = attn_c, mlp_c

    def pulse_u(self, t):
        tl = self.tl
        if tl.t1 <= t < tl.t2:
            p = tl.first_pass(t)
            for (p0, u0), (p1, u1) in zip(self.stations, self.stations[1:]):
                if p <= p1:
                    return u0 + (u1 - u0) * ((p - p0) / (p1 - p0) if p1 > p0 else 0)
            return 0.0
        return self.phase[min(len(self.phase) - 1, int(round(t / self.dt)))] % 1.0

    # ------------------------------------------------------------------ stages
    def stage_state(self, t):
        """The stage up at time t: (index, text, part, colour, alpha, scale), or None."""
        for i, (text, part, a, b) in enumerate(self.tl.stages()):
            if a <= t < b:
                fin = smooth(clamp((t - a) / 0.25))
                fout = 1 - smooth(clamp((t - (b - 0.15)) / 0.15))
                return i, text, part, STAGE_COLORS[i % len(STAGE_COLORS)], fin * fout, 0.92 + 0.08 * fin
        return None

    def draw_banner(self, img, st):
        """The stage text, large and bold in its colour, with a numbered badge, scaled and faded in."""
        if st is None or st[4] < 0.01:
            return img
        i, text, part, col, a, sc = st
        k, bn = SS, BANNER[self.kind]
        f = font("Bold", bn["size"] * k)
        fb = font("Bold", 20 * k)
        tmp = ImageDraw.Draw(img)
        lines = wrap(tmp, text, f, bn["maxw"] * k)
        if len(lines) == 2:                              # balance the two lines, no lone words
            words = text.split()
            best = min(range(1, len(words)), key=lambda c: max(tmp.textlength(" ".join(words[:c]), font=f),
                                                               tmp.textlength(" ".join(words[c:]), font=f)))
            lines = [" ".join(words[:best]), " ".join(words[best:])]
        r = bn["r"] * k
        w = int(2 * r + 14 * k + max(tmp.textlength(l, font=f) for l in lines) + 8 * k)
        h = int(max(2 * r, len(lines) * bn["lead"] * k) + 10 * k)
        layer = Image.new("RGBA", (w, h), (0, 0, 0, 0))
        ld = ImageDraw.Draw(layer)
        ld.ellipse((0, 0, 2 * r, 2 * r), fill=col + (255,))
        ld.text((r, r), str(i + 1), font=fb, fill=(255, 255, 255, 255), anchor="mm")
        for j, line in enumerate(lines):
            ld.text((2 * r + 14 * k, r + j * bn["lead"] * k), line, font=f, fill=col + (255,), anchor="lm")
        if sc < 0.999:
            layer = layer.resize((max(1, int(w * sc)), max(1, int(h * sc))), Image.LANCZOS)
        if a < 0.999:
            alpha = layer.getchannel("A").point(lambda v: int(v * a))
            layer.putalpha(alpha)
        base = img.convert("RGBA")
        x0, y0 = bn["x"] * k, bn["y"] * k
        base.alpha_composite(layer, (int(x0 + (1 - sc) * r), int(y0 + (1 - sc) * r)))
        return base.convert("RGB")

    # ------------------------------------------------------------------ the whole maze
    def cell_box(self, i):
        r, c = divmod(i, self.side)
        x0, y0 = self.gx + c * self.cell, self.gy + r * self.cell
        return x0, y0, x0 + self.cell, y0 + self.cell

    def draw_grid(self, dr, L, hl=None):
        k = SS
        n = self.side
        gx1, gy1 = self.gx + n * self.cell, self.gy + n * self.cell
        dr.rectangle((self.gx - 1.5 * k, self.gy - 1.5 * k, gx1 + 1.5 * k, gy1 + 1.5 * k), outline=FG, width=max(2, round(1.5 * k)))
        for i, ch in enumerate(self.q):
            box = self.cell_box(i)
            if ch == "#":
                dr.rectangle(box, fill=WALL)
                continue
            col = BG
            rl = self.reach[i]
            if rl >= 0 and L > rl - 1:                       # the wave reached this cell
                grow = clamp(L - (rl - 1))
                col = mix(BG, mix(WAVE_NEW, WAVE_OLD, clamp((L - rl) / 10)), grow)
            ol = self.on[i]
            if ol >= 0 and L > ol - 1 and i not in (self.S, self.G):
                col = mix(col, ACCENT, clamp(L - (ol - 1)))    # joins the path from the start
            if col != BG:
                dr.rectangle(box, fill=col)
        for j in range(n + 1):
            x = self.gx + j * self.cell
            y = self.gy + j * self.cell
            dr.line([(x, self.gy), (x, gy1)], fill=GRIDLINE, width=1)
            dr.line([(self.gx, y), (gx1, y)], fill=GRIDLINE, width=1)
        for idx, letter in ((self.S, "S"), (self.G, "G")):
            x0, y0, x1, y1 = self.cell_box(idx)
            dr.rectangle((x0, y0, x1, y1), fill=FG)
            dr.text(((x0 + x1) / 2, (y0 + y1) / 2), letter, font=self.f["letter"], fill=BG, anchor="mm")
        if hl and hl[0].startswith("maze"):
            part, col, a = hl
            dr.rectangle((self.gx - 4 * k, self.gy - 4 * k, gx1 + 4 * k, gy1 + 4 * k), outline=mix(BG, col, a),
                         width=round(3.5 * k))
            for idx, name in ((self.G, "maze_goal"), (self.S, "maze_start")):
                if part == name:
                    x0, y0, x1, y1 = self.cell_box(idx)
                    pad = self.cell * 0.7
                    dr.ellipse((x0 - pad, y0 - pad, x1 + pad, y1 + pad), outline=mix(BG, col, a), width=round(3 * k))

    # ------------------------------------------------------------------ the architecture
    def arrow(self, dr, p0, p1, col, w=2.0, head=9):
        k = SS
        dr.line([p0, p1], fill=col, width=round(w * k))
        ang = math.atan2(p1[1] - p0[1], p1[0] - p0[0])
        h = head * k
        a1, a2 = ang + math.radians(152), ang - math.radians(152)
        dr.polygon([p1, (p1[0] + h * math.cos(a1), p1[1] + h * math.sin(a1)),
                    (p1[0] + h * math.cos(a2), p1[1] + h * math.sin(a2))], fill=col)

    def box(self, dr, rect, fill, outline, label, ink, w=1.5):
        k = SS
        x0, y0, x1, y1 = (v * k for v in rect)
        dr.rounded_rectangle((x0, y0, x1, y1), radius=9 * k, fill=fill, outline=outline, width=round(w * k))
        if label:
            dr.text(((x0 + x1) / 2, (y0 + y1) / 2), label, font=self.f["box"], fill=ink, anchor="mm")

    def draw_arch(self, dr, t, L, hl=None):
        k, g, f, tl = SS, self.g, self.f, self.tl
        hp, hc, ha = hl if hl else (None, ACCENT, 0.0)
        def on(part):
            return ha if hp == part else 0.0
        def soft(col, a):
            return mix(BG, col, 0.16 * a)
        cx = g["cx"]
        up = g["flow"] == "up"
        loop_on = t >= tl.t1 - 0.05
        done = clamp(L - (tl.done - 0.5)) if L > tl.done - 0.5 else 0.0
        quiet = 1 - 0.45 * done                       # after the path is done, the loop runs on quietly
        u = self.pulse_u(t)
        moving = loop_on and (L < tl.R - 1e-6)
        inside = 1.0 if (moving and self.u_in <= u <= self.u_out) else 0.0
        frac_in = (u - self.u_in) / (self.u_out - self.u_in) if inside else -1.0

        # input cells: the first row of the real maze
        x0, y0, sz, gap = g["cells"]
        if on("input"):
            pad = 7
            dr.rounded_rectangle(((x0 - pad) * k, (y0 - pad) * k, (x0 + 12 * (sz + gap) - gap + pad) * k, (y0 + sz + pad) * k),
                                 radius=8 * k, fill=soft(hc, on("input")), outline=mix(BG, hc, on("input")), width=round(2.5 * k))
        for j in range(12):
            bx = (x0 + j * (sz + gap)) * k
            box = (bx, y0 * k, bx + sz * k, (y0 + sz) * k)
            if self.q[j] == "#":
                dr.rectangle(box, fill=WALL)
            else:
                dr.rectangle(box, fill=BG, outline=LINE, width=round(1.2 * k))
        lx, ly = g["cells_label"]
        dr.text((lx * k, ly * k), "maze, 900 cells", font=f["small"], fill=MUTED,
                anchor="ms" if up else "ls")
        # input -> embed -> block
        e0, e1 = g["embed"][1], g["embed"][3]
        if up:
            self.arrow(dr, (cx * k, (y0 - 5) * k), (cx * k, (e1 + 1) * k), LINE)
            self.arrow(dr, (cx * k, e0 * k), (cx * k, (g["block"][3] + 1) * k), LINE)
        else:
            self.arrow(dr, (cx * k, (y0 + sz + 5) * k), (cx * k, (e0 - 1) * k), LINE)
            self.arrow(dr, (cx * k, e1 * k), (cx * k, (g["block"][1] - 1) * k), LINE)
        self.box(dr, g["embed"], mix(TAG, mix(BG, hc, 0.3), on("embed")), mix(LINE, hc, on("embed")), "embed", FG,
                 w=1.5 + 1.5 * on("embed"))
        nx, ny = g["embed_note"]
        dr.text((nx * k, ny * k), "35 numbers per cell", font=f["small"], fill=MUTED, anchor="ls")

        # feed: before the first loop, a pulse rises from the maze to embed, waits, then enters the block
        if t < tl.t1:
            fd = tl.feed(t)
            ya = (y0 - 5) if up else (y0 + sz + 5)
            ym = g["embed"][1] if up else g["embed"][3]   # embed's exit edge
            yb = g["io"][0]
            y = ya + (ym - ya) * (fd / 0.5) if fd <= 0.5 else ym + (yb - ym) * ((fd - 0.5) / 0.5)
            if not (g["embed"][1] + 2 < y < g["embed"][3] - 2):     # hidden while it crosses the embed box
                self.dot(dr, (cx * k, y * k), 1.0)

        # the loop arrow, drawn under the pulse
        col = mix(mix(BG, ACCENT, 0.45 + 0.25 * quiet), hc, on("loop"))
        pts = self.loop_pts
        dr.line(pts[:-1] + [pts[-1]], fill=col, width=round((2.4 + 1.6 * on("loop")) * k), joint="curve")
        tip = pts[-1]
        prev = pts[-2]
        back = (tip[0] + (prev[0] - tip[0]) * 0.001, tip[1] + (prev[1] - tip[1]) * 0.001)
        self.arrow(dr, (tip[0] + (prev[0] - tip[0]) * 0.2, tip[1] + (prev[1] - tip[1]) * 0.2), tip, col, w=2.4, head=10)

        # the block: lit while a pass runs through it
        glow = inside * (0.55 + 0.45 * quiet)
        bx0, by0, bx1, by1 = g["block"]
        dr.rounded_rectangle((bx0 * k, by0 * k, bx1 * k, by1 * k), radius=16 * k,
                             fill=mix(ACCENT_SOFT, GLOW, glow), outline=ACCENT, width=round((2.2 + 1.6 * glow) * k))
        tx, ty = g["tag"]
        dr.text((tx * k, ty * k), "one block", font=f["small"], fill=MUTED, anchor="ls")
        # attention: four heads, one per neighbour
        dr.text((cx * k, g["attn_label"] * k), "attention, 4 heads", font=f["lab"], fill=FG, anchor="ms")
        hy, hs, hg = g["heads"]
        hx0 = cx - (4 * hs + 3 * hg) / 2
        attn_c, mlp_c = self.attn_c, self.mlp_c       # where attention and MLP sit along the pass (0 in, 1 out)
        a_glow = bump(frac_in, attn_c, 0.3) * (0.6 + 0.4 * quiet) if inside else 0.0
        m_glow = bump(frac_in, mlp_c, 0.3) * (0.6 + 0.4 * quiet) if inside else 0.0
        for j, (dx, dy) in enumerate(((0, -1), (-1, 0), (1, 0), (0, 1))):     # up, left, right, down
            x = hx0 + j * (hs + hg)
            ah = on("attn")
            dr.rounded_rectangle((x * k, hy * k, (x + hs) * k, (hy + hs) * k), radius=6 * k,
                                 fill=mix(mix(BG, GLOW, a_glow), soft(hc, 1), ah),
                                 outline=mix(mix(LINE, ACCENT, a_glow), hc, ah), width=round((1.4 + 1.2 * ah) * k))
            mx, my = (x + hs / 2) * k, (hy + hs / 2) * k
            r = hs * 0.3 * k
            self.arrow(dr, (mx - dx * r, my - dy * r), (mx + dx * r, my + dy * r),
                       mix(mix(MUTED, ACCENT, a_glow), hc, ah), w=2.0, head=hs * 0.22)
        # MLP: 86 neurons in their five jobs
        dr.text((cx * k, g["mlp_label"] * k), "MLP, 86 neurons", font=f["lab"], fill=FG, anchor="ms")
        r1, r2, sp = g["dots"]
        cols = [n // 2 for n in GROUPS]
        width = sum(cols) * sp + (len(cols) - 1) * sp
        x = cx - width / 2 + sp / 2
        dim = mix(BG, ACCENT, 0.35)
        for n in cols:
            for _ in range(n):
                for yy in (r1, r2):
                    rr = 2.3 * k
                    dr.ellipse((x * k - rr, yy * k - rr, x * k + rr, yy * k + rr),
                               fill=mix(mix(dim, ACCENT, m_glow), hc, on("mlp")))
                x += sp
            x += sp
        # block -> readout -> maze
        o_y = g["io"][1]
        rx0, ry0, rx1, ry1 = g["readout"]
        if up:
            dr.line([(cx * k, by0 * k), (cx * k, o_y * k)], fill=col, width=round(2.4 * k))
            self.arrow(dr, (cx * k, o_y * k), (cx * k, (ry1 + 1) * k), LINE)
        else:
            dr.line([(cx * k, by1 * k), (cx * k, o_y * k)], fill=col, width=round(2.4 * k))
            self.arrow(dr, (cx * k, o_y * k), (cx * k, (ry0 - 1) * k), LINE)
        if on("readout"):
            pad = 7
            dr.rounded_rectangle(((rx0 - pad) * k, (ry0 - pad) * k, (rx1 + pad) * k, (ry1 + pad) * k), radius=13 * k,
                                 outline=mix(BG, hc, on("readout")), width=round(3 * k))
        self.box(dr, g["readout"], mix(TAG, ACCENT, done), mix(LINE, ACCENT, done), None, FG)
        lab_x = (rx0 + rx1) / 2 - 10 * done
        dr.text((lab_x * k, (ry0 + ry1) / 2 * k), "readout", font=f["box"], fill=mix(FG, BG, done), anchor="mm")
        if done > 0.02:                                  # a small check once the path is complete
            ck = mix(ACCENT, BG, done)
            cxk = (rx1 - 20) * k
            cyk = (ry0 + ry1) / 2 * k
            dr.line([(cxk - 6 * k, cyk), (cxk - 2 * k, cyk + 5 * k), (cxk + 7 * k, cyk - 6 * k)], fill=ck,
                    width=round(2.4 * k), joint="curve")
        p0, p1 = g["out"]
        self.arrow(dr, (p0[0] * k, p0[1] * k), (p1[0] * k, p1[1] * k), mix(LINE, ACCENT, done), w=2.0 + done)

        # the loop counter
        cxn, cyn = g["count"]
        n = int(math.floor(L + 1e-9))
        text_fixed(dr, (cxn * k, cyn * k), f"loop {n:,}", f["count"], mix(FG, hc, on("count")), anchor="ls")
        dr.text((cxn * k, (cyn + 26) * k), f"of {tl.R:,}", font=f["of"], fill=MUTED, anchor="ls")

        # the pulse, outside the block
        if moving and not inside:
            self.dot(dr, self.cycle.at(u), 0.55 + 0.45 * quiet)

    # ------------------------------------------------------------------ the thought bubble (stages 3 and 4)
    def bubble_alpha(self, t):
        """(input fade, bubble) between 0 and 1. In with stage 3 (the wave), out with the end of stage 4 (the path).
        The input strip and embed fade out first and the bubble fades in after, so the two never show at once."""
        st = self.tl.stages()
        a, b = st[2][2], st[3][3]
        hide = smooth(clamp((t - a) / 0.3)) * (1 - smooth(clamp((t - (b - 0.3)) / 0.3)))
        show = smooth(clamp((t - a - 0.3) / 0.4)) * (1 - smooth(clamp((t - (b - 0.75)) / 0.4)))
        return hide, show

    def make_cloud(self):
        k, g = SS, self.g
        layer = Image.new("RGBA", (self.W, self.H), (0, 0, 0, 0))
        dr = ImageDraw.Draw(layer)
        cx, cy, R, A = g["bubble"]
        base = [(x * k, y * k) for x, y in cloud_points(cx, cy, R, A)]
        dr.polygon(base, fill=BUBBLE_FILL + (255,))
        sketch(dr, base, BUBBLE_LINE + (255,), round(2.0 * k), 2.2 * k, seed=11)
        for j, (x, y, r) in enumerate(g["trail"]):
            pts = [(px * k, py * k) for px, py in circle_points(x, y, r)]
            dr.polygon(pts, fill=BUBBLE_FILL + (255,))
            sketch(dr, pts, BUBBLE_LINE + (255,), round(1.8 * k), 0.9 * k, seed=20 + j)
        return layer

    def make_note(self):
        """'loop imagination', handwritten in dark orange, slightly tilted."""
        k, g = SS, self.g
        x, y, size, lines, rot = g["note"]
        f = ImageFont.truetype(str(CAVEAT), int(size * k))
        f.set_variation_by_axes([650])
        tmp = ImageDraw.Draw(Image.new("RGBA", (1, 1)))
        lead = size * 1.0 * k
        w = int(max(tmp.textlength(l, font=f) for l in lines) + 20 * k)
        h = int(lead * len(lines) + 24 * k)
        layer = Image.new("RGBA", (w, h), (0, 0, 0, 0))
        d = ImageDraw.Draw(layer)
        for j, line in enumerate(lines):
            d.text((w / 2, 10 * k + lead * (j + 0.5)), line, font=f, fill=NOTE_INK + (255,), anchor="mm")
        layer = layer.rotate(-rot, resample=Image.BICUBIC, expand=True)
        return layer, (int(x * k - layer.width / 2), int(y * k - layer.height / 2))

    def cell_state(self, i, L):
        """(fill colour, digit shown 0..1, arrow shown 0..1, path 0..1) of cell i at loop L, from the trace."""
        tr = self.ins
        if self.q[i] == "#":
            return WALL, 0.0, 0.0, 0.0
        col, grow, arrow, path = BG, 0.0, 0.0, 0.0
        rl = tr["reach_loop"][i]
        if rl >= 0:
            grow = clamp(L - (rl - 1)) if rl > 0 else 1.0
            col = mix(BG, mix(WAVE_NEW, WAVE_OLD, clamp((L - rl) / 10)), grow)
        sl = tr["step_loop"][i]
        if sl >= 0:
            arrow = clamp(L - (sl - 1))
        ol = tr["on_loop"][i]
        if ol >= 0:
            path = clamp(L - (ol - 1)) if ol > 0 else 1.0
            col = mix(col, ACCENT, path)
        return col, grow, arrow, path

    def zbox(self, r, c):
        x, y, s = (v * SS for v in self.g["zoom"])
        x0, y0 = x + (c - COLS[0]) * s, y + (r - ROWS[0]) * s
        return x0, y0, x0 + s, y0 + s

    def draw_zoom(self, dr, L):
        """The close-up: distance counted in threes, each cell's next step, and the path, as the block holds them."""
        k, s = SS, self.g["zoom"][2] * SS
        a_tip, a_base, a_half = (v * k for v in self.g["arrow"])
        for r in ROWS:
            for c in COLS:
                dr.rectangle(self.zbox(r, c), fill=self.cell_state(r * self.side + c, L)[0] + (255,))
        x0, y0, _, _ = self.zbox(ROWS[0], COLS[0])
        _, _, x1, y1 = self.zbox(ROWS[-1], COLS[-1])
        for j in range(len(COLS) + 1):
            dr.line([(x0 + j * s, y0), (x0 + j * s, y1)], fill=CLOSE_GRID + (255,), width=max(1, k))
        for j in range(len(ROWS) + 1):
            dr.line([(x0, y0 + j * s), (x1, y0 + j * s)], fill=CLOSE_GRID + (255,), width=max(1, k))
        for r in ROWS:
            for c in COLS:
                i = r * self.side + c
                if self.q[i] == "#":
                    continue
                bx0, by0, bx1, by1 = box = self.zbox(r, c)
                col, grow, arrow, path = self.cell_state(i, L)
                cx, cy = (bx0 + bx1) / 2, (by0 + by1) / 2 - 1.5 * k
                ink = mix(FG, BG, path)
                if grow > 0.01:
                    dr.text((cx, cy), str(self.ins["phase"][i]), font=self.f["digit"], fill=mix(col, ink, grow) + (255,),
                            anchor="mm")
                if arrow > 0.01:
                    dy, dx = DIRS[self.ins["step"][i]]
                    ccy = (by0 + by1) / 2
                    tip = (cx + dx * (s / 2 - a_tip), ccy + dy * (s / 2 - a_tip))
                    base = (cx + dx * (s / 2 - a_base), ccy + dy * (s / 2 - a_base))
                    px, py = -dy * a_half, dx * a_half
                    dr.polygon([tip, (base[0] + px, base[1] + py), (base[0] - px, base[1] - py)],
                               fill=mix(col, mix(MUTED, BG, path), arrow) + (255,))
                if i == self.ins["goal"]:
                    dr.text((bx0 + 2 * k, by0 + 1 * k), "G", font=self.f["corner"], fill=ink + (255,), anchor="lt")
                    dr.rectangle(box, outline=FG + (255,), width=2 * k)
        dr.rectangle((x0 - k, y0 - k, x1 + k, y1 + k), outline=MARK + (255,), width=3 * k)

    def draw_bubble(self, img, t, L):
        """The block's thoughts: a bubble with the close-up, the note and its swirl, and a focus box on the maze."""
        hide, a = self.bubble_alpha(t)
        if hide < 0.005:
            return img
        k = SS
        base = img.convert("RGBA")
        zone = tuple(int(v * k) for v in self.g["zone"])               # the input strip and embed fade out
        region = base.crop(zone)
        base.paste(Image.blend(region, Image.new("RGBA", region.size, BG + (255,)), hide), zone[:2])
        if a < 0.005:
            return base.convert("RGB")
        over = self.cloud.copy()
        dr = ImageDraw.Draw(over)
        self.draw_zoom(dr, L)
        note, (nx, ny) = self.note
        over.alpha_composite(note, (nx, ny))
        st = self.tl.stages()
        u = smooth(clamp((t - (st[2][2] + 0.5)) / 1.1))          # the swirl is written on after the bubble appears
        if u > 0:
            for pts in self.swirl:
                n = max(2, int(len(pts) * u))
                dr.line(pts[:n], fill=NOTE_INK + (255,), width=round(2.2 * k), joint="curve")
            if u >= 0.98:
                pts = self.swirl[0]
                (xa, ya), (xb, yb) = pts[-6], pts[-1]
                ang = math.atan2(yb - ya, xb - xa)
                for side in (-1, 1):
                    q = ang + math.pi + side * 0.5
                    dr.line([(xb, yb), (xb + 11 * k * math.cos(q), yb + 11 * k * math.sin(q))], fill=NOTE_INK + (255,),
                            width=round(2.2 * k))
        n = self.side                                              # the focus box on the maze
        fx0, fy0 = self.gx + COLS[0] * self.cell, self.gy + ROWS[0] * self.cell
        fx1, fy1 = self.gx + (COLS[-1] + 1) * self.cell, self.gy + (ROWS[-1] + 1) * self.cell
        dr.rectangle((fx0 - k, fy0 - k, fx1 + k, fy1 + k), outline=MARK + (255,), width=round(2.5 * k))
        if a < 0.999:
            over.putalpha(over.getchannel("A").point(lambda v: int(v * a)))
        base.alpha_composite(over)
        return base.convert("RGB")

    def dot(self, dr, p, a):
        k = SS
        x, y = p
        for r, c in ((11 * k, mix(BG, GLOW, a)), (5.5 * k, mix(BG, ACCENT, a))):
            dr.ellipse((x - r, y - r, x + r, y + r), fill=c)

    def frame(self, tl, t):
        img = Image.new("RGB", (self.W, self.H), BG)
        dr = ImageDraw.Draw(img)
        L = tl.loop(t)
        st = self.stage_state(t)
        hl = (st[2], st[3], st[4]) if st else None
        self.draw_grid(dr, L, hl)
        self.draw_arch(dr, t, L, hl)
        img = self.draw_bubble(img, t, L)
        img = self.draw_banner(img, st)
        return img.reduce(SS)


def render(scene, tl, outputs, poster=None):
    n_frames = round(tl.length * FPS)
    W, H = FORMATS[scene.kind]
    procs = [subprocess.Popen(["ffmpeg", "-loglevel", "error", "-y", "-f", "rawvideo", "-pix_fmt", "rgb24",
                               "-s", f"{W}x{H}", "-r", str(FPS), "-i", "-", *args, str(out)], stdin=subprocess.PIPE)
             for out, args in outputs]
    first = scene.frame(tl, 0.0)
    loop_from = tl.length - T_LOOP
    for i in range(n_frames):
        t = i / FPS
        img = scene.frame(tl, t)
        if poster and abs(t - (loop_from - 1 / FPS)) < 0.5 / FPS:
            img.save(poster, optimize=True)
        if t >= loop_from:
            img = Image.blend(img, first, smooth((t - loop_from) / T_LOOP))
        for p in procs:
            p.stdin.write(img.tobytes())
    for p in procs:
        p.stdin.close()
        p.wait()
        if p.returncode:
            sys.exit("ffmpeg failed")
    return n_frames


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--only", choices=list(FORMATS))
    ap.add_argument("--frames", type=Path)
    a = ap.parse_args()
    tr = json.loads((HERE / "maze_trace.json").read_text())
    ins = json.loads((HERE / "maze_inside_trace.json").read_text())
    assert tr["exact_match"], "the network's final grid is not the dataset's answer"
    assert ins["question"] == tr["question"] and ins["reach_loop"] == tr["reach_loop"] and ins["on_loop"] == tr["on_loop"]
    tl = Timeline(tr)
    print(f"test maze {tr['test_index'] + 1}: wave reaches the start at loop {tl.D}, path done at loop {tl.done}, "
          f"{tl.R} loops; length {tl.length:.2f} s")
    kinds = [a.only] if a.only else list(FORMATS)
    if a.frames:
        a.frames.mkdir(parents=True, exist_ok=True)
        picks = {f"stage{j + 1:02d}": (a + b) / 2 for j, (_, _, a, b) in enumerate(tl.stages())}
        picks["zz_end"] = tl.length - T_LOOP - 0.05
        for kind in kinds:
            sc = Scene(kind, tr, tl)
            for name, t in picks.items():
                sc.frame(tl, t).save(a.frames / f"maze_{kind}_{name}.png")
        print(f"frames in {a.frames}")
        return
    jobs = {"wide": [(HERE / "maze_hero.mp4", MP4), (HERE / "maze_hero.webm", WEBM)],
            "tall": [(HERE / "maze_hero_tall.mp4", MP4), (HERE / "maze_hero_tall.webm", WEBM)]}
    posters = {"wide": HERE / "maze_hero_poster.png", "tall": HERE / "maze_hero_tall_poster.png"}
    for kind, outs in jobs.items():
        if kind in kinds:
            n = render(Scene(kind, tr, tl), tl, outs, posters.get(kind))
            print(f"{kind}: {n} frames, {n / FPS:.1f} s")


if __name__ == "__main__":
    main()
