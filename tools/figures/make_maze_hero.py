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
import subprocess
import sys
from pathlib import Path

from PIL import Image, ImageDraw

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
FORMATS = {"wide": (1280, 720), "tall": (720, 850)}

# timeline (seconds). The opening is slow so each stage can be read; then the loops speed up.
T_LOOP = 0.45
W_STAGE = 2.4                                        # each banner stage is up this long (2 s fully visible)
T_PASS1 = 2 * W_STAGE                                # the first pass starts after stages 1 and 2
T_WAVE0 = 5 * W_STAGE                                # the wave starts after the first pass (stages 3 to 5)


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
        """(text, part it describes, start, end). One at a time in the banner."""
        w = W_STAGE
        t2, t3, t4 = self.t2, self.t3, self.t4
        return [
            ("Each maze cell becomes one token, 900 in all", "input", 0.0, w),
            ("Each token holds 35 numbers", "embed", w, 2 * w),
            ("Attention lets each cell read its four neighbours", "attn", 2 * w, 3 * w),
            ("Neurons update each cell's distance, pointer and path", "mlp", 3 * w, 4 * w),
            ("The same block runs again, loop after loop", "loop", 4 * w, 5 * w),
            ("A wave of distances spreads out from the goal", "maze_goal", t2, t3),
            (f"The wave reaches the start at loop {self.D}", "maze_start", t3, t3 + w),
            ("The path grows from the start back to the goal", "maze", t3 + w, t4),
            (f"Path complete at loop {self.done}", "readout", t4, t4 + w),
            (f"Extra loops leave the answer unchanged, up to loop {self.R:,}", "count", t4 + w, self.length - T_LOOP - 0.05),
        ]


GLOW = (250, 211, 194)                              # the block lit by a pass (light orange)
ACCENT_SOFT = (253, 241, 235)
CAP = 1.0                                           # passes per second shown at most: the pulse never strobes
GROUPS = [6, 24, 8, 8, 40]                          # MLP neurons by job: wave, pointer, pointer clear, chain, clean-up
# One colour per banner stage, from the palette's text shades: blue #1c5cab, aqua #0f7d56, yellow #a87200, orange #c4471a.
# Orange (the path) is kept for the stage where the path grows; neighbouring stages differ.
_B, _A, _Y, _O = (28, 92, 171), (15, 125, 86), (168, 114, 0), (196, 71, 26)
STAGE_COLORS = [_B, _A, _Y, _B, _A, _B, _Y, _O, _A, _Y]
BANNER = {"wide": dict(x=40, y=26, r=21, size=26, lead=35, maxw=520),
          "tall": dict(x=24, y=18, r=20, size=24, lead=32, maxw=620)}

# Every position is in output pixels. "flow" is the direction data moves through the diagram.
LAYOUT = {
    "wide": dict(
        flow="up", cx=270, fs=dict(lab=19, small=16, box=19, count=22, of=16),
        maze=(640, 52, 20.5),
        cells=(135, 646, 18, 5), cells_label=(270, 694),
        embed=(190, 570, 350, 610), embed_note=(362, 597),
        block=(70, 260, 470, 500), tag=(88, 286),
        mlp_label=316, dots=(336, 350, 8), attn_label=404, heads=(420, 40, 14),
        readout=(195, 150, 345, 190),
        io=(535, 225), loop_x=494, count=(506, 372),
        out=[(345, 170), (626, 170)],
    ),
    "tall": dict(
        flow="down", cx=340, fs=dict(lab=20, small=17, box=20, count=24, of=17),
        maze=(160, 440, 400 / 30),
        cells=(216, 112, 16, 5), cells_label=(476, 126),
        embed=(265, 150, 415, 182), embed_note=(428, 172),
        block=(140, 206, 540, 362), tag=(156, 232),
        attn_label=232, heads=(242, 32, 12), mlp_label=302, dots=(318, 332, 8),
        readout=(275, 384, 405, 414),
        io=(194, 374), loop_x=562, count=(580, 290),
        out=[(340, 414), (340, 436)],
    ),
}


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
                      of=font("Regular", fs["of"] * k), letter=font("Bold", 12 * k))
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
