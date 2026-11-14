#!/usr/bin/env python3
"""Render the "inside the maze network" animation from the real per-loop state (build tool, not public).

    python3 figures/make_maze_inside.py [--only wide|tall] [--frames FOLDER]

Input: figures/maze_inside_trace.json (from make_maze_inside_trace.py, which runs the public weights/run.py). For every
cell of test maze 1 it holds the loop at which the cell got its distance count (stored mod 3), the loop at which it got
its next step (D, L, R or U), and the loop at which it joined the path, all read from the network's residual stream
after every loop; the trace script checks that these events rebuild the stored state exactly at every loop. Every digit,
arrow and colour shown at loop t is that state. Staged: only the timing (loops play at varying speed) and the four
"reads its neighbours" arrows at loop 1, which mark what every cell does in every loop.

Outputs: figures/maze_inside(.mp4|.webm|_poster.png) and figures/maze_inside_tall(...). Needs Pillow, numpy and ffmpeg.
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
from make_hero import ACCENT, BG, FG, MP4, MUTED, SS, WEBM, clamp, font, mix, ramp, smooth, text_fixed  # noqa: E402
from make_maze_hero import WALL, WAVE_NEW, WAVE_OLD  # noqa: E402

HERE = Path(__file__).resolve().parent
FPS = 60
ORANGE = (214, 110, 26)
GRIDLINE = (226, 229, 233)
ROWS, COLS = range(10, 18), range(3, 11)          # the close-up: 8 x 8 cells around the goal
READER = (14, 4)                                    # the cell whose four reads are drawn at loop 1
DIRS = {"D": (1, 0), "L": (0, -1), "R": (0, 1), "U": (-1, 0)}
FORMATS = {"wide": (1280, 720), "tall": (720, 850)}
T_LOOP = 0.45

# (time in s, loop shown): linear between keys, except the eased ones
KEYS = [(0.0, 0), (1.0, 0), (2.6, 1), (3.0, 1), (6.0, 5), (8.6, 20), (11.0, 111), (11.7, 111), (13.6, 206),
        (17.6, 223), (18.2, 223), (20.0, 1800), (22.4, 1800)]
EASED = {(8.6, 11.0): "in", (18.2, 20.0): "fast"}
LENGTH = KEYS[-1][0] + T_LOOP


def loop_at(t):
    for (t0, l0), (t1, l1) in zip(KEYS, KEYS[1:]):
        if t < t1:
            x = (t - t0) / (t1 - t0)
            kind = EASED.get((t0, t1))
            x = x ** 2.2 if kind == "fast" else (0.35 * x + 0.65 * x * x) if kind == "in" else x
            return l0 + (l1 - l0) * x
    return float(KEYS[-1][1])


def on(t, a, b):
    """1 between times a and b, with short fades."""
    return ramp(t, a - 0.15, a + 0.15) * (1 - ramp(t, b - 0.15, b + 0.15))


NAVY = (21, 70, 119)
SKETCH_GRAY = (150, 156, 166)
CAVEAT = HERE / "fonts" / "Caveat-Variable.ttf"     # SIL Open Font License, fonts/Caveat-OFL.txt

# Output pixels. zoom: (x, y, cell); mini: (x, y, cell); bubble: (cx, cy, R, bump); trail: [(x, y, r)];
# char: (cx, cy, w, h); label: (x, y, anchor); count: (x, y).
GEOM = {
    "wide": dict(zoom=(160, 72, 42), mini=(680, 52, 18.5), bubble=(328, 240, 196, 20),
                 trail=[(240, 480, 13), (221, 514, 9), (207, 541, 6)], char=(196, 620, 92, 70),
                 label=(196, 700, "ms"), count=(680, 650), digit=20, arrow=(4, 12, 5), corner=12, letter=11,
                 count_size=20, label_size=30,
                 note=(482, 612, 38, ["loop imagination"], -4), swirl=((430, 582), (468, 432), 2)),
    "tall": dict(zoom=(224, 56, 34), mini=(210, 532, 10.0), bubble=(360, 192, 160, 18),
                 trail=[(240, 390, 10), (216, 413, 7), (198, 431, 5)], char=(170, 490, 84, 56),
                 label=(228, 506, "ls"), count=(524, 830), digit=17, arrow=(3.5, 10, 4), corner=10, letter=0,
                 count_size=18, label_size=27,
                 note=(622, 452, 34, ["loop", "imagination"], -5), swirl=((600, 404), (522, 330), 1)),
}


def swirl_points(p0, p1, loops, N=260):
    """A loop-de-loop from p0 to p1: a prolate cycloid laid along the line between them, starting and ending on
    a forward stroke, so the arrow leaves the note and arrives at the bubble head first."""
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


# ------------------------------------------------------------------ hand-drawn strokes (rough.js style)
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


def rrect_points(x0, y0, x1, y1, r, N=24):
    pts = []
    corners = [(x1 - r, y0 + r, -90), (x1 - r, y1 - r, 0), (x0 + r, y1 - r, 90), (x0 + r, y0 + r, 180)]
    for cx, cy, a0 in corners:
        for i in range(N + 1):
            a = math.radians(a0 + 90 * i / N)
            pts.append((cx + r * math.cos(a), cy + r * math.sin(a)))
    return pts


class Scene:
    """The whole maze with its loop counter; the network Claude wrote as a small hand-drawn character; and, in a
    thought bubble above it, the close-up of the cells around the goal: what the network works out, loop by loop."""

    def __init__(self, kind, tr):
        self.kind, self.tr, self.g = kind, tr, GEOM[kind]
        W, H = FORMATS[kind]
        self.W, self.H = W * SS, H * SS
        self.n = tr["side"]
        self.q = tr["question"]
        g, k = self.g, SS
        caveat = ImageFont.truetype(str(CAVEAT), int(g["label_size"] * k))
        caveat.set_variation_by_axes([600])
        self.f = dict(digit=font("SemiBold", g["digit"] * k), corner=font("Bold", g["corner"] * k),
                      letter=font("Bold", max(g["letter"], 8) * k), count=font("Regular", g["count_size"] * k),
                      label=caveat)
        self.bubble = self.make_bubble()
        self.note = self.make_note()
        sw = g["swirl"]
        base = [(x * k, y * k) for x, y in swirl_points(sw[0], sw[1], sw[2])]
        self.swirl = [jitter(base, 1.1 * k, seed=70 + j, closed=False) for j in range(2)]
        self.char_open, self.char_shut = self.make_char(1.0), self.make_char(0.12)

    # ------------------------------------------------------------------ hand-drawn layers, made once
    def make_bubble(self):
        k, g = SS, self.g
        layer = Image.new("RGBA", (self.W, self.H), (0, 0, 0, 0))
        dr = ImageDraw.Draw(layer)
        cx, cy, R, A = g["bubble"]
        base = [(x * k, y * k) for x, y in cloud_points(cx, cy, R, A)]
        dr.polygon(base, fill=BG + (255,))
        sketch(dr, base, SKETCH_GRAY + (255,), round(2.0 * k), 2.2 * k, seed=11)
        for j, (x, y, r) in enumerate(g["trail"]):
            pts = [(px * k, py * k) for px, py in circle_points(x, y, r)]
            dr.polygon(pts, fill=BG + (255,))
            sketch(dr, pts, SKETCH_GRAY + (255,), round(1.8 * k), 0.9 * k, seed=20 + j)
        return layer

    def make_note(self):
        """'loop imagination', handwritten in navy, slightly tilted."""
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
            d.text((w / 2, 10 * k + lead * (j + 0.5)), line, font=f, fill=NAVY + (255,), anchor="mm")
        layer = layer.rotate(-rot, resample=Image.BICUBIC, expand=True)
        return layer, (int(x * k - layer.width / 2), int(y * k - layer.height / 2))

    def draw_swirl(self, dr, t):
        """The arrow is written on after the first second, then stays."""
        u = smooth(clamp((t - 1.0) / 1.1))
        if u <= 0:
            return
        k = SS
        for pts in self.swirl:
            n = max(2, int(len(pts) * u))
            dr.line(pts[:n], fill=NAVY, width=round(2.2 * k), joint="curve")
        if u >= 0.98:
            pts = self.swirl[0]
            (xa, ya), (xb, yb) = pts[-6], pts[-1]
            ang = math.atan2(yb - ya, xb - xa)
            for side in (-1, 1):
                a = ang + math.pi + side * 0.5
                dr.line([(xb, yb), (xb + 11 * k * math.cos(a), yb + 11 * k * math.sin(a))], fill=NAVY, width=round(2.2 * k))

    def make_char(self, eye_open):
        """The network Claude wrote: a navy block, hatched by hand, with two eyes and a loop arrow on top."""
        k, g = SS, self.g
        cx, cy, w, h = g["char"]
        pad = 40
        size = (int((w + 2 * pad) * k), int((h + 2 * pad + 30) * k))
        layer = Image.new("RGBA", size, (0, 0, 0, 0))
        dr = ImageDraw.Draw(layer)
        ox, oy = (cx - w / 2 - pad) * k, (cy - h / 2 - pad - 30) * k   # layer origin in frame pixels
        X = lambda x: x * k - ox
        Y = lambda y: y * k - oy
        x0, y0, x1, y1 = cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2
        body = [(X(x), Y(y)) for x, y in rrect_points(x0, y0, x1, y1, min(w, h) * 0.28)]
        body_j = jitter([(x, y) for x, y in body], 1.2 * k, seed=31)
        dr.polygon(body_j, fill=mix(BG, NAVY, 0.58) + (255,))        # a lighter navy fill:
        hatch = Image.new("RGBA", size, (0, 0, 0, 0))                 # loose darker hatching on top,
        hd = ImageDraw.Draw(hatch)                                    # clipped to the body
        rnd = random.Random(41)
        step = 6 * k
        x = X(x0) - (y1 - y0) * k
        while x < X(x1) + 4 * k:
            a = (x + rnd.uniform(-1, 1) * k, Y(y1) + 2 * k)
            b = (x + (y1 - y0) * k + rnd.uniform(-1, 1) * k, Y(y0) - 2 * k)
            hd.line(jitter([a, ((a[0] + b[0]) / 2, (a[1] + b[1]) / 2), b], 0.8 * k, seed=int(x), closed=False),
                    fill=NAVY + (255,), width=round(2.0 * k))
            x += step
        mask = Image.new("L", size, 0)
        ImageDraw.Draw(mask).polygon(body_j, fill=255)
        layer.paste(hatch, (0, 0), Image.composite(hatch.getchannel("A"), Image.new("L", size, 0), mask))
        sketch(dr, body, NAVY + (255,), round(2.4 * k), 1.8 * k, seed=33)
        # eyes: hand-drawn white ovals with a small pupil
        for j, ex in enumerate((cx - w * 0.17, cx + w * 0.17)):
            ey = cy - h * 0.06
            rx, ry = w * 0.085, h * 0.17 * eye_open
            oval = [(X(ex + rx * math.cos(t)), Y(ey + ry * math.sin(t))) for t in [2 * math.pi * i / 40 for i in range(40)]]
            oval = jitter(oval, 0.5 * k, seed=50 + j)
            dr.polygon(oval, fill=(255, 255, 255, 255))
            if eye_open > 0.5:
                pr = rx * 0.45 * k
                px, py = X(ex + rx * 0.15), Y(ey + ry * 0.2)
                dr.ellipse((px - pr, py - pr, px + pr, py + pr), fill=NAVY + (255,))
        # a loop arrow on top: an open circle with an arrowhead
        lr = min(w, h) * 0.17
        lcx, lcy = cx, y0 - lr - 7
        arc = [(X(lcx + lr * math.cos(math.radians(a))), Y(lcy + lr * math.sin(math.radians(a))))
               for a in range(-60, 241, 6)]
        for pass_ in range(2):
            dr.line(jitter(arc, 0.6 * k, seed=60 + pass_, closed=False), fill=NAVY + (255,), width=round(2.2 * k),
                    joint="curve")
        ex_, ey_ = arc[-1]
        ang = math.radians(240 + 90)                                   # tangent at the arc's end
        for side in (-1, 1):
            hx = ex_ + 7 * k * math.cos(ang + math.pi + side * 0.55)
            hy = ey_ + 7 * k * math.sin(ang + math.pi + side * 0.55)
            dr.line([(ex_, ey_), (hx, hy)], fill=NAVY + (255,), width=round(2.2 * k))
        return layer, (int(ox), int(oy))

    # ------------------------------------------------------------------ the real state at (fractional) loop L
    def state(self, i, L):
        """(fill colour, count shown 0..1, arrow shown 0..1, path 0..1) of cell i."""
        tr = self.tr
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

    # ------------------------------------------------------------------ the close-up, inside the bubble
    def zbox(self, r, c):
        x, y, s = (v * SS for v in self.g["zoom"])
        x0, y0 = x + (c - COLS[0]) * s, y + (r - ROWS[0]) * s
        return x0, y0, x0 + s, y0 + s

    def draw_zoom(self, dr, L, t):
        k, s = SS, self.g["zoom"][2] * SS
        a_tip, a_base, a_half = (v * k for v in self.g["arrow"])
        for r in ROWS:
            for c in COLS:
                dr.rectangle(self.zbox(r, c), fill=self.state(r * self.n + c, L)[0])
        x0, y0, _, _ = self.zbox(ROWS[0], COLS[0])
        _, _, x1, y1 = self.zbox(ROWS[-1], COLS[-1])
        for j in range(len(COLS) + 1):
            dr.line([(x0 + j * s, y0), (x0 + j * s, y1)], fill=GRIDLINE, width=max(1, k))
        for j in range(len(ROWS) + 1):
            dr.line([(x0, y0 + j * s), (x1, y0 + j * s)], fill=GRIDLINE, width=max(1, k))
        for r in ROWS:
            for c in COLS:
                i = r * self.n + c
                if self.q[i] == "#":
                    continue
                bx0, by0, bx1, by1 = box = self.zbox(r, c)
                col, grow, arrow, path = self.state(i, L)
                cx, cy = (bx0 + bx1) / 2, (by0 + by1) / 2
                ink = mix(FG, BG, path)
                if grow > 0.01:
                    dr.text((cx, cy), str(self.tr["phase"][i]), font=self.f["digit"], fill=mix(col, ink, grow), anchor="mm")
                if arrow > 0.01:
                    dy, dx = DIRS[self.tr["step"][i]]
                    tip = (cx + dx * (s / 2 - a_tip), cy + dy * (s / 2 - a_tip))
                    base = (cx + dx * (s / 2 - a_base), cy + dy * (s / 2 - a_base))
                    px, py = -dy * a_half, dx * a_half
                    dr.polygon([tip, (base[0] + px, base[1] + py), (base[0] - px, base[1] - py)],
                               fill=mix(col, mix(MUTED, BG, path), arrow))
                if i == self.tr["goal"]:
                    dr.text((bx0 + 4 * k, by0 + 2 * k), "G", font=self.f["corner"], fill=ink, anchor="lt")
                    dr.rectangle(box, outline=FG, width=2 * k)
        dr.rectangle((x0 - k, y0 - k, x1 + k, y1 + k), outline=ORANGE, width=3 * k)

    def draw_reads(self, img, t):
        """The four reads of one cell at loop 1 (marks what every cell does in every loop)."""
        a = on(t, 1.15, 2.85)
        if a < 0.01:
            return
        k = SS
        over = Image.new("RGBA", img.size, (0, 0, 0, 0))
        od = ImageDraw.Draw(over)
        r, c = READER
        tx0, ty0, tx1, ty1 = self.zbox(r, c)
        tcx, tcy = (tx0 + tx1) / 2, (ty0 + ty1) / 2
        s = tx1 - tx0
        alpha = int(255 * a)
        for d, (dy, dx) in DIRS.items():
            src = self.zbox(r + dy, c + dx)
            scx, scy = (src[0] + src[2]) / 2, (src[1] + src[3]) / 2
            start = (scx - dx * s * 0.08, scy - dy * s * 0.08)
            end = (tcx + dx * s * 0.44, tcy + dy * s * 0.44)
            od.line([start, end], fill=ORANGE + (alpha,), width=3 * k)
            hx, hy = -dx * 9 * k, -dy * 9 * k
            px, py = -dy * 6 * k, dx * 6 * k
            od.polygon([(end[0] - hx * 0.2, end[1] - hy * 0.2), (end[0] + hx + px, end[1] + hy + py),
                        (end[0] + hx - px, end[1] + hy - py)], fill=ORANGE + (alpha,))
        od.rectangle((tx0, ty0, tx1, ty1), outline=ORANGE + (alpha,), width=3 * k)
        img.alpha_composite(over)

    # ------------------------------------------------------------------ the whole maze
    def draw_mini(self, dr, L, t):
        k = SS
        mx, my, cs = self.g["mini"]
        mx, my, cs = mx * k, my * k, cs * k
        n = self.n
        for i, ch in enumerate(self.q):
            r, c = divmod(i, n)
            col = self.state(i, L)[0]
            if i in (self.tr["start"], self.tr["goal"]):
                col = FG
            if col != BG:
                dr.rectangle((mx + c * cs, my + r * cs, mx + (c + 1) * cs, my + (r + 1) * cs), fill=col)
        for j in range(n + 1):
            dr.line([(mx + j * cs, my), (mx + j * cs, my + n * cs)], fill=(236, 238, 241), width=1)
            dr.line([(mx, my + j * cs), (mx + n * cs, my + j * cs)], fill=(236, 238, 241), width=1)
        dr.rectangle((mx - 1.5 * k, my - 1.5 * k, mx + n * cs + 1.5 * k, my + n * cs + 1.5 * k), outline=FG, width=max(2, round(1.5 * k)))
        if self.g["letter"]:
            for idx, letter in ((self.tr["start"], "S"), (self.tr["goal"], "G")):
                r, c = divmod(idx, n)
                dr.text((mx + (c + 0.5) * cs, my + (r + 0.5) * cs), letter, font=self.f["letter"], fill=BG, anchor="mm")
        dr.rectangle((mx + COLS[0] * cs - k, my + ROWS[0] * cs - k, mx + (COLS[-1] + 1) * cs + k, my + (ROWS[-1] + 1) * cs + k),
                     outline=ORANGE, width=2 * k)
        pulse = on(t, 10.9, 12.0)                          # the count reaches the start
        if pulse > 0.01:
            r, c = divmod(self.tr["start"], n)
            cx, cy, rad = mx + (c + 0.5) * cs, my + (r + 0.5) * cs, cs * (1.4 + 0.8 * smooth(clamp((t - 10.9) / 1.1)))
            dr.ellipse((cx - rad, cy - rad, cx + rad, cy + rad), outline=mix(BG, ORANGE, pulse), width=2 * k)

    # ------------------------------------------------------------------ the only text: counter and label
    def draw_text(self, dr, L, t):
        k, g = SS, self.g
        x, y = g["count"]
        text_fixed(dr, (x * k, y * k), f"loop {int(math.floor(L + 1e-9)):,}", self.f["count"], MUTED, anchor="ls")
        lx, ly, anc = g["label"]
        dr.text((lx * k, ly * k), "the network Claude wrote", font=self.f["label"], fill=MUTED, anchor=anc)

    def frame(self, t):
        img = Image.new("RGBA", (self.W, self.H), BG + (255,))
        L = loop_at(t)
        img.alpha_composite(self.bubble)
        bob = round(2.0 * SS * math.sin(2 * math.pi * t / 2.8))          # a gentle bob
        blink = (t % 3.7) > 3.52                                          # and a blink now and then
        layer, (ox, oy) = self.char_shut if blink else self.char_open
        img.alpha_composite(layer, (ox, oy + bob))
        a = smooth(clamp((t - 0.5) / 0.4))                             # the note fades in, the arrow is drawn on
        if a > 0:
            note, (nx, ny) = self.note
            if a < 0.999:
                note = note.copy()
                note.putalpha(note.getchannel("A").point(lambda v: int(v * a)))
            img.alpha_composite(note, (nx, ny))
        dr = ImageDraw.Draw(img)
        self.draw_swirl(dr, t)
        self.draw_zoom(dr, L, t)
        self.draw_mini(dr, L, t)
        self.draw_text(dr, L, t)
        self.draw_reads(img, t)
        return img.convert("RGB").reduce(SS)


def render(scene, outputs, poster=None):
    n_frames = round(LENGTH * FPS)
    W, H = FORMATS[scene.kind]
    procs = [subprocess.Popen(["ffmpeg", "-loglevel", "error", "-y", "-f", "rawvideo", "-pix_fmt", "rgb24",
                               "-s", f"{W}x{H}", "-r", str(FPS), "-i", "-", *args, str(out)], stdin=subprocess.PIPE)
             for out, args in outputs]
    first = scene.frame(0.0)
    loop_from = LENGTH - T_LOOP
    for i in range(n_frames):
        t = i / FPS
        img = scene.frame(t)
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
    tr = json.loads((HERE / "maze_inside_trace.json").read_text())
    assert tr["exact_match"] and tr["max_distance_from_whole_number"] == 0.0
    assert (tr["start_reached_at"], tr["readout_last_change"], tr["state_unchanged_from"]) == (111, 222, 223), \
        "the timeline's keys (111, 206-223) assume this trace"
    kinds = [a.only] if a.only else list(FORMATS)
    if a.frames:
        a.frames.mkdir(parents=True, exist_ok=True)
        picks = {"0_start": 0.5, "1_reads": 2.0, "2_wave": 4.4, "3_arrows": 8.0, "4_start": 11.3, "5_path": 16.0,
                 "6_end": LENGTH - T_LOOP - 0.05}
        for kind in kinds:
            sc = Scene(kind, tr)
            for name, t in picks.items():
                sc.frame(t).save(a.frames / f"inside_{kind}_{name}.png")
        print(f"frames in {a.frames}")
        return
    jobs = {"wide": [(HERE / "maze_inside.mp4", MP4), (HERE / "maze_inside.webm", WEBM)],
            "tall": [(HERE / "maze_inside_tall.mp4", MP4), (HERE / "maze_inside_tall.webm", WEBM)]}
    posters = {"wide": HERE / "maze_inside_poster.png", "tall": HERE / "maze_inside_tall_poster.png"}
    for kind in kinds:
        n = render(Scene(kind, tr), jobs[kind], posters[kind])
        print(f"{kind}: {n} frames, {n / FPS:.1f} s")


if __name__ == "__main__":
    main()
