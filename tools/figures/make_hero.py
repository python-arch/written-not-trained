#!/usr/bin/env python3
"""Render the Tower of Hanoi hero animation: the hand-written looped transformer writing each move, next to the pegs.

    python3 figures/make_hero.py ANSWER.txt[.gz] [--square-out PATH.mp4] [--only wide|tall|square]
    python3 figures/make_hero.py ANSWER.txt[.gz] --frames FOLDER      # sample frames only

Inputs:
- ANSWER: the network's saved 20-disk answer, "moves = [[disk, from, to], ...]".
- figures/hero_trace.json (from make_hero_trace.py): what the real network output after passes 1-4 for the first
  moves, and its real attention.
Before anything is drawn, a checker in this file replays every move of the answer (the disk must be on top of its
peg and may never land on a smaller disk, and the goal must be reached). The mistake counter is its count.

What comes from data, and what doesn't:
- Every move, peg position and counter comes from the saved answer. In the fast part, each frame blends several real
  positions from its slice of time, like a camera shutter, so fast small disks look blurred instead of flickering.
- The pass-by-pass readouts and the equal attention lines come from the trace, for the first moves.
- Only the timing is staged: the first moves are slowed down, then playback speeds up, and slows again for the last
  moves. In the fast part the network is dimmed, not animated.

Outputs: figures/hanoi_hero(.mp4|.webm|_poster.png) (wide, for desktop),
figures/hanoi_hero_tall(.mp4|.webm|_poster.png) (stacked, for phones), and the square version (--square-out).
Needs Pillow, numpy and ffmpeg (libx264, libvpx-vp9).
"""
from __future__ import annotations

import argparse
import bisect
import gzip
import json
import math
import re
import subprocess
import sys
from functools import lru_cache
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont

HERE = Path(__file__).resolve().parent
FONTS = HERE / "fonts"
FPS = 60
SS = 2                      # draw at 2x, then scale down: smooth edges and sub-pixel motion
PEGS = "ABC"

BG = (255, 255, 255)
FG = (29, 29, 31)
MUTED = (95, 99, 104)
FAINT = (150, 154, 160)
LINE = (215, 217, 221)
PEG = (190, 194, 200)
TAG = (236, 237, 239)
DISK = ((203, 207, 213), (184, 189, 197))
ACCENT = (21, 70, 119)       # MBZUAI blue
ACCENT_SOFT = (234, 238, 243)

# ---- timeline (seconds)
T_INTRO = 0.7
T_MOVE = (2.8, 2.1)         # the first two moves, pass by pass
T_RAMP = 11.0               # the rest: speeds up, then slows down for the last moves
T_EASE_IN, T_EASE_OUT, V_LAST = 2.6, 1.3, 0.9
T_HOLD, T_LOOP = 2.2, 0.45
V_DETAIL = (0.9, 3.2)       # moves per second: full detail below the first, none above the second
# ---- inside one move (fractions of the move)
U_PASS = (0.04, 0.19, 0.39, 0.51)
U_DONE = 0.63               # passes finished
U_CHIP = 0.30               # the move appears in the output
U_FLY = 0.68                # the token goes back into the input, the disk moves
U_CLEAR = 0.84              # readouts clear for the next move


# --------------------------------------------------------------------------- small helpers
def clamp(x, a=0.0, b=1.0):
    return a if x < a else b if x > b else x


def smooth(x):
    x = clamp(x)
    return x * x * (3 - 2 * x)


def smoother(x):
    x = clamp(x)
    return x * x * x * (x * (6 * x - 15) + 10)


def ramp(x, a, b):
    return smooth((x - a) / (b - a))


def mix(c0, c1, a):
    a = clamp(a)
    return tuple(int(round(p + (q - p) * a)) for p, q in zip(c0, c1))


def detail(v):
    """1 when moves are slow enough to show the network's steps, 0 when they are too fast to follow."""
    if v <= 0:
        return 1.0
    lo, hi = math.log(V_DETAIL[0]), math.log(V_DETAIL[1])
    return 1 - smooth((math.log(v) - lo) / (hi - lo))


@lru_cache(maxsize=None)
def font(weight, size):
    return ImageFont.truetype(str(FONTS / f"Inter-{weight}.ttf"), max(8, int(round(size))))


# --------------------------------------------------------------------------- data
def read_moves(path: Path):
    raw = gzip.open(path, "rt").read() if path.suffix == ".gz" else path.read_text()
    body = raw.split("=", 1)[1] if "=" in raw else raw
    return [(int(d), int(f), int(t)) for d, f, t in re.findall(r"\[\s*(\d+)\s*,\s*(\d+)\s*,\s*(\d+)\s*\]", body)]


def check(moves, n):
    """Replay every move. Returns (move numbers that were mistakes, goal_reached)."""
    pegs = [list(range(n, 0, -1)), [], []]
    bad = []
    for i, (d, f, t) in enumerate(moves, 1):
        ok = (1 <= d <= n and f in (0, 1, 2) and t in (0, 1, 2) and f != t
              and pegs[f] and pegs[f][-1] == d and (not pegs[t] or pegs[t][-1] > d))
        if not ok:
            bad.append(i)
            continue
        pegs[t].append(pegs[f].pop())
    goal = pegs[0] == [] and pegs[1] == [] and pegs[2] == list(range(n, 0, -1))
    return bad, goal


def snapshots(moves, n, wanted):
    """Peg positions after each wanted number of moves."""
    need = sorted(set(wanted))
    out, pegs, j = {}, [list(range(n, 0, -1)), [], []], 0
    while j < len(need) and need[j] == 0:
        out[0] = tuple(tuple(p) for p in pegs)
        j += 1
    for i, (d, f, t) in enumerate(moves, 1):
        if j == len(need):
            break
        pegs[t].append(pegs[f].pop())
        while j < len(need) and need[j] == i:
            out[i] = tuple(tuple(p) for p in pegs)
            j += 1
    return out


# --------------------------------------------------------------------------- timeline
class Timeline:
    """Maps time to moves done (a real number: 3.4 means move 4 is 40% done) and to moves per second.

    After the first moves, playback speeds up smoothly and evenly per power of ten until the halfway move, then slows
    down the same way, so the last few moves are slow again. Moves done before halfway grow as exp(F); after halfway,
    the moves left shrink as exp(-F). F rises at a rate that eases in, holds, and eases out."""

    def __init__(self, total):
        self.M = total
        self.n0 = len(T_MOVE)
        self.t0 = T_INTRO + sum(T_MOVE)
        self.t_end = self.t0 + T_RAMP
        self.length = self.t_end + T_HOLD + T_LOOP
        self.half = total / 2
        self.F_half = math.log(self.half / self.n0)
        F_end = self.F_half + math.log(self.half + 1)
        tau = np.linspace(0.0, T_RAMP, 400001)
        e_in = np.clip(tau / T_EASE_IN, 0, 1)
        e_in = e_in * e_in * (3 - 2 * e_in)
        e_out = np.clip((tau - (T_RAMP - T_EASE_OUT)) / T_EASE_OUT, 0, 1)
        e_out = e_out * e_out * (3 - 2 * e_out)
        g0 = 1.0 / (T_MOVE[-1] * self.n0)          # matches the speed of the last slow move
        base, unit = g0 * (1 - e_in) + V_LAST * e_out, e_in - e_out
        dt = tau[1] - tau[0]
        integ = lambda y: float(np.sum((y[1:] + y[:-1]) / 2) * dt)
        S = (F_end - integ(base)) / integ(unit)
        g = base + S * unit
        F = np.concatenate([[0.0], np.cumsum((g[1:] + g[:-1]) / 2 * dt)])
        F *= F_end / F[-1]
        self.tau, self.g, self.F, self.F_end = tau, g, F, F_end
        self.t_mid = self.t0 + float(np.interp(self.F_half, F, tau))

    def at(self, t):
        if t < T_INTRO:
            return 0.0, 1.0 / T_MOVE[0]
        x = t - T_INTRO
        for i, d in enumerate(T_MOVE):
            if x < d:
                return i + x / d, 1.0 / d
            x -= d
        if x >= T_RAMP:
            return float(self.M), 0.0
        F = float(np.interp(x, self.tau, self.F))
        g = float(np.interp(x, self.tau, self.g))
        if F <= self.F_half:
            m = self.n0 * math.exp(F)
            return m, g * m
        left = (self.half + 1) * math.exp(-(F - self.F_half))      # moves left + 1
        return min(self.M + 1 - left, self.M - 1e-9), g * left


# --------------------------------------------------------------------------- layouts (output pixels)
FORMATS = {"wide": (1280, 720), "tall": (720, 900), "square": (1080, 1080)}
GEOM = {
    "wide": dict(
        left=48, right=1236, header_y=44, panel_title=(48, 104),
        labels="left", cx0=160, net_right=716, in_y=220, blk_y=420, out_y=0,
        th=40, box=(170, 66, 34), cells=(44, 72, 54, 12), chip_h=52,
        att_label=(48, 318, "la", "reads all\ntokens equally"),
        divider=((745, 96), (745, 604)), pegs=(770, 100, 1236, 560),
        count_lab_y=598, count_y=650, prog_y=668, foot_y=702,
        f=dict(head=26, tag=19, panel=20, lane=18, tok=17, start=17, box=19, loop=14, cell=21, cell_lab=14,
               note=16, chip=30, att=15, count=50, count_lab=18, peg=22, final=44, foot=17)),
    "tall": dict(
        left=36, right=684, header_y=42, panel_title=None,
        labels="above", cx0=36, net_right=684, in_y=160, blk_y=345, out_y=0,
        th=50, box=(220, 74, 36), cells=(40, 88, 64, 12), chip_h=60,
        att_label=(684, 262, "rs", "reads all tokens equally"),
        divider=((36, 474), (684, 474)), pegs=(36, 484, 684, 770),
        count_lab_y=800, count_y=846, prog_y=862, foot_y=890,
        f=dict(head=30, tag=22, panel=0, lane=22, tok=24, start=19, box=26, loop=18, cell=26, cell_lab=18,
               note=21, chip=34, att=19, count=50, count_lab=21, peg=24, final=50, foot=18)),
    "square": dict(
        left=54, right=1026, header_y=182, panel_title=None, title_y=46,
        labels="above", cx0=54, net_right=1026, in_y=310, blk_y=500, out_y=0,
        th=50, box=(240, 74, 36), cells=(48, 100, 64, 14), chip_h=60,
        att_label=(1026, 418, "rs", "reads all tokens equally"),
        divider=((54, 642), (1026, 642)), pegs=(54, 654, 1026, 884),
        count_lab_y=922, count_y=968, prog_y=986, foot_y=None,
        f=dict(head=30, tag=24, panel=0, lane=24, tok=24, start=21, box=26, loop=18, cell=26, cell_lab=18,
               note=22, chip=36, att=20, count=52, count_lab=22, peg=26, final=54, foot=0, title=54, sub=26)),
}


class Scene:
    def __init__(self, kind, moves, trace, bad):
        self.kind = kind
        W, H = FORMATS[kind]
        self.W, self.H = W * SS, H * SS
        g = GEOM[kind]
        k = SS
        self.g = g
        self.moves, self.M, self.n = moves, len(moves), 20
        self.steps = trace["steps"]
        self.n_start = self.steps[0]["tokens_so_far"]
        self.bad = bad
        for st in self.steps:          # the notes under the passes describe this pattern, so check it holds
            assert "end" in st["output_by_pass"]["1"] and st["identical_to_pass_2"] == ["3", "4"], st
        self.F = {name: font(w, g["f"][name] * k) for name, w in (
            ("head", "SemiBold"), ("tag", "SemiBold"), ("panel", "SemiBold"), ("lane", "SemiBold"),
            ("tok", "Medium"), ("start", "Regular"), ("box", "SemiBold"), ("loop", "Regular"), ("cell", "Bold"),
            ("cell_lab", "Regular"), ("note", "Regular"), ("chip", "Bold"), ("att", "Regular"), ("count", "Bold"),
            ("count_lab", "Regular"), ("peg", "SemiBold"), ("final", "Bold"), ("foot", "Regular"),
            ("title", "Bold"), ("sub", "Regular")) if g["f"].get(name)}
        self.st1, self.st2 = max(1, round(1.5 * k)), max(2, round(2.2 * k))
        tl = ImageDraw.Draw(Image.new("RGB", (1, 1))).textlength
        # ---- network geometry
        self.left, self.right = g["left"] * k, g["right"] * k
        self.cx0, self.nx1 = g["cx0"] * k, g["net_right"] * k
        self.y = {"in": g["in_y"] * k, "blk": g["blk_y"] * k}
        self.th = g["th"] * k
        self.start_txt = f"start · {self.n_start} tokens"
        self.sw = tl(self.start_txt, font=self.F["start"]) + 22 * k
        self.mw = tl("d20 A→B", font=self.F["tok"]) + 20 * k
        self.gap, self.ell = 8 * k, 26 * k
        self.tx0 = self.cx0 + self.sw + self.gap + self.ell
        self.pitch = self.mw + self.gap
        self.fit = max(2, int((self.nx1 - self.tx0 + self.gap) // self.pitch))
        self.ty0 = self.y["in"] - self.th / 2
        bw, bh, ld = (v * k for v in g["box"])
        self.box = (self.cx0, self.y["blk"] - bh / 2, self.cx0 + bw, self.y["blk"] + bh / 2)
        self.loop_d = ld
        cgap, cw, ch, cg = (v * k for v in g["cells"])
        self.cell_x = [self.box[2] + cgap + i * (cw + cg) for i in range(4)]
        self.cw, self.ch = cw, ch
        self.cy0 = self.y["blk"] - ch / 2
        bx0, by0, bx1, by1 = self.box
        inset, ym = 0.18 * bw, by1 - 0.2 * bh          # the marker runs along the inside bottom of the block
        self.loop_path = [(bx0 + inset, ym), (bx1 - inset, ym), (bx1 - inset, by1 + ld),
                          (bx0 + inset, by1 + ld), (bx0 + inset, ym)]
        seg = [math.dist(a, b) for a, b in zip(self.loop_path, self.loop_path[1:])]
        self.loop_len = sum(seg)
        self.loop_seg = seg
        self.q_exit = seg[0] / self.loop_len           # the marker leaves the block here
        # ---- pegs
        x0, y0, x1, y1 = (v * k for v in g["pegs"])
        self.px0, self.px1 = x0, x1
        pw = x1 - x0
        self.pegs_x = [x0 + pw * j / 6 for j in (1, 3, 5)]
        self.base_y = y1 - 40 * k
        self.top_y = y0 + 16 * k
        self.max_w = pw / 3 * 0.92
        self.min_w = self.max_w * 0.18
        self.disk_h = (self.base_y - self.top_y) / (self.n + 1.5)
        self.region = (int(x0 - 6 * k), int(self.top_y - 2.4 * self.disk_h), int(x1 + 6 * k), int(self.base_y + 12 * k))
        self.peg_bg = self.make_peg_bg()
        self.static = self.make_static()

    # ------------------------------------------------------------------ static parts
    def make_peg_bg(self):
        k = SS
        rx0, ry0, rx1, ry1 = self.region
        im = Image.new("RGB", (rx1 - rx0, ry1 - ry0), BG)
        d = ImageDraw.Draw(im)
        d.rectangle((self.px0 - rx0, self.base_y - ry0, self.px1 - rx0, self.base_y + 8 * k - ry0), fill=PEG)
        for x in self.pegs_x:
            d.rectangle((x - 4 * k - rx0, self.top_y - ry0, x + 4 * k - rx0, self.base_y - ry0), fill=PEG)
        return im

    def make_static(self):
        g, k, F = self.g, SS, self.F
        img = Image.new("RGB", (self.W, self.H), BG)
        dr = ImageDraw.Draw(img)
        if self.kind == "square":
            dr.text((self.left, g["title_y"] * k), "Written, Not Trained", font=F["title"], fill=FG, anchor="lt")
            dr.text((self.left, (g["title_y"] + 66) * k),
                    "Claude wrote this network by hand. A separate program checks every move.",
                    font=F["sub"], fill=MUTED, anchor="lt")
        dr.text((self.left, g["header_y"] * k), "Tower of Hanoi, 20 disks", font=F["head"], fill=FG, anchor="lm")
        dr.line([(a * k, b * k) for a, b in g["divider"]], fill=LINE, width=self.st1)
        if g["panel_title"]:
            x, y = g["panel_title"]
            dr.text((x * k, y * k), "Hand-written network", font=F["panel"], fill=FG, anchor="ls")
        for x, name in zip(self.pegs_x, PEGS):
            dr.text((x, self.base_y + 30 * k), name, font=F["peg"], fill=MUTED, anchor="mm")
        dr.text((self.px0, g["count_lab_y"] * k), f"Moves, of {self.M:,}", font=F["count_lab"], fill=MUTED,
                anchor="ls")
        dr.text((self.px1, g["count_lab_y"] * k), "Mistakes", font=F["count_lab"], fill=MUTED, anchor="rs")
        if g["foot_y"]:
            dr.text(((self.px0 + self.px1) / 2 if self.kind == "wide" else self.W / 2, g["foot_y"] * k),
                    "Every move checked by a separate program", font=F["foot"], fill=MUTED, anchor="ms")
        return img

    # ------------------------------------------------------------------ pegs
    def disk_rect(self, d, cx, ybottom):
        width = self.min_w + (self.max_w - self.min_w) * (d - 1) / (self.n - 1)
        return (cx - width / 2, ybottom - self.disk_h * 0.86, cx + width / 2, ybottom)

    def flight(self, before, move, frac):
        d, f, t = move
        x0, y0 = self.pegs_x[f], self.base_y - (len(before[f]) - 1) * self.disk_h
        x1, y1 = self.pegs_x[t], self.base_y - len(before[t]) * self.disk_h
        top = self.top_y - self.disk_h * 0.4
        up, across, down = smoother(frac / 0.36), smoother((frac - 0.26) / 0.48), smoother((frac - 0.64) / 0.36)
        return d, x0 + (x1 - x0) * across, y0 + (top - y0) * up + (y1 - top) * down

    def peg_layer(self, sample, states):
        """One real position of the pegs: (moves done, fraction of the next move, when its flight starts, accent)."""
        k_done, u, w0, acc_cur, acc_prev = sample
        rx0, ry0 = self.region[:2]
        im = self.peg_bg.copy()
        d = ImageDraw.Draw(im)
        st = states[k_done]
        cur = self.moves[k_done] if k_done < self.M else None
        prev = self.moves[k_done - 1][0] if k_done > 0 else None
        flying = None
        if cur and u >= w0:
            flying = self.flight(st, cur, (u - w0) / (1 - w0))
            st = [list(p) for p in st]
            st[cur[1]].pop()

        def color(disk):
            a = acc_cur if cur and disk == cur[0] else acc_prev if disk == prev else 0.0
            return mix(DISK[disk % 2], ACCENT, a)

        for p, stack in enumerate(st):
            for level, disk in enumerate(stack):
                x0, y0, x1, y1 = self.disk_rect(disk, self.pegs_x[p], self.base_y - level * self.disk_h)
                d.rectangle((x0 - rx0, y0 - ry0, x1 - rx0, y1 - ry0), fill=color(disk))
        if flying:
            disk, cx, yb = flying
            x0, y0, x1, y1 = self.disk_rect(disk, cx, yb)
            d.rectangle((x0 - rx0, y0 - ry0, x1 - rx0, y1 - ry0), fill=color(disk))
        return im

    def pegs_image(self, samples, states):
        if len(samples) == 1:
            return self.peg_layer(samples[0], states)
        acc = None
        for s in samples:
            a = np.asarray(self.peg_layer(s, states), dtype=np.float32)
            acc = a if acc is None else acc + a
        return Image.fromarray(np.clip(acc / len(samples) + 0.5, 0, 255).astype(np.uint8))

    # ------------------------------------------------------------------ network
    def loop_point(self, q):
        dist = q * self.loop_len
        for (a, b), L in zip(zip(self.loop_path, self.loop_path[1:]), self.loop_seg):
            if dist <= L:
                f = dist / L if L else 0
                return a[0] + (b[0] - a[0]) * f, a[1] + (b[1] - a[1]) * f
            dist -= L
        return self.loop_path[-1]

    def next_x(self, kd, u, scroll, done):
        """Left edge of the empty "next" slot. After a token lands, the slot slides one place right."""
        if done:                                   # the track ends at the last move written
            return self.tx0 + (self.M - 1 - scroll) * self.pitch
        return self.tx0 + (kd - 1 + (smooth(u / 0.12) if kd else 1.0) - scroll) * self.pitch

    def tok(self, j):
        d, f, t = self.moves[j - 1]
        return f"d{d} {PEGS[f]}→{PEGS[t]}"

    def draw_network(self, dr, tl, t, m, v):
        k, F = SS, self.F
        M = self.M
        done = m >= M
        kd = M if done else int(m)            # moves done
        u = 0.0 if done else m - kd
        cur = kd + 1                          # the move being written
        det = 1.0 if done else detail(v)
        traced = det if (not done and cur <= len(self.steps)) else 0.0
        dim = 0.35 + 0.65 * det               # the network dims in the fast part
        settle = ramp(t, tl.t_end, tl.t_end + 0.4) if done else 0.0     # after the last move lands
        reading = traced * ramp(u, U_PASS[0], U_PASS[0] + 0.04) * (1 - ramp(u, U_DONE, U_DONE + 0.05))
        g = self.g
        # ---- lane labels
        for key, name in (("in", "Input"), ("blk", "Layer")):
            col = mix(BG, MUTED, dim)
            if g["labels"] == "left":
                dr.text((self.left, self.y[key]), name, font=F["lane"], fill=col, anchor="lm")
            else:
                half = {"in": self.th, "blk": self.box[3] - self.box[1]}[key] / 2
                dr.text((self.left, self.y[key] - half - 12 * k), name, font=F["lane"], fill=col, anchor="ls")
        # ---- input tape: the start, then every move written so far; older ones slide left under the start
        scroll = max(0.0, (M if done else kd + smooth(u / 0.12)) - self.fit)
        w0 = U_FLY * det
        ta = det * dim
        boxes = []
        if ta > 0.01:
            na_ = 0.0 if done else ta * (smooth(u / 0.12) if kd else 1.0)
            x = self.next_x(kd, u, scroll, done)
            if na_ > 0.01 and x + self.mw <= self.nx1 + 2 * k:
                dashed_rect(dr, (x, self.ty0, x + self.mw, self.ty0 + self.th), mix(BG, FAINT, na_), self.st1,
                            6 * k, 5 * k)
                dr.text((x + self.mw / 2, self.y["in"]), "next", font=F["start"], fill=mix(BG, FAINT, na_),
                        anchor="mm")
            j0 = max(1, int(scroll) - 1)
            for j in range(j0, kd + 1):
                x = self.tx0 + (j - 1 - scroll) * self.pitch
                ax = clamp((x - (self.tx0 - 0.75 * self.pitch)) / (0.75 * self.pitch))
                if ax <= 0 or x > self.nx1:
                    continue
                newest = 0.0
                if j == kd:
                    newest = (1 - settle) if done else (1 - ramp(u, U_PASS[1], U_PASS[1] + 0.06))
                a = ta * ax
                dr.rectangle((x, self.ty0, x + self.mw, self.ty0 + self.th), fill=BG,
                             outline=mix(BG, mix(LINE, ACCENT, newest), a), width=self.st2 if newest > 0.3 else self.st1)
                dr.text((x + self.mw / 2, self.y["in"]), self.tok(j), font=F["tok"], fill=mix(BG, FG, a), anchor="mm")
                boxes.append(x + self.mw / 2)
            ea = ta * smooth(scroll)
            if ea > 0.01:
                dr.text((self.cx0 + self.sw + self.gap + self.ell / 2, self.y["in"]), "…", font=F["tok"],
                        fill=mix(BG, MUTED, ea), anchor="mm")
        fa = (1 - det) * (0 if done else 1)
        if fa > 0.01:                       # fast part: the input is too long to show token by token
            x = self.cx0 + self.sw + self.gap
            dr.rectangle((x, self.ty0, self.nx1, self.ty0 + self.th), outline=mix(BG, LINE, fa), width=self.st1)
            text_fixed(dr, (x + 14 * k, self.y["in"]), f"+ {kd:,} moves written", F["tok"], mix(BG, MUTED, fa),
                       anchor="lm")
        dr.rectangle((self.cx0, self.ty0, self.cx0 + self.sw, self.ty0 + self.th), fill=BG,
                     outline=mix(BG, LINE, dim), width=self.st1)
        dr.text((self.cx0 + self.sw / 2, self.y["in"]), self.start_txt, font=F["start"], fill=mix(BG, FG, dim),
                anchor="mm")
        # ---- attention: real and equal, from the block to every token
        bx0, by0, bx1, by1 = self.box
        if reading > 0.01:
            ox = (bx0 + bx1) / 2
            for cxm in [self.cx0 + self.sw / 2] + boxes:
                dr.line([(ox, by0), (cxm, self.ty0 + self.th)], fill=mix(BG, FAINT, reading), width=self.st1)
            ax_, ay_, anc, txt = g["att_label"]
            lab_col = mix(BG, MUTED, reading)
            if anc == "rs":
                tw = max(dr.textlength(s, font=F["att"]) for s in txt.split("\n"))
                dr.rectangle((ax_ * k - tw - 6 * k, ay_ * k - F["att"].size, ax_ * k + 2 * k, ay_ * k + 6 * k), fill=BG)
            dr.multiline_text((ax_ * k, ay_ * k), txt, font=F["att"], fill=lab_col, anchor=anc, spacing=3 * k)
        # ---- the block, drawn once, with its loop
        dr.rectangle(self.box, fill=mix(BG, ACCENT_SOFT, reading), outline=mix(BG, mix(LINE, FG, 0.25 + 0.75 * reading), dim),
                     width=self.st2)
        dr.text(((bx0 + bx1) / 2, self.y["blk"]), "transformer", font=F["box"], fill=mix(BG, FG, dim), anchor="mm")
        lp = self.loop_path
        lc = mix(BG, MUTED, dim)
        dr.line([(lp[1][0], by1), lp[2], lp[3], (lp[3][0], by1 + 9 * k)], fill=lc, width=self.st1, joint="curve")
        tip = (lp[3][0], by1 + 1 * k)
        dr.polygon([tip, (tip[0] - 6 * k, tip[1] + 11 * k), (tip[0] + 6 * k, tip[1] + 11 * k)], fill=lc)
        dr.text(((bx0 + bx1) / 2, by1 + self.loop_d / 2 + 1 * k), "loop", font=F["loop"], fill=lc, anchor="mm")
        # passes: one trip round the loop each
        p_now, q = 0, 0.0
        for p in range(4):
            a, b = U_PASS[p], (U_PASS[p + 1] if p < 3 else U_DONE)
            if a <= u < b:
                p_now, q = p + 1, smooth((u - a) / (b - a))
        if traced > 0.01 and p_now:
            mx, my = self.loop_point(q)
            r = 6.5 * k
            dr.ellipse((mx - r, my - r, mx + r, my + r), fill=mix(BG, ACCENT, traced))
        # ---- results after each pass (real readouts)
        cl = mix(BG, MUTED, dim)
        arr_y = self.y["blk"]
        arrow(dr, (bx1 + 6 * k, arr_y), (self.cell_x[0] - 8 * k, arr_y), mix(BG, LINE, dim * 1.4), self.st1, 9 * k)
        for i, x in enumerate(self.cell_x):
            p = i + 1
            ra_in = ramp(u, U_PASS[i] + (U_PASS[i + 1] - U_PASS[i] if i < 3 else U_DONE - U_PASS[i]) * self.q_exit,
                         U_PASS[i] + (U_PASS[i + 1] - U_PASS[i] if i < 3 else U_DONE - U_PASS[i]) * (self.q_exit + 0.2))
            ra = traced * ra_in * (1 - ramp(u, U_CLEAR, U_CLEAR + 0.08))
            focus = traced * (1 if p == p_now else 0) * ra_in
            edge = mix(LINE, FG, focus)
            if p == 4:                         # the answer is read here, then goes back into the input
                ans = traced * ramp(u, w0, w0 + 0.06 * (1 - w0)) * (1 - ramp(u, w0 + 0.45 * (1 - w0), 1.0))
                edge = mix(edge, ACCENT, ans)
                focus = max(focus, ans)
            dr.rectangle((x, self.cy0, x + self.cw, self.cy0 + self.ch), fill=BG,
                         outline=mix(BG, edge, dim), width=self.st2 if focus > 0.5 else self.st1)
            dr.text((x + self.cw / 2, self.cy0 - 8 * k), f"loop {p}", font=F["cell_lab"], fill=cl, anchor="ms")
            if ra > 0.01:
                o = self.steps[cur - 1]["output_by_pass"][str(p)]
                txt = "end" if "end" in o else f"{PEGS[o['from']]}→{PEGS[o['to']]}"
                col = FAINT if p == 1 else ACCENT if p == 2 else FG
                rise = (1 - ra_in) * 8 * k
                dr.text((x + self.cw / 2, self.y["blk"] + rise), txt, font=F["cell"], fill=mix(BG, col, ra), anchor="mm")
        notes = ("loop 1: not ready yet", "loop 2: the move appears", "loop 3: same answer", "loop 4: same answer")
        for i, txt in enumerate(notes):
            a0, a1 = U_PASS[i], (U_PASS[i + 1] if i < 3 else U_DONE + 0.08)
            na = traced * ramp(u, a0 + 0.02, a0 + 0.05) * (1 - ramp(u, a1 - 0.03, a1))
            if na > 0.01:
                dr.text((self.cell_x[0], self.cy0 + self.ch + 26 * k), txt, font=F["note"], fill=mix(BG, MUTED, na),
                        anchor="ls")
        # ---- the move it wrote becomes its next input: the token rises from the last pass into the "next" slot
        if not done and det > 0.01 and u >= w0:
            fu = smoother((u - w0) / (1 - w0) * 1.08)
            sx = self.cell_x[3] + self.cw / 2                                         # from just above the last pass
            sy = self.cy0 - 30 * k - self.th / 2
            ex, ey = self.tx0 + (kd - scroll) * self.pitch + self.mw / 2, self.y["in"]   # to the next input slot
            cx_, cy_ = sx + (ex - sx) * fu, sy + (ey - sy) * fu
            a = det * ramp(fu, 0.0, 0.12)
            dr.rectangle((cx_ - self.mw / 2, cy_ - self.th / 2, cx_ + self.mw / 2, cy_ + self.th / 2), fill=BG,
                         outline=mix(BG, ACCENT, a), width=self.st2)
            dr.text((cx_, cy_), self.tok(cur), font=F["tok"], fill=mix(BG, FG, a), anchor="mm")

    # ------------------------------------------------------------------ header tag, counters, ending
    def draw_tag(self, dr, tl, t, det):
        k, F = SS, self.F
        ws = ramp(t, tl.t_end, tl.t_end + 0.3)
        fast = 1.0 if t >= tl.t_mid else (1 - det if t >= T_INTRO + T_MOVE[0] * 0.5 else 0.0)
        items = [("move by move", (1 - ws) * (1 - fast), TAG, FG), ("sped up", (1 - ws) * fast, ACCENT, BG),
                 ("solved", ws, TAG, FG)]
        width = sum(w * dr.textlength(s, font=F["tag"]) for s, w, _, _ in items) + 30 * k
        fill = tuple(int(round(sum(w * c[i] for _, w, c, _ in items))) for i in range(3))
        y = self.g["header_y"] * k
        dr.rectangle((self.right - width, y - 19 * k, self.right, y + 19 * k), fill=fill)
        for s, w, _, col in items:
            if w > 0.02:
                dr.text((self.right - width / 2, y), s, font=F["tag"], fill=mix(fill, col, w), anchor="mm")

    def draw_counters(self, dr, m):
        k, g = SS, self.g
        done = self.M if m >= self.M else int(m)
        text_fixed(dr, (self.px0, g["count_y"] * k), f"{done:,}", self.F["count"], FG, anchor="ls")
        text_fixed(dr, (self.px1, g["count_y"] * k), f"{bisect.bisect_right(self.bad, done)}", self.F["count"], FG,
                   anchor="rs")
        y = g["prog_y"] * k
        dr.rectangle((self.px0, y, self.px1, y + 3 * k), fill=TAG)
        if done:
            dr.rectangle((self.px0, y, self.px0 + (self.px1 - self.px0) * done / self.M, y + 3 * k), fill=FG)

    def draw_final(self, img, a):
        if a <= 0:
            return img
        k, F = SS, self.F
        top = img.copy()
        dr = ImageDraw.Draw(top)
        cy = self.top_y + (self.base_y - self.top_y) * 0.36 + (1 - a) * 12 * k
        lines = (f"{self.M:,} moves", f"{len(self.bad)} mistakes")
        x_lo, x_hi = self.px0 + 6 * k, self.pegs_x[2] - self.max_w / 2 - 16 * k     # the empty space above A and B
        f = F["final"]
        f = font("Bold", f.size * min(1.0, (x_hi - x_lo) / max(dr.textlength(s, font=f) for s in lines)))
        cx = (x_lo + x_hi) / 2
        lh = f.size * 1.22
        fw = max(dr.textlength(s, font=f) for s in lines)
        dr.rectangle((cx - fw / 2 - 16 * k, cy - lh - 6 * k, cx + fw / 2 + 16 * k, cy + lh + 4 * k), fill=BG)
        dr.text((cx, cy - lh / 2), lines[0], font=f, fill=ACCENT, anchor="mm")
        dr.text((cx, cy + lh / 2), lines[1], font=f, fill=ACCENT, anchor="mm")
        return Image.blend(img, top, a)

    # ------------------------------------------------------------------ one frame
    def samples(self, tl, t):
        """Real peg positions blended into one frame. More of them when disks move fast (a camera shutter)."""
        def one(tt):
            m, v = tl.at(tt)
            if m >= self.M:
                return (self.M, 0.0, 1.0, 0.0, 0.0), 0.0, 1.0
            det = detail(v)
            kd = int(m)
            u = m - kd
            w0 = U_FLY * det
            acc_cur = det * ramp(u, U_CHIP, U_CHIP + 0.06)
            acc_prev = det * (1 - ramp(u, 0.0, 0.15))
            return (kd, u, w0, acc_cur, acc_prev), v, det

        s0, v, det = one(t)
        if v <= 0:
            return [s0]
        frames_per_flight = FPS * (1 - U_FLY * det) / v
        n = int(clamp(math.ceil(48 / max(frames_per_flight, 1e-6)), 1, 32))
        if n == 1:
            return [s0]
        shutter = 0.85 / FPS
        return [one(t - shutter / 2 + shutter * (i + 0.5) / n)[0] for i in range(n)]

    def frame(self, tl, t, samples, states):
        img = self.static.copy()
        dr = ImageDraw.Draw(img)
        m, v = tl.at(t)
        det = 1.0 if m >= self.M else detail(v)
        self.draw_network(dr, tl, t, m, v)
        self.draw_tag(dr, tl, t, det)
        img.paste(self.pegs_image(samples, states), self.region[:2])
        self.draw_counters(dr, m)
        img = self.draw_final(img, ramp(t, tl.t_end + 0.15, tl.t_end + 0.55))
        return img.reduce(SS)


# --------------------------------------------------------------------------- drawing helpers
def text_fixed(dr, xy, s, f, fill, anchor="ls"):
    """Text whose digits all take the same width, so counters don't wobble."""
    cell = dr.textlength("0", font=f)
    widths = [cell if c.isdigit() else dr.textlength(c, font=f) for c in s]
    x, y = xy
    if anchor[0] == "r":
        x -= sum(widths)
    elif anchor[0] == "m":
        x -= sum(widths) / 2
    for c, w in zip(s, widths):
        dr.text((x + (w - dr.textlength(c, font=f)) / 2, y), c, font=f, fill=fill, anchor="l" + anchor[1])
        x += w


def dashed_rect(dr, box, fill, width, dash, gap):
    x0, y0, x1, y1 = box
    for (a, b, horiz, c) in ((x0, x1, True, y0), (x0, x1, True, y1), (y0, y1, False, x0), (y0, y1, False, x1)):
        t = a
        while t < b:
            e = min(b, t + dash)
            dr.line([(t, c), (e, c)] if horiz else [(c, t), (c, e)], fill=fill, width=width)
            t = e + gap


def arrow(dr, p0, p1, fill, width, head):
    dr.line([p0, p1], fill=fill, width=width)
    ang = math.atan2(p1[1] - p0[1], p1[0] - p0[0])
    a1, a2 = ang + math.radians(150), ang - math.radians(150)
    dr.polygon([p1, (p1[0] + head * math.cos(a1), p1[1] + head * math.sin(a1)),
                (p1[0] + head * math.cos(a2), p1[1] + head * math.sin(a2))], fill=fill)


# --------------------------------------------------------------------------- render
MP4 = ["-c:v", "libx264", "-pix_fmt", "yuv420p", "-crf", "27", "-preset", "slow", "-tune", "animation",
       "-movflags", "+faststart", "-an"]
WEBM = ["-c:v", "libvpx-vp9", "-pix_fmt", "yuv420p", "-crf", "43", "-b:v", "0", "-row-mt", "1", "-an"]


def plan(scene, tl, times):
    per_frame = [scene.samples(tl, t) for t in times]
    need = {s[0] for ss in per_frame for s in ss} | {0}
    return per_frame, snapshots(scene.moves, scene.n, need)


def render(scene, tl, outputs, poster=None):
    n_frames = round(tl.length * FPS)
    times = [i / FPS for i in range(n_frames)]
    per_frame, states = plan(scene, tl, times)
    W, H = FORMATS[scene.kind]
    procs = [subprocess.Popen(["ffmpeg", "-loglevel", "error", "-y", "-f", "rawvideo", "-pix_fmt", "rgb24",
                               "-s", f"{W}x{H}", "-r", str(FPS), "-i", "-", *args, str(out)],
                              stdin=subprocess.PIPE) for out, args in outputs]
    first = scene.frame(tl, 0.0, per_frame[0], states)
    loop_from = tl.t_end + T_HOLD
    for t, smp in zip(times, per_frame):
        img = scene.frame(tl, t, smp, states)
        if poster and abs(t - (loop_from - 1 / FPS)) < 0.5 / FPS:
            img.save(poster, optimize=True)
        if t >= loop_from:                      # cross-fade back to the first frame, so the loop is seamless
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
    ap.add_argument("answer", type=Path)
    ap.add_argument("--square-out", type=Path)
    ap.add_argument("--only", choices=list(FORMATS))
    ap.add_argument("--frames", type=Path, help="write sample frames (PNG) to this folder and stop")
    a = ap.parse_args()

    moves = read_moves(a.answer)
    bad, goal = check(moves, 20)
    trace = json.loads((HERE / "hero_trace.json").read_text())
    traced = [(s["written"]["disk"], s["written"]["from"], s["written"]["to"]) for s in trace["steps"]]
    assert traced == moves[: len(traced)], "trace and answer disagree"
    print(f"checker: {len(moves):,} moves, {len(bad)} mistakes, goal reached: {goal}, "
          f"fewest possible: {len(moves) == 2 ** 20 - 1}; first {len(traced)} moves traced from the real network")
    tl = Timeline(len(moves))
    print(f"length {tl.length:.2f} s; halfway move at {tl.t_mid:.2f} s; last move lands at {tl.t_end:.2f} s")

    if a.frames:
        a.frames.mkdir(parents=True, exist_ok=True)
        t1 = T_INTRO
        picks = {"1_reading": t1 + T_MOVE[0] * 0.12, "2_pass2": t1 + T_MOVE[0] * 0.30,
                 "3_chip": t1 + T_MOVE[0] * 0.55, "4_moving": t1 + T_MOVE[0] * 0.86,
                 "5_move2": t1 + T_MOVE[0] + T_MOVE[1] * 0.45, "6_speeding": tl.t0 + 1.4,
                 "7_fast": tl.t_mid - 1.2, "8_halfway": tl.t_mid, "9_landing": tl.t_end - 0.45,
                 "10_end": tl.t_end + T_HOLD - 0.02}
        kinds = [a.only] if a.only else list(FORMATS)
        for kind in kinds:
            sc = Scene(kind, moves, trace, bad)
            per, states = plan(sc, tl, list(picks.values()))
            for (name, t), smp in zip(picks.items(), per):
                sc.frame(tl, t, smp, states).save(a.frames / f"{kind}_{name}.png")
        print(f"frames in {a.frames}")
        return

    jobs = {"wide": [(HERE / "hanoi_hero.mp4", MP4), (HERE / "hanoi_hero.webm", WEBM)],
            "tall": [(HERE / "hanoi_hero_tall.mp4", MP4), (HERE / "hanoi_hero_tall.webm", WEBM)]}
    posters = {"wide": HERE / "hanoi_hero_poster.png", "tall": HERE / "hanoi_hero_tall_poster.png"}
    if a.square_out:
        a.square_out.parent.mkdir(parents=True, exist_ok=True)
        jobs["square"] = [(a.square_out, MP4)]
    for kind, outs in jobs.items():
        if a.only and kind != a.only:
            continue
        n = render(Scene(kind, moves, trace, bad), tl, outs, posters.get(kind))
        print(f"{kind}: {n} frames, {n / FPS:.1f} s")


if __name__ == "__main__":
    main()
