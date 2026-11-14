"""Two-panel concept figure: Claude solving a maze itself (fails) vs Claude writing the network's weights (solves).

Usage: python3 -I make_concept.py <out.html>   (writes an inline <figure> with an <svg>)
"""
import math, random, sys
from collections import deque

NAVY, GOLD, INK, MUTED, LINE, SOFT, WALL, RED = "#154677", "#8a6a1f", "#1d1d1f", "#5f6368", "#d7d9dd", "#f6f6f7", "#3b4048", "#c0392b"
W, H, PW = 1000, 430, 490
N, CS = 11, 18           # maze side in cells, cell size in px
rng = random.Random(11)

# ---------------------------------------------------------------- a small perfect maze, then a few extra openings
g = [["#"] * N for _ in range(N)]
g[1][1] = " "
todo = [(1, 1)]
while todo:
    r, c = todo[-1]
    nxt = [(r + a, c + b) for a, b in ((2, 0), (-2, 0), (0, 2), (0, -2)) if 0 < r + a < N - 1 and 0 < c + b < N - 1 and g[r + a][c + b] == "#"]
    if not nxt:
        todo.pop(); continue
    r2, c2 = rng.choice(nxt)
    g[(r + r2) // 2][(c + c2) // 2] = g[r2][c2] = " "
    todo.append((r2, c2))
S, G = (1, 1), (N - 2, N - 2)


def bfs(a, b):
    prev, q = {a: None}, deque([a])
    while q:
        u = q.popleft()
        if u == b: break
        for dr, dc in ((1, 0), (0, -1), (0, 1), (-1, 0)):
            v = (u[0] + dr, u[1] + dc)
            if 0 <= v[0] < N and 0 <= v[1] < N and g[v[0]][v[1]] != "#" and v not in prev:
                prev[v] = u; q.append(v)
    path, u = [], b
    while u is not None:
        path.append(u); u = prev[u]
    return path[::-1]


good = bfs(S, G)
# a wrong path: follow the right one for a while, wander into a side branch, then cut through a wall
k = max(3, len(good) // 3)
wrong = good[:k]
cur, seen = wrong[-1], set(good)
for _ in range(6):
    opts = [(cur[0] + a, cur[1] + b) for a, b in ((1, 0), (0, 1), (-1, 0), (0, -1))
            if 0 <= cur[0] + a < N and 0 <= cur[1] + b < N and g[cur[0] + a][cur[1] + b] != "#" and (cur[0] + a, cur[1] + b) not in wrong and (cur[0] + a, cur[1] + b) not in seen]
    if not opts: break
    cur = opts[0]; wrong.append(cur)
# step into a wall next to the last cell
wall = next(((cur[0] + a, cur[1] + b) for a, b in ((0, 1), (1, 0), (0, -1), (-1, 0))
             if 0 < cur[0] + a < N - 1 and 0 < cur[1] + b < N - 1 and g[cur[0] + a][cur[1] + b] == "#"), None)


def maze(ox, oy, path, color, crash=None, solved=False):
    out = [f'<rect x="{ox - 6}" y="{oy - 6}" width="{N * CS + 12}" height="{N * CS + 12}" rx="10" fill="#fff" stroke="{LINE}"/>']
    for r in range(N):
        for c in range(N):
            if g[r][c] == "#":
                out.append(f'<rect x="{ox + c * CS + 1}" y="{oy + r * CS + 1}" width="{CS - 2}" height="{CS - 2}" rx="3" fill="{WALL}"/>')
    pts = " ".join(f"{ox + c * CS + CS / 2},{oy + r * CS + CS / 2}" for r, c in path)
    if crash:
        pts += f" {ox + crash[1] * CS + CS / 2},{oy + crash[0] * CS + CS / 2}"
    out.append(f'<polyline points="{pts}" fill="none" stroke="{color}" stroke-width="6" stroke-linecap="round" stroke-linejoin="round" opacity=".92"/>')
    for (r, c), lab in ((S, "S"), (G, "G")):
        out.append(f'<rect x="{ox + c * CS + 1}" y="{oy + r * CS + 1}" width="{CS - 2}" height="{CS - 2}" rx="4" fill="#fff" stroke="{INK}" stroke-width="1.5"/>')
        out.append(f'<text x="{ox + c * CS + CS / 2}" y="{oy + r * CS + CS / 2 + 4.5}" text-anchor="middle" class="cx-s">{lab}</text>')
    if crash:
        cx, cy = ox + crash[1] * CS + CS / 2, oy + crash[0] * CS + CS / 2
        out.append(f'<circle cx="{cx}" cy="{cy}" r="15" fill="{RED}"/>')
        out.append(f'<path d="M{cx - 6},{cy - 6} L{cx + 6},{cy + 6} M{cx + 6},{cy - 6} L{cx - 6},{cy + 6}" stroke="#fff" stroke-width="3" stroke-linecap="round"/>')
    if solved:
        cx, cy = ox + G[1] * CS + CS / 2 + 16, oy + G[0] * CS + CS / 2 + 16
        out.append(f'<circle cx="{cx}" cy="{cy}" r="15" fill="{NAVY}"/>')
        out.append(f'<path d="M{cx - 6.5},{cy + 0.5} L{cx - 2},{cy + 5} L{cx + 7},{cy - 5}" fill="none" stroke="#fff" stroke-width="3" stroke-linecap="round" stroke-linejoin="round"/>')
    return "".join(out)



ORANGE, ORANGE_D, DARK = "#d97757", "#b85f42", "#2b1d16"


def spark(cx, cy, r=40, seed=3):
    """A Claude-style orange spark (an original mascot body, not Anthropic's logo)."""
    rr = random.Random(seed)
    out = []
    n = 11
    for i in range(n):
        a = math.radians(i * 360 / n + rr.uniform(-6, 6) - 90)
        L = r * rr.uniform(0.82, 1.0)
        out.append(f'<line x1="{cx:.1f}" y1="{cy:.1f}" x2="{cx + L * math.cos(a):.1f}" y2="{cy + L * math.sin(a):.1f}" '
                   f'stroke="{ORANGE}" stroke-width="{r * 0.3:.1f}" stroke-linecap="round"/>')
    out.append(f'<circle cx="{cx}" cy="{cy}" r="{r * 0.5:.1f}" fill="{ORANGE}"/>')
    return "".join(out)


def claude_dumb(cx, cy):
    """The Claude that tries to solve the maze itself: googly eyes, tongue out, sweating, confused."""
    o = [spark(cx, cy, 40, seed=3)]
    o.append(f'<circle cx="{cx - 9}" cy="{cy - 3}" r="7" fill="#fff" stroke="{DARK}" stroke-width="1.6"/>')
    o.append(f'<circle cx="{cx - 12.5}" cy="{cy + 0.5}" r="3" fill="{DARK}"/>')
    o.append(f'<circle cx="{cx + 9}" cy="{cy - 6}" r="4.6" fill="#fff" stroke="{DARK}" stroke-width="1.6"/>')
    o.append(f'<circle cx="{cx + 10.5}" cy="{cy - 8}" r="2" fill="{DARK}"/>')
    o.append(f'<path d="M{cx - 9},{cy + 8} Q{cx},{cy + 6} {cx + 9},{cy + 10}" fill="none" stroke="{DARK}" stroke-width="2.4" stroke-linecap="round"/>')
    o.append(f'<path d="M{cx + 1},{cy + 9} q2,9 7,5 q2,-3 -1,-6 z" fill="#e8798f" stroke="{DARK}" stroke-width="1.2"/>')
    o.append(f'<path d="M{cx + 30},{cy - 26} q-6,9 0,13 q6,-4 0,-13 z" fill="#8fc8ea"/>')
    for (dx, dy, fs, rot) in ((40, -34, 26, 12), (54, -6, 18, -10), (-50, -36, 20, -14)):
        o.append(f'<text x="{cx + dx}" y="{cy + dy}" font-size="{fs}" font-weight="800" fill="{ORANGE_D}" '
                 f'transform="rotate({rot} {cx + dx} {cy + dy})" class="cx-q">?</text>')
    o.append(f'<text x="{cx}" y="{cy + 64}" text-anchor="middle" class="cx-name">Claude</text>')
    return "".join(o)


def claude_villain(cx, cy):
    """The Claude that writes the weights: a scheming mastermind with a monocle."""
    o = [spark(cx, cy, 40, seed=8)]
    o.append(f'<line x1="{cx - 17}" y1="{cy - 14}" x2="{cx - 4}" y2="{cy - 8}" stroke="{DARK}" stroke-width="3.4" stroke-linecap="round"/>')
    o.append(f'<line x1="{cx + 17}" y1="{cy - 14}" x2="{cx + 4}" y2="{cy - 8}" stroke="{DARK}" stroke-width="3.4" stroke-linecap="round"/>')
    for ex in (-9, 9):
        o.append(f'<ellipse cx="{cx + ex}" cy="{cy - 1}" rx="5.2" ry="3.6" fill="#fff" stroke="{DARK}" stroke-width="1.4"/>')
        o.append(f'<circle cx="{cx + ex + 1.5}" cy="{cy}" r="2.1" fill="{DARK}"/>')
    o.append(f'<circle cx="{cx + 9}" cy="{cy - 1}" r="8" fill="none" stroke="{DARK}" stroke-width="1.6"/>')
    o.append(f'<path d="M{cx + 16},{cy + 3} q8,8 2,20" fill="none" stroke="{DARK}" stroke-width="1" stroke-dasharray="1.5 2"/>')
    o.append(f'<path d="M{cx - 9},{cy + 9} Q{cx + 1},{cy + 15} {cx + 11},{cy + 6}" fill="none" stroke="{DARK}" stroke-width="2.6" stroke-linecap="round"/>')
    o.append(f'<path d="M{cx + 11},{cy + 6} l3,-3" stroke="{DARK}" stroke-width="2.6" stroke-linecap="round"/>')
    o.append(f'<text x="{cx + 52}" y="{cy + 5}" class="cx-name">Claude</text>')
    return "".join(o)


def glove(hx, hy):
    """A white cartoon glove gripping the control bar at (hx, hy)."""
    return (f'<rect x="{hx - 14}" y="{hy - 13}" width="28" height="20" rx="9" fill="#fff" stroke="{DARK}" stroke-width="1.8"/>'
            f'<path d="M{hx - 5},{hy - 12} v9 M{hx + 2},{hy - 12} v9 M{hx + 9},{hy - 11} v8" stroke="{DARK}" stroke-width="1.3" stroke-linecap="round"/>'
            f'<rect x="{hx - 10}" y="{hy - 20}" width="20" height="8" rx="3" fill="#fff" stroke="{DARK}" stroke-width="1.6"/>')

def claude(cx, cy, side="below"):
    """Plain gold 'mind' icon, no longer used: a circle with three thinking dots."""
    lab = (f'<text x="{cx}" y="{cy + 56}" text-anchor="middle" class="cx-name">Claude</text>' if side == "below" else
           f'<text x="{cx + 46}" y="{cy + 5}" class="cx-name">Claude</text>')
    return (f'<circle cx="{cx}" cy="{cy}" r="34" fill="{GOLD}"/>'
            f'<circle cx="{cx - 12}" cy="{cy}" r="4.5" fill="#fff"/><circle cx="{cx}" cy="{cy}" r="4.5" fill="#fff"/>'
            f'<circle cx="{cx + 12}" cy="{cy}" r="4.5" fill="#fff"/>' + lab)


def outcome(x, y, big, color):
    return (f'<text x="{x}" y="{y}" class="cx-big" fill="{color}">{big}</text>'
            f'<text x="{x}" y="{y + 24}" class="cx-small">test mazes solved</text>')


my = 150                      # maze top
mx_a, mx_b = 258, PW + 20 + 258
parts = [f'<rect x="0" y="0" width="{PW}" height="{H}" rx="18" fill="{SOFT}"/>',
         f'<rect x="{PW + 20}" y="0" width="{PW}" height="{H}" rx="18" fill="{SOFT}"/>',
         f'<text x="28" y="46" class="cx-title" fill="{INK}">Asked to solve the maze</text>',
         f'<text x="{PW + 48}" y="46" class="cx-title" fill="{NAVY}">Asked to write the network</text>']

# panel A
ca = (110, 140)
parts.append(claude_dumb(*ca))
parts.append(f'<path d="M{ca[0] + 40},{ca[1] - 6} C{ca[0] + 95},{ca[1] - 30} {mx_a - 50},{my - 20} {mx_a - 10},{my + 20}" fill="none" stroke="{GOLD}" stroke-width="2.5" stroke-dasharray="2 7" stroke-linecap="round"/>')
parts.append(maze(mx_a, my, wrong, GOLD, crash=wall))
parts.append(outcome(30, 376, "26%", GOLD))

# panel B: Claude holds a control bar; strings run to the weight grid; the looped network solves the maze
cb = (PW + 20 + 110, 104)
parts.append(claude_villain(*cb))
bar_y, bx0, bx1 = 166, cb[0] - 70, cb[0] + 70
for side in (-1, 1):
    hx = cb[0] + side * 52
    parts.append(f'<path d="M{cb[0] + side * 22},{cb[1] + 26} C{cb[0] + side * 46},{cb[1] + 30} {hx + side * 8},{bar_y - 40} {hx},{bar_y - 20}" fill="none" stroke="{ORANGE}" stroke-width="7" stroke-linecap="round"/>')
parts.append(f'<line x1="{bx0}" y1="{bar_y}" x2="{bx1}" y2="{bar_y}" stroke="{GOLD}" stroke-width="5" stroke-linecap="round"/>')
parts.append(glove(cb[0] - 52, bar_y + 2))
parts.append(glove(cb[0] + 52, bar_y + 2))
gx0, gy0, gc, gr, gs = cb[0] - 76, 200, 8, 4, 19
for i in range(gc):
    sx = bx0 + (bx1 - bx0) * i / (gc - 1)
    tx = gx0 + i * gs + 8
    parts.append(f'<line x1="{sx}" y1="{bar_y}" x2="{tx}" y2="{gy0}" stroke="{GOLD}" stroke-width="1" opacity=".75"/>')
wr = random.Random(5)
for r in range(gr):
    for c in range(gc):
        op = wr.choice((0.12, 0.25, 0.45, 0.7, 1.0))
        parts.append(f'<rect x="{gx0 + c * gs}" y="{gy0 + r * gs}" width="{gs - 3}" height="{gs - 3}" rx="3" fill="{NAVY}" opacity="{op}"/>')
parts.append(f'<text x="{gx0 + gc * gs / 2}" y="{gy0 + gr * gs + 18}" text-anchor="middle" class="cx-small">weights</text>')
# arrow with a loop from the weights to the maze
ax0, ay = gx0 + gc * gs + 4, gy0 + gr * gs / 2
parts.append(f'<path d="M{ax0},{ay} L{mx_b - 18},{ay}" stroke="{NAVY}" stroke-width="2.5" stroke-linecap="round"/>')
parts.append(f'<path d="M{mx_b - 26},{ay - 6} L{mx_b - 17},{ay} L{mx_b - 26},{ay + 6}" fill="none" stroke="{NAVY}" stroke-width="2.5" stroke-linecap="round" stroke-linejoin="round"/>')
lx, ly = (ax0 + mx_b - 18) / 2, ay - 22
parts.append(f'<path d="M{lx + 9},{ly - 7} A11,11 0 1,0 {lx + 11},{ly + 2}" fill="none" stroke="{NAVY}" stroke-width="2.2" stroke-linecap="round"/>')
parts.append(f'<path d="M{lx + 6},{ly - 10} L{lx + 10},{ly - 6} L{lx + 14},{ly - 11}" fill="none" stroke="{NAVY}" stroke-width="2.2" stroke-linecap="round" stroke-linejoin="round"/>')
parts.append(maze(mx_b, my, good, NAVY, solved=True))
parts.append(outcome(PW + 48, 376, "100%", NAVY))

svg = (f'<svg class="concept-svg" viewBox="0 0 {W} {H}" role="img" aria-label="Two panels. On the left, a confused cartoon Claude is asked to solve a maze '
       f'itself, draws a path that goes wrong and cuts through a wall, and solves 26% of test mazes. On the right, a scheming cartoon Claude holds a '
       f'control bar whose strings set the weights of a small network, which loops and finds the right path, solving 100% of test mazes.">'
       + "".join(parts) + "</svg>")
html = ('<figure class="concept">\n  ' + svg + '\n  <figcaption>When Claude solves a maze itself, it often goes wrong. When it writes '
        'the network instead, setting every weight by hand, the looped network finds the path in every test maze.</figcaption>\n</figure>')
open(sys.argv[1], "w", encoding="utf-8").write(html)
print("path lengths", len(good), len(wrong), "wall", wall)
