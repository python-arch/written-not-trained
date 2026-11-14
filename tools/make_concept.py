"""Two-panel concept figure: Claude solving a maze itself (fails) vs Claude writing the network's weights (solves).

Usage: python3 -I make_concept.py <out.html>   (writes an inline <figure> with an <svg>)
"""
import random, sys
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


def claude(cx, cy, side="below"):
    """A generic gold 'mind' (not Anthropic's logo): a circle with three thinking dots."""
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
parts.append(claude(*ca))
parts.append(f'<path d="M{ca[0] + 40},{ca[1] - 6} C{ca[0] + 95},{ca[1] - 30} {mx_a - 50},{my - 20} {mx_a - 10},{my + 20}" fill="none" stroke="{GOLD}" stroke-width="2.5" stroke-dasharray="2 7" stroke-linecap="round"/>')
parts.append(maze(mx_a, my, wrong, GOLD, crash=wall))
parts.append(outcome(30, 376, "13 / 50", GOLD))

# panel B: Claude holds a control bar; strings run to the weight grid; the looped network solves the maze
cb = (PW + 20 + 110, 104)
parts.append(claude(*cb, side="right"))
bar_y, bx0, bx1 = 166, cb[0] - 70, cb[0] + 70
parts.append(f'<line x1="{cb[0]}" y1="{cb[1] + 34}" x2="{cb[0]}" y2="{bar_y}" stroke="{GOLD}" stroke-width="3" stroke-linecap="round"/>')
parts.append(f'<line x1="{bx0}" y1="{bar_y}" x2="{bx1}" y2="{bar_y}" stroke="{GOLD}" stroke-width="5" stroke-linecap="round"/>')
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
parts.append(outcome(PW + 48, 376, "1,000 / 1,000", NAVY))

svg = (f'<svg class="concept-svg" viewBox="0 0 {W} {H}" role="img" aria-label="Two panels. On the left, Claude is asked to solve a maze '
       f'itself, draws a path that goes wrong and cuts through a wall, and solves 13 of 50 test mazes. On the right, Claude holds a '
       f'control bar whose strings set the weights of a small network, which loops and finds the right path, solving all 1,000 test mazes.">'
       + "".join(parts) + "</svg>")
html = ('<figure class="concept">\n  ' + svg + '\n  <figcaption>When Claude solves a maze itself, it often goes wrong. When it writes '
        'the network instead, setting every weight by hand, the looped network finds the path in every test maze.</figcaption>\n</figure>')
open(sys.argv[1], "w", encoding="utf-8").write(html)
print("path lengths", len(good), len(wrong), "wall", wall)
