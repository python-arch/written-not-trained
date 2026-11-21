// Interactive explorer for the maze transformer (Figure 5 of the blog post).
// It runs the published weights (window.MAZE_MODEL, from demo/maze_model.js) in 64-bit numbers, with the same
// arithmetic as demo/maze.html, and splits every loop into the three things the block does:
//   1. read     the four attention heads copy 5 numbers from each neighbour into the cell's 20 scratch numbers
//   2. update   46 neurons update the cell's 8 search numbers (count, pointer, path flag)
//   3. clean up 40 neurons wipe the 20 scratch numbers
// In the real block, steps 2 and 3 happen in the same MLP at the same time; they are shown one after the other.
(function () {
  "use strict";
  const root = document.getElementById("xp");
  if (!root || !window.MAZE_MODEL) return;

  // ------------------------------------------------------------------ the model, as dense small matrices
  const M = window.MAZE_MODEL, D = M.config.d_model, CS = M.config.charset;   // "# SGo"
  const dense = (t) => { const a = new Float64Array(t.shape.reduce((x, y) => x * y, 1)); t.idx.forEach((i, j) => { a[i] = t.val[j]; }); return a; };
  const EMB = M.embed.map(dense), L0 = M.layers[0];
  const HEADS = L0.heads.map((h) => {
    const H = { Wq: dense(h.Wq), Wk: dense(h.Wk), Wv: dense(h.Wv), Wo: dense(h.Wo), dk: h.Wq.shape[1] };
    const used = (W, b) => { for (let a = 0; a < D; a++) if (W[a * H.dk + b]) return true; return false; };
    H.qk = [...Array(H.dk).keys()].filter((b) => used(H.Wq, b) && used(H.Wk, b));   // q.k sums only these dims
    H.vd = [...Array(H.dk).keys()].filter((b) => used(H.Wv, b));                     // v is zero elsewhere
    return H;
  });
  const W1 = dense(L0.W1), B1 = dense(L0.b1), W2 = dense(L0.W2), NH = L0.W1.shape[1];   // 86 neurons
  const WOUT = dense(M.W_out), NV = M.W_out.shape[1];
  // where things live in a cell's 35 numbers (read from the weights; see the math boxes)
  const IDX = { one: 0, row: 1, col: 2, sq: 3, wall: 4, start: 5, goal: 6, m: [7, 8, 9], p: [10, 11, 12, 13], path: 14, scratch: 15 };
  const DIRS = [[1, 0, "down", "below"], [0, -1, "left", "to the left"], [0, 1, "right", "to the right"], [-1, 0, "up", "above"]];
  const ARROW = ["↓", "←", "→", "↑"];
  const HEAD_COL = ["#9002FF", "#9002FF", "#9002FF", "#9002FF"];      // one strong accent, as in the visual guides
  const INKC = "#2c2c2c", PURPLE = "#9002FF", LAV = "#EEDAFF", PEACH = "#F4D7C1", PINK = "#D81B60";
  const GROUPS = [
    { name: "wave", from: 0, to: 6, col: "#9002FF", rule: "Give the cell count k+1 when it is not a wall, has no count yet, and a neighbour has count k." },
    { name: "pointer", from: 6, to: 30, col: "#0847A6", rule: "Point at the first neighbour, in the order down, left, right, up, whose count is one less." },
    { name: "pointer reset", from: 30, to: 38, col: "#7FA8E0", rule: "Erase last loop's pointer, so the pointer neurons can write it afresh." },
    { name: "path", from: 38, to: 46, col: "#D81B60", rule: "Join the path when a neighbour already on the path points at this cell." },
    { name: "clean-up", from: 46, to: 86, col: "#7F7F7F", rule: "Subtract each of the 20 copied numbers, which sets them back to 0." }];

  // ------------------------------------------------------------------ mazes
  const EXAMPLE = ["###########", "#G#       #", "# # ### # #", "#   # #   #", "##### # ###", "#         #",
                   "# ####### #", "#     #   #", "# # # # # #", "#  S#   # #", "###########"];
  function randomMaze(side) {
    const k = (side - 1) / 2, g = Array.from({ length: side }, () => Array(side).fill("#"));
    const stack = [[1, 1]]; g[1][1] = " ";
    while (stack.length) {
      const [r, c] = stack[stack.length - 1];
      const opts = [[2, 0], [-2, 0], [0, 2], [0, -2]].filter(([dr, dc]) => r + dr > 0 && r + dr < side - 1 && c + dc > 0 && c + dc < side - 1 && g[r + dr][c + dc] === "#");
      if (!opts.length) { stack.pop(); continue; }
      const [dr, dc] = opts[Math.floor(Math.random() * opts.length)];
      g[r + dr / 2][c + dc / 2] = " "; g[r + dr][c + dc] = " "; stack.push([r + dr, c + dc]);
    }
    const walls = []; for (let r = 1; r < side - 1; r++) for (let c = 1; c < side - 1; c++) if (g[r][c] === "#" && (r % 2) !== (c % 2)) walls.push([r, c]);
    for (let e = 0; e < 3 && walls.length; e++) { const [r, c] = walls.splice(Math.floor(Math.random() * walls.length), 1)[0]; g[r][c] = " "; }
    const corners = [[1, 1], [1, side - 2], [side - 2, 1], [side - 2, side - 2]], G = corners[Math.floor(Math.random() * 4)];
    g[G[0]][G[1]] = "G";
    const dist = bfs(g.map((row) => row.join("")));
    let S = null, best = -1;
    for (let i = 0; i < side * side; i++) if (dist[i] > best && g[Math.floor(i / side)][i % side] === " ") { best = dist[i]; S = i; }
    g[Math.floor(S / side)][S % side] = "S";
    void k; return g.map((row) => row.join(""));
  }
  function bfs(rows) {                              // distance of every open cell to G (-1 if unreachable)
    const side = rows.length, n = side * side, dist = new Int32Array(n).fill(-1), q = [];
    for (let i = 0; i < n; i++) if (rows[Math.floor(i / side)][i % side] === "G") { dist[i] = 0; q.push(i); }
    for (let h = 0; h < q.length; h++) {
      const i = q[h], r = Math.floor(i / side), c = i % side;
      for (const [dr, dc] of DIRS) {
        const nr = r + dr, nc = c + dc; if (nr < 0 || nc < 0 || nr >= side || nc >= side) continue;
        const j = nr * side + nc; if (rows[nr][nc] === "#" || dist[j] >= 0) continue;
        dist[j] = dist[i] + 1; q.push(j);
      }
    }
    return dist;
  }

  // ------------------------------------------------------------------ the transformer, loop by loop
  let rows, side, n, R, frames, mids, acts, bfsDist, Dsg, reachedAt;
  function embed() {
    const H = new Float64Array(n * D);
    for (let i = 0; i < n; i++) {
      const tok = [Math.floor(i / side), i % side, CS.indexOf(rows[Math.floor(i / side)][i % side])];
      for (let f = 0; f < 3; f++) for (let c = 0; c < D; c++) H[i * D + c] += EMB[f][tok[f] * D + c];
    }
    return H;
  }
  const S_ = new Float64Array(4096);
  function attention(H) {                           // returns H + attention output (the "read" step)
    const out = H.slice();
    for (const h of HEADS) {
      const dk = h.dk, Q = new Float64Array(n * dk), K = new Float64Array(n * dk), V = new Float64Array(n * dk);
      for (let i = 0; i < n; i++) for (let a = 0; a < D; a++) {
        const x = H[i * D + a]; if (x === 0) continue;
        for (let b = 0; b < dk; b++) { Q[i * dk + b] += x * h.Wq[a * dk + b]; K[i * dk + b] += x * h.Wk[a * dk + b]; V[i * dk + b] += x * h.Wv[a * dk + b]; }
      }
      const sc = Math.sqrt(dk), o = new Float64Array(dk);
      for (let i = 0; i < n; i++) {
        let m = -Infinity;
        for (let j = 0; j < n; j++) { let s = 0; for (const b of h.qk) s += Q[i * dk + b] * K[j * dk + b]; s /= sc; S_[j] = s; if (s > m) m = s; }
        let sum = 0; o.fill(0);
        for (let j = 0; j < n; j++) { const x = S_[j] - m; if (x < -746) { S_[j] = 0; continue; } S_[j] = Math.exp(x); sum += S_[j]; }
        for (let j = 0; j < n; j++) if (S_[j]) { const p = S_[j] / sum; for (const b of h.vd) o[b] += p * V[j * dk + b]; }
        for (let b = 0; b < dk; b++) { const x = o[b]; if (x === 0) continue; for (let c = 0; c < D; c++) out[i * D + c] += x * h.Wo[b * D + c]; }
      }
    }
    return out;
  }
  function neurons(Hm) {                            // ReLU activations of the 86 neurons at every cell
    const A = new Float64Array(n * NH);
    for (let i = 0; i < n; i++) for (let u = 0; u < NH; u++) {
      let z = B1[u]; for (let a = 0; a < D; a++) { const w = W1[a * NH + u]; if (w) z += Hm[i * D + a] * w; }
      A[i * NH + u] = z > 0 ? z : 0;
    }
    return A;
  }
  function addNeurons(Hm, A, from, to) {            // Hm + sum over neurons [from, to) of activation x W2 row
    const out = Hm.slice();
    for (let i = 0; i < n; i++) for (let u = from; u < to; u++) {
      const a = A[i * NH + u]; if (!a) continue;
      for (let c = 0; c < D; c++) { const w = W2[u * D + c]; if (w) out[i * D + c] += a * w; }
    }
    return out;
  }
  function ensure(t) {                              // frames[0..t] exist; frame t = state after loop t
    while (frames.length <= t) {
      const k = frames.length, Hm = attention(frames[k - 1]), A = neurons(Hm);
      mids[k] = Hm; acts[k] = A; frames.push(addNeurons(Hm, A, 0, NH));
      for (let i = 0; i < n; i++) if (reachedAt[i] < 0 && countOf(frames[k], i) >= 0) reachedAt[i] = k;
    }
  }
  const countOf = (H, i) => { for (let k = 0; k < 3; k++) if (H[i * D + IDX.m[k]] > 0.5) return k; return -1; };
  const pointerOf = (H, i) => { for (let k = 0; k < 4; k++) if (H[i * D + IDX.p[k]] > 0.5) return k; return -1; };
  const onPath = (H, i) => H[i * D + IDX.path] > 0.5;
  function readoutOf(H) {
    const out = new Int32Array(n);
    for (let i = 0; i < n; i++) {
      let best = 0, bv = -Infinity;
      for (let v = 0; v < NV; v++) { let s = 0; for (let c = 0; c < D; c++) { const w = WOUT[c * NV + v]; if (w) s += H[i * D + c] * w; } if (s > bv) { bv = s; best = v; } }
      out[i] = best;
    }
    return out;
  }
  function load(newRows) {
    rows = newRows.slice(); side = rows.length; n = side * side; R = 2 * n;
    frames = [embed()]; mids = []; acts = []; reachedAt = new Int32Array(n).fill(-1);
    for (let i = 0; i < n; i++) if (countOf(frames[0], i) >= 0) reachedAt[i] = 0;
    bfsDist = bfs(rows); const s = rows.join("").indexOf("S"); Dsg = s >= 0 ? bfsDist[s] : -1;
    T = 0; PH = 0; sel = s >= 0 ? s : 0;
    el("xp-scrub").max = R; el("xp-R").textContent = R.toLocaleString("en-US");
    marks();
  }

  // ------------------------------------------------------------------ what is on screen
  // Cursor: loop T and phase PH. PH 0 = state after loop T (after the embedding when T = 0);
  // PH 1 = loop T+1 has read its neighbours; PH 2 = loop T+1 has updated its search numbers (scratch still full).
  let T = 0, PH = 0, sel = 0, detailed = true, playing = false, editing = false, brush = "wall", hover = -1, tween = 1, tweenStart = 0;
  function view() {                                 // {H, grid state source, phase name}
    if (PH === 0) { ensure(T); return { H: frames[T], prev: T > 0 ? frames[T - 1] : null, act: T > 0 ? acts[T] : null }; }
    ensure(T + 1);
    if (PH === 1) return { H: mids[T + 1], prev: frames[T], act: null };
    const H = frames[T + 1].slice(); for (let i = 0; i < n; i++) for (let c = IDX.scratch; c < D; c++) H[i * D + c] = mids[T + 1][i * D + c];
    return { H, prev: frames[T], act: acts[T + 1] };
  }
  const el = (id) => document.getElementById(id);

  // ------------------------------------------------------------------ drawing the grid
  const cv = el("xp-grid"), cx = cv.getContext("2d");
  let px = 0, cell = 0;
  function resize() {
    const w = Math.min(cv.parentElement.clientWidth, 560), dpr = window.devicePixelRatio || 1;
    cv.style.width = w + "px"; cv.style.height = w + "px"; cv.width = Math.round(w * dpr); cv.height = Math.round(w * dpr);
    cx.setTransform(dpr, 0, 0, dpr, 0, 0); px = w; cell = w / side; draw();
  }
  const TINT = ["#EEDAFF", "#F4D7C1", "#D7EBDD"], INK = ["#7B2CBF", "#A0522D", "#2E7D4F"];
  function rrect(x, y, w, h, r) { cx.beginPath(); cx.moveTo(x + r, y); cx.arcTo(x + w, y, x + w, y + h, r); cx.arcTo(x + w, y + h, x, y + h, r); cx.arcTo(x, y + h, x, y, r); cx.arcTo(x, y, x + w, y, r); cx.closePath(); }
  function arrow(x0, y0, x1, y1, col, w, head) {
    cx.strokeStyle = col; cx.fillStyle = col; cx.lineWidth = w; cx.lineCap = "round";
    cx.beginPath(); cx.moveTo(x0, y0); cx.lineTo(x1, y1); cx.stroke();
    const a = Math.atan2(y1 - y0, x1 - x0); cx.beginPath();
    cx.moveTo(x1 + Math.cos(a) * head * 0.4, y1 + Math.sin(a) * head * 0.4);
    cx.lineTo(x1 - Math.cos(a - 0.5) * head, y1 - Math.sin(a - 0.5) * head);
    cx.lineTo(x1 - Math.cos(a + 0.5) * head, y1 - Math.sin(a + 0.5) * head); cx.closePath(); cx.fill();
  }
  function draw() {
    if (!cell) return;
    const v = view(), H = v.H, e = Math.min(1, tween), fontPx = Math.max(9, Math.round(cell * 0.28));
    cx.clearRect(0, 0, px, px);
    for (let i = 0; i < n; i++) {
      const r = Math.floor(i / side), c = i % side, x = c * cell, y = r * cell, ch = rows[r][c];
      if (ch === "#") { cx.fillStyle = "#363636"; cx.fillRect(x, y, cell + 0.5, cell + 0.5); continue; }
      cx.fillStyle = "#ffffff"; cx.fillRect(x, y, cell, cell);
      const k = countOf(H, i), fresh = v.prev && countOf(v.prev, i) < 0 && k >= 0;
      if (onPath(H, i) && ch !== "S" && ch !== "G") {
        const newly = v.prev && !onPath(v.prev, i);
        cx.globalAlpha = newly ? e : 1; cx.fillStyle = PINK; rrect(x + 2, y + 2, cell - 4, cell - 4, cell * 0.14); cx.fill(); cx.globalAlpha = 1;
      } else if (k >= 0) {
        cx.globalAlpha = fresh ? e : 1; cx.fillStyle = TINT[k]; rrect(x + 2, y + 2, cell - 4, cell - 4, cell * 0.14); cx.fill(); cx.globalAlpha = 1;
      }
      if (k >= 0 && ch !== "G" && ch !== "S") {
        cx.font = `600 ${fontPx}px Inter, system-ui, sans-serif`; cx.textAlign = "left"; cx.textBaseline = "top";
        cx.fillStyle = onPath(H, i) ? "rgba(255,255,255,0.92)" : INK[k]; cx.fillText(String(k), x + cell * 0.1, y + cell * 0.08);
      }
      if (fresh) { cx.strokeStyle = `rgba(144,2,255,${0.75 * (1 - e) + 0.25})`; cx.lineWidth = 2; rrect(x + 2, y + 2, cell - 4, cell - 4, cell * 0.14); cx.stroke(); }
    }
    // grid lines, S and G
    cx.strokeStyle = "#EDEDED"; cx.lineWidth = 1;
    for (let k = 1; k < side; k++) { cx.beginPath(); cx.moveTo(k * cell, 0); cx.lineTo(k * cell, px); cx.moveTo(0, k * cell); cx.lineTo(px, k * cell); cx.stroke(); }
    for (const L of ["S", "G"]) {
      const i = rows.join("").indexOf(L); if (i < 0) continue;
      const x = (i % side) * cell, y = Math.floor(i / side) * cell;
      cx.fillStyle = INKC; rrect(x + 2, y + 2, cell - 4, cell - 4, cell * 0.16); cx.fill();
      cx.fillStyle = "#fff"; cx.font = `700 ${Math.round(cell * 0.42)}px Inter, system-ui, sans-serif`; cx.textAlign = "center"; cx.textBaseline = "middle";
      cx.fillText(L, x + cell / 2, y + cell / 2 + 1);
      const k = countOf(H, i);
      if (k >= 0) { cx.font = `600 ${fontPx}px Inter, system-ui, sans-serif`; cx.textAlign = "left"; cx.textBaseline = "top"; cx.fillStyle = "rgba(255,255,255,0.8)"; cx.fillText(String(k), x + cell * 0.1, y + cell * 0.08); }
    }
    // pointers
    for (let i = 0; i < n; i++) {
      const p = pointerOf(H, i); if (p < 0) continue;
      const r = Math.floor(i / side), c = i % side, x = (c + 0.5) * cell, y = (r + 0.5) * cell, [dr, dc] = DIRS[p];
      const col = onPath(H, i) || rows[r][c] === "S" || rows[r][c] === "G" ? "rgba(255,255,255,0.95)" : INKC;
      arrow(x - dc * cell * 0.1, y - dr * cell * 0.1, x + dc * cell * 0.28, y + dr * cell * 0.28, col, Math.max(1.8, cell * 0.055), cell * 0.17);
    }
    // the selected cell, and what its heads read in step 1
    if (sel >= 0 && !editing) {
      const r = Math.floor(sel / side), c = sel % side;
      if (PH === 1 && rows[r][c] !== "#") {
        DIRS.forEach(([dr, dc], h) => {
          const nr = r + dr, nc = c + dc; if (nr < 0 || nc < 0 || nr >= side || nc >= side) return;
          const wall = rows[nr][nc] === "#", t0 = Math.min(1, e * 1.25);
          const xs = (nc + 0.5) * cell, ys = (nr + 0.5) * cell, xe = (c + 0.5) * cell - dc * cell * 0.22, ye = (r + 0.5) * cell - dr * cell * 0.22;
          cx.globalAlpha = wall ? 0.35 : 1;
          arrow(xs, ys, xs + (xe - xs) * t0, ys + (ye - ys) * t0, HEAD_COL[h], Math.max(2, cell * 0.07), cell * 0.2);
          cx.globalAlpha = 1;
        });
      }
      cx.strokeStyle = PURPLE; cx.lineWidth = 3.5; rrect(c * cell + 1.75, r * cell + 1.75, cell - 3.5, cell - 3.5, cell * 0.16); cx.stroke();
    }
    if (hover >= 0 && hover !== sel) {
      const r = Math.floor(hover / side), c = hover % side;
      cx.strokeStyle = editing ? PINK : "rgba(144,2,255,0.45)"; cx.lineWidth = 2; rrect(c * cell + 1, r * cell + 1, cell - 2, cell - 2, cell * 0.16); cx.stroke();
    }
  }

  // ------------------------------------------------------------------ the side panel
  const NAMES = ["1", "row", "col", "r²+c²", "wall", "start", "goal", "≡0", "≡1", "≡2", "↓", "←", "→", "↑", "path"];
  const SCR = ["≡0", "≡1", "≡2", "path", "⤺"];
  function buildPanel() {
    const strip = el("xp-strip"); strip.innerHTML = "";
    const groups = [["What and where", 0, 7, "fixed"], ["Search state", 7, 15, "search"], ["Scratch, copied from the neighbours", 15, 35, "scratch"]];
    for (const [title, a, b, kind] of groups) {
      const g = document.createElement("div"); g.className = "xp-g xp-" + kind;
      g.innerHTML = `<div class="xp-gt">${title} <span>${b - a} numbers</span></div><div class="xp-cells"></div>`;
      const box = g.querySelector(".xp-cells");
      for (let d = a; d < b; d++) {
        const s = document.createElement("div"); s.className = "xp-n"; s.id = "xp-n" + d;
        if (kind === "scratch") { const h = Math.floor((d - 15) / 5); s.style.setProperty("--hc", HEAD_COL[h]); if ((d - 15) % 5 === 0) s.classList.add("xp-first"); }
        const lab = kind === "scratch" ? (((d - 15) % 5 === 0 ? ARROW[Math.floor((d - 15) / 5)] + " " : "") + SCR[(d - 15) % 5]) : NAMES[d];
        s.innerHTML = `<b></b><i>${lab}</i>`; box.appendChild(s);
      }
      strip.appendChild(g);
    }
    buildArch();
  }

  // ------------------------------------------------------------------ the transformer diagram (an SVG that follows the loop)
  const SVGNS = "http://www.w3.org/2000/svg";
  function sv(tag, attrs, parent, text) {
    const e = document.createElementNS(SVGNS, tag);
    for (const k in attrs) e.setAttribute(k, attrs[k]);
    if (text !== undefined) e.textContent = text;
    if (parent) parent.appendChild(e);
    return e;
  }
  function head(parent, x, y, dir, cls) {            // a small arrowhead pointing in dir (radians)
    const s = 7, a = dir;
    return sv("path", { d: `M${x},${y} L${x - s * Math.cos(a - 0.45)},${y - s * Math.sin(a - 0.45)} L${x - s * Math.cos(a + 0.45)},${y - s * Math.sin(a + 0.45)} Z`, class: "ah " + (cls || "") }, parent);
  }
  function box(parent, id, x, y, w, h, title, lines) {
    const g = sv("g", { id, class: "blk" }, parent);
    sv("rect", { x, y, width: w, height: h, rx: 7, class: "bx" }, g);
    sv("text", { x: x + w / 2, y: y + 20, class: "t1", "text-anchor": "middle" }, g, title);
    (lines || []).forEach((ln, k) => sv("text", { x: x + w / 2, y: y + 38 + 14 * k, class: "t2", "text-anchor": "middle" }, g, ln));
    return g;
  }
  function buildArch() {
    const host = el("xp-arch"); host.innerHTML = "";
    const svg = sv("svg", { viewBox: "0 0 1200 222", class: "xp-svg", role: "img",
      "aria-label": "The maze transformer. The cells are embedded as 35 numbers each, then one block runs again and again: an attention layer with four heads, each adding what it reads to the residual stream, then a layer of 86 ReLU neurons that also add to it. After the last loop, a linear readout writes one character per cell." }, host);
    const Y = 120, thick = (x1, x2, parent) => { sv("line", { x1, y1: Y, x2: x2 - 9, y2: Y, class: "wire" }, parent || svg); head(parent || svg, x2, Y, 0); };
    const peach = (id, x, w, title, sub) => {
      const g = sv("g", { id, class: "blk peach" }, svg);
      sv("rect", { x, y: Y - 33, width: w, height: 66, rx: 10, class: "bx" }, g);
      sv("text", { x: x + w / 2, y: Y - 6, class: "t1", "text-anchor": "middle" }, g, title);
      sv("text", { x: x + w / 2, y: Y + 14, class: "t2", "text-anchor": "middle" }, g, sub);
      return g;
    };
    peach("xp-a-in", 8, 112, "Input", "121 cells"); svg.querySelector("#xp-a-in text.t2").id = "xp-a-n";
    thick(120, 150);
    peach("xp-a-emb", 150, 140, "Embedding", "→ 35 numbers per cell");
    thick(290, 316);
    // the looped block: a dark container with white inner boxes
    sv("text", { x: 318, y: 24, class: "t0" }, svg, "Looped block");
    sv("text", { x: 432, y: 24, class: "t2 ital" }, svg, "the same weights every loop, no normalisation");
    const lt = sv("text", { x: 990, y: 24, class: "t2", "text-anchor": "end" }, svg);
    sv("tspan", {}, lt, "loop "); sv("tspan", { id: "xp-t", class: "num" }, lt, "0"); sv("tspan", {}, lt, " of "); sv("tspan", { id: "xp-R", class: "num" }, lt, "242");
    const F = sv("g", { id: "xp-a-block" }, svg);
    sv("rect", { x: 316, y: 36, width: 676, height: 172, rx: 16, class: "dark" }, F);
    sv("line", { x1: 316, y1: Y, x2: 992, y2: Y, class: "stream" }, F);
    const sk1 = sv("g", { id: "xp-a-skip1", class: "wireg" }, F);
    sv("path", { d: `M332,${Y} V70 H604 V${Y - 13}`, class: "skip" }, sk1); head(sk1, 604, Y - 13, Math.PI / 2, "skiph");
    const sk2 = sv("g", { id: "xp-a-skip2", class: "wireg" }, F);
    sv("path", { d: `M618,${Y} V70 H958 V${Y - 13}`, class: "skip" }, sk2); head(sk2, 958, Y - 13, Math.PI / 2, "skiph");
    sv("text", { x: 468, y: 66, class: "t3 lite ital", "text-anchor": "middle" }, F, "skip connection");
    sv("text", { x: 788, y: 66, class: "t3 lite ital", "text-anchor": "middle" }, F, "skip connection");
    for (const [id, cx] of [["xp-a-plus1", 604], ["xp-a-plus2", 958]]) {
      const g = sv("g", { id, class: "plus" }, F);
      sv("circle", { cx, cy: Y, r: 12 }, g); sv("text", { x: cx, y: Y + 5.5, "text-anchor": "middle", class: "t1" }, g, "+");
    }
    const A = sv("g", { id: "xp-a-attn", class: "blk inner" }, F);
    sv("rect", { x: 346, y: 80, width: 238, height: 112, rx: 11, class: "bx" }, A);
    sv("text", { x: 465, y: 101, class: "t1", "text-anchor": "middle" }, A, "Attention");
    ["below", "left", "right", "above"].forEach((nm, h) => {
      const x = 356 + h * 56, g = sv("g", { class: "chip", id: "xp-a-h" + h }, A);
      sv("rect", { x, y: 110, width: 51, height: 72, rx: 7 }, g);
      sv("text", { x: x + 25.5, y: 128, "text-anchor": "middle", class: "t2 strong" }, g, ARROW[h]);
      sv("text", { x: x + 25.5, y: 142, "text-anchor": "middle", class: "t3 strong" }, g, nm);
      sv("text", { x: x + 25.5, y: 160, "text-anchor": "middle", class: "t3", id: "xp-a-c" + h }, g, "");
      sv("text", { x: x + 25.5, y: 173, "text-anchor": "middle", class: "t3", id: "xp-a-d" + h }, g, "");
    });
    const Mx = 626, Mlp = sv("g", { id: "xp-a-mlp", class: "blk inner" }, F);
    sv("rect", { x: Mx, y: 80, width: 312, height: 112, rx: 11, class: "bx" }, Mlp);
    sv("text", { x: Mx + 156, y: 101, class: "t1", "text-anchor": "middle" }, Mlp, "MLP · 86 neurons");
    const layout = [[0, 0, 6], [1, 60, 12], [2, 168, 8], [3, 240, 8]];
    for (const [gi, dx, cols] of layout) {
      const G = GROUPS[gi], gx = Mx + 12 + dx;
      const g = sv("g", { class: "ngrp" }, Mlp); sv("title", {}, g, G.name + ". " + G.rule);
      sv("text", { x: gx, y: 120, class: "t3 strong" }, g, `${G.name}`);
      for (let u = G.from; u < G.to; u++) { const k = u - G.from; sv("circle", { cx: gx + 4 + 8 * (k % cols), cy: 130 + 8 * Math.floor(k / cols), r: 3.2, class: "dot", id: "xp-u" + u, style: `--nc:${G.col}` }, g); }
    }
    { const G = GROUPS[4], gx = Mx + 12, g = sv("g", { class: "ngrp" }, Mlp); sv("title", {}, g, G.name + ". " + G.rule);
      sv("text", { x: gx, y: 168, class: "t3 strong" }, g, G.name);
      for (let u = G.from; u < G.to; u++) { const k = u - G.from; sv("circle", { cx: gx + 64 + 8 * (k % 20), cy: 165 + 8 * Math.floor(k / 20), r: 3.2, class: "dot", id: "xp-u" + u, style: `--nc:${G.col}` }, g); } }
    // the loop back into the same block
    const lp = sv("g", { id: "xp-a-loop", class: "wireg loopg" }, F);
    sv("path", { d: `M972,${Y + 12} V200 H330 V${Y + 14}`, class: "loopwire" }, lp); head(lp, 330, Y + 14, -Math.PI / 2, "looph");
    sv("rect", { x: 560, y: 192, width: 180, height: 16, rx: 8, class: "looptag" }, lp);
    sv("text", { x: 650, y: 204, "text-anchor": "middle", class: "t3 lite strong" }, lp, "back into the same block");
    // readout
    thick(992, 1028);
    peach("xp-a-out", 1028, 166, "Readout", "35 → 5 characters per cell");
  }
  function archState() {
    const set = (id, on) => { const e = el(id); if (e) e.classList.toggle("on", !!on); };
    const emb = T === 0 && PH === 0, read = PH === 1, upd = PH === 2, clean = PH === 0 && T > 0;
    set("xp-a-in", emb); set("xp-a-emb", emb); set("xp-a-attn", read); set("xp-a-skip1", read); set("xp-a-plus1", read);
    set("xp-a-mlp", upd || clean); set("xp-a-skip2", upd || clean); set("xp-a-plus2", upd || clean); set("xp-a-loop", clean);
    set("xp-a-out", Dsg > 0 && T >= 2 * Dsg && PH === 0);
    el("xp-a-n").textContent = `${n} cells`;
    // what each head copies for the selected cell
    const v = view(), i = sel, r = Math.floor(i / side), c = i % side, isWall = rows[r][c] === "#";
    DIRS.forEach(([dr, dc], h) => {
      let a = "", b = "";
      if (read && !isWall) {
        const nr = r + dr, nc = c + dc, j = nr * side + nc;
        if (rows[nr][nc] === "#") a = "wall";
        else {
          const k = countOf(v.prev, j); a = k >= 0 ? "count " + k : "no count";
          const bits = []; if (onPath(v.prev, j)) bits.push("path"); if (pointerOf(v.prev, j) === [3, 2, 1, 0][h]) bits.push("→ here");
          b = bits.join(", ");
        }
      }
      el("xp-a-c" + h).textContent = a; el("xp-a-d" + h).textContent = b;
      el("xp-a-h" + h).classList.toggle("on", read && !isWall);
    });
  }
  const fmtNum = (x) => { const r = Math.round(x * 100) / 100; return Math.abs(r) < 1e-9 ? "0" : (Number.isInteger(r) ? String(r) : r.toFixed(2)); };
  function panel() {
    const v = view(), H = v.H, i = sel, r = Math.floor(i / side), c = i % side, ch = rows[r][c];
    const kind = ch === "#" ? "wall" : ch === "S" ? "the start" : ch === "G" ? "the goal" : "an open cell";
    el("xp-cellname").textContent = `Row ${r}, column ${c}, ${kind}`;
    for (let d = 0; d < D; d++) {
      const s = el("xp-n" + d), x = H[i * D + d], on = d >= 4 ? Math.abs(x) > 0.5 : false;
      s.querySelector("b").textContent = d < 4 ? fmtNum(x) : (Math.abs(x) < 1e-9 ? "" : fmtNum(x));
      s.classList.toggle("on", on); s.classList.toggle("lit", d >= 4 && on);
    }
    const A = v.act, showA = PH === 2 ? [0, 46] : (PH === 0 && T > 0 ? [46, 86] : null);
    for (let u = 0; u < NH; u++) {
      const d = el("xp-u" + u), a = A && showA && u >= showA[0] && u < showA[1] ? A[i * NH + u] : 0;
      d.style.opacity = a > 0 ? String(Math.min(1, 0.35 + a * 0.65)) : ""; d.classList.toggle("on", a > 0);
    }
    el("xp-nhint").textContent = PH === 2 ? "Lit: the update neurons firing at this cell in this loop." :
      (PH === 0 && T > 0 ? "Lit: the clean-up neurons, erasing the copied numbers." : "No neuron fires in this step.");
    // what the transformer would write if it stopped here
    const out = readoutOf(frames[Math.min(T, frames.length - 1)]), pre = el("xp-out");
    let html = "";
    for (let rr = 0; rr < side; rr++) {
      for (let cc = 0; cc < side; cc++) {
        const k = out[rr * side + cc], chr = CS[k];
        html += chr === "o" ? '<b>o</b>' : chr === " " ? '<i>·</i>' : chr === "#" ? '<u>#</u>' : `<s>${chr}</s>`;
      }
      html += "\n";
    }
    pre.innerHTML = html;
    const pathCells = [...out].filter((k) => CS[k] === "o").length;
    el("xp-outnote").textContent = pathCells ? `${pathCells} path cells so far.` : (Dsg > 0 && T < Dsg ? "Nothing yet. The path only starts once the wave visits S." : "No path cells.");
  }

  // ------------------------------------------------------------------ the narration
  function story() {
    const v = view(), H = v.H, i = sel, r = Math.floor(i / side), c = i % side, ch = rows[r][c];
    const loopNo = PH === 0 ? T : T + 1, reachedNow = [...Array(n).keys()].filter((j) => countOf(H, j) >= 0).length;
    let step, say;
    if (PH === 0 && T === 0) { step = "Embedding"; say = "Each cell's 35 numbers start with its position and type. The goal gets count 0, and the start cell gets a path flag of 1. Press play, or step through one loop at a time."; }
    else if (PH === 1) step = `Loop ${loopNo}, step 1 of 3, read`;
    else if (PH === 2) step = `Loop ${loopNo}, step 2 of 3, update`;
    else step = `Loop ${loopNo}, step 3 of 3, clean up`;
    if (!(PH === 0 && T === 0)) {
      if (ch === "#") say = "This is a wall. Walls do not get a count, so the wave goes around them.";
      else if (PH === 1) {
        const parts = DIRS.map(([dr, dc, , where], h) => {
          const nr = r + dr, nc = c + dc;
          if (rows[nr][nc] === "#") return `${ARROW[h]} a wall`;
          const k = countOf(v.prev, nr * side + nc), on = onPath(v.prev, nr * side + nc), back = pointerOf(v.prev, nr * side + nc) === [3, 2, 1, 0][h];
          return `${ARROW[h]} ${k >= 0 ? "count " + k : "no count"}${on ? ", on the path" : ""}${back ? ", points here" : ""}`;
        });
        say = `Each head copies five numbers from one neighbour into the scratch space. This cell reads: ${parts.join(" · ")}.`;
      } else if (PH === 2) {
        const before = v.prev, kb = countOf(before, i), ka = countOf(H, i), pb = pointerOf(before, i), pa = pointerOf(H, i);
        const bits = [];
        if (kb < 0 && ka >= 0) {
          const from = DIRS.map((d, h) => h).filter((h) => { const nr = r + DIRS[h][0], nc = c + DIRS[h][1]; return countOf(before, nr * side + nc) === (ka + 2) % 3; });
          bits.push(`The wave arrives. A neighbour ${from.map((h) => DIRS[h][3]).join(" and ")} has count ${(ka + 2) % 3}, so the wave neurons give this cell count ${ka}.`);
        }
        if (pa >= 0 && pa !== pb) {
          const closer = DIRS.map((d, h) => h).filter((h) => { const nr = r + DIRS[h][0], nc = c + DIRS[h][1]; return rows[nr][nc] !== "#" && countOf(before, nr * side + nc) === (kb + 2) % 3; });
          bits.push(closer.length > 1
            ? `Neighbours ${closer.map((h) => DIRS[h][3]).join(" and ")} are both one step closer. The pointer neurons pick ${DIRS[pa][2]}, the first in the order down, left, right, up.`
            : `The neighbour ${DIRS[pa][3]} is one step closer, so the pointer neurons point ${DIRS[pa][2]}.`);
        }
        if (!onPath(before, i) && onPath(H, i)) {
          const from = DIRS.map((d, h) => h).find((h) => { const nr = r + DIRS[h][0], nc = c + DIRS[h][1]; const j = nr * side + nc; return onPath(before, j) && pointerOf(before, j) === [3, 2, 1, 0][h]; });
          bits.push(`The neighbour ${from !== undefined ? DIRS[from][3] : ""} is on the path and points here, so this cell joins the path.`);
        }
        if (!bits.length) bits.push(ka >= 0 ? "Nothing new for this cell. Its count and pointer stay as they are." : "The wave has not visited this cell yet, so nothing changes.");
        say = bits.join(" ");
      } else {
        say = "The 40 clean-up neurons subtract every copied number, so the scratch space is empty again for the next loop. The search numbers stay at 0 or 1.";
      }
    }
    el("xp-step").textContent = step;
    el("xp-say").textContent = say;
    let status;
    if (Dsg < 0) status = "S cannot reach G here, so the wave does not visit S and the transformer leaves the maze unchanged.";
    else if (T < Dsg) status = `${reachedNow} cells visited. The wave visits S at loop ${Dsg}.`;
    else if (T < 2 * Dsg) status = `The wave visited S at loop ${Dsg}. The path grows one cell per loop and is complete at loop ${2 * Dsg}.`;
    else status = `Solved at loop ${2 * Dsg}. Every later loop leaves all the numbers as they are.`;
    el("xp-status").textContent = status;
    archState();
    el("xp-t").textContent = T.toLocaleString("en-US");
    el("xp-scrub").value = T;
  }
  function marks() {
    const m = el("xp-marks"); m.innerHTML = "";
    if (Dsg > 0) for (const [t, lab] of [[Dsg, `loop ${Dsg}, the wave visits S`], [2 * Dsg, `loop ${2 * Dsg}, the path is complete`]]) {
      const s = document.createElement("span"); s.style.left = (100 * t / R) + "%"; s.textContent = lab; s.title = lab;
      s.addEventListener("click", () => { pause(); go(t, 0); }); m.appendChild(s);
    }
  }
  function render() { story(); panel(); draw(); }

  // ------------------------------------------------------------------ moving through the loops
  function go(t, ph, animate) {
    T = Math.max(0, Math.min(R, t)); PH = T === R ? 0 : ph;
    tween = animate ? 0 : 1; tweenStart = performance.now(); render();
    if (animate) requestAnimationFrame(anim);
  }
  function anim(now) { tween = (now - tweenStart) / 380; draw(); if (tween < 1) requestAnimationFrame(anim); }
  function next(animate) {
    if (!detailed) return go(T + 1, 0, animate);
    if (PH === 0) return go(T, T >= R ? 0 : 1, animate);
    if (PH === 1) return go(T, 2, animate);
    return go(T + 1, 0, animate);
  }
  function prev() {
    if (!detailed) return go(T - 1, 0);
    if (PH === 2) return go(T, 1);
    if (PH === 1) return go(T, 0);
    if (T === 0) return go(0, 0);
    return go(T - 1, 2);
  }
  let timer = null;
  function speedMs() { const s = +el("xp-speed").value; return detailed ? [1400, 900, 550, 300, 160][s] : [500, 250, 120, 60, 25][s]; }
  function tick() {
    if (!playing) return;
    const end = Dsg > 0 ? Math.min(R, 2 * Dsg + 1) : R;
    if (T >= end) { pause(); return; }
    next(true); timer = setTimeout(tick, speedMs());
  }
  function play() { if (editing) toggleEdit(); playing = true; el("xp-play").textContent = "Pause"; el("xp-play").classList.add("on"); if (T >= (Dsg > 0 ? 2 * Dsg + 1 : R)) go(0, 0); tick(); }
  function pause() { playing = false; clearTimeout(timer); el("xp-play").textContent = "Play"; el("xp-play").classList.remove("on"); }

  // ------------------------------------------------------------------ editing the maze
  function toggleEdit() {
    editing = !editing; pause(); if (editing) { const keep = sel; load(rows); sel = keep; }
    root.classList.toggle("xp-editing", editing); el("xp-edit").textContent = editing ? "Done editing" : "Edit the maze";
    if (!editing) { const keep = sel; load(rows); sel = keep; }
    render();
  }
  function paint(i) {
    const r = Math.floor(i / side), c = i % side; if (r === 0 || c === 0 || r === side - 1 || c === side - 1) return;
    const g = rows.map((row) => row.split(""));
    if (brush === "wall") { if (g[r][c] === "S" || g[r][c] === "G") return; g[r][c] = paintTo; }
    else { for (const row of g) for (let k = 0; k < side; k++) if (row[k] === (brush === "start" ? "S" : "G")) row[k] = " "; g[r][c] = brush === "start" ? "S" : "G"; }
    const keep = sel; load(g.map((row) => row.join(""))); sel = keep; render();
  }
  let painting = false, paintTo = "#";
  const cellAt = (e) => { const b = cv.getBoundingClientRect(), c = Math.floor((e.clientX - b.left) / cell), r = Math.floor((e.clientY - b.top) / cell); return r >= 0 && c >= 0 && r < side && c < side ? r * side + c : -1; };
  cv.addEventListener("pointerdown", (e) => {
    const i = cellAt(e); if (i < 0) return;
    if (editing) { painting = true; const r = Math.floor(i / side), c = i % side; paintTo = rows[r][c] === "#" ? " " : "#"; paint(i); cv.setPointerCapture(e.pointerId); return; }
    sel = i; render();
  });
  cv.addEventListener("pointermove", (e) => { const i = cellAt(e); if (i !== hover) { hover = i; if (editing && painting && i >= 0 && brush === "wall") paint(i); draw(); } });
  cv.addEventListener("pointerup", () => { painting = false; });
  cv.addEventListener("pointerleave", () => { hover = -1; painting = false; draw(); });

  // ------------------------------------------------------------------ controls
  el("xp-play").addEventListener("click", () => (playing ? pause() : play()));
  el("xp-next").addEventListener("click", () => { pause(); next(true); });
  el("xp-prev").addEventListener("click", () => { pause(); prev(); });
  el("xp-reset").addEventListener("click", () => { pause(); go(0, 0); });
  el("xp-scrub").addEventListener("input", (e) => { pause(); go(+e.target.value, 0); });
  el("xp-detailed").addEventListener("change", (e) => { detailed = e.target.checked; if (!detailed) go(T, 0); });
  el("xp-edit").addEventListener("click", toggleEdit);
  el("xp-new").addEventListener("click", () => { pause(); if (editing) toggleEdit(); load(randomMaze(11)); render(); });
  el("xp-example").addEventListener("click", () => { pause(); if (editing) toggleEdit(); load(EXAMPLE); render(); });
  root.querySelectorAll("[data-brush]").forEach((b) => b.addEventListener("click", () => {
    brush = b.dataset.brush; root.querySelectorAll("[data-brush]").forEach((x) => x.classList.toggle("on", x === b));
  }));
  root.addEventListener("keydown", (e) => {
    if (e.target.tagName === "INPUT" && e.target.type === "range") return;
    if (e.key === " ") { e.preventDefault(); playing ? pause() : play(); }
    else if (e.key === "ArrowRight") { e.preventDefault(); pause(); next(true); }
    else if (e.key === "ArrowLeft") { e.preventDefault(); pause(); prev(); }
  });
  window.addEventListener("resize", resize);

  buildPanel(); load(EXAMPLE);
  // deep links for screenshots and tests: ?xp=loop,phase,cell
  const q = new URLSearchParams(location.search).get("xp");
  if (q) { const [t, ph, c] = q.split(",").map(Number); if (c >= 0) sel = c; go(t || 0, ph || 0); }
  resize(); render();
  root.dataset.ready = "1";
  window.__xp = { get T() { return T; }, readout: () => [...readoutOf((ensure(R), frames[R]))].map((k) => CS[k]).join(""), rows: () => rows.slice() };
})();
