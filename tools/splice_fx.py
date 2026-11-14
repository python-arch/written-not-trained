"""Swap the page's old image charts for the inline instrument cards.

Usage: python3 -I splice_fx.py <index.html> <fx dir>
"""
import os, re, sys

page, fx = sys.argv[1], sys.argv[2]
s = open(page, encoding="utf-8").read()

# 1. fonts: every figure uses the page's sans-serif, so drop any monospace font load
s = s.replace("&family=JetBrains+Mono:wght@500", "")

# 2. styles (replace a previous copy if present)
css = open(os.path.join(fx, "fx.css"), encoding="utf-8").read()
s = re.sub(r"\n  /\* ---------- Instrument figures.*?(?=\n</style>)", "", s, flags=re.S)
s = s.replace("\n</style>", css.rstrip() + "\n</style>", 1)

# 3. figures: each old <figure> is found by the image it shows
OLD = {
    "maze_training": "figures/maze_training_data.svg",
    "maze_loops": "figures/maze_loops_data.svg",
    "written_vs_trained": "figures/written_vs_trained_data.svg",
    "glance": "figures/glance_data.svg",
    "hanoi": "figures/hanoi_data.svg",
    "maze_size": "figures/maze_size_data.svg",
    "maze_compute": "figures/maze_compute_data.svg",
    "maze_block": "figures/maze_block.svg",
    "ouro_clock": "figures/ouro_clock.svg",
}
for name, src in OLD.items():
    new = open(os.path.join(fx, f"{name}.html"), encoding="utf-8").read()
    pat_old = r'<figure[^>]*>(?:(?!</figure>).)*?' + re.escape(src) + r'.*?</figure>'
    pat_new = r'<figure class="fx reveal"[^>]*>(?:(?!</figure>).)*?aria-label="[^"]*"(?:(?!</figure>).)*?</figure>'
    s, n = re.subn(pat_old, lambda m: new, s, count=1, flags=re.S)
    if n == 0:  # already swapped once: replace the card that sits in the same place
        marker = {"maze_training": 'id="written-vs-trained-maze"'}.get(name)
        cands = [m for m in re.finditer(pat_new, s, flags=re.S) if f'url(#wash-{name})' in m.group(0)
                 or f'glow-{name}"' in m.group(0) or (marker and marker in m.group(0))]
        assert len(cands) == 1, (name, len(cands))
        m = cands[0]
        s = s[:m.start()] + new + s[m.end():]
    print("swapped", name)

# 4. tooltip script, once
JS = """<script>
  /* Instrument figures: one tooltip for every element with data-tip. */
  (function () {
    var tip = document.createElement("div"); tip.className = "fx-tip"; tip.setAttribute("aria-hidden", "true");
    document.body.appendChild(tip);
    function owner(e) { return e.target && e.target.closest ? e.target.closest(".fx [data-tip]") : null; }
    document.addEventListener("pointerover", function (e) { var t = owner(e); if (t) { tip.textContent = t.getAttribute("data-tip"); tip.classList.add("on"); } });
    document.addEventListener("pointermove", function (e) { if (tip.classList.contains("on")) { tip.style.left = e.clientX + "px"; tip.style.top = e.clientY + "px"; } });
    document.addEventListener("pointerout", function (e) { if (owner(e)) { tip.classList.remove("on"); } });
  })();
</script>
"""
if "Instrument figures: one tooltip" not in s:
    s = s.replace("</body>", JS + "</body>", 1)
open(page, "w", encoding="utf-8").write(s)
print("done")
