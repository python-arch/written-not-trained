"""Copy the plotted curves for the training figure from the handoff into figures/maze_training.json."""
import json
src = json.load(open("/Users/python/Desktop/ai702_project/loopws/logs/fig1_gd/fig1_time_data.json"))
H = 3600.0
trm = [[0.0, 0.0]] + [[round(p["est_secs"] / H, 4), round(p["exact_pct"], 1), p["step"]] for p in src["trm"]["points"]]
order = ["base_lr3e-4", "base_lr1e-3", "base_lr3e-3", "loops2", "blocks2", "blocks4", "width4", "width16"]
gd = {}
for r in order:
    pts = sorted(src["gradient_descent_same_arch"]["runs"][r]["points"], key=lambda p: p["step"])
    gd[r] = dict(params=src["gradient_descent_same_arch"]["runs"][r]["config"]["params"],
                 points=[[round(p["est_secs"] / H, 4), p["exact_pct"], p["step"]] for p in pts])
std = [[round(p["est_secs"] / H, 4), round(p["exact_pct"], 1), p["step"]] for p in src["standard_transformer"]["points"]]
out = dict(_source="loopws logs/fig1_gd/fig1_time_data.json (2026-11-14). Hours = updates x time per update on one H200.",
           claude_at_hours=1.0, trm=trm, gd=gd, std=std,
           std_params=src["standard_transformer"]["config"]["params"],
           trm_released=round(src["trm"]["released_checkpoint"]["exact_pct"], 1))
json.dump(out, open("/Users/python/Desktop/written-not-trained-blog/figures/maze_training.json", "w"), indent=1)
print("trm", trm)
print("std", std, out["std_params"], "released", out["trm_released"])
for r, v in gd.items():
    print(r, v["params"], len(v["points"]), v["points"][-1], "max exact", max(p[1] for p in v["points"]))
