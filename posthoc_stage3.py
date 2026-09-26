"""Post-hoc after the Stage 3 receipt (not a gate): is the state/FAST gain from the sender's
LONG history, or from having a second trained processing stage? Same architecture and capacity,
but the sender only has the short modes (tau < 6 beats), so it has no more history than the receiver."""
import json
import numpy as np
from run_stage2_common import world
from ra.twomodule import TwoModule, FAST, train2, predict2

W = world(8, 18)
J = W["J"]
POST = [J, J + 1, J + 2]
out = {"post": [], "whole": []}
for seed in [0, 1, 2]:
    m = train2(TwoModule("state", FAST, seed=seed, sender_modes=FAST), W["train"], noise=0.006, seed=seed)
    p, w = [], []
    for s in (0.6, 1.0, 1.6):
        yh, y = predict2(m, W["ev"][s], noise=0.006)
        c = (yh == y).float()
        p.append(c[:, POST].mean().item())
        w.append(c.mean().item())
    out["post"].append(float(np.mean(p)))
    out["whole"].append(float(np.mean(w)))
    print(seed, out["post"][-1], out["whole"][-1], flush=True)
out["mean"] = {k: float(np.mean(v)) for k, v in out.items() if k in ("post", "whole")}
print(out["mean"])
json.dump(out, open("results/posthoc_stage3_fast_sender.json", "w"), indent=1)
