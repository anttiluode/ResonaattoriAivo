"""S5 confirmation (see GATES_STAGE2.md). Writes results/receipt_stage2_s5.json."""
import json, time
import numpy as np
from run_stage2_common import world
from ra.plastic import PlasticRA, train, accuracy

ARMS = {"fixed": (False, False), "coupling-only": (False, True),
        "plastic-geom": (True, False), "plastic-geom+coupling": (True, True)}
SEEDS, NOISE, T3 = [3, 4, 5], 0.01, (0.6, 1.0, 1.6)
t0 = time.time()
A, B = world(10, 6), world(8, 18)
post = [B["J"], B["J"] + 1, B["J"] + 2]
res = {a: {"A_whole": [], "B_post": []} for a in ARMS}
for arm, (lg, lc) in ARMS.items():
    for seed in SEEDS:
        m = train(PlasticRA(learn_geom=lg, learn_coupling=lc, seed=seed), A["train"], noise=NOISE, seed=seed)
        res[arm]["A_whole"].append(float(np.mean([accuracy(m, A["ev"][s], noise=NOISE, seed=400 + seed) for s in T3])))
        m = train(PlasticRA(learn_geom=lg, learn_coupling=lc, seed=seed), B["train"], noise=NOISE, seed=seed)
        res[arm]["B_post"].append(float(np.mean([accuracy(m, B["ev"][s], noise=NOISE, beats=post, seed=500 + seed) for s in T3])))
        print(f"[{time.time()-t0:5.0f}s]", arm, seed, res[arm]["A_whole"][-1], res[arm]["B_post"][-1], flush=True)
mean = {a: {k: float(np.mean(v)) for k, v in r.items()} for a, r in res.items()}
fa, pa = res["fixed"]["A_whole"], res["plastic-geom"]["A_whole"]
OUT = {"per_seed": res, "mean": mean,
       "S5a": {"PASS": bool(mean["plastic-geom"]["A_whole"] >= mean["fixed"]["A_whole"] + 0.15
                            and all(p > f for p, f in zip(pa, fa)))},
       "S5b": {"PASS": bool(mean["plastic-geom+coupling"]["A_whole"] >= mean["coupling-only"]["A_whole"] + 0.05)}}
print("SUMMARY", mean, OUT["S5a"], OUT["S5b"], flush=True)
json.dump(OUT, open("results/receipt_stage2_s5.json", "w"), indent=1)
