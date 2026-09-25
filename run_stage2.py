"""Stage 2 gates S1-S3 (+ reported S4 sweep). See GATES_STAGE2.md. Writes results/receipt_stage2.json."""
import json
import time
import numpy as np
from ra.world import make_junction_songs, render, tempo_profile, prefix_oracle
from ra.plastic import PlasticRA, prepare, train, accuracy

TEMPOS = [round(0.6 + 0.1 * i, 1) for i in range(11)]
ARMS = {"fixed": (False, False), "plastic-geom": (True, False), "plastic-geom+coupling": (True, True)}
SEEDS = [0, 1, 2]
NOISE = 0.003
t0 = time.time()
OUT = {}


def log(*a):
    print(f"[{time.time() - t0:6.0f}s]", *a, flush=True)


def world(pre, shared):
    songs, J = make_junction_songs(0, pre=pre, shared=shared)
    rng = np.random.default_rng(1)
    train_p = [prepare(render(s, tempo_profile("const", 1.0), rng)) for s in songs for _ in range(8)]
    r = np.random.default_rng(7)
    ev = {s: [prepare(render(x, tempo_profile("const", s), r)) for x in songs for _ in range(2)] for s in TEMPOS}
    ceil = float(np.mean([prefix_oracle(songs, s).mean() for s in songs]))
    return dict(songs=songs, J=J, train=train_p, ev=ev, ceil=ceil)


def fit(arm, W, seed, noise):
    lg, lc = ARMS[arm]
    m = PlasticRA(learn_geom=lg, learn_coupling=lc, seed=seed)
    return train(m, W["train"], noise=noise, seed=seed)


def geom(m):
    _, tau = m.poles()
    return {"tau": [round(float(v), 3) for v in tau.detach()], "nu": [round(float(v), 4) for v in m.nu.detach()]}


A = world(10, 6)
B = world(8, 18)
log("worlds ready; ceilings", A["ceil"], B["ceil"])

# ---------------- S1 / S2 on world A
resA = {arm: {s: [] for s in TEMPOS} for arm in ARMS}
geoms = {arm: [] for arm in ARMS}
for arm in ARMS:
    for seed in SEEDS:
        m = fit(arm, A, seed, NOISE)
        for s in TEMPOS:
            resA[arm][s].append(accuracy(m, A["ev"][s], noise=NOISE, seed=100 + seed))
        geoms[arm].append(geom(m))
        log(arm, seed, "mean", np.mean([resA[arm][s][-1] for s in TEMPOS]))
curve = {arm: {s: float(np.mean(v)) for s, v in resA[arm].items()} for arm in ARMS}
mean = {arm: float(np.mean(list(curve[arm].values()))) for arm in ARMS}
per_seed_mean = {arm: [float(np.mean([resA[arm][s][i] for s in TEMPOS])) for i in range(len(SEEDS))] for arm in ARMS}
pg = "plastic-geom"
OUT["ceiling_A"] = A["ceil"]
OUT["S1"] = {"mean_over_tempos": mean, "per_seed": per_seed_mean, "plastic_frac_of_ceiling": mean[pg] / A["ceil"],
             "PASS": bool(mean[pg] >= mean["fixed"] + 0.10 and mean[pg] / A["ceil"] >= 0.85)}
drop = curve[pg][1.0] - min(curve[pg].values())
OUT["S2"] = {"curve": curve, "plastic_worst_drop": drop, "PASS": bool(drop <= 0.10)}
OUT["learned_geometry"] = geoms
log("S1", OUT["S1"]); log("S2 drop", drop)

# ---------------- S3 on world B (18-beat shared segment)
post = [B["J"], B["J"] + 1, B["J"] + 2]
resB = {arm: [] for arm in ARMS}
for arm in ARMS:
    for seed in SEEDS:
        m = fit(arm, B, seed, NOISE)
        resB[arm].append(float(np.mean([accuracy(m, B["ev"][s], noise=NOISE, beats=post, seed=200 + seed)
                                        for s in (0.6, 1.0, 1.6)])))
        log("B", arm, seed, resB[arm][-1])
s3 = {arm: float(np.mean(v)) for arm, v in resB.items()}
OUT["S3"] = {"post_junction": s3, "per_seed": resB,
             "PASS": bool(s3[pg] >= 0.80 and s3[pg] >= s3["fixed"] + 0.15)}
log("S3", OUT["S3"])

# ---------------- S4 reported: noise sweep, seed 0
sweep = {}
for noise in [0.0, 0.003, 0.01, 0.02]:
    for arm in ARMS:
        mA = fit(arm, A, 0, noise)
        whole = float(np.mean([accuracy(mA, A["ev"][s], noise=noise, seed=300) for s in (0.6, 1.0, 1.6)]))
        mB = fit(arm, B, 0, noise)
        pj = float(np.mean([accuracy(mB, B["ev"][s], noise=noise, beats=post, seed=300) for s in (0.6, 1.0, 1.6)]))
        sweep[f"{arm}@{noise}"] = {"whole_song_A": whole, "post_junction_B": pj}
        log("sweep", arm, noise, sweep[f"{arm}@{noise}"])
OUT["S4_noise_sweep"] = sweep
OUT["summary"] = {g: OUT[g]["PASS"] for g in ("S1", "S2", "S3")}
log("SUMMARY", OUT["summary"])
json.dump(OUT, open("results/receipt_stage2.json", "w"), indent=1)
