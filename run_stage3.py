"""Stage 3 gates T1-T5 (see GATES_STAGE3.md). Writes results/receipt_stage3.json."""
import json
import time
import numpy as np
import torch
from run_stage2_common import world
from ra.world import render, tempo_profile
from ra.plastic import prepare, batch
from ra.twomodule import TwoModule, FAST, SLOW, train2, predict2, mean_event_shape

NOISE, SEEDS, T3 = 0.006, [0, 1, 2], (0.6, 1.0, 1.6)
G4 = float(np.exp(-4))
t0 = time.time()


def log(*a):
    print(f"[{time.time() - t0:6.0f}s]", *a, flush=True)


W = world(8, 18)
J = W["J"]
POST = [J, J + 1, J + 2]


def score(m, noise=NOISE, gain=1.0, **kw):
    post, whole = [], []
    for s in T3:
        yh, y = predict2(m, W["ev"][s], noise=noise, gain=gain, **kw)
        c = (yh == y).float()
        post.append(c[:, POST].mean().item())
        whole.append(c.mean().item())
    return float(np.mean(post)), float(np.mean(whole))


ARMS = {"none/FAST": ("none", FAST, 1.0), "identity/FAST": ("identity", FAST, 1.0),
        "state/FAST": ("state", FAST, 1.0), "none/SLOW": ("none", SLOW, 1.0),
        "state/SLOW": ("state", SLOW, 1.0), "state/FAST@4λ": ("state", FAST, G4)}
res = {a: {"post": [], "whole": []} for a in ARMS}
attacks = {"clamp": [], "shuffle": []}
models = {}
for seed in SEEDS:
    for arm, (mode, rm, gain) in ARMS.items():
        m = train2(TwoModule(mode, rm, seed=seed), W["train"], noise=NOISE, gain=gain, seed=seed)
        p, w = score(m, gain=gain)
        res[arm]["post"].append(p)
        res[arm]["whole"].append(w)
        log(seed, arm, "post", round(p, 3), "whole", round(w, 3))
        if arm == "state/FAST":
            models[seed] = m
            ms = mean_event_shape(m, W["train"])
            attacks["clamp"].append(score(m, attack="clamp", mean_shape=ms)[0])
            attacks["shuffle"].append(score(m, attack="shuffle")[0])
            log(seed, "attacks post: clamp", round(attacks["clamp"][-1], 3), "shuffle", round(attacks["shuffle"][-1], 3))

mean = {a: {k: float(np.mean(v)) for k, v in r.items()} for a, r in res.items()}
P = {a: mean[a]["post"] for a in ARMS}
OUT = {"per_seed": res, "mean": mean, "attacks_post_per_seed": attacks}
OUT["T1"] = {"PASS": bool(P["state/FAST"] >= P["none/FAST"] + 0.15 and P["state/FAST"] >= P["identity/FAST"] + 0.15)}
cl, sh = float(np.mean(attacks["clamp"])), float(np.mean(attacks["shuffle"]))
OUT["T2"] = {"clamp_post": cl, "shuffle_post": sh,
             "PASS": bool(P["state/FAST"] - cl >= 0.15 and P["state/FAST"] - sh >= 0.15)}
OUT["T4"] = {"PASS": bool(P["state/FAST@4λ"] <= P["none/FAST"] + 0.05)}
OUT["T5"] = {"gain_with_own_memory": P["state/SLOW"] - P["none/SLOW"],
             "PASS": bool(P["state/SLOW"] - P["none/SLOW"] < 0.03)}
log("T1", OUT["T1"], "T2", OUT["T2"], "T4", OUT["T4"], "T5", OUT["T5"])

# ---------------- T3 transplant
songs = W["songs"]
pairs = [(i, i + 1) for i in range(0, len(songs), 2)]
cases, follow = 0, 0
for seed, m in models.items():
    for s in T3:
        for a_, b_ in pairs:
            for a, b in ((a_, b_), (b_, a_)):
                pa = prepare(render(songs[a], tempo_profile("const", s), None, jitter=0))
                pb = prepare(render(songs[b], tempo_profile("const", s), None, jitter=0))
                ya, _ = predict2(m, [pa], seed=900 + seed)
                yb, _ = predict2(m, [pb], seed=900 + seed)
                if ya[0, J] != songs[a][J] or yb[0, J] != songs[b][J]:
                    continue
                # sender state from beta, receiver hears alpha: same events up to the prediction frame
                gen = torch.Generator().manual_seed(900 + seed)
                xb, db, _, _ = batch([pb])
                xa, _, _, _ = batch([pa])
                F = min(xa.shape[1], xb.shape[1])
                Eb = m.sender(xb, db, NOISE, gen)
                Eo = torch.zeros(1, xa.shape[1], 3)
                Eo[:, :F] = Eb[:, :F]
                yt, _ = predict2(m, [pa], E_override=Eo, seed=900 + seed)
                cases += 1
                follow += int(yt[0, J] == songs[b][J])
OUT["T3"] = {"cases": cases, "follows_transplanted_history": follow / max(cases, 1),
             "PASS": bool(cases > 0 and follow / cases >= 0.70)}
log("T3", OUT["T3"])

# ---------------- reported: gain curve and noise levels (seed 0)
rep = {}
for gname, g in (("e^-1", float(np.exp(-1))), ("e^-2", float(np.exp(-2)))):
    m = train2(TwoModule("state", FAST, seed=0), W["train"], noise=NOISE, gain=g, seed=0)
    rep[f"state/FAST gain {gname}"] = score(m, gain=g)
    log("gain", gname, rep[f"state/FAST gain {gname}"])
for noise in (0.003, 0.01):
    for mode in ("none", "state"):
        m = train2(TwoModule(mode, FAST, seed=0), W["train"], noise=noise, seed=0)
        rep[f"{mode}/FAST noise {noise}"] = score(m, noise=noise)
        log("noise", noise, mode, rep[f"{mode}/FAST noise {noise}"])
OUT["reported_seed0_(post,whole)"] = rep
OUT["summary"] = {g: OUT[g]["PASS"] for g in ("T1", "T2", "T3", "T4", "T5")}
log("SUMMARY", OUT["summary"])
json.dump(OUT, open("results/receipt_stage3.json", "w"), indent=1)
