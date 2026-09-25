"""Run pre-registered gates G0-G6 (see GATES.md) and write results/receipt.json."""
import json
import time
import numpy as np
from ra.world import (make_junction_songs, make_novel_songs, render, tempo_profile, prefix_oracle,
                      P, HOLD, COUNT_IN, SONG_BEATS)
from ra.model import ResonaattoriAivo, DEFAULT_MODES
from ra.baselines import train_gru, gru_probs, NGram, GRUNet, n_params

TEMPOS = [round(0.6 + 0.1 * i, 1) for i in range(11)]
SEED = 0
OUT = {}
t0 = time.time()


def log(*a):
    print(f"[{time.time() - t0:6.0f}s]", *a, flush=True)


songs, J = make_junction_songs(SEED)
novel = make_novel_songs(123, 40)
ceil = {i: prefix_oracle(songs, s) for i, s in enumerate(songs)}
CEIL = float(np.mean([c.mean() for c in ceil.values()]))
log(f"{len(songs)} songs, junction at beat {J}, ceiling {CEIL:.3f}")

# ---------------------------------------------------------------- training
rng = np.random.default_rng(SEED + 1)
train_rds = [render(s, tempo_profile("const", 1.0), rng) for s in songs for _ in range(4)]

models = {}
models["RA"] = ResonaattoriAivo(seed=SEED).fit(train_rds, epochs=150)
models["RA-frozen"] = ResonaattoriAivo(frozen_clock=True, seed=SEED).fit(train_rds, epochs=150)
models["RA-noslow"] = ResonaattoriAivo(modes=[m for m in DEFAULT_MODES if m[0] < 6], seed=SEED).fit(
    train_rds, epochs=150)
log("RA models trained")
ra_trainable = models["RA"].W.size

grus = {}
grus["GRU-1"] = train_gru(songs, H=32, tempo_range=(1.0, 1.0), seed=SEED)
log("GRU-1 trained")
grus["GRU-aug"] = train_gru(songs, H=32, tempo_range=(0.7, 1.4), seed=SEED)
log("GRU-aug trained")
grus["GRU-big"] = train_gru(songs, H=128, tempo_range=(0.7, 1.4), seed=SEED)
log("GRU-big trained")
ngram = NGram(songs, k=3)
OUT["params"] = {"RA_trainable": int(ra_trainable),
                 "RA_fixed_random": int(models["RA"].Win.size + models["RA"].bin.size),
                 **{k: n_params(v) for k, v in grus.items()}}
log("params", OUT["params"])


# ---------------------------------------------------------------- evaluation helpers
def eval_set(kind, s=1.0, reps=2, seed=7):
    r = np.random.default_rng(seed)
    return [(i, render(song, tempo_profile(kind, s), r)) for i, song in enumerate(songs) for _ in range(reps)]


def probs_all(rds):
    out = {name: [m.predict(rd)[0] for _, rd in rds] for name, m in models.items()}
    for name, net in grus.items():
        pr = gru_probs(net, [rd for _, rd in rds])
        out[name] = list(pr)
    out["ngram-oracle"] = [ngram.probs(list(rd["tokens"])) for _, rd in rds]
    return out


def acc(pr_list, rds, beats=None):
    a = []
    for p, (_, rd) in zip(pr_list, rds):
        c = p.argmax(1) == rd["tokens"]
        a.append(c[beats].mean() if beats is not None else c.mean())
    return float(np.mean(a))


# ---------------------------------------------------------------- G0 + G1 + G2
table = {}
clock_err, slips, n_runs = [], 0, 0
for s in TEMPOS:
    rds = eval_set("const", s)
    pr = probs_all(rds)
    table[s] = {k: acc(v, rds) for k, v in pr.items()}
    for _, rd in rds:                                   # clock diagnostics from RA
        _, clock = models["RA"].listen(rd)
        n_runs += 1
        onset_beats = [COUNT_IN + b for b, tok in enumerate(rd["tokens"]) if tok != HOLD]
        logged = clock.errors[2:]                        # clicks 3 and 4 come first
        slipped = False
        for (assigned, e), true_b in zip(logged, onset_beats):
            if true_b >= COUNT_IN + 8:
                clock_err.append(abs(e))
                if assigned != true_b:
                    slipped = True
        slips += slipped
    log(f"s={s}: " + "  ".join(f"{k} {v:.3f}" for k, v in table[s].items()))
OUT["accuracy_by_tempo"] = table
OUT["ceiling"] = CEIL

med_err = float(np.median(clock_err))
slip_rate = slips / n_runs
OUT["G0"] = {"median_abs_phase_error_beats": med_err, "slip_rate": slip_rate,
             "PASS": bool(med_err < 0.05 and slip_rate < 0.05)}

ra1, gru1 = table[1.0]["RA"], table[1.0]["GRU-1"]
OUT["G1"] = {"RA_at_1.0": ra1, "RA_frac_of_ceiling": ra1 / CEIL, "GRU-1_at_1.0": gru1,
             "PASS": bool(ra1 / CEIL >= 0.90 and ra1 >= gru1 - 0.05)}

mean = {k: float(np.mean([table[s][k] for s in TEMPOS])) for k in table[1.0]}
worst_ra = min(table[s]["RA"] for s in TEMPOS)
frozen_drop = table[1.0]["RA-frozen"] - mean["RA-frozen"]
g2_pass = (mean["RA"] / CEIL >= 0.85 and (ra1 - worst_ra) <= 0.10 and frozen_drop > 0.30)
if mean["RA"] > mean["GRU-aug"] + 0.05:
    verdict = "RA beats GRU-aug"
elif mean["RA"] >= mean["GRU-aug"] - 0.05:
    verdict = "RA matches GRU-aug"
else:
    verdict = "RA loses to GRU-aug"
OUT["G2"] = {"mean_over_tempos": mean, "RA_frac_of_ceiling": mean["RA"] / CEIL,
             "RA_worst_tempo_drop": ra1 - worst_ra, "RA-frozen_mean_drop": frozen_drop,
             "vs_GRU-aug": verdict, "PASS": bool(g2_pass)}
log("G0", OUT["G0"]); log("G1", OUT["G1"]); log("G2", OUT["G2"])

# ---------------------------------------------------------------- G3 tempo drift
g3 = {}
for kind in ["accel", "rit"]:
    rds = eval_set(kind)
    pr = probs_all(rds)
    g3[kind] = {k: acc(v, rds) for k, v in pr.items()}
OUT["G3"] = {**g3, "PASS": bool(all(g3[k]["RA"] / CEIL >= 0.80 for k in g3))}
log("G3", OUT["G3"])

# ---------------------------------------------------------------- G4 junctions
post = [J, J + 1, J + 2]
g4 = {}
for s in [1.0] + [t for t in TEMPOS if t != 1.0]:
    rds = eval_set("const", s)
    pr = probs_all(rds)
    g4[s] = {k: acc(v, rds, post) for k, v in pr.items()}
OUT["G4"] = {"at_1.0": g4[1.0], "mean_over_tempos": {k: float(np.mean([g4[s][k] for s in g4]))
                                                     for k in g4[1.0]},
             "PASS": bool(g4[1.0]["RA"] >= 0.80 and g4[1.0]["RA-noslow"] <= 0.60)}
log("G4", OUT["G4"])


# ---------------------------------------------------------------- G5 residue = surprise
def surprisal(m, rd):
    p, _ = m.predict(rd)
    return -np.log(p[np.arange(len(rd["tokens"])), rd["tokens"]] + 1e-9)


def auroc(pos, neg):
    pos, neg = np.asarray(pos), np.asarray(neg)
    return float(((pos[:, None] > neg[None, :]).sum() + 0.5 * (pos[:, None] == neg[None, :]).sum())
                 / (len(pos) * len(neg)))


r5 = np.random.default_rng(55)
ra = models["RA"]
fam = [surprisal(ra, render(s, tempo_profile("const", 1.0), r5)).mean() for s in songs for _ in range(2)]
nov = [surprisal(ra, render(s, tempo_profile("const", 1.0), r5)).mean() for s in novel]
au = auroc(nov, fam)
hits, trials, example = 0, 100, None
for k in range(trials):
    i = r5.integers(len(songs))
    s = list(songs[i])
    cand = [b for b in range(12, 28) if s[b] != HOLD]
    b = int(r5.choice(cand))
    s[b] = int(r5.choice([q for q in range(P) if q != s[b]]))
    sp = surprisal(ra, render(s, tempo_profile("const", r5.choice(TEMPOS)), r5))
    peak = 4 + int(np.argmax(sp[4:]))
    hits += abs(peak - b) <= 1
    if example is None:
        example = {"song": i, "beat": b, "surprisal": sp.tolist()}
OUT["G5"] = {"AUROC_novel_vs_familiar": au, "familiar_mean_surprisal": float(np.mean(fam)),
             "novel_mean_surprisal": float(np.mean(nov)), "substitution_localized": hits / trials,
             "PASS": bool(au >= 0.95 and hits / trials >= 0.80), "example": example}
log("G5", {k: v for k, v in OUT["G5"].items() if k != "example"})

# ---------------------------------------------------------------- G6 free-run
g6 = {}
r6 = np.random.default_rng(66)
for s in TEMPOS:
    accs = []
    for i, song in enumerate(songs):
        rd = render(song, tempo_profile("const", s), r6)
        pub = ra.free_run(rd, cue_beats=12)
        accs.append(float((pub == rd["tokens"][12:]).mean()))
    g6[s] = float(np.mean(accs))
OUT["G6"] = {"by_tempo": g6, "mean": float(np.mean(list(g6.values()))),
             "PASS": bool(g6[1.0] >= 0.80 and np.mean(list(g6.values())) >= 0.70)}
log("G6", OUT["G6"])

OUT["summary"] = {g: OUT[g]["PASS"] for g in ["G0", "G1", "G2", "G3", "G4", "G5", "G6"]}
log("SUMMARY", OUT["summary"])
json.dump(OUT, open("results/receipt.json", "w"), indent=1, default=float)
