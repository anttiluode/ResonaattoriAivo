"""Post-hoc follow-ups, written AFTER the gate receipt. Not gates; reported as exploration.

A. GRU-big-1: the capable GRU (hidden 128) trained at ONE tempo only. This is the arm the
   main claim actually needs: the parameter-matched GRUs failed to train even at s = 1.0.
B. Junction length sweep: G4 failed because RA-noslow handled a 6-beat shared segment.
   How long must the shared segment be before the slow bands matter?
"""
import json
import numpy as np
from ra.world import make_junction_songs, render, tempo_profile, HOLD
from ra.model import ResonaattoriAivo, DEFAULT_MODES
from ra.baselines import train_gru, gru_probs

OUT = {}
TEMPOS = [round(0.6 + 0.1 * i, 1) for i in range(11)]

# ---------------- B: junction sweep (fast, numpy only)
sweep = {}
for shared in [6, 12, 18]:
    songs, J = make_junction_songs(0, pre=8, shared=shared)
    rng = np.random.default_rng(1)
    tr = [render(s, tempo_profile("const", 1.0), rng) for s in songs for _ in range(4)]
    row = {}
    for name, modes in [("RA", DEFAULT_MODES), ("RA-noslow", [m for m in DEFAULT_MODES if m[0] < 6])]:
        m = ResonaattoriAivo(modes=modes, seed=0).fit(tr, epochs=150)
        a = []
        for s in [0.6, 1.0, 1.6]:
            r = np.random.default_rng(7)
            for song in songs:
                rd = render(song, tempo_profile("const", s), r)
                p, _ = m.predict(rd)
                a.append((p.argmax(1)[J:J + 3] == rd["tokens"][J:J + 3]).mean())
        row[name] = float(np.mean(a))
    sweep[shared] = row
    print("shared", shared, row, flush=True)
OUT["B_junction_sweep_post3_acc"] = sweep

# ---------------- A: GRU-big trained at one tempo
songs, J = make_junction_songs(0)
net = train_gru(songs, H=128, tempo_range=(1.0, 1.0), seed=0)
curve = {}
for s in TEMPOS:
    r = np.random.default_rng(7)
    rds = [render(song, tempo_profile("const", s), r) for song in songs for _ in range(2)]
    pr = gru_probs(net, rds)
    curve[s] = float(np.mean([(p.argmax(1) == rd["tokens"]).mean() for p, rd in zip(pr, rds)]))
    print("GRU-big-1", s, curve[s], flush=True)
OUT["A_GRU-big-1_by_tempo"] = curve
json.dump(OUT, open("results/posthoc.json", "w"), indent=1)
