"""Post-hoc C (after receipt): does the short-memory model's junction recall survive state noise?
Hypothesis: RA-noslow reads song identity from e^-(L/3)-sized leftovers that standardization blows up;
small noise on the resonator state should erase them, while the slow bands keep the context.
First attempt without an sd floor was invalid: imag parts of nu=0 modes are exactly 0 in training,
so standardization multiplied the noise by 1e6. sd_floor=0.05 fixes that."""
import json, numpy as np
from ra.world import make_junction_songs, render, tempo_profile
from ra.model import ResonaattoriAivo, DEFAULT_MODES
out = {}
for shared in [6, 12, 18]:
    songs, J = make_junction_songs(0, pre=8, shared=shared)
    rng = np.random.default_rng(1)
    tr = [render(s, tempo_profile("const", 1.0), rng) for s in songs for _ in range(4)]
    row = {}
    for noise in [0.0, 0.003, 0.01]:
        for name, modes in [("RA", DEFAULT_MODES), ("RA-noslow", [m for m in DEFAULT_MODES if m[0] < 6])]:
            m = ResonaattoriAivo(modes=modes, seed=0, noise=noise, sd_floor=0.05).fit(tr, epochs=150)
            a, whole = [], []
            r = np.random.default_rng(7)
            for s in [0.6, 1.0, 1.6]:
                for song in songs:
                    rd = render(song, tempo_profile("const", s), r)
                    p, _ = m.predict(rd)
                    a.append((p.argmax(1)[J:J+3] == rd["tokens"][J:J+3]).mean())
                    whole.append((p.argmax(1) == rd["tokens"]).mean())
            row[f"{name}@noise{noise}"] = {"post_junction": float(np.mean(a)), "whole_song": float(np.mean(whole))}
            print(shared, name, noise, row[f"{name}@noise{noise}"], flush=True)
    out[shared] = row
json.dump(out, open("results/posthoc_noise.json", "w"), indent=1)
