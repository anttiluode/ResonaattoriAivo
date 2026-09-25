"""World construction shared by the Stage 2 runners."""
import numpy as np
from ra.world import make_junction_songs, render, tempo_profile, prefix_oracle
from ra.plastic import prepare

TEMPOS = [round(0.6 + 0.1 * i, 1) for i in range(11)]


def world(pre, shared):
    songs, J = make_junction_songs(0, pre=pre, shared=shared)
    rng = np.random.default_rng(1)
    train_p = [prepare(render(s, tempo_profile("const", 1.0), rng)) for s in songs for _ in range(8)]
    r = np.random.default_rng(7)
    ev = {s: [prepare(render(x, tempo_profile("const", s), r)) for x in songs for _ in range(2)] for s in TEMPOS}
    ceil = float(np.mean([prefix_oracle(songs, s).mean() for s in songs]))
    return dict(songs=songs, J=J, train=train_p, ev=ev, ceil=ceil)
