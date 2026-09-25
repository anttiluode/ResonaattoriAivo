"""ResonaattoriAivo: entrained clock -> clock-slaved resonators -> pyramids -> residue-learned readout.

Pieces, named after the idea they stand in for:

  Clock      basket-like: a phase-locked loop seeded by the count-in. Its phase is time *in beats*.
  Resonators the dendrite/axon "resonance windows": damped complex modes per input channel whose
             decay tau and frequency nu are in beats, so they advance by the clock's phase, not by frames.
             Timing is geometry: tau and nu are the lengths.
  Pyramids   a fixed random tanh layer ("the initial layers set capability").
  Readout    learned only from the residue r = onehot(actual) - p (predict what comes, subtract it,
             learn from what is left).
  Chandelier winner-take-all per clock slot during free-run: exactly one token is published per beat.
"""
import numpy as np
from .world import P, HOLD, NTOK, NCH, T0, COUNT_IN, SONG_BEATS

# (tau, nu) in beats / cycles-per-beat.  Slow bands = tau >= 6.
DEFAULT_MODES = [(0.7, 0.0), (1.5, 0.0), (3.0, 0.0), (6.0, 0.0), (12.0, 0.0), (24.0, 0.0),
                 (3.0, 0.25), (8.0, 0.125)]


class Clock:
    def __init__(self, frozen=False, alpha=0.6, beta=0.15):
        self.frozen, self.alpha, self.beta = frozen, alpha, beta
        self.reset()

    def reset(self):
        self.phi = None          # beats since first click (None = not started)
        self.f = 1.0 / T0        # beats per frame
        self.clicks = []
        self.errors = []         # (phi_before_correction_target, error) log

    def step(self, t, onset_any, click):
        """Advance one frame. Returns dphi (beats advanced during this frame)."""
        if self.phi is None:
            if click:
                self.phi = 0.0
                self.clicks.append(t)
            return 0.0
        dphi = self.f
        self.phi += dphi
        if self.frozen:
            if click and len(self.clicks) < 1:
                self.clicks.append(t)
            return dphi
        if click and len(self.clicks) == 1:           # seed tempo from the first two clicks
            self.clicks.append(t)
            self.f = 1.0 / (t - self.clicks[0])
            self.phi = 1.0
            return dphi
        if onset_any:
            e = self.phi - round(self.phi)
            self.errors.append((round(self.phi), e))
            self.phi -= self.alpha * e
            self.f *= (1.0 - self.beta * e)
        return dphi


class ResonatorBank:
    def __init__(self, modes=DEFAULT_MODES, noise=0.0, rng=None):
        self.modes = list(modes)
        self.noise = noise
        self.rng = rng if rng is not None else np.random.default_rng()
        tau = np.array([m[0] for m in modes])
        nu = np.array([m[1] for m in modes])
        self.log_pole = (-1.0 / tau + 2j * np.pi * nu)[None, :]   # per beat
        self.reset()

    def reset(self):
        self.z = np.zeros((NCH, len(self.modes)), complex)

    def step(self, dphi, x):
        self.z = self.z * np.exp(self.log_pole * dphi) + x[:, None]
        if self.noise > 0:                       # optional state noise (post-hoc test C)
            self.z = self.z + self.noise * (self.rng.standard_normal(self.z.shape)
                                            + 1j * self.rng.standard_normal(self.z.shape))

    def features(self):
        return np.concatenate([self.z.real.ravel(), self.z.imag.ravel()])


class ResonaattoriAivo:
    def __init__(self, modes=DEFAULT_MODES, n_pyr=512, frozen_clock=False, seed=0, gain=1.5, noise=0.0, sd_floor=1e-6):
        self.modes = list(modes)
        self.noise = noise
        self.sd_floor = sd_floor   # gates ran with 1e-6; the noise test needs a real floor
        self.noise_rng = np.random.default_rng(seed + 999)
        self.frozen = frozen_clock
        rng = np.random.default_rng(seed)
        D = NCH * len(self.modes) * 2
        self.Win = rng.normal(0, gain / np.sqrt(D), (n_pyr, D))
        self.bin = rng.normal(0, 0.5, n_pyr)
        self.W = np.zeros((NTOK, n_pyr + 1))
        self.mu = np.zeros(D)
        self.sd = np.ones(D)

    # ---------- listening: run the stream, sample features at requested frames ----------
    def listen(self, rd, frames=None):
        """Run clock + resonators over a rendered stream. Returns features at frames and the clock."""
        x = rd["x"]
        frames = rd["pred_frames"] if frames is None else frames
        want = {int(f): i for i, f in enumerate(frames)}
        clock = Clock(frozen=self.frozen)
        bank = ResonatorBank(self.modes, self.noise, self.noise_rng)
        out = np.zeros((len(frames), NCH * len(self.modes) * 2))
        for t in range(x.shape[0]):
            xt = x[t]
            dphi = clock.step(t, xt[:P].any() or xt[P] > 0, xt[P] > 0)
            bank.step(dphi, xt)
            if t in want:
                out[want[t]] = bank.features()
        return out, clock

    def pyramids(self, feats):
        u = (feats - self.mu) / self.sd
        h = np.tanh(u @ self.Win.T + self.bin)
        return np.concatenate([h, np.ones((h.shape[0], 1))], 1)

    def probs(self, H):
        a = H @ self.W.T
        a -= a.max(1, keepdims=True)
        e = np.exp(a)
        return e / e.sum(1, keepdims=True)

    # ---------- learning: training is discussion ----------
    def fit(self, renders, epochs=300, lr=0.05, seed=0, verbose=False):
        """Online residue learning: after every beat, W += lr * residue * pyramids."""
        F = [self.listen(rd)[0] for rd in renders]
        allf = np.concatenate(F)
        self.mu, self.sd = allf.mean(0), np.maximum(allf.std(0), self.sd_floor)
        Hs = [self.pyramids(f) for f in F]
        Ys = [rd["tokens"] for rd in renders]
        rng = np.random.default_rng(seed)
        for ep in range(epochs):
            for i in rng.permutation(len(Hs)):
                H, y = Hs[i], Ys[i]
                for b in range(len(y)):                    # beat by beat, in order
                    h = H[b:b + 1]
                    p = self.probs(h)[0]
                    r = -p
                    r[y[b]] += 1.0                          # residue = what came - what was expected
                    self.W += lr * np.outer(r, h[0])
            if verbose and ep % 50 == 0:
                acc = np.mean([(self.probs(H).argmax(1) == y).mean() for H, y in zip(Hs, Ys)])
                print(f"  epoch {ep} train acc {acc:.3f}")
        return self

    def predict(self, rd):
        f, clock = self.listen(rd)
        return self.probs(self.pyramids(f)), clock

    # ---------- free-run: the chandelier publishes one token per clock slot ----------
    def free_run(self, rd, cue_beats=12, max_frames=4000):
        x = rd["x"]
        clock = Clock(frozen=self.frozen)
        bank = ResonatorBank(self.modes)
        cue_end = COUNT_IN + cue_beats - 0.25        # clock phase where we stop listening
        published = []
        next_pred = cue_beats
        pending = None                               # (phase at which to fire, token)
        listening = True
        t = 0
        while len(published) < SONG_BEATS - cue_beats and t < max_frames:
            if listening and clock.phi is not None and clock.phi >= cue_end:
                listening = False
            xt = x[t] if (listening and t < x.shape[0]) else np.zeros(NCH, np.float32)
            inj = np.zeros(NCH, np.float32)
            if listening:
                dphi = clock.step(t, xt[:P].any() or xt[P] > 0, xt[P] > 0)
            else:
                dphi = clock.f
                clock.phi += dphi
                # predict a quarter beat before each slot, publish at the slot
                if pending is None and clock.phi >= COUNT_IN + next_pred - 0.25:
                    h = self.pyramids(bank.features()[None])
                    tok = int(self.probs(h)[0].argmax())   # chandelier: winner takes the slot
                    pending = (COUNT_IN + next_pred, tok)
                if pending is not None and clock.phi >= pending[0]:
                    tok = pending[1]
                    published.append(tok)
                    if tok != HOLD:
                        inj[tok] = 1.0                      # its own output is its next input
                    next_pred += 1
                    pending = None
            bank.step(dphi, xt + inj)
            t += 1
        return np.array(published)
