"""Stage 2: resonator geometry that the residue can reshape.

Same clock as Stage 1 (run once per render, it does not learn). The resonators run in PyTorch
so that decay tau, frequency nu and the mode-to-pyramid coupling can receive gradients of the
prediction error. At the output that gradient is exactly the residue onehot - p.
"""
import numpy as np
import torch
import torch.nn as nn
from .world import NCH, NTOK, P
from .model import Clock, DEFAULT_MODES

torch.set_num_threads(2)


def prepare(rd):
    """Run the PLL clock over a render; return what the torch model needs."""
    x = rd["x"]
    clock = Clock()
    dphi = np.zeros(x.shape[0], np.float32)
    for t in range(x.shape[0]):
        xt = x[t]
        dphi[t] = clock.step(t, xt[:P].any() or xt[P] > 0, xt[P] > 0)
    return dict(x=x, dphi=dphi, pred=rd["pred_frames"], y=rd["tokens"])


def batch(preps):
    F = max(int(p["pred"].max()) + 1 for p in preps)
    B = len(preps)
    x = np.zeros((B, F, NCH), np.float32)
    d = np.zeros((B, F), np.float32)
    for i, p in enumerate(preps):
        n = min(F, p["x"].shape[0])
        x[i, :n] = p["x"][:n]
        d[i, :n] = p["dphi"][:n]
    pred = np.stack([p["pred"] for p in preps])
    y = np.stack([p["y"] for p in preps])
    return (torch.from_numpy(x), torch.from_numpy(d), torch.from_numpy(pred).long(),
            torch.from_numpy(y).long())


class PlasticRA(nn.Module):
    def __init__(self, modes=DEFAULT_MODES, n_pyr=512, learn_geom=False, learn_coupling=False,
                 seed=0, gain=1.5):
        super().__init__()
        g = torch.Generator().manual_seed(seed)
        K = len(modes)
        D = NCH * K * 2
        self.logtau = nn.Parameter(torch.log(torch.tensor([m[0] for m in modes])),
                                   requires_grad=learn_geom)
        self.nu = nn.Parameter(torch.tensor([float(m[1]) for m in modes]), requires_grad=learn_geom)
        self.Win = nn.Parameter(torch.randn(n_pyr, D, generator=g) * gain / np.sqrt(D),
                                requires_grad=learn_coupling)
        self.bin = nn.Parameter(torch.randn(n_pyr, generator=g) * 0.5, requires_grad=learn_coupling)
        self.out = nn.Linear(n_pyr, NTOK)
        self.register_buffer("mu", torch.zeros(D))       # running feature mean
        self.register_buffer("sd", torch.ones(D))        # running feature std, floored at 0.05 so a
        self.mu_init = False                             # near-constant feature cannot amplify noise
        with torch.no_grad():
            self.out.weight.zero_()
            self.out.bias.zero_()

    def poles(self):
        tau = torch.exp(self.logtau.clamp(np.log(0.3), np.log(64.0)))
        return torch.complex(-1.0 / tau, 2 * np.pi * self.nu), tau

    def forward(self, x, dphi, pred, noise=0.0, gen=None):
        pole, tau = self.poles()
        B, F, C = x.shape
        m = torch.exp(pole[None, None, :] * dphi[..., None].to(torch.complex64))   # [B,F,K]
        z = torch.zeros(B, C, pole.shape[0], dtype=torch.complex64)
        xs = x.to(torch.complex64)
        zs = []
        for t in range(F):
            z = z * m[:, t, None, :] + xs[:, t, :, None]
            if noise > 0:
                n = torch.randn(B, C, pole.shape[0], 2, generator=gen) * noise
                z = z + torch.complex(n[..., 0], n[..., 1])
            zs.append(z)
        Z = torch.stack(zs, 1)                                               # [B,F,C,K]
        Zp = Z[torch.arange(B)[:, None], pred]                               # [B,beats,C,K]
        s = (2.0 / torch.sqrt(tau))[None, None, None, :]
        u = torch.cat([(Zp.real * s).flatten(2), (Zp.imag * s).flatten(2)], -1)
        if self.training:
            with torch.no_grad():
                flat = u.reshape(-1, u.shape[-1])
                bm, bs = flat.mean(0), flat.std(0).clamp_min(0.05)
                if self.mu_init:
                    bm, bs = 0.9 * self.mu + 0.1 * bm, 0.9 * self.sd + 0.1 * bs
                self.mu.copy_(bm)
                self.sd.copy_(bs)
                self.mu_init = True
        u = (u - self.mu) / self.sd
        h = torch.tanh(u @ self.Win.T + self.bin)
        return self.out(h)


def train(model, preps, steps=300, noise=0.0, seed=0, bs=48, lr=1e-2, lr_geom=2e-2, verbose=False):
    rng = np.random.default_rng(seed)
    gen = torch.Generator().manual_seed(seed + 17)
    geom = [p for p in (model.logtau, model.nu) if p.requires_grad]
    rest = [p for n, p in model.named_parameters() if p.requires_grad and n not in ("logtau", "nu")]
    groups = [{"params": rest, "lr": lr}]
    if geom:
        groups.append({"params": geom, "lr": lr_geom})
    opt = torch.optim.Adam(groups)
    model.train()
    lossf = nn.CrossEntropyLoss()
    for step in range(steps):
        idx = rng.choice(len(preps), size=min(bs, len(preps)), replace=False)
        x, d, pr, y = batch([preps[i] for i in idx])
        logits = model(x, d, pr, noise=noise, gen=gen)
        loss = lossf(logits.reshape(-1, NTOK), y.reshape(-1))   # d loss / d logits = p - onehot = -residue
        opt.zero_grad()
        loss.backward()
        opt.step()
        if verbose and step % 100 == 0:
            print(f"   step {step} loss {loss.item():.3f}", flush=True)
    return model


@torch.no_grad()
def accuracy(model, preps, noise=0.0, beats=None, seed=123):
    model.eval()
    gen = torch.Generator().manual_seed(seed)
    x, d, pr, y = batch(preps)
    c = (model(x, d, pr, noise=noise, gen=gen).argmax(-1) == y).float()
    if beats is not None:
        c = c[:, beats]
    return c.mean().item()
