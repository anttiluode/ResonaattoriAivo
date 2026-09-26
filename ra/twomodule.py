"""Stage 3: two modules that share only events.

Sender A hears the melody stream with the full resonator bank (incl. slow modes).
At every onset A fires; the event keeps its timing and identity (the digital spike) and may
carry a small shape vector e in [-1, 1]^3 computed from A's state (the analog part).
Receiver B has only short modes (tau < 6 beats). It hears the events: identity + timing on
the usual channels, and the shape on 3 extra channels at the event frame.

The shape is an abstract 3-number descriptor per event, not a simulated voltage trace.

Distance: the shape is attenuated by g = exp(-d / lambda) and receiver noise is added after
attenuation, as with analog-digital facilitation beyond its length constant.
"""
import numpy as np
import torch
import torch.nn as nn
from .world import NCH, NTOK, P
from .model import DEFAULT_MODES
from .plastic import batch

SLOW = DEFAULT_MODES
FAST = [m for m in DEFAULT_MODES if m[0] < 6]
NSH = 3


def poles(modes):
    tau = torch.tensor([m[0] for m in modes])
    nu = torch.tensor([float(m[1]) for m in modes])
    return torch.complex(-1.0 / tau, 2 * np.pi * nu), tau


def run_bank(inp, dphi, modes, noise, gen, collect=None):
    """inp [B,F,C] real; returns stacked states at all frames [B,F,C,K] (or only `collect` frames)."""
    pole, tau = poles(modes)
    B, F, C = inp.shape
    m = torch.exp(pole[None, None, :] * dphi[..., None].to(torch.complex64))
    z = torch.zeros(B, C, len(modes), dtype=torch.complex64)
    zs = []
    xs = inp.to(torch.complex64)
    for t in range(F):
        z = z * m[:, t, None, :] + xs[:, t, :, None]
        if noise > 0:
            n = torch.randn(B, C, len(modes), 2, generator=gen) * noise
            z = z + torch.complex(n[..., 0], n[..., 1])
        zs.append(z)
    return torch.stack(zs, 1), tau


class Norm(nn.Module):
    """Running standardization with floored std (Stage 2)."""
    def __init__(self, D):
        super().__init__()
        self.register_buffer("mu", torch.zeros(D))
        self.register_buffer("sd", torch.ones(D))
        self.init = False

    def forward(self, u):
        if self.training:
            with torch.no_grad():
                f = u.reshape(-1, u.shape[-1])
                bm, bs = f.mean(0), f.std(0).clamp_min(0.05)
                if self.init:
                    bm, bs = 0.9 * self.mu + 0.1 * bm, 0.9 * self.sd + 0.1 * bs
                self.mu.copy_(bm)
                self.sd.copy_(bs)
                self.init = True
        return (u - self.mu) / self.sd


def feats(Z, tau):
    s = (2.0 / torch.sqrt(tau))[None, None, None, :]
    return torch.cat([(Z.real * s).flatten(2), (Z.imag * s).flatten(2)], -1)


class TwoModule(nn.Module):
    """shape_mode: 'none' (timestamps+identity only), 'state' (shape from A's state),
    'identity' (shape from the current event's identity only, no history).
    recv_modes: FAST (receiver lacks long memory) or SLOW (receiver has its own)."""

    def __init__(self, shape_mode="state", recv_modes=FAST, n_pyr=512, seed=0, sender_modes=SLOW):
        super().__init__()
        g = torch.Generator().manual_seed(seed)
        self.shape_mode, self.recv_modes, self.sender_modes = shape_mode, recv_modes, sender_modes
        DA = NCH * len(sender_modes) * 2
        self.WA = nn.Parameter(torch.randn(256, DA, generator=g) * 1.5 / np.sqrt(DA), requires_grad=False)
        self.bA = nn.Parameter(torch.randn(256, generator=g) * 0.5, requires_grad=False)
        self.normA = Norm(DA)
        self.emit = nn.Linear(256 if shape_mode == "state" else NCH, NSH)
        CB = NCH + (NSH if shape_mode != "none" else 0)
        DB = CB * len(recv_modes) * 2
        self.WB = nn.Parameter(torch.randn(n_pyr, DB, generator=g) * 1.5 / np.sqrt(DB), requires_grad=False)
        self.bB = nn.Parameter(torch.randn(n_pyr, generator=g) * 0.5, requires_grad=False)
        self.normB = Norm(DB)
        self.out = nn.Linear(n_pyr, NTOK)
        with torch.no_grad():
            self.out.weight.zero_()
            self.out.bias.zero_()

    def sender(self, x, dphi, noise, gen):
        """Shape per frame [B,F,3], zero where there is no onset."""
        onset = (x.sum(-1) > 0).float()[..., None]
        if self.shape_mode == "identity":
            return torch.tanh(self.emit(x)) * onset
        Z, tau = run_bank(x, dphi, self.sender_modes, noise, gen)
        h = torch.tanh(self.normA(feats(Z, tau)) @ self.WA.T + self.bA)
        return torch.tanh(self.emit(h)) * onset

    def receiver(self, x, dphi, pred, E, gain, shape_noise, noise, gen):
        inp = x
        if self.shape_mode != "none":
            onset = (x.sum(-1) > 0).float()[..., None]
            n = torch.randn(E.shape, generator=gen) * shape_noise * onset
            inp = torch.cat([x, gain * E + n], -1)
        Z, tau = run_bank(inp, dphi, self.recv_modes, noise, gen)
        Zp = Z[torch.arange(x.shape[0])[:, None], pred]
        h = torch.tanh(self.normB(feats(Zp, tau)) @ self.WB.T + self.bB)
        return self.out(h)

    def forward(self, x, dphi, pred, noise=0.0, gain=1.0, shape_noise=0.05, gen=None, E=None):
        if self.shape_mode != "none" and E is None:
            E = self.sender(x, dphi, noise, gen)
        return self.receiver(x, dphi, pred, E, gain, shape_noise, noise, gen)


def train2(model, preps, steps=300, noise=0.003, gain=1.0, seed=0, bs=48, lr=1e-2):
    rng = np.random.default_rng(seed)
    gen = torch.Generator().manual_seed(seed + 17)
    opt = torch.optim.Adam([p for p in model.parameters() if p.requires_grad], lr=lr)
    lossf = nn.CrossEntropyLoss()
    model.train()
    for _ in range(steps):
        idx = rng.choice(len(preps), size=min(bs, len(preps)), replace=False)
        x, d, pr, y = batch([preps[i] for i in idx])
        loss = lossf(model(x, d, pr, noise=noise, gain=gain, gen=gen).reshape(-1, NTOK), y.reshape(-1))
        opt.zero_grad()
        loss.backward()
        opt.step()
    model.eval()
    return model


@torch.no_grad()
def predict2(model, preps, noise=0.003, gain=1.0, attack=None, E_override=None, seed=123, mean_shape=None):
    """attack: None | 'clamp' (every event gets the training-mean shape) | 'shuffle' (shapes permuted
    across events within each song). E_override: shapes computed from another stream (transplant)."""
    gen = torch.Generator().manual_seed(seed)
    x, d, pr, y = batch(preps)
    E = None
    if model.shape_mode != "none":
        E = E_override if E_override is not None else model.sender(x, d, noise, gen)
        onset = (x.sum(-1) > 0)
        if attack == "clamp":
            E = torch.where(onset[..., None], mean_shape[None, None, :].expand_as(E), torch.zeros_like(E))
        elif attack == "shuffle":
            E = E.clone()
            g2 = torch.Generator().manual_seed(seed + 1)
            for b in range(E.shape[0]):
                idx = torch.nonzero(onset[b]).squeeze(1)
                E[b, idx] = E[b, idx[torch.randperm(len(idx), generator=g2)]]
    logits = model.receiver(x, d, pr, E, gain, 0.05, noise, gen) if E is not None else \
        model(x, d, pr, noise=noise, gain=gain, gen=gen)
    return logits.argmax(-1), y


@torch.no_grad()
def mean_event_shape(model, preps, noise=0.003):
    gen = torch.Generator().manual_seed(5)
    x, d, pr, y = batch(preps)
    E = model.sender(x, d, noise, gen)
    onset = x.sum(-1) > 0
    return E[onset].mean(0)
