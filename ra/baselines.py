"""Baselines: GRU on the raw frame stream, and a beat-grid n-gram reference."""
import numpy as np
import torch
import torch.nn as nn
from .world import NCH, NTOK, render, tempo_profile

torch.set_num_threads(2)


class GRUNet(nn.Module):
    def __init__(self, H):
        super().__init__()
        self.gru = nn.GRU(NCH, H, batch_first=True)
        self.out = nn.Linear(H, NTOK)

    def forward(self, x, idx):
        h, _ = self.gru(x)                                  # [B, F, H]
        g = torch.gather(h, 1, idx[..., None].expand(-1, -1, h.shape[-1]))
        return self.out(g)                                  # [B, beats, NTOK]


def n_params(m):
    return sum(p.numel() for p in m.parameters())


def _batch(rds):
    F = max(r["x"].shape[0] for r in rds)
    x = np.zeros((len(rds), F, NCH), np.float32)
    for i, r in enumerate(rds):
        x[i, :r["x"].shape[0]] = r["x"]
    idx = np.stack([r["pred_frames"] for r in rds])
    y = np.stack([r["tokens"] for r in rds])
    return torch.from_numpy(x), torch.from_numpy(idx).long(), torch.from_numpy(y).long()


def train_gru(songs, H=32, tempo_range=(1.0, 1.0), steps=1500, reps=2, lr=3e-3, seed=0, verbose=False):
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    net = GRUNet(H)
    opt = torch.optim.Adam(net.parameters(), lr=lr)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, steps)
    lossf = nn.CrossEntropyLoss()
    for step in range(steps):
        rds = [render(s, tempo_profile("const", rng.uniform(*tempo_range)), rng)
               for s in songs for _ in range(reps)]
        x, idx, y = _batch(rds)
        logits = net(x, idx)
        loss = lossf(logits.reshape(-1, NTOK), y.reshape(-1))
        opt.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(net.parameters(), 1.0)
        opt.step()
        sched.step()
        if verbose and step % 250 == 0:
            acc = (logits.argmax(-1) == y).float().mean().item()
            print(f"  GRU{H} step {step} loss {loss.item():.3f} acc {acc:.3f}")
    net.eval()
    return net


@torch.no_grad()
def gru_probs(net, rds):
    x, idx, _ = _batch(rds)
    return torch.softmax(net(x, idx), -1).numpy()


class NGram:
    """Order-k beat-token Markov model with backoff. It is handed the beat grid (tokens)."""
    def __init__(self, songs, k=3):
        self.k = k
        self.tables = [dict() for _ in range(k + 1)]
        for s in songs:
            seq = [-1] * k + list(s)
            for i in range(k, len(seq)):
                for o in range(k + 1):
                    ctx = tuple(seq[i - o:i])
                    d = self.tables[o].setdefault(ctx, np.zeros(NTOK))
                    d[seq[i]] += 1

    def probs(self, song):
        seq = [-1] * self.k + list(song)
        out = []
        for i in range(self.k, len(seq)):
            for o in range(self.k, -1, -1):
                d = self.tables[o].get(tuple(seq[i - o:i]))
                if d is not None:
                    out.append(d / d.sum())
                    break
        return np.array(out)
