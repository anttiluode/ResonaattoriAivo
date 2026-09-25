"""Song world: junction-paired melodies rendered to a frame stream at any tempo.

Tokens per beat: 0..P-1 = pitch onset on this beat, P = HOLD (previous note sustains).
Frame stream channels: 0..P-1 = pitch onset impulses, P = count-in click.
"""
import numpy as np

P = 8            # pitches
HOLD = P         # hold token
NTOK = P + 1     # prediction classes
NCH = P + 1      # input channels (pitches + click)
T0 = 20.0        # frames per beat at tempo factor s = 1.0
COUNT_IN = 4     # clicks before the song
SONG_BEATS = 32


def notes_to_tokens(notes):
    out = []
    for p, d in notes:
        out.append(p)
        out.extend([HOLD] * (d - 1))
    return out


def segment(rng, n_beats, p_long=0.3):
    """Random note list filling exactly n_beats; no immediate pitch repeats."""
    notes, left, last = [], n_beats, -1
    while left > 0:
        d = 2 if (rng.random() < p_long and left >= 2) else 1
        p = rng.integers(P)
        while p == last:
            p = rng.integers(P)
        notes.append((int(p), d))
        left -= d
        last = p
    return notes_to_tokens(notes)


def make_junction_songs(seed=0, n_pairs=6, pre=10, shared=6):
    """Songs come in pairs that share a middle segment X at the same position."""
    rng = np.random.default_rng(seed)
    songs, junctions = [], []
    post = SONG_BEATS - pre - shared
    for _ in range(n_pairs):
        X = segment(rng, shared)
        for _ in range(2):
            s = segment(rng, pre) + X + segment(rng, post)
            # the first beat after X must differ between the pair (else no junction)
            songs.append(s)
        a, b = songs[-2], songs[-1]
        j = pre + shared
        while b[j] == a[j]:            # segments always start with an onset
            b[j:] = segment(rng, post)
    return songs, pre + shared


def make_novel_songs(seed=123, n=40):
    rng = np.random.default_rng(seed)
    return [segment(rng, SONG_BEATS) for _ in range(n)]


def tempo_profile(kind, s=1.0):
    """Returns tempo factor per beat for count-in + song."""
    n = COUNT_IN + SONG_BEATS
    if kind == "const":
        return np.full(n, s)
    if kind == "accel":
        return np.concatenate([np.full(COUNT_IN, 0.75), np.linspace(0.75, 1.35, SONG_BEATS)])
    if kind == "rit":
        return np.concatenate([np.full(COUNT_IN, 1.35), np.linspace(1.35, 0.75, SONG_BEATS)])
    raise ValueError(kind)


def render(tokens, tempo, rng=None, jitter=1.0):
    """Render a song (list of SONG_BEATS tokens) to frames.

    tempo: array of tempo factors per beat (len COUNT_IN + SONG_BEATS).
    Returns dict with x [F, NCH], beat_times (float frame of each song beat),
    pred_frames (int frame at which each beat is predicted), tokens.
    """
    periods = T0 / np.asarray(tempo, float)
    beat_t = np.concatenate([[0.0], np.cumsum(periods)])[:-1] + 5.0  # start at frame 5
    F = int(np.ceil(beat_t[-1] + periods[-1] + 5))
    x = np.zeros((F, NCH), np.float32)

    def put(t, ch):
        tj = t + (rng.uniform(-jitter, jitter) if (rng is not None and jitter > 0) else 0.0)
        f = int(np.clip(round(tj), 0, F - 1))
        x[f, ch] = 1.0
        return f

    onset_frames = []
    for i in range(COUNT_IN):
        onset_frames.append(put(beat_t[i], P))
    song_t = beat_t[COUNT_IN:]
    song_T = periods[COUNT_IN:]
    for b, tok in enumerate(tokens):
        if tok != HOLD:
            onset_frames.append(put(song_t[b], tok))
    pred = np.floor(song_t - song_T / 4.0).astype(int)
    return dict(x=x, beat_times=song_t, periods=song_T, pred_frames=pred,
                tokens=np.array(tokens), onset_frames=np.array(onset_frames))


def prefix_oracle(train_songs, song):
    """Majority next token among training songs sharing the exact prefix -> ceiling per beat."""
    correct = []
    for b in range(len(song)):
        cands = [s[b] for s in train_songs if s[:b] == list(song[:b])]
        if not cands:
            correct.append(0.0)
            continue
        vals, cnt = np.unique(cands, return_counts=True)
        best = cnt.max()
        winners = vals[cnt == best]
        correct.append(float(song[b] in winners) / len(winners))
    return np.array(correct)
