# ResonaattoriAivo — pre-registered gates

Written **before** any gate was run. Thresholds below are frozen; the receipt
(`results/receipt.json`) reports PASS/FAIL against exactly these numbers. If a
threshold is later changed, the change is recorded in the ledger at the bottom of
this file with the reason, and the original number is kept.

## The one claim under test

> A memory whose resonators are **slaved to an entrained clock** (time measured in
> beats, not frames) recalls learned rhythmic sequences **independently of tempo**,
> and needs only one tempo to learn them. A generic recurrent network needs to have
> seen the tempos to cope with them.

Everything else (junctions, residue, free-run) is secondary and gated separately.

## World

- 8 pitches + HOLD token per beat. Songs are 32 beats long, built from notes of 1 or 2 beats.
- 12 training songs in 6 **junction pairs**: `prefixA · X · suffixA` and `prefixB · X · suffixB`,
  where `X` is a shared 6-beat segment. Right after `X`, only context from before `X` says which way to go.
- Songs are rendered to a 100 Hz frame stream of onset impulses (one channel per pitch, plus a
  click channel carrying a 4-click count-in). Base beat period is 20 frames (tempo factor s = 1.0).
  Timing jitter is ±1 frame per onset.
- A prediction is made for every beat `b` at time `t_b − T/4` (a quarter beat early), from the
  stream seen so far. Target: the pitch that starts on beat `b`, or HOLD.
- **Ceiling** = the prefix oracle: the majority next token among training songs that share the exact
  token prefix. Accuracies are reported raw and as a fraction of this ceiling.

## Models

| name | what it is |
|---|---|
| **RA** | ResonaattoriAivo: PLL clock → clock-slaved complex resonators (τ, ν in beats) → fixed random pyramids (tanh) → readout learned online from the residue (onehot − p) |
| RA-frozen | same, but clock frozen at the training tempo (resonators run in frames). Isolates the clock |
| RA-noslow | same, but resonators with τ ≥ 6 beats removed. Isolates the slow bands |
| GRU-1 | GRU on the frame stream, parameter-matched to RA's trainable readout, trained at s = 1.0 only |
| GRU-aug | same size, trained with tempos drawn uniformly from s ∈ [0.7, 1.4] |
| GRU-big | hidden 128, trained with the same augmentation (an unconstrained reference) |
| ngram-oracle | order-3 beat-token Markov model that is *given the beat grid* (a reference, not a competitor) |

RA and the RA ablations are trained on s = 1.0 only.

## Gates

**G0 — the clock entrains.** Over s ∈ {0.6, 0.7, …, 1.6} × all songs, onsets after beat 8 of the song:
median |phase error| < 0.05 beat **and** fraction of songs ending in an octave/slip error < 5 %.

**G1 — recall at the training tempo.** RA accuracy at s = 1.0 ≥ 90 % of ceiling, and
not more than 5 points below GRU-1.

**G2 — tempo transfer (the main claim).** Test tempos s ∈ {0.6, …, 1.6}.
- RA mean accuracy ≥ 85 % of ceiling, and RA's worst tempo is ≤ 10 points below its s = 1.0 accuracy.
- RA-frozen must lose > 30 points on average (otherwise the clock is not what does it).
- Reported, not gated: RA vs GRU-1, GRU-aug, GRU-big. "RA matches an augmented GRU" is only claimed
  if RA ≥ GRU-aug − 5 points on the mean, and "beats" only if RA > GRU-aug + 5 points.

**G3 — tempo drift.** Accelerando s: 0.75 → 1.35 and ritardando 1.35 → 0.75 across the song.
RA accuracy ≥ 80 % of ceiling on both.

**G4 — slow bands carry context.** Accuracy on the 3 beats after each shared segment X:
RA ≥ 80 % and RA-noslow ≤ 60 %.

**G5 — the anti-phase residue is surprise.** Surprisal = −log p(actual token).
- Per-song mean surprisal separates familiar from 40 novel songs: AUROC ≥ 0.95.
- One pitch substituted at a random beat in [12, 28): the surprisal peak lands within ±1 beat of it
  in ≥ 80 % of trials.

**G6 — free-run continuation.** Cue = the first 12 beats at the test tempo, then RA runs on its own
clock, the chandelier (winner-take-all per slot) publishes one token per beat, and the published
token is fed back as input. Exact-token accuracy over beats 12–31 ≥ 80 % at s = 1.0 and ≥ 70 % mean
over test tempos.

## Ledger of threshold changes

- *Clarification, written before G5 was run:* the G5 substitution peak is searched over beats 4-31.
  Beats 0-3 are high-surprisal for every song by construction (nothing yet says which song it is),
  so a whole-song argmax would measure that, not the substitution. Threshold unchanged.
- *After the receipt (no threshold changed):* GRU-1 and GRU-aug (hidden 32) failed to train (43-47 % at s = 1.0),
  so their comparisons in G1/G2 are against a broken baseline. The post-hoc arm GRU-128 trained at s = 1.0 only
  (`posthoc.py`) is the comparison the main claim needs; it is reported in the README, not as a gate.
