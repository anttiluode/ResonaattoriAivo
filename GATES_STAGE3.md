# Stage 3 — two modules, events only (pre-registered)

Written and committed **before** any Stage 3 gate run. Receipt: `results/receipt_stage3.json`.

## Question

When two systems share nothing but events, can a small state-dependent **shape** on each event
carry history that the receiver cannot hold itself? Is it the *state-to-shape mapping* that does it
(not extra bandwidth)? Does geometry, meaning attenuation with distance as in analog-digital facilitation's
length constant, decide who receives it?

Motivation: Martin-Burgos et al., bioRxiv 2026.09.15.751814 (spike waveforms vary with input and
state) + analog-digital facilitation (somatic state reaches terminals only within ~150–700 µm).

## Setup

- World: 18-beat shared segment (pre = 8). Right after the shared segment, only history older than
  18 beats says which song this is.
- **Sender A** hears the melody with the full resonator bank (τ up to 24 beats), fixed geometry,
  fixed random 256-unit layer. At every onset it fires. The event keeps timing and identity
  (the digital spike) and carries a shape e ∈ [-1, 1]³ from a trainable emission head.
- **Receiver B** hears the events: identity and timing on the usual channels, the shape on 3 extra
  channels at the event frame, attenuated by gain g, plus receiver shape noise 0.05.
  B has only the short modes (τ < 6 beats) unless stated.
- The shape is an abstract 3-number descriptor per event, **not** a simulated voltage trace.
- Training: tempo 1.0 only, 300 Adam steps, lr 1e-2, loss = B's next-beat cross-entropy. The sender's
  emission head learns only through B's error. State noise **0.006** per frame in both modules.
- Test: tempos {0.6, 1.0, 1.6}, seeds 0, 1, 2; seed means reported.
  **post** = accuracy on the 3 beats after the shared segment; **whole** = whole song.

## Arms

| arm | shape carried by each event | receiver memory |
|---|---|---|
| none/FAST | none (timestamps + identity only) | short |
| identity/FAST | from the current event's identity only (no history) | short |
| **state/FAST** | from the sender's state | short |
| none/SLOW | none | long (own slow bands) |
| state/SLOW | from the sender's state | long |
| state/FAST @ 4λ | from the sender's state, gain e⁻⁴ | short |

## Gates

**T1 — the shape carries history.** post(state/FAST) ≥ post(none/FAST) + 15 **and**
≥ post(identity/FAST) + 15.

**T2 — it is the state-to-shape mapping.** On state/FAST, replacing every shape with the training
mean (clamp) and permuting shapes across events within each song (shuffle, timing and identity
untouched) each lower post by ≥ 15 points.

**T3 — same present, different history, different continuation.** Pairs (α, β) that share the
18-beat segment, rendered without jitter so every event in the segment is identical in time and
identity. B hears α; the shapes come from A run on β (sender-state transplant). Among cases where
state/FAST is correct at the first post-segment beat on both α and β, it predicts β's token in ≥ 70 %.

**T4 — geometry decides who hears.** Trained and tested at gain e⁻⁴ (distance 4 length constants):
post ≤ post(none/FAST) + 5.

**T5 — the prediction written in the previous turn.** A receiver that has its own long memory gains
little: post(state/SLOW) − post(none/SLOW) < 3 points.

Reported, not gated: gain curve {e⁻¹, e⁻²} (seed 0); none/FAST and state/FAST at noise 0.003 and 0.01
(seed 0); whole-song numbers for every arm.

## Ledger

- *Pilot before the gates, baseline arm only (none/FAST, seed 0):* post-junction accuracy was
  85 % at noise 0.003 (too little headroom for T1), 65 % at 0.006, 47 % at 0.01. Noise 0.006 was
  chosen from this alone. The state arm was only smoke-run for 5 steps to check the code; its
  accuracy was not computed.
- *After the receipt (no threshold changed):* one post-hoc control, `posthoc_stage3.py`: the same
  state/FAST architecture, but the **sender also has only the short modes**, so it holds no more
  history than the receiver. It scored post 71.3 % / whole 82.9 % (seeds 0–2), the same as state/FAST
  (71.5 % / 83.4 %). The gain is therefore not history transfer; it comes from the sender acting as a
  second trained stage that sends a cleaner summary of recent context. This is consistent with T3 failing.
- T4 passed, but part of why: at gain e⁻⁴ the shape channels are almost pure receiver noise, and the
  receiver did *worse* than with no shape channels (49.5 % vs 65.7 %). "No benefit" is shown;
  "harmless when out of range" is not.
