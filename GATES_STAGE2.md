# Stage 2 — plastic geometry (pre-registered)

Written and committed **before** any Stage 2 run. Receipt: `results/receipt_stage2.json`.

## Why this stage exists

Stage 1 learned only the readout. The resonators (τ, ν) and the random pyramid layer were fixed:
the brain "starts with geometry". Two Stage 1 results point at that fixed geometry as the weak part:

- with state noise 0.003 per frame, whole-song accuracy fell from ~95 % to 73–78 %;
- under noise, the slow bands stopped helping at junctions.

The claim under test: **letting the residue reshape the geometry itself** (what the system is
susceptible to) makes the memory robust where fixed geometry is fragile, without losing tempo transfer.

"The residue reshapes the geometry" is implemented as backprop of the prediction error
(cross-entropy; its gradient at the output is exactly the residue onehot − p) into the resonator
parameters. It is not a local plasticity rule. That is a known limitation of this stage.

## Model changes (all arms share them)

- Resonators are simulated in PyTorch so their parameters can receive gradients. The clock is the same
  PLL as Stage 1, run once per render; its phase increments drive the resonators.
- Features are scaled by 1/√τ (the steady-state size of a mode), not by data standardization, so the
  noise is never amplified by a near-zero training variance (the Stage 1 bug).
- Readout trained with Adam on cross-entropy. The Stage 1 delta rule is not used here.
- 8 modes initialised at Stage 1's (τ, ν). 512 pyramids.

## Arms

| arm | trainable |
|---|---|
| **fixed** | readout only (Stage 1 geometry, same optimizer and budget) |
| **plastic-geom** | readout + per-mode log τ and ν (16 extra numbers) |
| **plastic-geom+coupling** | readout + τ, ν + the pyramid input matrix (how modes combine nonlinearly; +74k numbers) |

All arms train at tempo 1.0 only, 3 seeds each; reported numbers are seed means.
The coupling arm has many more trainable numbers than the others, so a win by it alone
is not evidence for geometry; the gates are placed on **plastic-geom**.

## Gates (noise = 0.003 per frame in training and test unless stated)

**S1 — robustness.** Whole-song accuracy, mean over tempos 0.6–1.6:
plastic-geom ≥ fixed + 10 points, **and** plastic-geom ≥ 85 % of ceiling.

**S2 — tempo transfer survives.** plastic-geom: worst tempo ≤ 10 points below its 1.0 accuracy.

**S3 — context through a long junction.** World with an 18-beat shared segment (pre = 8):
accuracy on the 3 beats after it, mean over tempos {0.6, 1.0, 1.6}:
plastic-geom ≥ 80 % **and** ≥ fixed + 15 points.

**S4 — reported, not gated.** The learned τ and ν per mode, the coupling arm's numbers, and all arms
at noise 0 and 0.01.

## Ledger of changes

- *Before the gate run — pilot on the fixed arm only (seed 0, tempo 1.0), used to set the training budget:*
  Adam on a readout fed by features scaled with 1/√τ alone converged slowly (≤ 75 % after 300 steps).
  Switching to running standardization with the std floored at 0.05 (so noise is amplified at most 20×,
  not 10⁶× as in the Stage 1 bug) converged in ~100 steps. With it, the **fixed** geometry reached
  **94.9 % at noise 0.003**, i.e. Stage 1's drop to 73–78 % was mostly the delta rule / feature scaling,
  not the fixed geometry. S1 as written (plastic ≥ fixed + 10) is therefore expected to FAIL; it is run
  unchanged. Budget frozen for all arms: 300 Adam steps, readout/coupling lr 1e-2, geometry lr 2e-2.
  Because the premise weakened, a noise sweep {0, 0.003, 0.01, 0.02} over all arms is added as
  **reported, not gated**.
- *After the S1–S4 receipt:* S1 and S3 failed because the fixed arm was already strong at noise 0.003.
  The reported sweep (seed 0 only) showed the split opening at higher noise: at 0.01, whole-song
  fixed 59 %, plastic-geom 88 %, plastic-geom+coupling 96 %. One seed is not evidence, so a
  confirmation stage S5 is pre-registered below, on fresh seeds, before it is run.

## S5 — confirmation at noise 0.01 (pre-registered after the S4 sweep, before running S5)

Seeds 3, 4, 5 (never used before). World A (6-beat junction). Train and test at noise 0.01.
Whole-song accuracy, mean over tempos {0.6, 1.0, 1.6}. Budget unchanged (300 steps).
New arm **coupling-only**: fixed Stage 1 geometry, trainable pyramid input matrix. It separates
"the geometry learned" from "74k extra trainable numbers".

- **S5a** plastic-geom ≥ fixed + 15 points (seed mean), and better than fixed on every seed.
- **S5b** geometry adds beyond coupling: plastic-geom+coupling ≥ coupling-only + 5 points.
- Reported: world B post-junction accuracy for all four arms, same seeds and noise.
