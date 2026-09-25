"""results/noise_sweep.png from receipt_stage2.json (seed 0 sweep) and S5 seed means."""
import json
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

R = json.load(open("results/receipt_stage2.json"))["S4_noise_sweep"]
S5 = json.load(open("results/receipt_stage2_s5.json"))["mean"]
noises = [0.0, 0.003, 0.01, 0.02]
arms = [("fixed", "fixed geometry", "#eb6834"),
        ("plastic-geom", "plastic τ, ν (16 numbers)", "#2a78d6"),
        ("plastic-geom+coupling", "plastic τ, ν + coupling (74k)", "#1baf7a")]
fig, ax = plt.subplots(figsize=(8.2, 4.6), dpi=150)
for key, label, c in arms:
    ax.plot(noises, [R[f"{key}@{n}"]["whole_song_A"] for n in noises], "-o", color=c, lw=2, ms=4, label=label + " (seed 0)")
s5 = [("fixed", "#eb6834"), ("coupling-only", "#8c9bad"), ("plastic-geom", "#2a78d6"), ("plastic-geom+coupling", "#1baf7a")]
for i, (k, c) in enumerate(s5):
    ax.plot(0.01 + (i - 1.5) * 0.0006, S5[k]["A_whole"], "D", color=c, ms=7, mec="white", zorder=4)
ax.annotate("S5: seeds 3–5 at 0.01\n(grey = coupling only)", (0.0112, 0.70), fontsize=8, color="#5b6b80")
ax.set_xlabel("state noise per frame")
ax.set_ylabel("whole-song accuracy (tempos 0.6, 1.0, 1.6)")
ax.set_ylim(0.4, 1.0)
ax.set_xticks(noises)
ax.grid(axis="y", color="#e3e6ea", lw=0.8)
for s in ["top", "right"]:
    ax.spines[s].set_visible(False)
ax.legend(frameon=False, fontsize=8.5, loc="lower left")
ax.set_title("Under state noise, a tuned response geometry holds and the fixed one collapses", loc="left", fontsize=11)
fig.tight_layout()
fig.savefig("results/noise_sweep.png")
print("saved")
