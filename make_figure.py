"""results/tempo_transfer.png from receipt.json (+ posthoc.json if present)."""
import json, os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

R = json.load(open("results/receipt.json"))
tab = R["accuracy_by_tempo"]
T = sorted(tab, key=float)
x = [float(t) for t in T]
series = [
    ("RA (trained at 1.0 only)", [tab[t]["RA"] for t in T], "#2a78d6", "-"),
    ("RA, clock frozen", [tab[t]["RA-frozen"] for t in T], "#eb6834", "-"),
    ("GRU-128, trained on 0.7–1.4", [tab[t]["GRU-big"] for t in T], "#1baf7a", "-"),
    ("GRU-32, trained on 0.7–1.4", [tab[t]["GRU-aug"] for t in T], "#eda100", "-"),
]
if os.path.exists("results/posthoc.json"):
    ph = json.load(open("results/posthoc.json"))["A_GRU-big-1_by_tempo"]
    series.insert(2, ("GRU-128, trained at 1.0 only (post hoc)", [ph[t] for t in T], "#e87ba4", "--"))

fig, ax = plt.subplots(figsize=(8.6, 4.8), dpi=150)
ax.axvspan(0.7, 1.4, color="#e6e9ee", zorder=0)
ax.text(1.05, 0.03, "GRU augmentation range", ha="center", fontsize=8, color="#5b6b80")
ax.axhline(R["ceiling"], color="#8c9bad", lw=1, ls=":", zorder=1)
ax.text(0.6, R["ceiling"] + 0.012, "ceiling (prefix oracle)", fontsize=8, color="#5b6b80")
for name, y, c, ls in series:
    ax.plot(x, y, ls, color=c, lw=2, marker="o", ms=4, label=name, zorder=3)
ax.set_xlabel("test tempo (× training tempo)")
ax.set_ylabel("next-beat accuracy")
ax.set_ylim(0, 1.05)
ax.set_xticks(x)
ax.grid(axis="y", color="#e3e6ea", lw=0.8)
for s in ["top", "right"]:
    ax.spines[s].set_visible(False)
ax.legend(loc="lower center", bbox_to_anchor=(0.5, -0.42), ncol=2, frameon=False, fontsize=8.5)
ax.set_title("Recall of learned melodies at unseen tempos", loc="left", fontsize=11)
fig.tight_layout()
fig.savefig("results/tempo_transfer.png")
print("saved")
