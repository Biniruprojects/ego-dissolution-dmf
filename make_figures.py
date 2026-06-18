"""Publication figures for the preprint, from the saved firming result tables."""
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

plt.rcParams.update({"font.size": 11, "axes.spines.top": False, "axes.spines.right": False})
C = {"LSD": "#d1495b", "N2O": "#4682b4", "COMBO": "#6a4c93"}
order = ["LSD", "N2O", "COMBO"]
labmap = {"LSD": "5-HT2A\n(LSD/DMT)", "N2O": "N2O\n(disinhib.)", "COMBO": "COMBO"}

ms = pd.read_csv(r"E:\BiniruProjects\psyche-sim\results_multiseed.csv").set_index("Condition").loc[order].reset_index()
dose = pd.read_csv(r"E:\BiniruProjects\psyche-sim\results_dose.csv")
cols = [C[c] for c in ms.Condition]
xlab = [labmap[c] for c in ms.Condition]

fig, ax = plt.subplots(1, 3, figsize=(13.5, 4.3))

# Panel A: complexity (the near-cancellation)
ax[0].bar(xlab, ms.dLZc_mean, yerr=ms.dLZc_std, capsize=5, color=cols, edgecolor="k", linewidth=0.6)
ax[0].axhline(0, color="k", lw=0.8)
ax[0].set_ylabel("Δ Lempel-Ziv complexity (%)")
ax[0].set_title("A  Signal complexity", loc="left", fontweight="bold")
for i, (m, s) in enumerate(zip(ms.dLZc_mean, ms.dLZc_std)):
    ax[0].text(i, m + np.sign(m) * (s + 0.4), "%+.1f" % m, ha="center",
               va="bottom" if m >= 0 else "top", fontsize=9)

# Panel B: integration (the robust collapse)
ax[1].bar(xlab, ms.dInteg_mean, yerr=ms.dInteg_std, capsize=5, color=cols, edgecolor="k", linewidth=0.6)
ax[1].axhline(0, color="k", lw=0.8)
ax[1].set_ylabel("Δ functional integration (%)")
ax[1].set_title("B  Functional integration", loc="left", fontweight="bold")
for i, (m, s) in enumerate(zip(ms.dInteg_mean, ms.dInteg_std)):
    ax[1].text(i, m - (s + 2), "%+.0f" % m, ha="center", va="top", fontsize=9)

# Panel C: dose-response
axc = ax[2]
axc.plot(dose.dose, dose.dInteg_pct, "o-", color=C["COMBO"], lw=2, label="integration")
axc.axhline(0, color="k", lw=0.8)
axc.set_xlabel("combined dose (fraction of full)")
axc.set_ylabel("Δ integration (%)", color=C["COMBO"])
axc.tick_params(axis="y", labelcolor=C["COMBO"])
axc.set_title("C  Dose-response (combo)", loc="left", fontweight="bold")
ax2 = axc.twinx()
ax2.spines["top"].set_visible(False)
ax2.plot(dose.dose, dose.dLZc_pct, "s--", color=C["LSD"], lw=1.6, label="complexity")
ax2.set_ylabel("Δ complexity (%)", color=C["LSD"])
ax2.tick_params(axis="y", labelcolor=C["LSD"])
ax2.set_ylim(-3, 6)

fig.suptitle("Combined 5-HT2A agonism + NMDA antagonism is dissociation-dominated\n"
             "(whole-brain DMF + FIC, rate-matched to 3 Hz; n=6 seeds, error bars = SD)",
             fontsize=12, y=1.02)
fig.tight_layout()
fig.savefig(r"E:\BiniruProjects\psyche-sim\FIG1_results.png", dpi=200, bbox_inches="tight")
print("Saved FIG1_results.png")

# Figure 2: the mechanism plane (conditions on the complexity x integration plane)
fig2, axp = plt.subplots(figsize=(6, 5))
pts = {"Baseline": (0.0, 0.0)}
for c in order:
    row = ms[ms.Condition == c].iloc[0]
    pts[c] = (row.dInteg_mean, row.dLZc_mean)
for name, (x, y) in pts.items():
    col = C.get(name, "#444")
    axp.scatter(x, y, s=160, color=col, edgecolor="k", zorder=3)
    axp.annotate(name, (x, y), fontsize=10, xytext=(7, 5), textcoords="offset points")
axp.axhline(0, color="gray", lw=0.7, ls=":")
axp.axvline(0, color="gray", lw=0.7, ls=":")
axp.set_xlabel("Δ functional integration (%)   ← dissociation")
axp.set_ylabel("Δ complexity (%)   entropy →")
axp.set_title("Mechanism plane (rate-matched)\nCOMBO: integration collapses, complexity ~preserved")
axp.grid(alpha=0.25)
fig2.tight_layout()
fig2.savefig(r"E:\BiniruProjects\psyche-sim\FIG2_plane.png", dpi=200, bbox_inches="tight")
print("Saved FIG2_plane.png")
