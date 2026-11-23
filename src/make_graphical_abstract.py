"""Graphical abstract (paper/submission_files/graphical_abstract.pdf). Values from results/analysis_summary.txt."""
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch
fig = plt.figure(figsize=(13, 5)); ax = fig.add_axes([0, 0, 1, 1]); ax.set_xlim(0, 13); ax.set_ylim(0, 5); ax.axis("off")
C1, C2, C3 = "#e8f0f7", "#1f5f8b", "#9ab8d0"
def box(x, y, w, h, title, lines, fc=C1):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.05,rounding_size=0.15", fc=fc, ec=C2, lw=1.5))
    ax.text(x + w / 2, y + h - 0.3, title, ha="center", va="top", fontsize=12, weight="bold", color=C2)
    for i, l in enumerate(lines): ax.text(x + 0.2, y + h - 0.85 - i * 0.42, l, ha="left", va="top", fontsize=9.5)
box(0.2, 0.6, 3.4, 3.9, "Data stream", ["Daily national consumption (REN)", "2015-2025, 3,926 days", "Weather of 6 Portuguese cities", "(Open-Meteo), calendar, lags", "Horizons: 1 and 7 days", "Predict -> score -> update"])
box(4.3, 0.6, 4.2, 3.9, "AR-ENFS", ["Evolving Takagi-Sugeno rule base", "Conjunctive Gaussian activation", "Direct + residual consequents (WRLS)", "Seasonal regimes on consequents", "Novelty AND (error OR uncertainty)", "  -> bounded rule growth"], fc="#f3f7fb")
for x in (3.65, 8.55): ax.annotate("", xy=(x + 0.55, 2.55), xytext=(x, 2.55), arrowprops=dict(arrowstyle="-|>", lw=2, color=C2))
box(9.2, 0.6, 3.6, 3.9, "Findings", [], fc=C1)
axb = fig.add_axes([0.775, 0.40, 0.19, 0.32])
axb.bar([0, 1], [5.844, 7.229], width=0.35, color=C2, label="AR-ENFS A7"); axb.bar([0.37, 1.37], [5.820, 7.538], width=0.35, color=C3, label="ePL")
axb.set_xticks([0.18, 1.18]); axb.set_xticklabels(["1 day\n(tie)", "7 days\n(-4.1%)"], fontsize=8.5); axb.set_ylim(5, 8); axb.set_ylabel("Mean RMSE (GWh)", fontsize=8.5)
axb.tick_params(labelsize=8); axb.legend(fontsize=7.5, frameon=False, loc="upper left"); axb.spines[["top", "right"]].set_visible(False)
ax.text(9.4, 0.95, "Rules = warm / cool-season regimes", fontsize=9.5); ax.text(9.4, 0.65, "Trade-off: more sensitive to input noise", fontsize=9.5)
fig.savefig("graphical_abstract.pdf")
