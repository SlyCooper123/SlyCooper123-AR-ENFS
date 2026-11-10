"""Data figures (Figs. 1-4) and benchmark figure (Fig. 5) of the paper."""
import sys, numpy as np, pandas as pd, matplotlib
matplotlib.use("Agg"); import matplotlib.pyplot as plt
plt.rcParams.update({"font.size": 11, "axes.spines.top": False, "axes.spines.right": False})
csv = sys.argv[1] if len(sys.argv) > 1 else "../data/dataset_meteo_com_consumo.csv.gz"
res = sys.argv[2] if len(sys.argv) > 2 else "../results/out_no_year"
d = pd.read_csv(csv).dropna(subset=["consumo_gwh"]); d["dt"] = pd.to_datetime(d.date, format="%Y-%m-%d")
nat = d.groupby("dt").agg(c=("consumo_gwh", "first"), t=("tmean_c", "mean")).reset_index(); C = "#1f5f8b"
fig, ax = plt.subplots(figsize=(8, 3.6)); ax.plot(nat.dt, nat.c, lw=.6, color=C); ax.set_ylabel("Daily consumption (GWh)"); ax.set_xlabel("Date"); fig.tight_layout(); fig.savefig("consumo_timeseries.png", dpi=300)
m = nat.groupby(nat.dt.dt.month).c.mean(); fig, ax = plt.subplots(figsize=(7, 3.6)); ax.bar(m.index, m.values, color=C); ax.set_xticks(range(1, 13))
ax.set_xticklabels(["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]); ax.set_ylim(110, 160); ax.set_ylabel("Mean daily consumption (GWh)"); fig.tight_layout(); fig.savefig("consumo_by_month.png", dpi=300)
fig, ax = plt.subplots(figsize=(6.5, 4)); ax.scatter(nat.t, nat.c, s=5, alpha=.4, color=C, edgecolors="none")
ax.set_xlabel("Mean air temperature, average of the 50 municipalities (°C)"); ax.set_ylabel("Daily consumption (GWh)"); fig.tight_layout(); fig.savefig("scatter_consumo_vs_tmean_c.png", dpi=300)
s = nat.c.astype(float); a = [1.0] + [s.autocorr(k) for k in range(1, 61)]
fig, ax = plt.subplots(figsize=(7, 3.6)); ax.vlines(range(61), 0, a, color=C, lw=1.2); ax.plot(range(61), a, "o", ms=3, color=C)
ci = 1.96 / np.sqrt(len(s)); ax.axhspan(-ci, ci, color="grey", alpha=.2, lw=0); ax.axhline(0, color="k", lw=.6)
for k in range(7, 61, 7): ax.axvline(k, color="grey", ls=":", lw=.7)
ax.set_xlabel("Lag (days)"); ax.set_ylabel("Autocorrelation"); fig.tight_layout(); fig.savefig("acf_consumo_lag60.png", dpi=300)
plt.rcParams.update({"font.size": 10})
ab = pd.read_csv(f"{res}/ablation/summary.csv"); ex = pd.read_csv(f"{res}/benchmark_external/summary.csv")
b = pd.concat([ex, ab[ab.Model.isin(["AR_ENFS_A0", "AR_ENFS_A1", "AR_ENFS_A7"])]])
names = {"AR_ENFS_A7": "AR-ENFS A7", "AR_ENFS_A0": "AR-ENFS A0", "AR_ENFS_A1": "AR-ENFS A1", "ePL": "ePL", "ePL_plus": "ePL+", "exTS": "exTS", "Simpl_eTS": "Simpl_eTS", "eMG": "eMG",
         "OnlineRLS": "Online SGD regression", "Persistence": "Persistence", "SeasonalPersistence7": "Seasonal persistence"}
t = b.pivot_table(index="Model", columns="HorizonDays", values="RMSE", aggfunc="mean"); t["m"] = t.mean(axis=1); t = t.sort_values("m", ascending=False)
fig, ax = plt.subplots(figsize=(7, 4.4)); y = range(len(t))
ax.barh([i + .2 for i in y], t[1], height=.38, color=C, label="1-day horizon"); ax.barh([i - .2 for i in y], t[7], height=.38, color="#9ab8d0", label="7-day horizon")
ax.set_yticks(list(y)); ax.set_yticklabels([names[k] for k in t.index]); ax.set_xlabel("Mean RMSE across the six cities (GWh)"); ax.legend(frameon=False, loc="upper right")
for i, k in enumerate(t.index):
    ax.text(t.loc[k, 1] + .08, i + .2, f"{t.loc[k, 1]:.2f}", va="center", fontsize=7.5); ax.text(t.loc[k, 7] + .08, i - .2, f"{t.loc[k, 7]:.2f}", va="center", fontsize=7.5)
ax.set_xlim(0, 13.8); fig.tight_layout(); fig.savefig("mean_rmse_by_model.png", dpi=300)
