#!/usr/bin/env python3
"""Aggregate all experiment outputs of one variant into the numbers used in the paper."""
import sys, json
from pathlib import Path
import numpy as np, pandas as pd
from scipy import stats

V = Path(sys.argv[1])
abl = pd.read_csv(V / "ablation/summary.csv")
ext = pd.read_csv(V / "benchmark_external/summary.csv")
bench = pd.concat([ext, abl[abl.Model.isin(["AR_ENFS_A0", "AR_ENFS_A1", "AR_ENFS_A7"])]], ignore_index=True)
pa = pd.read_csv(V / "ablation/predictions.csv.gz", parse_dates=["Date"])
pe = pd.read_csv(V / "benchmark_external/predictions.csv.gz", parse_dates=["Date"])
pred = pd.concat([pe, pa[pa.Model.isin(["AR_ENFS_A0", "AR_ENFS_A1", "AR_ENFS_A7"])]], ignore_index=True)
out = {}
pd.set_option("display.width", 250)


def agg(df):
    g = df.groupby("Model")
    t = pd.DataFrame({"MeanRMSE": g.RMSE.mean(), "MedianRMSE": g.RMSE.median(), "StdRMSE": g.RMSE.std(ddof=0), "WorstRMSE": g.RMSE.max(),
                      "MeanMAE": g.MAE.mean(), "MeanR2": g.R2.mean(), "RelPers": g.RMSE_ratio_to_persistence.mean(),
                      "FinalRules": g.FinalRules.mean(), "Runtime": g.TotalTime_s.mean(),
                      "WinsVsPers": g.apply(lambda x: int((x.RMSE_ratio_to_persistence < 1).sum()), include_groups=False)})
    r = df.copy(); r["rank"] = r.groupby(["City", "HorizonDays"]).RMSE.rank(method="average")
    t["MeanRank"] = r.groupby("Model")["rank"].mean()
    return t.sort_values("MeanRMSE")

print("=== ABLATION by horizon"); print(abl.pivot_table(index="Model", columns="HorizonDays", values="RMSE", aggfunc="mean").round(3).sort_values(1))
print(agg(abl).round(3))
w = abl.loc[abl.groupby(["City", "HorizonDays"]).RMSE.idxmin()][["City", "HorizonDays", "Model", "RMSE"]]; print(w.to_string()); print(w.Model.value_counts())
print(abl.pivot_table(index=["City", "HorizonDays"], columns="Model", values="RMSE").round(3).to_string())

print("=== BENCHMARK global"); print(agg(bench).round(3).to_string())
for h in (1, 7):
    print(f"=== BENCHMARK h{h}"); print(agg(bench[bench.HorizonDays == h]).round(3).to_string())
wb = bench.loc[bench.groupby(["City", "HorizonDays"]).RMSE.idxmin()][["City", "HorizonDays", "Model", "RMSE"]]; print(wb.to_string()); print(wb.Model.value_counts())
cmp = bench[bench.Model.isin(["AR_ENFS_A7", "ePL"])].pivot_table(index=["City", "HorizonDays"], columns="Model", values="RMSE")
cmp["RelDiffPct"] = (cmp.ePL - cmp.AR_ENFS_A7) / cmp.ePL * 100; print(cmp.round(3).to_string())


def sig(c):
    res = {}
    for h in (1, 7):
        d = (c.xs(h, level="HorizonDays").AR_ENFS_A7 - c.xs(h, level="HorizonDays").ePL).values
        t = stats.ttest_rel(c.xs(h, level="HorizonDays").AR_ENFS_A7, c.xs(h, level="HorizonDays").ePL)
        wcx = stats.wilcoxon(d)
        res[h] = dict(mean=d.mean(), sd=d.std(ddof=1), t=t.statistic, p=t.pvalue, wp=wcx.pvalue, n_better=int((d < 0).sum()))
    return res
print("=== SIG full", sig(cmp))

# hold-out window: forecast origin >= 2023-01-01 (Date column = forecast origin)
ho = pred[pred.Date >= "2023-01-01"].copy()
hr = ho.groupby(["Model", "City", "HorizonDays"]).apply(lambda x: np.sqrt(np.mean((x.y_true - x.y_pred) ** 2)), include_groups=False).rename("RMSE").reset_index()
n = ho.groupby(["Model", "City", "HorizonDays"]).size().groupby(["Model", "HorizonDays"]).sum()
print("=== HOLDOUT n per model/h", n.loc["ePL"].to_dict())
ht = hr.pivot_table(index="Model", columns="HorizonDays", values="RMSE", aggfunc="mean"); ht["Pooled"] = ht.mean(axis=1); print(ht.sort_values("Pooled").round(3))
hw = hr.loc[hr.groupby(["City", "HorizonDays"]).RMSE.idxmin()]; print(hw.to_string()); print(hw.Model.value_counts())
hc = hr[hr.Model.isin(["AR_ENFS_A7", "ePL"])].pivot_table(index=["City", "HorizonDays"], columns="Model", values="RMSE"); print(hc.round(3)); print("=== SIG holdout", sig(hc))
print("origin range", ho.Date.min(), ho.Date.max())

# warm-up
rows = []
for w_ in (300, 430):
    s = pd.read_csv(V / f"warmup_{w_}/summary.csv"); s["Warmup"] = w_; rows.append(s)
base = bench[bench.Model.isin(["AR_ENFS_A7", "ePL"])].copy(); base["Warmup"] = 365; rows.append(base[s.columns])
wu = pd.concat(rows)
cv = wu.groupby(["Model", "City", "HorizonDays"]).RMSE.agg(lambda x: x.std(ddof=0) / x.mean() * 100).groupby("Model").mean(); print("=== WARMUP CV", cv.round(3).to_dict())
pv = wu.pivot_table(index=["City", "HorizonDays", "Warmup"], columns="Model", values="RMSE"); pv["winner"] = np.where(pv.AR_ENFS_A7 < pv.ePL, "A7", "ePL")
ref = pv.xs(365, level="Warmup").winner
flips = [(c, h, w_) for (c, h, w_), r in pv.iterrows() if r.winner != ref.loc[(c, h)]]
print("=== WARMUP flips", flips, len(pv)); print(pv.round(3).to_string())

# perturbation if available
if (V / "extra/perturbation.csv").exists():
    p = pd.read_csv(V / "extra/perturbation.csv"); p["deg"] = (p.RMSE_noisy / p.RMSE_clean - 1) * 100
    cvs = p.groupby(["Model", "City", "HorizonDays"]).RMSE_noisy.agg(lambda x: x.std(ddof=0) / x.mean() * 100)
    print("=== PERTURB", p.groupby(["Model", "HorizonDays"]).deg.mean().round(2).to_dict(), cvs.groupby(["Model", "HorizonDays"]).mean().round(2).to_dict(),
          p.groupby("Model").deg.mean().round(2).to_dict())
if (V / "extra/rules_physical.csv").exists():
    r = pd.read_csv(V / "extra/rules_physical.csv"); sel = []
    for (c, h), g in r.groupby(["City", "HorizonDays"]):
        top = g.sort_values("support", ascending=False).head(3)
        sel.append(top.loc[top.tmean_c.idxmax()].to_dict() | {"Regime": "Warm"}); sel.append(top.loc[top.tmean_c.idxmin()].to_dict() | {"Regime": "Cool"})
    print("=== RULES"); print(pd.DataFrame(sel)[["City", "HorizonDays", "Regime", "support", "month", "tmean_c", "day_length_hours", "ConsumptionLag1"]].round(2).to_string())
