#!/usr/bin/env python3
"""Run ablation + benchmark + validation experiments for AR-ENFS with ONE code base.

Usage: python run_all.py --csv data.csv --out OUT --variant {with_year,no_year} --part {ablation,benchmark,warmup}   (perturbation and rules: run_extra.py)
"""
import argparse, json, sys, time, copy
from pathlib import Path
import numpy as np, pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import run_final_benchmark as R  # noqa: E402

CITIES = ["BRAGA", "COIMBRA", "EVORA", "FARO", "LISBOA", "PORTO"]
HORIZONS = [1, 7]

BASE = dict(scale_x=True, regime_shrinkage_tau=6.0)
ABLATION = {
    "A0": dict(scale_y=False, prediction_mode="mixed", residual_gate=1.0, max_rules=50, seasonal_regimes=3, seasonal_mode="hard"),
    "A1": dict(scale_y=True, prediction_mode="mixed", residual_gate=1.0, max_rules=50, seasonal_regimes=3, seasonal_mode="hard"),
    "A2": dict(scale_y=True, prediction_mode="residual", residual_gate=1.0, max_rules=20, seasonal_regimes=0, seasonal_mode="hard"),
    "A3": dict(scale_y=True, prediction_mode="direct", residual_gate=0.0, max_rules=20, seasonal_regimes=0, seasonal_mode="hard"),
    "A4": dict(scale_y=True, prediction_mode="mixed", residual_gate=0.5, max_rules=20, seasonal_regimes=0, seasonal_mode="hard"),
    "A5": dict(scale_y=True, prediction_mode="mixed", residual_gate=0.75, max_rules=12, seasonal_regimes=0, seasonal_mode="hard"),
    "A6": dict(scale_y=True, prediction_mode="mixed", residual_gate=0.75, max_rules=12, seasonal_regimes=2, seasonal_mode="hard"),
    "A7": dict(scale_y=True, prediction_mode="mixed", residual_gate=0.75, max_rules=12, seasonal_regimes=3, seasonal_mode="hard"),
    "A8": dict(scale_y=True, prediction_mode="mixed", residual_gate=0.75, max_rules=12, seasonal_regimes=4, seasonal_mode="fuzzy"),
    "A9": dict(scale_y=True, prediction_mode="mixed", residual_gate=0.75, max_rules=20, seasonal_regimes=3, seasonal_mode="hard"),
}


def cfg_for(names):
    c = {"ar_enfs_configurations": {}, "rolling_window": 30}
    for k in names:
        d = dict(BASE); d.update(ABLATION[k]); c["ar_enfs_configurations"][f"AR_ENFS_{k}"] = d
    return c


def load(csv):
    df = pd.read_csv(csv)
    df["date"] = pd.to_datetime(df["date"], format="%Y-%m-%d").dt.strftime("%Y-%m-%d")
    return df


def data_for(raw, city, h, drop_year):
    citydf = R.select_city(raw, city)
    X, y, a, ts = R.make_data(citydf, h)
    if drop_year:
        X = X.drop(columns=["year"])
    return X, y, a, ts


def external_models():
    from evolvingfuzzysystems.eFS import ePL, ePL_plus, exTS, Simpl_eTS, eMG
    return {c.__name__: c for c in [ePL, ePL_plus, exTS, Simpl_eTS, eMG]}


def run_set(raw, models, cfg, out, drop_year, warmup=365, tag=""):
    out.mkdir(parents=True, exist_ok=True)
    ext = external_models()
    rows, preds, snaps = [], [], []
    for city in CITIES:
        for h in HORIZONS:
            X, y, a, ts = data_for(raw, city, h, drop_year)
            for m in models:
                t0 = time.time()
                rr = R.run_model(m, X, y, a, ts, warmup, h, cfg, ext)
                s = rr.summary; s["City"] = city; rows.append(s)
                p = rr.predictions[["Date", "y_true", "y_pred", "Persistence"]].copy(); p["City"] = city; p["HorizonDays"] = h; p["Model"] = m; preds.append(p)
                hist = rr.rule_history
                if m.startswith("AR_ENFS"):
                    snap = rr.rule_snapshot.copy(); snap.insert(0, "HorizonDays", h); snap.insert(0, "City", city); snap.insert(0, "Model", m); snaps.append(snap)
                print(f"{tag} {city} h{h} {m} RMSE={s['RMSE']:.4f} rules={s['FinalRules']} t={time.time()-t0:.1f}s", flush=True)
                pd.DataFrame(rows).to_csv(out / "summary.csv", index=False)
    pd.concat(preds).to_csv(out / "predictions.csv.gz", index=False, compression="gzip")
    if snaps:
        pd.concat(snaps).to_csv(out / "rule_snapshots.csv.gz", index=False, compression="gzip")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", required=True); ap.add_argument("--out", required=True)
    ap.add_argument("--variant", choices=["with_year", "no_year"], required=True)
    ap.add_argument("--part", choices=["ablation", "benchmark", "warmup"], required=True)
    a = ap.parse_args()
    raw = load(a.csv); drop = a.variant == "no_year"; out = Path(a.out)
    if a.part == "ablation":
        names = list(ABLATION)
        run_set(raw, [f"AR_ENFS_{k}" for k in names], cfg_for(names), out / "ablation", drop, tag="abl")
    elif a.part == "benchmark":
        models = ["Persistence", "SeasonalPersistence7", "OnlineRLS", "ePL", "ePL_plus", "exTS", "Simpl_eTS", "eMG"]
        run_set(raw, models, cfg_for([]), out / "benchmark_external", drop, tag="ext")
    elif a.part == "warmup":
        for w in (300, 430):
            run_set(raw, ["AR_ENFS_A7", "ePL"], cfg_for(["A7"]), out / f"warmup_{w}", drop, warmup=w, tag=f"w{w}")


if __name__ == "__main__":
    main()
