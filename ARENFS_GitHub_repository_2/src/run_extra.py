#!/usr/bin/env python3
"""Perturbation robustness and physical-unit rule recovery for AR-ENFS A7 (and ePL)."""
import argparse, math, sys, time
from pathlib import Path
import numpy as np, pandas as pd
from sklearn.preprocessing import MinMaxScaler

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import run_final_benchmark as R  # noqa: E402
from run_all import CITIES, HORIZONS, cfg_for, load, data_for, external_models  # noqa: E402

SIGMA = 0.02
SEEDS = [0, 1, 2, 3, 4]


def prepare(X, warmup):
    Xraw = X.to_numpy(float); sc = MinMaxScaler().fit(Xraw[:warmup]); Xs = np.nan_to_num(sc.transform(Xraw), nan=0, posinf=1, neginf=0)
    return Xs, sc


def build(name, X, y, a, h, Xs, warmup, ext):
    if name == "ePL":
        m = R.ExternalAdapter(name, ext[name]); m.fit(Xs[:warmup], y[:warmup]); return m
    cfg = cfg_for(["A7"]); m = R.ARENFSAdapter("AR_ENFS_A7", cfg["ar_enfs_configurations"]["AR_ENFS_A7"], X.columns, h)
    m.fit(Xs[:warmup], y[:warmup].to_numpy(), a[:warmup].to_numpy()); return m


def prequential(name, X, y, a, h, warmup, ext, seed=None):
    Xs, sc = prepare(X, warmup); m = build(name, X, y, a, h, Xs, warmup, ext)
    rng = None if seed is None else np.random.default_rng(seed)
    yt, yp = [], []
    for t in range(warmup, len(y)):
        xq = Xs[t] if rng is None else Xs[t] + rng.normal(0.0, SIGMA, size=Xs.shape[1])
        p = m.predict(xq, a.iloc[t]) if name != "ePL" else m.predict(xq)
        if not np.isfinite(p): p = float(a.iloc[t])
        yt.append(float(y.iloc[t])); yp.append(p)
        if name == "ePL": m.update(Xs[t], y.iloc[t])
        else: m.update(Xs[t], y.iloc[t], a.iloc[t])
    rmse = math.sqrt(np.mean((np.array(yt) - np.array(yp)) ** 2))
    return rmse, m, sc


def perturb(raw, out, drop):
    ext = external_models(); rows = []
    for city in CITIES:
        for h in HORIZONS:
            X, y, a, ts = data_for(raw, city, h, drop)
            for name in ("AR_ENFS_A7", "ePL"):
                base, _, _ = prequential(name, X, y, a, h, 365, ext)
                for s in SEEDS:
                    r, _, _ = prequential(name, X, y, a, h, 365, ext, seed=s)
                    rows.append(dict(City=city, HorizonDays=h, Model=name, Seed=s, RMSE_clean=base, RMSE_noisy=r))
                print(city, h, name, base, flush=True)
                pd.DataFrame(rows).to_csv(out / "perturbation.csv", index=False)


def rules(raw, out, drop):
    ext = {}; rows = []
    for city in CITIES:
        for h in HORIZONS:
            X, y, a, ts = data_for(raw, city, h, drop)
            rmse, m, sc = prequential("AR_ENFS_A7", X, y, a, h, 365, ext)
            mod = m.model; names = list(X.columns)
            for r in mod.rules:
                z = np.asarray(r.center, float)
                xs = z * mod.x_scaler.scale + mod.x_scaler.mean if mod.scale_x else z
                xr = sc.inverse_transform(xs.reshape(1, -1))[0]
                d = dict(zip(names, xr))
                ang = math.atan2(d["SeasonSin"], d["SeasonCos"]) % (2 * math.pi)
                doy = ang / (2 * math.pi) * 365.25 + 1
                month = pd.Timestamp("2021-01-01") + pd.Timedelta(days=float(doy) - 1)
                rows.append(dict(City=city, HorizonDays=h, rule_id=r.rule_id, support=r.support, month=month.strftime("%b"),
                                 tmean_c=d["tmean_c"], day_length_hours=d["day_length_hours"], ConsumptionLag1=d["ConsumptionLag1"], RMSE=rmse))
            print("rules", city, h, rmse, flush=True)
    pd.DataFrame(rows).to_csv(out / "rules_physical.csv", index=False)


if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("--csv", required=True); ap.add_argument("--out", required=True)
    ap.add_argument("--variant", choices=["with_year", "no_year"], required=True); ap.add_argument("--part", choices=["perturb", "rules"], required=True)
    a = ap.parse_args(); raw = load(a.csv); out = Path(a.out); out.mkdir(parents=True, exist_ok=True); drop = a.variant == "no_year"
    perturb(raw, out, drop) if a.part == "perturb" else rules(raw, out, drop)
