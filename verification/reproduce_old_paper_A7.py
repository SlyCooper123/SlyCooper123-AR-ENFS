"""Shows where the AR-ENFS numbers of the OLD manuscript (Applied Energy version, with `year` as input) came from.

Run from the src/ folder:  python ../verification/reproduce_old_paper_A7.py ../data/dataset_meteo_com_consumo.csv.gz
1) Final code (sum/conjunctive activation, regime tau = 6) WITH `year`  -> reproduces the old paper's A7 exactly
   (mean RMSE 6.4235 over the 12 scenarios; Lisboa 5.7858 / 7.0083).
2) Original run_pilot code (mean activation) on Lisboa                 -> different numbers, i.e. it was not the
   code that produced the old paper.
"""
import sys, importlib, shutil, tempfile
from pathlib import Path
import numpy as np, pandas as pd
sys.path.insert(0, ".")
import run_all as A, run_final_benchmark as R

csv = sys.argv[1] if len(sys.argv) > 1 else "../data/dataset_meteo_com_consumo.csv.gz"
raw = A.load(csv); cfg = A.cfg_for(["A7"])
rows = []
for city in A.CITIES:
    for h in A.HORIZONS:
        X, y, a, ts = A.data_for(raw, city, h, drop_year=False)
        rows.append((city, h, R.run_model("AR_ENFS_A7", X, y, a, ts, 365, h, cfg, {}).summary["RMSE"]))
df = pd.DataFrame(rows, columns=["City", "h", "RMSE"]); print(df.round(4).to_string())
print("Mean RMSE A7 with year (old paper: 6.4235):", round(df.RMSE.mean(), 4))

orig = Path(__file__).resolve().parent / "original_code/enfs_energy_experimental_ORIGINAL_mean_activation.py"
tmp = Path(tempfile.mkdtemp()); shutil.copy(orig, tmp / "enfs_energy_experimental.py"); sys.path.insert(0, str(tmp))
del sys.modules["enfs_energy_experimental"]; importlib.invalidate_caches()
for h in A.HORIZONS:
    X, y, a, ts = A.data_for(raw, "LISBOA", h, drop_year=False)
    print(f"ORIGINAL mean-activation code, Lisboa h={h}:", round(R.run_model("AR_ENFS_A7", X, y, a, ts, 365, h, cfg, {}).summary["RMSE"], 4),
          "(old paper: 5.7858 / 7.0083)")
