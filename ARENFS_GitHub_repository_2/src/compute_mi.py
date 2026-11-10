"""Mutual information table (Table 3 of the paper)."""
import sys, numpy as np, pandas as pd
from sklearn.feature_selection import mutual_info_regression
d = pd.read_csv(sys.argv[1] if len(sys.argv) > 1 else "../data/dataset_meteo_com_consumo.csv.gz").dropna(subset=["consumo_gwh"])
f = ["day_length_hours", "sunshine_sec", "rad_solar", "tmean_c", "tmax_c", "HDD18", "tmin_c", "wind_speed_max", "nebulosidade_media", "precip_mm", "humidade_relativa", "CDD22"]
mi = mutual_info_regression(d[f], d.consumo_gwh, random_state=0)
print(pd.Series(mi, f).sort_values(ascending=False).round(3).to_string())
