#!/usr/bin/env python3
"""Final multi-city evolving-fuzzy benchmark with structural interpretability.

Protocol: chronological warm-up followed by strict predict -> score -> update.
The validated AR-ENFS implementation must be placed beside this file as
``enfs_energy_experimental.py`` and expose ``ENFSEnergyRegressor``.
"""
from __future__ import annotations
import argparse, importlib, inspect, json, math, platform, sys, time, traceback, unicodedata
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import SGDRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.preprocessing import MinMaxScaler, StandardScaler

EXTERNAL = ["ePL", "ePL_plus", "exTS", "Simpl_eTS", "eMG"]
BASELINES = ["Persistence", "SeasonalPersistence7", "OnlineRLS"]


def token(x: str) -> str:
    return unicodedata.normalize("NFKD", str(x)).encode("ascii", "ignore").decode().strip().upper().replace(" ", "_").replace("-", "_")


def read_csv(path: Path) -> pd.DataFrame:
    for sep in (",", ";", None):
        try:
            kw = dict(encoding="utf-8-sig", low_memory=False)
            if sep is None: kw.update(sep=None, engine="python")
            else: kw["sep"] = sep
            df = pd.read_csv(path, **kw)
            if df.shape[1] > 1:
                df.columns = [str(c).strip().lstrip("\ufeff") for c in df.columns]
                return df
        except Exception:
            pass
    raise ValueError(f"Could not parse {path}")


def select_city(df: pd.DataFrame, city: str) -> pd.DataFrame:
    z = df.copy(); z["date"] = pd.to_datetime(z["date"], errors="coerce"); z = z.dropna(subset=["date"])
    z["_zone"] = z["zone"].map(token); c = token(city)
    if c not in set(z["_zone"]): raise ValueError(f"City {city} not found")
    return z.loc[z["_zone"] == c].drop(columns="_zone").sort_values("date").drop_duplicates("date").reset_index(drop=True)


def make_data(df: pd.DataFrame, h: int) -> Tuple[pd.DataFrame,pd.Series,pd.Series,pd.Series]:
    w = df.sort_values("date").reset_index(drop=True).copy(); y0 = pd.to_numeric(w["consumo_gwh"], errors="coerce")
    w["CurrentConsumption"] = y0
    w["ConsumptionLag1"] = y0.shift(1); w["ConsumptionLag7"] = y0.shift(7)
    w["ConsumptionRoll7"] = y0.shift(1).rolling(7, min_periods=3).mean(); w["ConsumptionRoll30"] = y0.shift(1).rolling(30, min_periods=7).mean()
    w["ConsumptionDelta1"] = y0.shift(1)-y0.shift(2)
    dt = pd.to_datetime(w["date"]); doy = dt.dt.dayofyear.astype(float); dow = dt.dt.dayofweek.astype(float)
    w["SeasonSin"] = np.sin(2*np.pi*(doy-1)/365.25); w["SeasonCos"] = np.cos(2*np.pi*(doy-1)/365.25)
    w["WeekSin"] = np.sin(2*np.pi*dow/7); w["WeekCos"] = np.cos(2*np.pi*dow/7)
    target = y0.shift(-h); anchor = w["CurrentConsumption"]
    excluded={"time","date","zone","sunrise","sunset","latitude","longitude","consumo_gwh","target"}
    cols=[c for c in w.columns if c not in excluded]
    X=w[cols].apply(pd.to_numeric,errors="coerce"); valid=X.notna().all(axis=1)&target.notna()&anchor.notna()
    return X.loc[valid].reset_index(drop=True),target.loc[valid].astype(float).reset_index(drop=True),anchor.loc[valid].astype(float).reset_index(drop=True),dt.loc[valid].reset_index(drop=True)


def construct(cls: type, kwargs: Dict[str,Any]) -> Any:
    sig=inspect.signature(cls)
    if any(p.kind==p.VAR_KEYWORD for p in sig.parameters.values()): return cls(**kwargs)
    return cls(**{k:v for k,v in kwargs.items() if k in sig.parameters})

class ExternalAdapter:
    """Version-tolerant adapter for evolvingfuzzysystems models.

    Different releases/models disagree about target shape and prediction return
    type. This adapter tests the supported conventions without changing the
    scientific protocol.
    """
    def __init__(self, name, cls):
        self.name = name
        self.cls = cls
        self.model = cls()

    def fit(self, X, y):
        X = np.ascontiguousarray(X, dtype=np.float64)
        y = np.ascontiguousarray(y, dtype=np.float64).reshape(-1)
        errors = []
        for yy in (y, y.reshape(-1, 1)):
            try:
                self.model.fit(X, yy)
                return
            except Exception as exc:
                errors.append(repr(exc))

        # Some implementations cannot initialize from the complete warm-up
        # block. Seed with a few observations and then evolve sequentially.
        for seed in (2, 3, 5, 10):
            if len(y) <= seed:
                continue
            try:
                self.model = self.cls()
                initialized = False
                for yy in (y[:seed], y[:seed].reshape(-1, 1)):
                    try:
                        self.model.fit(X[:seed], yy)
                        initialized = True
                        break
                    except Exception as exc:
                        errors.append(f"seed={seed}: {exc!r}")
                if not initialized:
                    continue
                for i in range(seed, len(y)):
                    self.update(X[i], y[i])
                return
            except Exception as exc:
                errors.append(f"seed={seed}: {exc!r}")
        raise RuntimeError("external fit failed: " + " | ".join(errors[-8:]))

    @staticmethod
    def _prediction_scalar(value):
        # Some models return (prediction, auxiliary_information).
        if isinstance(value, tuple):
            value = value[0]
        elif isinstance(value, list) and value:
            first = value[0]
            if np.isscalar(first) or np.asarray(first).size == 1:
                value = first
        arr = np.asarray(value, dtype=np.float64)
        if arr.size == 0:
            raise ValueError("empty prediction")
        return float(arr.reshape(-1)[0])

    def predict(self, x):
        row = np.ascontiguousarray(x, dtype=np.float64)
        errors = []
        calls = [
            lambda: self.model.predict(row.reshape(1, -1)),
            lambda: self.model.predict(row.reshape(-1)),
        ]
        if hasattr(self.model, "predict_one"):
            calls.append(lambda: self.model.predict_one(row.reshape(-1)))
        for call in calls:
            try:
                return self._prediction_scalar(call())
            except Exception as exc:
                errors.append(repr(exc))
        raise RuntimeError("external prediction failed: " + " | ".join(errors[-4:]))

    def update(self, x, y):
        X = np.ascontiguousarray(x, dtype=np.float64).reshape(1, -1)
        y1 = np.ascontiguousarray([float(y)], dtype=np.float64)
        errors = []
        for yy in (y1, y1.reshape(-1, 1)):
            try:
                self.model.evolve(X, yy)
                return
            except Exception as exc:
                errors.append(repr(exc))
        raise RuntimeError("external update failed: " + " | ".join(errors[-4:]))

    def rules(self): return generic_rule_count(self.model)
    def snapshot(self): return introspect_rules(self.model)

class OnlineRLS:
    def __init__(self,n): self.model=SGDRegressor(loss="squared_error",penalty=None,learning_rate="constant",eta0=.005,random_state=42); self.ready=False
    def fit(self,X,y): self.model.partial_fit(X,y); self.ready=True
    def predict(self,x): return float(self.model.predict(np.asarray(x).reshape(1,-1))[0]) if self.ready else 0.0
    def update(self,x,y): self.model.partial_fit(np.asarray(x).reshape(1,-1),[y]); self.ready=True
    def rules(self): return 1.0
    def snapshot(self): return pd.DataFrame()

class ARENFSAdapter:
    def __init__(self,name,cfg,names,h):
        mod=importlib.import_module("enfs_energy_experimental"); cls=getattr(mod,"ENFSEnergyRegressor")
        kw=dict(cfg); kw.update(n_features=len(names),feature_names=list(names),forecast_horizon_days=h,
            anchor_feature_index=list(names).index("CurrentConsumption"),circular_sin_feature_index=list(names).index("SeasonSin"),
            circular_cos_feature_index=list(names).index("SeasonCos"),record_history=True)
        self.model=construct(cls,kw); self.name=name
    def fit(self,X,y,a):
        for xi,yi,ai in zip(X,y,a): self.model.learn_one(xi,float(yi),baseline_value=float(ai))
    def predict(self,x,a): return float(self.model.predict_one(x,baseline_value=float(a)))
    def update(self,x,y,a): self.model.learn_one(x,float(y),baseline_value=float(a))
    def rules(self): return float(getattr(self.model,"n_rules_",np.nan))
    def snapshot(self):
        try: return pd.DataFrame(self.model.rule_diagnostics())
        except Exception: return pd.DataFrame()
    def cells(self):
        try: return pd.DataFrame(self.model.regime_cell_diagnostics())
        except Exception: return pd.DataFrame()
    def history(self): return pd.DataFrame(getattr(self.model,"history",[]))


def generic_rule_count(obj):
    for attr in ("n_rules_","n_rules","parameters_list","parameters","rules","rule_base","clusters"):
        if hasattr(obj,attr):
            try:
                v=getattr(obj,attr); v=v() if callable(v) else v
                return float(v) if np.isscalar(v) else float(len(v))
            except Exception: pass
    return np.nan


def introspect_rules(obj):
    rows=[]
    for attr in ("parameters_list","parameters","rules","rule_base","clusters"):
        if not hasattr(obj,attr): continue
        try:
            seq=getattr(obj,attr); seq=seq() if callable(seq) else seq
            if isinstance(seq,dict): seq=list(seq.values())
            for i,r in enumerate(seq):
                row={"rule_id":i}
                if isinstance(r,dict):
                    for k,v in r.items():
                        if np.isscalar(v): row[str(k)]=v
                        elif isinstance(v,(list,tuple,np.ndarray)) and np.asarray(v).size<=30: row[str(k)]=json.dumps(np.asarray(v).tolist())
                else:
                    for k in dir(r):
                        if k.startswith("_"): continue
                        try:
                            v=getattr(r,k)
                            if np.isscalar(v): row[k]=v
                        except Exception: pass
                rows.append(row)
            break
        except Exception: pass
    return pd.DataFrame(rows)


def metrics(y,p,persist):
    e=np.asarray(y)-np.asarray(p); ae=np.abs(e); rmse=math.sqrt(mean_squared_error(y,p)); pr=math.sqrt(mean_squared_error(y,persist))
    return dict(RMSE=rmse,MAE=mean_absolute_error(y,p),NRMSE=rmse/max(np.std(y),1e-12),R2=r2_score(y,p),
        RMSE_ratio_to_persistence=rmse/max(pr,1e-12),MaxAbsError=float(ae.max()),P95AbsError=float(np.percentile(ae,95)),InvalidPredictions=int((~np.isfinite(p)).sum()))

@dataclass
class RunResult:
    summary: Dict[str,Any]; predictions: pd.DataFrame; rule_history: pd.DataFrame; rule_snapshot: pd.DataFrame; cell_snapshot: pd.DataFrame


def run_model(name,Xdf,y,a,ts,warmup,h,cfg,external):
    Xraw=Xdf.to_numpy(float); scaler=MinMaxScaler().fit(Xraw[:warmup]); X=scaler.transform(Xraw)
    below=float((X<0).mean()); above=float((X>1).mean()); X=np.nan_to_num(X,nan=0,posinf=1,neginf=0)
    start=time.perf_counter(); rule_hist=[]
    if name=="Persistence": model=None
    elif name=="SeasonalPersistence7": model=None
    elif name=="OnlineRLS": model=OnlineRLS(X.shape[1]); model.fit(X[:warmup],y[:warmup])
    elif name in external: model=ExternalAdapter(name,external[name]); model.fit(X[:warmup],y[:warmup])
    elif name.startswith("AR_ENFS"):
        model=ARENFSAdapter(name,cfg["ar_enfs_configurations"][name],Xdf.columns,h); model.fit(X[:warmup],y[:warmup].to_numpy(),a[:warmup].to_numpy())
    else: raise ValueError(name)
    warm=time.perf_counter()-start; yt=[]; yp=[]; pp=[]; dates=[]; pt=0.; ut=0.
    for t in range(warmup,len(y)):
        tick=time.perf_counter()
        if name=="Persistence": pred=float(a.iloc[t])
        elif name=="SeasonalPersistence7": pred=float(y.iloc[t-7]) if t>=7 else float(a.iloc[t])
        elif name.startswith("AR_ENFS"): pred=model.predict(X[t],a.iloc[t])
        else: pred=model.predict(X[t])
        pt+=time.perf_counter()-tick
        if not np.isfinite(pred): pred=float(a.iloc[t])
        yt.append(float(y.iloc[t])); yp.append(pred); pp.append(float(a.iloc[t])); dates.append(ts.iloc[t])
        tick=time.perf_counter()
        if name not in BASELINES[:2]:
            if name.startswith("AR_ENFS"): model.update(X[t],y.iloc[t],a.iloc[t])
            else: model.update(X[t],y.iloc[t])
        ut+=time.perf_counter()-tick
        rule_hist.append({"Date":ts.iloc[t],"Rules":0 if model is None else model.rules()})
    m=metrics(np.array(yt),np.array(yp),np.array(pp)); total=warm+pt+ut
    m.update(Model=name,HorizonDays=h,Warmup=warmup,EvaluatedSamples=len(yt),FinalRules=rule_hist[-1]["Rules"] if rule_hist else np.nan,
        MaxRules=float(pd.DataFrame(rule_hist)["Rules"].max()) if rule_hist else 0,MeanRules=float(pd.DataFrame(rule_hist)["Rules"].mean()) if rule_hist else 0,
        WarmupTime_s=warm,PredictionTime_s=pt,UpdateTime_s=ut,TotalTime_s=total,MeanPrediction_ms=1000*pt/max(len(yt),1),MeanUpdate_ms=1000*ut/max(len(yt),1),
        ScaledBelowZeroFraction=below,ScaledAboveOneFraction=above)
    pred_df=pd.DataFrame({"Date":dates,"y_true":yt,"y_pred":yp,"Persistence":pp}); pred_df["Error"]=pred_df.y_true-pred_df.y_pred; pred_df["AbsError"]=pred_df.Error.abs(); pred_df["RollingRMSE"]=np.sqrt(pred_df.Error.pow(2).rolling(cfg["rolling_window"],min_periods=1).mean())
    snap=pd.DataFrame() if model is None else model.snapshot(); cells=model.cells() if name.startswith("AR_ENFS") else pd.DataFrame()
    return RunResult(m,pred_df,pd.DataFrame(rule_hist),snap,cells)


def report(out:Path,summary:pd.DataFrame):
    s=summary.sort_values(["City","HorizonDays","RMSE"]); lines=["# Final Evolving-Fuzzy Benchmark Report","","## Best models by context",""]
    for (c,h),g in s.groupby(["City","HorizonDays"]):
        r=g.iloc[0]; lines += [f"### {c} — {int(h)}-day horizon",f"- Best model: **{r.Model}**",f"- RMSE: **{r.RMSE:.4f}**",f"- RMSE/persistence: **{r.RMSE_ratio_to_persistence:.4f}**",f"- Final/max rules: **{r.FinalRules:.0f}/{r.MaxRules:.0f}**",""]
    lines += ["## Interpretation boundary","","Structural diagnostics describe observable model behavior. They are not causal conclusions and must be interpreted together with controlled experiments and domain knowledge."]
    (out/"FINAL_AUTOMATIC_REPORT.md").write_text("\n".join(lines),encoding="utf-8")


def figures(out:Path,allpred:pd.DataFrame,summary:pd.DataFrame,rules:pd.DataFrame):
    fd=out/"figures"; fd.mkdir(exist_ok=True)
    for (c,h),g in allpred.groupby(["City","HorizonDays"]):
        plt.figure(figsize=(11,5))
        for model,r in g.groupby("Model"): plt.plot(r.Date,r.RollingRMSE,label=model,linewidth=.8)
        plt.title(f"Rolling RMSE — {c} — t+{h}d"); plt.legend(ncol=3,fontsize=6); plt.tight_layout(); plt.savefig(fd/f"rolling_rmse_{c}_h{h}.png",dpi=180); plt.close()
    for h,g in summary.groupby("HorizonDays"):
        piv=g.pivot(index="City",columns="Model",values="RMSE")
        plt.figure(figsize=(12,5)); plt.imshow(piv.values,aspect="auto"); plt.xticks(range(len(piv.columns)),piv.columns,rotation=70,ha="right",fontsize=7); plt.yticks(range(len(piv.index)),piv.index); plt.colorbar(label="RMSE"); plt.title(f"RMSE heatmap — t+{h}d"); plt.tight_layout(); plt.savefig(fd/f"rmse_heatmap_h{h}.png",dpi=180); plt.close()


def main():
    ap=argparse.ArgumentParser(); ap.add_argument("--csv",required=True); ap.add_argument("--profile",choices=["smoke","pilot","paper"],default="smoke"); ap.add_argument("--config",default="final_benchmark_config.json"); ap.add_argument("--output",default="final_benchmark_results"); ap.add_argument("--models",nargs="+"); args=ap.parse_args()
    root=Path(__file__).resolve().parent; cfg=json.loads((root/args.config).read_text()); prof=cfg["profiles"][args.profile]; models=args.models or cfg["models"]
    out=root/args.output; out.mkdir(parents=True,exist_ok=True); raw=read_csv(Path(args.csv)); required={"date","zone","consumo_gwh"};
    if missing:=required-set(raw.columns): raise ValueError(f"Missing {missing}")
    try:
        from evolvingfuzzysystems.eFS import ePL,ePL_plus,exTS,Simpl_eTS,eMG
        ext={c.__name__:c for c in [ePL,ePL_plus,exTS,Simpl_eTS,eMG]}
    except Exception as exc:
        ext={}; print(f"External models unavailable: {exc}")
    summaries=[]; preds=[]; histories=[]; failures=[]
    total=len(prof["cities"])*len(prof["horizons"])*len(models); done=0; began=time.time()
    for city in prof["cities"]:
        citydf=select_city(raw,city)
        if prof.get("max_rows"): citydf=citydf.tail(int(prof["max_rows"])).reset_index(drop=True)
        for h in prof["horizons"]:
            X,y,a,ts=make_data(citydf,int(h)); warm=min(int(prof["warmup"]),max(30,len(y)//3))
            for name in models:
                done+=1; print(f"[{done}/{total}] {city} h={h} {name} ...",flush=True)
                try:
                    rr=run_model(name,X,y,a,ts,warm,int(h),cfg,ext); rr.summary["City"]=city; summaries.append(rr.summary)
                    p=rr.predictions; p["City"]=city;p["HorizonDays"]=h;p["Model"]=name;preds.append(p)
                    rh=rr.rule_history;rh["City"]=city;rh["HorizonDays"]=h;rh["Model"]=name;histories.append(rh)
                    d=out/"interpretability"/city/f"h{h}"/name; d.mkdir(parents=True,exist_ok=True); rr.rule_snapshot.to_csv(d/"rule_snapshot.csv",index=False); rr.cell_snapshot.to_csv(d/"regime_cell_snapshot.csv",index=False)
                    print(f"  RMSE={rr.summary['RMSE']:.4f} R2={rr.summary['R2']:.4f} rules={rr.summary['FinalRules']:.0f} time={rr.summary['TotalTime_s']:.2f}s")
                except Exception as exc:
                    failures.append({"City":city,"HorizonDays":h,"Model":name,"Error":repr(exc),"Traceback":traceback.format_exc()}); print(f"  FAILED: {exc}")
                pd.DataFrame(summaries).to_csv(out/"benchmark_summary_checkpoint.csv",index=False); pd.DataFrame(failures).to_csv(out/"failures.csv",index=False)
    summary=pd.DataFrame(summaries); allpred=pd.concat(preds,ignore_index=True) if preds else pd.DataFrame(); rules=pd.concat(histories,ignore_index=True) if histories else pd.DataFrame()
    if not summary.empty:
        summary["Rank_RMSE"]=summary.groupby(["City","HorizonDays"])["RMSE"].rank(); summary["Rank_Rules"]=summary.groupby(["City","HorizonDays"])["MaxRules"].rank(); summary["Rank_Time"]=summary.groupby(["City","HorizonDays"])["TotalTime_s"].rank(); summary["MeanRank"]=summary[["Rank_RMSE","Rank_Rules","Rank_Time"]].mean(axis=1)
        summary.to_csv(out/"final_benchmark_summary.csv",index=False); allpred.to_csv(out/"final_predictions.csv",index=False); rules.to_csv(out/"final_rule_history.csv",index=False); figures(out,allpred,summary,rules); report(out,summary)
    meta=dict(profile=args.profile,models=models,cities=prof["cities"],horizons=prof["horizons"],protocol="warm-up then strict predict-score-update",python=sys.version,platform=platform.platform(),elapsed_sec=time.time()-began)
    (out/"run_metadata.json").write_text(json.dumps(meta,indent=2),encoding="utf-8"); print(f"\nDone: {out.resolve()}")
if __name__=="__main__": main()
