# AR-ENFS: Adaptive Residual Evolving Neuro-Fuzzy System

Code, data, results and manuscript source for
*"AR-ENFS: An Adaptive Residual Evolving Neuro-Fuzzy System for Online Electricity-Demand Forecasting"*
(P. I. Catarro, P. V. de Campos Souza, I. Scott; submitted to *Applied Soft Computing*).

## Repository map

```
data/
  dataset_meteo_com_consumo.csv.gz        municipality-day panel (50 municipalities; consumption 2015-01-01 .. 2025-09-30)
src/                                       everything needed to reproduce the paper
  enfs_energy_experimental.py              AR-ENFS model (class ENFSEnergyRegressor)
  run_final_benchmark.py                   data preparation, prequential loop, adapters for all models
  run_all.py                               ablation A0-A9, external benchmark, warm-up sensitivity
  run_extra.py                             input-perturbation study, rules in physical units
  analyze.py                               turns the results into every number/table of Section 5
  compute_mi.py                            mutual-information table
  make_figures.py                          Figures 1-5
  make_graphical_abstract.py               graphical abstract
results/
  out_no_year/                             outputs used in the paper (inputs WITHOUT the calendar year)
    ablation/                              A0-A9: summary.csv, predictions.csv.gz, rule_snapshots.csv.gz (final rule bases)
    benchmark_external/                    ePL, ePL+, exTS, Simpl_eTS, eMG, persistence, seasonal persistence, online SGD
    warmup_300/, warmup_430/               warm-up sensitivity (A7 and ePL)
    extra/perturbation.csv                 input-noise experiment (A7 and ePL, 5 noise realizations)
    extra/rules_physical.csv               rule centres of the final A7 models in physical units
  analysis_summary.txt                     output of analyze.py (all numbers reported in Section 5)
paper/
  main.tex, sample.bib, figures/           LaTeX source of the submitted manuscript
  submission_files/                        highlights, graphical abstract
verification/                              how the previous manuscript version was checked (see below)
```

## Where each result of the paper comes from

| Paper item | File(s) | Produced by |
|---|---|---|
| Table 1 (hyperparameters) | `src/enfs_energy_experimental.py` (constructor defaults), `BASE` in `src/run_all.py` | -- |
| Table 2 (panel overview), Figs. 1-4 | `data/dataset_meteo_com_consumo.csv.gz` | `make_figures.py` |
| Table 3 (mutual information) | same data | `compute_mi.py` |
| Table 4 (ablation configurations) | `ABLATION` in `src/run_all.py` | -- |
| Table 5 (ablation results) | `results/out_no_year/ablation/summary.csv` | `run_all.py --part ablation`, `analyze.py` |
| Table 6, Fig. 5 (benchmark) | `ablation/summary.csv` (A0, A1, A7) + `benchmark_external/summary.csv` | `run_all.py --part benchmark`, `analyze.py` |
| Table 7 (winners, A7 vs ePL) and significance tests | same | `analyze.py` ("=== SIG full") |
| Table 8 (2023-2025 scoring) and tests | `*/predictions.csv.gz` | `analyze.py` ("=== HOLDOUT", "=== SIG holdout") |
| Table 9 (rules in physical units) | `extra/rules_physical.csv` | `run_extra.py --part rules`, `analyze.py` ("=== RULES") |
| Table 10 (perturbation) | `extra/perturbation.csv` | `run_extra.py --part perturb`, `analyze.py` ("=== PERTURB") |
| Warm-up sensitivity (Sec. 5.3.4) | `warmup_300/`, `warmup_430/` | `run_all.py --part warmup`, `analyze.py` ("=== WARMUP") |

## Installation and full reproduction

```bash
python -m venv venv && source venv/bin/activate      # Windows: venv\Scripts\activate
pip install -r requirements.txt
cd src
DATA=../data/dataset_meteo_com_consumo.csv.gz
python run_all.py   --csv $DATA --out ../results/out_no_year --variant no_year --part ablation
python run_all.py   --csv $DATA --out ../results/out_no_year --variant no_year --part benchmark
python run_all.py   --csv $DATA --out ../results/out_no_year --variant no_year --part warmup
python run_extra.py --csv $DATA --out ../results/out_no_year/extra --variant no_year --part rules
python run_extra.py --csv $DATA --out ../results/out_no_year/extra --variant no_year --part perturb
python analyze.py ../results/out_no_year
python compute_mi.py $DATA
python make_figures.py $DATA ../results/out_no_year
```

All runs are deterministic (AR-ENFS has no stochastic component). The whole pipeline takes about one hour on a
laptop, dominated by eMG. `--variant no_year` removes the calendar-year counter from the inputs (29 inputs), as in the
paper; `--variant with_year` keeps it (30 inputs) and reproduces the previous manuscript version.

## Data sources

* National daily electricity consumption (GWh): REN Data Hub, Electricity - Daily balance,
  https://datahub.ren.pt/en/electricity/daily-balance/
* Daily meteorological variables: Open-Meteo historical weather API, https://open-meteo.com/ (doi:10.5281/zenodo.7970649).
  Heating/cooling degree days (bases 18 and 22 degC), thermal amplitude, day length and calendar variables were derived
  by the authors. The same national target is repeated for every municipality. The `is_holiday` column is zero for
  every day and therefore uninformative.

## Protocol in brief

* Six cities (Braga, Coimbra, Evora, Faro, Lisboa, Porto) x two horizons (1 and 7 days).
* 365-sample warm-up (learned, never scored), then strict predict -> score -> update.
* Min-max scaling fitted on the warm-up block only; AR-ENFS additionally standardizes online.
* External evolving fuzzy systems from `evolvingfuzzysystems` 0.1.5 with default hyperparameters.

## Verification of the previous manuscript version (`verification/`)

The previous version of the manuscript (Applied Energy format) included the calendar year among the inputs and its
AR-ENFS numbers came from an updated model version that was not the one distributed in the original benchmark kit.

* `verification/original_code/` - the original model file (rule activation based on the **mean** of squared
  deviations), the original benchmark configuration (regime parameters named `n_regimes`/`regime_mode`, which the model
  silently ignored), and `DIFF_original_vs_final_model.txt` (the only change in the final model: **sum** of squared
  deviations, i.e. conjunctive activation, as described in the paper).
* `verification/reproduce_old_paper_A7.py` - run from `src/`: with the final model, regime confidence constant 6 and
  `year` included, it reproduces the old manuscript's AR-ENFS A7 results exactly (mean RMSE 6.4235 over the 12
  scenarios; e.g. Lisboa 5.7858 / 7.0083), whereas the original mean-activation code does not.
* `verification/results_with_year/` - the old synthesis tables, the old benchmark run, the reproduction run, and the
  ablation re-run with `year` for comparison.
