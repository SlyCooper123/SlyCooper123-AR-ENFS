"""Standalone experimental ENFS for daily energy forecasting.

This module is deliberately independent from the original agricultural class
hierarchy. It keeps the main ideas (evolving anisotropic fuzzy rules, local
Takagi-Sugeno consequents, residual/direct paths and optional circular regimes)
while removing soil-moisture assumptions.

The implementation supports leakage-safe online standardization of X and y,
unbounded outputs, day-based horizon profiles, and detailed diagnostics for
ablation studies.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np


_EPS = 1e-12


class OnlineVectorStandardizer:
    def __init__(self, n_features: int, min_scale: float = 1e-6) -> None:
        self.n_features = int(n_features)
        self.min_scale = float(min_scale)
        self.n = 0
        self.mean = np.zeros(self.n_features, dtype=float)
        self.m2 = np.zeros(self.n_features, dtype=float)

    @property
    def scale(self) -> np.ndarray:
        if self.n < 2:
            return np.ones(self.n_features, dtype=float)
        var = self.m2 / max(self.n - 1, 1)
        return np.sqrt(np.maximum(var, self.min_scale**2))

    def transform(self, x: Sequence[float]) -> np.ndarray:
        a = np.asarray(x, dtype=float).ravel()
        if a.size != self.n_features:
            raise ValueError(f"Expected {self.n_features} features, got {a.size}.")
        if self.n < 2:
            return np.zeros_like(a)
        return (a - self.mean) / self.scale

    def update(self, x: Sequence[float]) -> None:
        a = np.asarray(x, dtype=float).ravel()
        if a.size != self.n_features:
            raise ValueError(f"Expected {self.n_features} features, got {a.size}.")
        self.n += 1
        delta = a - self.mean
        self.mean += delta / self.n
        self.m2 += delta * (a - self.mean)


class OnlineScalarStandardizer:
    def __init__(self, min_scale: float = 1e-6) -> None:
        self.min_scale = float(min_scale)
        self.n = 0
        self.mean = 0.0
        self.m2 = 0.0

    @property
    def scale(self) -> float:
        if self.n < 2:
            return 1.0
        return float(math.sqrt(max(self.m2 / max(self.n - 1, 1), self.min_scale**2)))

    def transform(self, value: float) -> float:
        if self.n < 2:
            return 0.0
        return float((float(value) - self.mean) / self.scale)

    def inverse(self, value: float) -> float:
        if self.n < 2:
            return float(self.mean if self.n else value)
        return float(self.mean + self.scale * float(value))

    def update(self, value: float) -> None:
        value = float(value)
        self.n += 1
        delta = value - self.mean
        self.mean += delta / self.n
        self.m2 += delta * (value - self.mean)


class WeightedRLS:
    """Small weighted recursive least-squares learner."""

    def __init__(
        self,
        dimension: int,
        ridge: float = 1e-2,
        prior_variance: float = 10.0,
        lambda_min: float = 0.97,
        lambda_max: float = 0.999,
        adaptive_forgetting: bool = True,
        forgetting_sensitivity: float = 1.0,
        beta_max: float = 50.0,
    ) -> None:
        self.dimension = int(dimension)
        self.ridge = float(ridge)
        self.lambda_min = float(lambda_min)
        self.lambda_max = float(lambda_max)
        self.adaptive_forgetting = bool(adaptive_forgetting)
        self.forgetting_sensitivity = float(forgetting_sensitivity)
        self.beta_max = float(beta_max)
        precision = self.ridge + 1.0 / max(float(prior_variance), _EPS)
        self.P = np.eye(self.dimension, dtype=float) / precision
        self.beta = np.zeros(self.dimension, dtype=float)
        self.n_updates = 0
        self.error_mean = 0.0
        self.error_m2 = 0.0
        self.update_mass = 0.0

    @property
    def error_std(self) -> float:
        if self.n_updates < 2:
            return 0.0
        return float(math.sqrt(max(self.error_m2 / (self.n_updates - 1), 0.0)))

    def _forgetting(self, abs_error: float) -> float:
        if not self.adaptive_forgetting or self.n_updates < 8:
            return self.lambda_max
        sd = self.error_std
        if sd <= _EPS:
            return self.lambda_max
        z = max(0.0, (math.log1p(abs_error) - self.error_mean) / sd)
        drift = 1.0 - math.exp(-self.forgetting_sensitivity * z)
        return float(self.lambda_max - (self.lambda_max - self.lambda_min) * drift)

    def predict(self, phi: Sequence[float]) -> float:
        return float(np.asarray(phi, dtype=float).ravel() @ self.beta)

    def update(self, phi: Sequence[float], target: float, weight: float = 1.0) -> None:
        w = float(np.clip(weight, 0.0, 1.0))
        if w <= 1e-8:
            return
        p = np.asarray(phi, dtype=float).ravel()
        if p.size != self.dimension:
            raise ValueError("RLS feature dimension mismatch.")
        pred = self.predict(p)
        abs_error = abs(float(target) - pred)
        lam = self._forgetting(abs_error)
        pw = math.sqrt(w) * p
        denom = lam + float(pw @ self.P @ pw)
        gain = (self.P @ pw) / max(denom, _EPS)
        innovation = math.sqrt(w) * (float(target) - pred)
        self.beta += gain * innovation
        self.beta = np.clip(self.beta, -self.beta_max, self.beta_max)
        self.P = (self.P - np.outer(gain, pw) @ self.P) / max(lam, _EPS)
        self.P = 0.5 * (self.P + self.P.T)
        self.n_updates += 1
        self.update_mass += w
        logged = math.log1p(abs_error)
        delta = logged - self.error_mean
        self.error_mean += delta / self.n_updates
        self.error_m2 += delta * (logged - self.error_mean)


@dataclass
class EnergyRule:
    rule_id: int
    center: np.ndarray
    sigma: np.ndarray
    support: int
    birth_step: int
    last_update_step: int
    cumulative_activation: float = 0.0
    winner_count: int = 0
    residual: Optional[WeightedRLS] = None
    direct: Optional[WeightedRLS] = None
    regime_residual: Dict[int, WeightedRLS] = field(default_factory=dict)
    regime_direct: Dict[int, WeightedRLS] = field(default_factory=dict)

    def update_geometry(self, x: np.ndarray, min_sigma: float, step: int) -> None:
        self.support += 1
        eta = 1.0 / max(self.support, 1)
        delta = x - self.center
        self.center = self.center + eta * delta
        target_sigma = np.maximum(np.abs(delta), min_sigma)
        self.sigma = np.maximum((1.0 - eta) * self.sigma + eta * target_sigma, min_sigma)
        self.last_update_step = int(step)


DEFAULT_ENERGY_PROFILES: Dict[int, Dict[str, float]] = {
    1: {
        "residual_gate": 0.75,
        "local_ridge": 1e-2,
        "direct_ridge": 1e-2,
        "lambda_min": 0.97,
        "lambda_max": 0.999,
        "maturity_tau": 8.0,
    },
    7: {
        "residual_gate": 0.75,
        "local_ridge": 2e-2,
        "direct_ridge": 2e-2,
        "lambda_min": 0.98,
        "lambda_max": 0.9995,
        "maturity_tau": 12.0,
    },
}


class ENFSEnergyRegressor:
    """Standalone evolving neuro-fuzzy regressor for energy forecasting."""

    def __init__(
        self,
        n_features: int,
        feature_names: Optional[Sequence[str]] = None,
        forecast_horizon_days: int = 1,
        anchor_feature_index: Optional[int] = None,
        scale_x: bool = True,
        scale_y: bool = True,
        prediction_mode: str = "mixed",
        residual_gate: Optional[float] = None,
        max_rules: int = 12,
        initial_sigma: float = 1.0,
        min_sigma: float = 0.05,
        novelty_threshold: float = 1.5,
        error_threshold: float = 1.0,
        uncertainty_threshold: float = 0.35,
        min_samples_before_growth: int = 10,
        min_rule_support_for_update: int = 1,
        local_ridge: Optional[float] = None,
        direct_ridge: Optional[float] = None,
        prior_variance: float = 10.0,
        lambda_min: Optional[float] = None,
        lambda_max: Optional[float] = None,
        adaptive_forgetting: bool = True,
        forgetting_sensitivity: float = 1.0,
        maturity_tau: Optional[float] = None,
        minimum_consequent_updates: int = 2,
        beta_max: float = 50.0,
        seasonal_regimes: int = 0,
        seasonal_mode: str = "hard",
        circular_sin_feature_index: Optional[int] = None,
        circular_cos_feature_index: Optional[int] = None,
        regime_kappa: float = 4.0,
        regime_shrinkage_tau: float = 8.0,
        regime_minimum_updates: int = 3,
        regime_minimum_activation: float = 0.005,
        random_state: int = 42,
        record_history: bool = True,
    ) -> None:
        self.n_features = int(n_features)
        self.feature_names = list(feature_names) if feature_names is not None else [f"x{i}" for i in range(n_features)]
        if len(self.feature_names) != self.n_features:
            raise ValueError("feature_names length must equal n_features.")
        self.forecast_horizon_days = int(forecast_horizon_days)
        profile = DEFAULT_ENERGY_PROFILES.get(
            self.forecast_horizon_days,
            DEFAULT_ENERGY_PROFILES[min(DEFAULT_ENERGY_PROFILES, key=lambda h: abs(h - self.forecast_horizon_days))],
        )
        self.anchor_feature_index = None if anchor_feature_index is None else int(anchor_feature_index)
        self.scale_x = bool(scale_x)
        self.scale_y = bool(scale_y)
        self.prediction_mode = str(prediction_mode).lower().strip()
        if self.prediction_mode not in {"residual", "direct", "mixed"}:
            raise ValueError("prediction_mode must be residual, direct or mixed.")
        self.residual_gate = float(profile["residual_gate"] if residual_gate is None else residual_gate)
        self.residual_gate = float(np.clip(self.residual_gate, 0.0, 1.0))
        self.max_rules = int(max_rules)
        self.initial_sigma = float(initial_sigma)
        self.min_sigma = float(min_sigma)
        self.novelty_threshold = float(novelty_threshold)
        self.error_threshold = float(error_threshold)
        self.uncertainty_threshold = float(uncertainty_threshold)
        self.min_samples_before_growth = int(min_samples_before_growth)
        self.min_rule_support_for_update = int(min_rule_support_for_update)
        self.local_ridge = float(profile["local_ridge"] if local_ridge is None else local_ridge)
        self.direct_ridge = float(profile["direct_ridge"] if direct_ridge is None else direct_ridge)
        self.prior_variance = float(prior_variance)
        self.lambda_min = float(profile["lambda_min"] if lambda_min is None else lambda_min)
        self.lambda_max = float(profile["lambda_max"] if lambda_max is None else lambda_max)
        self.adaptive_forgetting = bool(adaptive_forgetting)
        self.forgetting_sensitivity = float(forgetting_sensitivity)
        self.maturity_tau = float(profile["maturity_tau"] if maturity_tau is None else maturity_tau)
        self.minimum_consequent_updates = int(minimum_consequent_updates)
        self.beta_max = float(beta_max)
        self.seasonal_regimes = max(0, int(seasonal_regimes))
        self.seasonal_mode = str(seasonal_mode).lower().strip()
        if self.seasonal_mode not in {"hard", "fuzzy"}:
            raise ValueError("seasonal_mode must be hard or fuzzy.")
        self.circular_sin_feature_index = circular_sin_feature_index
        self.circular_cos_feature_index = circular_cos_feature_index
        self.regime_kappa = float(regime_kappa)
        self.regime_shrinkage_tau = float(regime_shrinkage_tau)
        self.regime_minimum_updates = int(regime_minimum_updates)
        self.regime_minimum_activation = float(regime_minimum_activation)
        self.record_history = bool(record_history)
        self.rng = np.random.default_rng(random_state)

        self.x_scaler = OnlineVectorStandardizer(self.n_features)
        self.y_scaler = OnlineScalarStandardizer()
        self.rules: List[EnergyRule] = []
        self.step = 0
        self.next_rule_id = 0
        self.target_mean = 0.0
        self.target_count = 0
        self.last_prediction = 0.0
        self.last_prediction_scaled = 0.0
        self.last_details: Dict[str, object] = {}
        self.history: List[Dict[str, float]] = []

    def _new_learner(self, ridge: float) -> WeightedRLS:
        return WeightedRLS(
            dimension=self.n_features + 1,
            ridge=ridge,
            prior_variance=self.prior_variance,
            lambda_min=self.lambda_min,
            lambda_max=self.lambda_max,
            adaptive_forgetting=self.adaptive_forgetting,
            forgetting_sensitivity=self.forgetting_sensitivity,
            beta_max=self.beta_max,
        )

    def _new_rule(self, x: np.ndarray) -> EnergyRule:
        r = EnergyRule(
            rule_id=self.next_rule_id,
            center=x.copy(),
            sigma=np.full(self.n_features, self.initial_sigma, dtype=float),
            support=1,
            birth_step=self.step,
            last_update_step=self.step,
            residual=self._new_learner(self.local_ridge),
            direct=self._new_learner(self.direct_ridge),
        )
        self.next_rule_id += 1
        self.rules.append(r)
        return r

    def _prepare_x(self, x_raw: Sequence[float]) -> np.ndarray:
        x = np.asarray(x_raw, dtype=float).ravel()
        if x.size != self.n_features or not np.all(np.isfinite(x)):
            raise ValueError("Invalid feature vector.")
        return self.x_scaler.transform(x) if self.scale_x else x

    def _anchor_scaled(self, x_raw: np.ndarray, baseline_value: Optional[float]) -> float:
        if baseline_value is not None:
            anchor = float(baseline_value)
        elif self.anchor_feature_index is not None:
            anchor = float(x_raw[self.anchor_feature_index])
        else:
            anchor = float(self.target_mean)
        return self.y_scaler.transform(anchor) if self.scale_y else anchor

    def _activation(self, rule: EnergyRule, x: np.ndarray) -> float:
        z = (x - rule.center) / np.maximum(rule.sigma, self.min_sigma)
        return float(math.exp(-0.5 * float(np.sum(z**2))))

    def _activations(self, x: np.ndarray) -> np.ndarray:
        if not self.rules:
            return np.zeros(0, dtype=float)
        raw = np.asarray([self._activation(r, x) for r in self.rules], dtype=float)
        total = float(raw.sum())
        if total <= _EPS:
            return np.full(len(self.rules), 1.0 / len(self.rules), dtype=float)
        return raw / total

    def _regime_memberships(self, x: np.ndarray) -> np.ndarray:
        if self.seasonal_regimes <= 0:
            return np.ones(1, dtype=float)
        if self.circular_sin_feature_index is None or self.circular_cos_feature_index is None:
            return np.full(self.seasonal_regimes, 1.0 / self.seasonal_regimes, dtype=float)
        s = float(x[int(self.circular_sin_feature_index)])
        c = float(x[int(self.circular_cos_feature_index)])
        angle = math.atan2(s, c)
        if angle < 0:
            angle += 2.0 * math.pi
        centers = (np.arange(self.seasonal_regimes, dtype=float) + 0.5) * (2.0 * math.pi / self.seasonal_regimes)
        if self.seasonal_mode == "hard":
            idx = int(angle / (2.0 * math.pi / self.seasonal_regimes)) % self.seasonal_regimes
            out = np.zeros(self.seasonal_regimes, dtype=float)
            out[idx] = 1.0
            return out
        logits = self.regime_kappa * np.cos(angle - centers)
        logits -= logits.max()
        out = np.exp(logits)
        return out / max(float(out.sum()), _EPS)

    def _confidence(self, learner: WeightedRLS) -> float:
        if learner.n_updates < self.minimum_consequent_updates:
            return 0.0
        return 1.0 if self.maturity_tau <= 0 else float(learner.update_mass / (learner.update_mass + self.maturity_tau))

    def _regime_confidence(self, learner: WeightedRLS) -> float:
        if learner.n_updates < self.regime_minimum_updates:
            return 0.0
        return 1.0 if self.regime_shrinkage_tau <= 0 else float(learner.update_mass / (learner.update_mass + self.regime_shrinkage_tau))

    def _predict_scaled(self, x_raw: np.ndarray, baseline_value: Optional[float]) -> Tuple[float, Dict[str, object]]:
        x = self._prepare_x(x_raw)
        anchor = self._anchor_scaled(x_raw, baseline_value)
        if not self.rules:
            fallback = anchor if self.prediction_mode != "direct" else (self.y_scaler.transform(self.target_mean) if self.scale_y else self.target_mean)
            return float(fallback), {"anchor": anchor, "n_rules": 0, "effective_rules": 0.0}

        acts = self._activations(x)
        phi = np.concatenate(([1.0], x))
        memberships = self._regime_memberships(x)
        residual_levels: List[float] = []
        direct_levels: List[float] = []

        for rule in self.rules:
            assert rule.residual is not None and rule.direct is not None
            global_res = self._confidence(rule.residual) * rule.residual.predict(phi)
            global_dir = self._confidence(rule.direct) * rule.direct.predict(phi)
            if self.seasonal_regimes > 0:
                reg_res = global_res
                reg_dir = global_dir
                specialized_res = 0.0
                specialized_dir = 0.0
                for j, m in enumerate(memberships):
                    lr = rule.regime_residual.get(j)
                    ld = rule.regime_direct.get(j)
                    if lr is not None:
                        cr = self._regime_confidence(lr)
                        specialized_res += float(m) * ((1.0 - cr) * global_res + cr * lr.predict(phi))
                    else:
                        specialized_res += float(m) * global_res
                    if ld is not None:
                        cd = self._regime_confidence(ld)
                        specialized_dir += float(m) * ((1.0 - cd) * global_dir + cd * ld.predict(phi))
                    else:
                        specialized_dir += float(m) * global_dir
                reg_res, reg_dir = specialized_res, specialized_dir
            else:
                reg_res, reg_dir = global_res, global_dir
            residual_levels.append(anchor + reg_res)
            direct_levels.append(reg_dir)

        residual_pred = float(acts @ np.asarray(residual_levels))
        direct_pred = float(acts @ np.asarray(direct_levels))
        if self.prediction_mode == "residual":
            mixed = residual_pred
        elif self.prediction_mode == "direct":
            mixed = direct_pred
        else:
            mixed = self.residual_gate * residual_pred + (1.0 - self.residual_gate) * direct_pred
        entropy = -float(np.sum(np.clip(acts, _EPS, 1.0) * np.log(np.clip(acts, _EPS, 1.0))))
        effective_rules = float(math.exp(entropy))
        details = {
            "anchor": anchor,
            "activations": acts,
            "regime_memberships": memberships,
            "residual_prediction": residual_pred,
            "direct_prediction": direct_pred,
            "mixed_prediction": mixed,
            "n_rules": len(self.rules),
            "effective_rules": effective_rules,
            "max_activation": float(acts.max()),
        }
        return mixed, details

    def predict_one(self, x: Sequence[float], baseline_value: Optional[float] = None) -> float:
        x_raw = np.asarray(x, dtype=float).ravel()
        pred_scaled, details = self._predict_scaled(x_raw, baseline_value)
        pred = self.y_scaler.inverse(pred_scaled) if self.scale_y else pred_scaled
        self.last_prediction = float(pred)
        self.last_prediction_scaled = float(pred_scaled)
        self.last_details = details
        return float(pred)

    def _should_grow(self, x: np.ndarray, y_scaled: float, pred_scaled: float, acts: np.ndarray) -> bool:
        if not self.rules:
            return True
        if len(self.rules) >= self.max_rules or self.step < self.min_samples_before_growth:
            return False
        distances = [float(np.sqrt(np.mean(((x - r.center) / np.maximum(r.sigma, self.min_sigma)) ** 2))) for r in self.rules]
        novelty = min(distances)
        error = abs(float(y_scaled) - float(pred_scaled))
        uncertainty = 1.0 - float(acts.max()) if acts.size else 1.0
        return bool(novelty >= self.novelty_threshold and (error >= self.error_threshold or uncertainty >= self.uncertainty_threshold))

    def learn_one(self, x: Sequence[float], y: float, baseline_value: Optional[float] = None) -> "ENFSEnergyRegressor":
        x_raw = np.asarray(x, dtype=float).ravel()
        y_raw = float(y)
        pred_scaled, details = self._predict_scaled(x_raw, baseline_value)
        pred_raw = self.y_scaler.inverse(pred_scaled) if self.scale_y else pred_scaled
        x_model = self._prepare_x(x_raw)
        y_scaled = self.y_scaler.transform(y_raw) if self.scale_y else y_raw
        anchor_scaled = self._anchor_scaled(x_raw, baseline_value)

        if not self.rules:
            self._new_rule(x_model)
        acts = self._activations(x_model)
        if self._should_grow(x_model, y_scaled, pred_scaled, acts):
            self._new_rule(x_model)
            acts = self._activations(x_model)

        phi = np.concatenate(([1.0], x_model))
        memberships = self._regime_memberships(x_model)
        winner = int(np.argmax(acts)) if acts.size else 0
        for i, (rule, activation) in enumerate(zip(self.rules, acts)):
            a = float(activation)
            rule.cumulative_activation += a
            if i == winner:
                rule.winner_count += 1
            if a > 0.01 and rule.support >= self.min_rule_support_for_update:
                rule.update_geometry(x_model, self.min_sigma, self.step)
            assert rule.residual is not None and rule.direct is not None
            rule.residual.update(phi, y_scaled - anchor_scaled, a)
            rule.direct.update(phi, y_scaled, a)
            if self.seasonal_regimes > 0:
                for j, membership in enumerate(memberships):
                    w = a * float(membership)
                    if w < self.regime_minimum_activation:
                        continue
                    if j not in rule.regime_residual:
                        rule.regime_residual[j] = self._new_learner(self.local_ridge)
                        rule.regime_direct[j] = self._new_learner(self.direct_ridge)
                    rule.regime_residual[j].update(phi, y_scaled - anchor_scaled, w)
                    rule.regime_direct[j].update(phi, y_scaled, w)

        self.target_count += 1
        self.target_mean += (y_raw - self.target_mean) / self.target_count
        if self.record_history:
            immature = self.immature_regime_cell_fraction()
            self.history.append({
                "step": float(self.step),
                "y": y_raw,
                "prediction": float(pred_raw),
                "abs_error": abs(y_raw - pred_raw),
                "n_rules": float(len(self.rules)),
                "effective_rules": float(details.get("effective_rules", 1.0)),
                "max_activation": float(details.get("max_activation", 1.0)),
                "immature_regime_cell_fraction": immature,
            })
        if self.scale_x:
            self.x_scaler.update(x_raw)
        if self.scale_y:
            self.y_scaler.update(y_raw)
        self.step += 1
        return self

    @property
    def n_rules_(self) -> int:
        return len(self.rules)

    def immature_regime_cell_fraction(self) -> float:
        cells: List[WeightedRLS] = []
        for rule in self.rules:
            cells.extend(rule.regime_residual.values())
            cells.extend(rule.regime_direct.values())
        if not cells:
            return 0.0
        immature = sum(1 for learner in cells if learner.n_updates < self.regime_minimum_updates)
        return float(immature / len(cells))

    def rule_diagnostics(self) -> List[Dict[str, float]]:
        total_steps = max(self.step, 1)
        rows: List[Dict[str, float]] = []
        for rule in self.rules:
            rows.append({
                "rule_id": float(rule.rule_id),
                "support": float(rule.support),
                "birth_step": float(rule.birth_step),
                "age": float(max(self.step - rule.birth_step, 0)),
                "mean_activation": float(rule.cumulative_activation / total_steps),
                "winner_share": float(rule.winner_count / total_steps),
                "residual_updates": float(rule.residual.n_updates if rule.residual else 0),
                "direct_updates": float(rule.direct.n_updates if rule.direct else 0),
                "regime_cells": float(len(rule.regime_residual) + len(rule.regime_direct)),
            })
        return rows

    def regime_cell_diagnostics(self) -> List[Dict[str, float]]:
        rows: List[Dict[str, float]] = []
        for rule in self.rules:
            for kind, mapping in (("residual", rule.regime_residual), ("direct", rule.regime_direct)):
                for regime, learner in mapping.items():
                    rows.append({
                        "rule_id": float(rule.rule_id),
                        "regime": float(regime),
                        "path": 0.0 if kind == "residual" else 1.0,
                        "n_updates": float(learner.n_updates),
                        "update_mass": float(learner.update_mass),
                        "confidence": float(self._regime_confidence(learner)),
                    })
        return rows
