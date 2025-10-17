#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
ARIA-style (Spec-Driven) Data Science Runner
"""
import argparse
import os
import sys
import json
import pickle
import importlib
from datetime import datetime
from typing import List, Literal, Tuple, Dict, Any

import numpy as np
import pandas as pd
import yaml
from pydantic import BaseModel, field_validator
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import (
    r2_score,
    mean_absolute_error,
    mean_squared_error,
    roc_auc_score,
    average_precision_score,
    f1_score,
    accuracy_score,
)
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.ensemble import (
    GradientBoostingRegressor,
    GradientBoostingClassifier,
    RandomForestRegressor,
    RandomForestClassifier,
)
from sklearn.inspection import permutation_importance
import matplotlib.pyplot as plt

TaskT = Literal["regression", "classification"]

# --- Spec schema ------------------------------------------------------------
class RunCfg(BaseModel):
    out_dir: str = "./runs/aria_demo"
    random_state: int = 42
    test_size: float = 0.2
    cv_folds: int = 5
    task: TaskT = "regression"


class DataCfg(BaseModel):
    csv_path: str
    target: str
    features: List[str]


class ConstraintsCfg(BaseModel):
    allow_models: List[Literal["linear", "gbdt", "rf", "xgboost", "lightgbm"]] = ["linear", "gbdt", "rf"]
    scale_numeric: bool = True
    handle_unknown_category: Literal["ignore", "infrequent_if_exist", "error"] = "ignore"


class PIcfg(BaseModel):
    enabled: bool = True
    n_repeats: int = 5


class ExplainCfg(BaseModel):
    permutation_importance: PIcfg = PIcfg()


class DeliverCfg(BaseModel):
    artifacts: List[str] = ["model.pkl", "metrics.json", "feature_importance.png", "lineage.json", "report.md"]


class Spec(BaseModel):
    run: RunCfg
    data: DataCfg
    constraints: ConstraintsCfg = ConstraintsCfg()
    explainability: ExplainCfg = ExplainCfg()
    deliverables: DeliverCfg = DeliverCfg()

    @field_validator("data")
    @classmethod
    def _chk_csv(cls, v: DataCfg):
        if not os.path.exists(v.csv_path):
            raise ValueError(f"CSV not found: {v.csv_path}")
        return v


# --- Utils -----------------------------------------------------------------
def ts_outdir(base_dir: str) -> str:
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    out = os.path.join(base_dir.rstrip("/"), ts)
    os.makedirs(os.path.join(out, "artifacts"), exist_ok=True)
    return out


def split_cols(df: pd.DataFrame, feats: List[str]) -> Tuple[List[str], List[str]]:
    num, cat = [], []
    for c in feats:
        if pd.api.types.is_numeric_dtype(df[c]):
            num.append(c)
        else:
            cat.append(c)
    return num, cat


def has_pkg(name: str) -> bool:
    try:
        importlib.import_module(name)
        return True
    except Exception:
        return False


def build_candidates(task: TaskT, allow: List[str], rs: int) -> Dict[str, Any]:
    candidates: Dict[str, Any] = {}

    if "linear" in allow:
        candidates["linear"] = LinearRegression() if task == "regression" else LogisticRegression(max_iter=300)

    if "gbdt" in allow:
        candidates["gbdt"] = (
            GradientBoostingRegressor(random_state=rs)
            if task == "regression"
            else GradientBoostingClassifier(random_state=rs)
        )

    if "rf" in allow:
        candidates["rf"] = (
            RandomForestRegressor(n_estimators=300, random_state=rs, n_jobs=-1)
            if task == "regression"
            else RandomForestClassifier(n_estimators=400, random_state=rs, n_jobs=-1)
        )

    if "xgboost" in allow and has_pkg("xgboost"):
        import xgboost as xgb  # type: ignore

        candidates["xgboost"] = (
            xgb.XGBRegressor(random_state=rs)
            if task == "regression"
            else xgb.XGBClassifier(random_state=rs)
        )

    if "lightgbm" in allow and has_pkg("lightgbm"):
        import lightgbm as lgb  # type: ignore

        candidates["lightgbm"] = (
            lgb.LGBMRegressor(random_state=rs)
            if task == "regression"
            else lgb.LGBMClassifier(random_state=rs)
        )

    return candidates


# --- Metrics ---------------------------------------------------------------
def reg_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    return {
        "r2": float(r2_score(y_true, y_pred)),
        "mae": float(mean_absolute_error(y_true, y_pred)),
        "rmse": float(mean_squared_error(y_true, y_pred) ** 0.5),
    }


def cls_metrics(y_true: np.ndarray, y_prob: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    is_binary = len(np.unique(y_true)) == 2
    rocauc = roc_auc_score(y_true, y_prob) if is_binary else np.nan
    prauc = average_precision_score(y_true, y_prob) if is_binary else np.nan
    return {
        "roc_auc": float(rocauc),
        "prauc": float(prauc),
        "f1": float(f1_score(y_true, y_pred)),
        "accuracy": float(accuracy_score(y_true, y_pred)),
    }


# --- Main runner -----------------------------------------------------------
def run_from_spec(spec_path: str) -> None:
    with open(spec_path, "r") as f:
        raw = yaml.safe_load(f)
    spec = Spec.model_validate(raw)

    out_dir = ts_outdir(spec.run.out_dir)
    art = os.path.join(out_dir, "artifacts")

    df = pd.read_csv(spec.data.csv_path)
    X, y = df[spec.data.features], df[spec.data.target]

    num, cat = split_cols(df, spec.data.features)
    prep = ColumnTransformer(
        transformers=[
            ("num", StandardScaler() if spec.constraints.scale_numeric else "passthrough", num),
            ("cat", OneHotEncoder(handle_unknown=spec.constraints.handle_unknown_category), cat),
        ]
    )

    Xtr, Xte, ytr, yte = train_test_split(
        X,
        y,
        test_size=spec.run.test_size,
        random_state=spec.run.random_state,
        shuffle=True,
    )

    cands = build_candidates(spec.run.task, spec.constraints.allow_models, spec.run.random_state)

    ladder: List[Dict[str, float]] = []
    for name, estimator in cands.items():
        pipe = Pipeline([("prep", prep), ("model", estimator)])
        scoring = "r2" if spec.run.task == "regression" else "roc_auc"
        sc = cross_val_score(pipe, Xtr, ytr, cv=spec.run.cv_folds, scoring=scoring, n_jobs=-1)
        ladder.append({"name": name, "cv_mean": float(sc.mean()), "cv_std": float(sc.std())})

    ladder = sorted(ladder, key=lambda d: d["cv_mean"], reverse=True)
    best_estimator = cands[ladder[0]["name"]]
    pipe = Pipeline([("prep", prep), ("model", best_estimator)])
    pipe.fit(Xtr, ytr)

    ypred = pipe.predict(Xte)
    if spec.run.task == "regression":
        metrics = reg_metrics(yte, ypred)
    else:
        if hasattr(pipe, "predict_proba"):
            yprob = pipe.predict_proba(Xte)
            yprob_pos = yprob[:, 1] if yprob.ndim == 2 and yprob.shape[1] > 1 else yprob.ravel()
        elif hasattr(pipe, "decision_function"):
            yprob_pos = pipe.decision_function(Xte)
        else:
            yprob_pos = ypred
        metrics = cls_metrics(yte, yprob_pos, pipe.predict(Xte))

    if spec.explainability.permutation_importance.enabled:
        pi = permutation_importance(
            pipe,
            Xte,
            yte,
            n_repeats=spec.explainability.permutation_importance.n_repeats,
            random_state=spec.run.random_state,
            n_jobs=-1,
        )
        # Use original input features; permutation_importance returns importances per original input column
        feats = list(spec.data.features)
        if len(feats) != len(pi.importances_mean):
            raise RuntimeError(
                f"PI length mismatch: features={len(feats)} importances={len(pi.importances_mean)}"
            )
        imp = pd.DataFrame({"feature": feats, "importance": pi.importances_mean}).sort_values(
            "importance", ascending=False
        )
        plt.figure(figsize=(8, 4))
        top = imp.head(10)
        plt.barh(list(reversed(top.feature.tolist())), list(reversed(top.importance.tolist())))
        plt.title("Top-10 Permutation Importance (Test)")
        plt.tight_layout()
        plt.savefig(os.path.join(art, "feature_importance.png"), dpi=160)
        plt.close()

    with open(os.path.join(art, "model.pkl"), "wb") as f:
        pickle.dump(pipe, f)

    with open(os.path.join(art, "metrics.json"), "w") as f:
        json.dump({"cv": ladder, "best_model": ladder[0]["name"], "test_metrics": metrics}, f, indent=2)

    with open(os.path.join(art, "lineage.json"), "w") as f:
        json.dump({"spec": spec_path, "random_state": spec.run.random_state}, f, indent=2)

    with open(os.path.join(art, "report.md"), "w") as f:
        f.write(f"# Report\n\nBest model: {ladder[0]['name']}\n\nMetrics:\n{metrics}\n")

    print("Done ->", art)


def generate_data() -> None:
    os.makedirs("data", exist_ok=True)
    rng = np.random.default_rng(7)
    n = 1000
    rooms = rng.integers(1, 7, size=n)
    distance = rng.normal(10, 4, size=n).clip(0.5, None)
    crime = rng.lognormal(-1.2, 0.6, size=n)
    age = rng.integers(1, 120, size=n)
    inc = rng.normal(60, 15, size=n).clip(10, None)
    school = rng.normal(7, 1.4, size=n).clip(1, 10)
    park = rng.integers(0, 2, size=n)
    sub = rng.integers(0, 2, size=n)
    city = rng.choice(["Alpha", "Beta", "Gamma", "Delta"], size=n, p=[0.35, 0.3, 0.25, 0.1])
    price = (
        20
        + 8 * rooms
        - 0.7 * distance
        - 5 * np.log1p(crime)
        - 0.03 * age
        + 0.6 * inc
        + 2.2 * school
        + 3.5 * park
        + 2.8 * sub
        + np.where(
            city == "Alpha",
            4,
            np.where(city == "Beta", 2, np.where(city == "Gamma", 0, -2)),
        )
        + rng.normal(0, 5, size=n)
    )
    df = pd.DataFrame(
        dict(
            rooms=rooms,
            distance=distance,
            crime_rate=crime,
            age=age,
            income=inc,
            school_score=school,
            has_park=park,
            near_subway=sub,
            city=city,
            house_price=price,
        )
    )
    df.to_csv("data/housing.csv", index=False)
    print("Generated data/housing.csv")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--spec", help="YAML spec path")
    ap.add_argument("--generate-data", action="store_true", help="Generate example data")
    a = ap.parse_args()
    if a.generate_data:
        generate_data()
    elif a.spec:
        run_from_spec(a.spec)
    else:
        print(
            "Usage:\n python aria_run.py --generate-data\n python aria_run.py --spec analysis.spec.yaml"
        )