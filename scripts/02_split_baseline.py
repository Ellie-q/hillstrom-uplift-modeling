"""Project 1 — step 2: split + two uplift baselines + Qini/AUUC evaluation.

Run:  PYTHONPATH=~/workspace/projects/scikit-uplift .venv/bin/python scripts/02_split_baseline.py
      (from the repo root; the script also adds the scikit-uplift fork to
      sys.path automatically when it sits at ~/workspace/projects/scikit-uplift)

Dataset: Hillstrom email-marketing stand-in (64,000 rows). Criteo's official
S3 download is dead (404, verified 2026-11-14/15) — see README. Nothing here
is reported as Criteo data or Criteo numbers.

Design choices (all explicit, all changeable in one place):
- Treatment is BINARY: any email (Mens E-Mail + Womens E-Mail) = 1,
  No E-Mail = 0. The original segment is 3-arm; collapsing the two email
  arms is a modelling choice, stated here and in the README.
- Primary outcome is `visit` (base rate ~14.7%); `conversion` (~0.9%) is
  reported as a secondary outcome because it is too sparse to rank on.
- Features: recency, history, mens, womens, newbie + one-hot
  history_segment / zip_code / channel. `segment` is the treatment
  definition, `spend` is post-treatment — both excluded (leakage).
- Split: train 60% / validation 20% / test 20%, stratified on
  treatment x visit, random_state=42. Models are fit on train only;
  Qini/AUUC are reported on the held-out test set.
- Baselines: (1) Two-Model Logistic Regression, (2) Single-Model
  (SoloModel) Logistic Regression, (3) Two-Model Random Forest
  (300 trees). A random-uplift baseline has Qini/AUUC ~ 0
  by construction, so any positive score is signal above chance.

Outputs: results/results.json (all numbers), results/test_predictions.csv
(treatment, outcomes, predicted uplift per model, test set only).
"""
import json
import os
import sys

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA = os.path.join(HERE, "data")
RESULTS = os.path.join(HERE, "results")
FORK = os.path.expanduser("~/workspace/projects/scikit-uplift")
if os.path.isdir(FORK) and FORK not in sys.path:
    sys.path.insert(0, FORK)

from sklearn.ensemble import RandomForestClassifier  # noqa: E402
from sklearn.linear_model import LogisticRegression  # noqa: E402
from sklearn.model_selection import train_test_split  # noqa: E402
from sklift.metrics import qini_auc_score, uplift_auc_score  # noqa: E402
from sklift.models import SoloModel, TwoModels  # noqa: E402

RANDOM_STATE = 42


def load():
    path = os.path.join(DATA, "hillstrom_no_indices.csv.gz")
    df = pd.read_csv(path)
    df["treatment"] = (df["segment"] != "No E-Mail").astype(int)
    return df


def make_features(df):
    X = df[["recency", "history", "mens", "womens", "newbie"]].copy()
    X = pd.concat(
        [X, pd.get_dummies(df[["history_segment", "zip_code", "channel"]],
                           dtype=float)],
        axis=1,
    )
    return X


def top_decile_uplift(y, uplift, treatment):
    """Observed uplift (treated minus control outcome rate) in the top
    10% by predicted uplift, minus the overall ATE. Positive = the model
    ranks persuadables above average."""
    order = np.argsort(-uplift)
    top = order[: max(1, len(order) // 10)]
    yt, yc = y[top][treatment[top] == 1], y[top][treatment[top] == 0]
    if len(yt) == 0 or len(yc) == 0:
        return None
    ate = y[treatment == 1].mean() - y[treatment == 0].mean()
    return float(yt.mean() - yc.mean() - ate)


def main():
    df = load()
    X = make_features(df)
    y_visit = df["visit"].values
    y_conv = df["conversion"].values
    t = df["treatment"].values
    strata = t * 2 + y_visit

    idx = np.arange(len(df))
    idx_train, idx_temp = train_test_split(
        idx, test_size=0.40, random_state=RANDOM_STATE, stratify=strata)
    idx_val, idx_test = train_test_split(
        idx_temp, test_size=0.50, random_state=RANDOM_STATE,
        stratify=strata[idx_temp])
    print(f"train/val/test: {len(idx_train)}/{len(idx_val)}/{len(idx_test)}")
    print(f"treated share (test): {t[idx_test].mean():.3f}, "
          f"visit rate (test): {y_visit[idx_test].mean():.4f}")
    ate_visit = (y_visit[idx_test][t[idx_test] == 1].mean()
                 - y_visit[idx_test][t[idx_test] == 0].mean())
    ate_conv = (y_conv[idx_test][t[idx_test] == 1].mean()
                - y_conv[idx_test][t[idx_test] == 0].mean())
    print(f"ATE visit (test): {ate_visit:.4f}  ATE conversion (test): {ate_conv:.4f}")

    models = {
        "two_model_logreg": TwoModels(
            LogisticRegression(max_iter=1000), LogisticRegression(max_iter=1000)),
        "solo_model_logreg": SoloModel(LogisticRegression(max_iter=1000)),
        "two_model_rf": TwoModels(
            RandomForestClassifier(n_estimators=300, n_jobs=-1,
                                   random_state=RANDOM_STATE),
            RandomForestClassifier(n_estimators=300, n_jobs=-1,
                                   random_state=RANDOM_STATE)),
    }
    out = {
        "dataset": "hillstrom (stand-in; Criteo S3 dead 2026-11-14/15)",
        "n_rows": int(len(df)),
        "treatment": "any email (Mens + Womens E-Mail) vs No E-Mail",
        "primary_outcome": "visit",
        "secondary_outcome": "conversion",
        "split": "train/val/test 60/20/20, stratified treatment x visit, seed 42",
        "n_train": int(len(idx_train)), "n_val": int(len(idx_val)),
        "n_test": int(len(idx_test)),
        "ate_visit_test": round(float(ate_visit), 4),
        "ate_conversion_test": round(float(ate_conv), 4),
        "models": {},
    }
    preds = pd.DataFrame({
        "treatment": t[idx_test],
        "visit": y_visit[idx_test],
        "conversion": y_conv[idx_test],
    })
    for name, model in models.items():
        model.fit(X.iloc[idx_train], y_visit[idx_train], t[idx_train])
        uplift = model.predict(X.iloc[idx_test])
        preds[name + "_uplift"] = uplift
        qini = qini_auc_score(y_visit[idx_test], uplift, t[idx_test])
        auuc = uplift_auc_score(y_visit[idx_test], uplift, t[idx_test])
        top10 = top_decile_uplift(y_visit[idx_test], uplift, t[idx_test])
        out["models"][name] = {
            "qini_auc_visit_test": round(float(qini), 4),
            "uplift_auc_visit_test": round(float(auuc), 4),
            "top_decile_uplift_minus_ate_visit": (
                round(top10, 4) if top10 is not None else None),
            "mean_predicted_uplift": round(float(np.mean(uplift)), 4),
        }
        print(f"{name}: Qini={qini:.4f}  AUUC={auuc:.4f}  "
              f"top-decile uplift-ATE={top10:.4f}")

    os.makedirs(RESULTS, exist_ok=True)
    with open(os.path.join(RESULTS, "results.json"), "w") as f:
        json.dump(out, f, indent=2)
    preds.to_csv(os.path.join(RESULTS, "test_predictions.csv"), index=False)
    print("wrote results/results.json + results/test_predictions.csv")


if __name__ == "__main__":
    main()
