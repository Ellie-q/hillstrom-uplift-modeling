"""Project 1 — step 3: Qini curve plot from the saved test predictions.

Run: .venv/bin/python scripts/03_plot_qini.py   (from the repo root)
Reads results/test_predictions.csv (written by 02_split_baseline.py) and
saves results/qini_curve.png — one curve per model, plus random (flat at 0)
and perfect baselines from sklift.metrics.
"""
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import sys  # noqa: E402

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
FORK = os.path.expanduser("~/workspace/projects/scikit-uplift")
if os.path.isdir(FORK) and FORK not in sys.path:
    sys.path.insert(0, FORK)
from sklift.metrics import perfect_qini_curve, qini_curve  # noqa: E402

MODELS = {
    "two_model_logreg_uplift": "Two-Model LogReg (Qini 0.0361)",
    "solo_model_logreg_uplift": "Solo-Model LogReg (Qini 0.0217)",
    "two_model_rf_uplift": "Two-Model RF (Qini -0.0288)",
}


def main():
    preds = pd.read_csv(os.path.join(HERE, "results", "test_predictions.csv"))
    y = preds["visit"].values
    t = preds["treatment"].values

    plt.figure(figsize=(7, 5))
    x, y_perf = perfect_qini_curve(y, t)
    plt.plot(x, y_perf, "--", color="grey", label="Perfect")
    plt.plot([0, 1], [0, 0], ":", color="grey", label="Random")
    for col, label in MODELS.items():
        x_c, y_c = qini_curve(y, preds[col].values, t)
        plt.plot(x_c, y_c, label=label)
    plt.xlabel("Share of population targeted (ranked by predicted uplift)")
    plt.ylabel("Qini curve (incremental visits)")
    plt.title("Qini curves — Hillstrom, test set (n=12,800), outcome = visit")
    plt.legend(fontsize=8, loc="upper left")
    plt.tight_layout()
    out = os.path.join(HERE, "results", "qini_curve.png")
    plt.savefig(out, dpi=150)
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
