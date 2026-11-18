"""Project 1 — step 1: load the data and print its shape.

Loads the Criteo uplift file if it is present at data/criteo.csv.gz,
otherwise loads the Hillstrom stand-in (data/hillstrom_no_indices.csv.gz),
which has the same shape of problem: treatment / outcome / features.

Run:  python3 scripts/01_load_data.py        (from the repo root)
Needs: pandas only.
"""
import os
import sys

import pandas as pd

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA = os.path.join(HERE, "data")
CRITEO = os.path.join(DATA, "criteo.csv.gz")
HILLSTROM = os.path.join(DATA, "hillstrom_no_indices.csv.gz")


def load_criteo(path):
    dtypes = {"exposure": "Int8", "treatment": "Int8",
              "conversion": "Int8", "visit": "Int8"}
    df = pd.read_csv(path, dtype=dtypes)
    print(f"dataset: criteo (file: {os.path.basename(path)})")
    print(f"shape: {df.shape}")
    print(f"columns: {list(df.columns)}")
    print("treatment share:")
    print(df["treatment"].value_counts(normalize=True).to_string())
    print("visit rate by treatment:")
    print(df.groupby("treatment")["visit"].mean().to_string())
    print("conversion rate by treatment:")
    print(df.groupby("treatment")["conversion"].mean().to_string())
    return df


def load_hillstrom(path):
    df = pd.read_csv(path)
    print("dataset: hillstrom (STAND-IN — Criteo S3 bucket is dead, see README)")
    print(f"shape: {df.shape}")
    print(f"columns: {list(df.columns)}")
    print("segment counts:")
    print(df["segment"].value_counts().to_string())
    print("visit rate by segment:")
    print(df.groupby("segment")["visit"].mean().to_string())
    print("conversion rate by segment:")
    print(df.groupby("segment")["conversion"].mean().to_string())
    return df


def main():
    if os.path.exists(CRITEO):
        df = load_criteo(CRITEO)
    elif os.path.exists(HILLSTROM):
        df = load_hillstrom(HILLSTROM)
    else:
        print("No data file found. Expected one of:")
        print(f"  {CRITEO}")
        print(f"  {HILLSTROM}")
        sys.exit(1)
    print(f"rows: {len(df):,}  cols: {len(df.columns)}")
    print("OK — data loads.")


if __name__ == "__main__":
    main()
