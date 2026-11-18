# Project 1 — Uplift Modeling on Hillstrom Email Data (Criteo stand-in)

Public repo, runnable pipeline + uplift modeling + Qini / AUUC evaluation.
Scope (from 2026-11-02 plan): pipeline → uplift modeling → Qini/AUUC evaluation.

## ⚠️ Read this first — the Criteo download link is dead (checked Sat 2026-11-14)

The download URL baked into `scikit-uplift` (`sklift/datasets/datasets.py`,
`fetch_criteo`) is:

- `https://criteo-bucket.s3.eu-central-1.amazonaws.com/criteo.csv.gz` (full)
- `https://criteo-bucket.s3.eu-central-1.amazonaws.com/criteo10.csv.gz` (10%)

Both return **404 `NoSuchBucket` — the S3 bucket `criteo-bucket` no longer
exists** (verified by direct request, 2026-11-14 18:46 UTC; the file the
request returns is a 323-byte XML error, not data). The Criteo AI Lab page
(https://ailab.criteo.com/criteo-uplift-prediction-dataset/, crawled the same
day) describes the dataset (25M rows, 12 features, treatment + visit /
conversion labels) but its extracted text contains **no download link** —
the file is probably behind a form/button, or the hosting moved.

**What this means for tonight:** if you run `fetch_criteo()` you will get a
confusing hash-mismatch / "file is broken" error. That is the dead bucket,
not your code. Do not burn the 30-minute block debugging it.

Per the fork's own docs, the full file would have been 297 MB compressed /
3.2 GB uncompressed, 13,979,592 rows — too big to casually load on a laptop
anyway. The 10% file was the sane starting point, and it is gone too.

## Tonight's light launch (Sat 2026-11-14, 20:40–21:30, 30 min cap)

Goal: repo exists, data loads, shape prints. Nothing more.

1. Copy this folder to your Mac (or `git init` here and push to a new public
   GitHub repo — Project 1 must be public).
2. Run the stand-in load (Hillstrom, already verified — see below):
   `python3 scripts/01_load_data.py`
   Expected output: `shape: (64000, 12)`, columns, treatment shares, visit
   rates by segment. That is the light launch done — pipeline skeleton runs
   end to end on real uplift data with the same shape of problem
   (treatment / outcome / features).
3. In parallel (5 min, not blocking): open the AI Lab page in your Chrome
   and look for the download button/form. If a working Criteo link turns
   up, save it to `data/` as `criteo.csv.gz` and re-run the script — it
   picks up a local Criteo file automatically when present.

## Fallback ladder for the Criteo file itself

1. **AI Lab page** (above) — check for a form/button in the browser; the
   text extraction found no link, so this is unverified.
2. **Kaggle mirror `evgeniypolin/criteo-uplift-v2-1`** — 3,248,115,221 bytes
   (~3.2 GB), 144 downloads, updated 2024-02-20. Needs a Kaggle account;
   too big for tonight — overnight download only, and note it is v2.1,
   not the 2018 v1 the paper describes.
3. **Kaggle mirror `guilinzhao/criteo-uplift-parquet`** — 109,605,645 bytes
   (~110 MB), updated 2026-10-30, only 3 downloads. Contents **not
   verified** (Kaggle does not list its files without login) — check the
   file list in the browser before trusting it.
4. **If no Criteo file is obtainable by Sun 2026-11-15:** build the full
   pipeline on Hillstrom (64k rows, verified working) and note the swap in
   the README. The modeling + Qini/AUUC code is dataset-agnostic; swapping
   in Criteo later is a one-line change in `01_load_data.py`. Do not let
   the download block the Nov 17 deadline.

## Verified stand-in: Hillstrom (already in `data/`)

- Source: `https://hillstorm1.s3.us-east-2.amazonaws.com/hillstorm_no_indices.csv.gz`
  (the URL `fetch_hillstrom` in your scikit-uplift fork uses — alive, 206).
- Size: 443,070 bytes gz. Verified load 2026-11-14: **64,000 rows × 12 cols**.
- Columns: recency, history_segment, history, mens, womens, zip_code,
  newbie, channel, segment (treatment: Womens E-Mail / Mens E-Mail /
  No E-Mail), visit, conversion, spend.
- Sanity signal (real uplift shape): visit rate Mens E-Mail 18.3%,
  Womens E-Mail 15.1%, No E-Mail 10.6%.
- The fork's other buckets are alive too (Lenta 144.7 MB gz, MegaFon) —
  only Criteo's bucket is dead.

## Results — Tue 2026-11-17 (ran by the assistant before the 14:00 block)

Reproduce: `.venv/bin/python scripts/02_split_baseline.py`
then `.venv/bin/python scripts/03_plot_qini.py`. The venv already has
scikit-learn + scikit-uplift fork on its path (see `scripts/02_*` header).

- Split: train 38,400 / val 12,800 / test 12,800 (60/20/20, stratified
  treatment × visit, seed 42). Treatment = any email vs No E-Mail.
- Test-set ATE: **visit +6.10 pp** (16.7% treated vs 10.6% control),
  conversion +0.42 pp.
- Test-set Qini / AUUC (outcome = visit):

| model | Qini | AUUC | top-decile uplift − ATE |
|---|---:|---:|---:|
| Two-Model Logistic Regression | **0.0361** | **0.0216** | +0.0406 |
| Solo-Model Logistic Regression | 0.0217 | 0.0139 | +0.0407 |
| Two-Model Random Forest (300 trees) | −0.0288 | −0.0167 | −0.0168 |

- Read: both logistic models rank persuadables above chance; the
  Two-Model LogReg is best. The Random Forest scores **below random** —
  it overfits the two arms and ranks worse than chance, so it is reported
  as a failed baseline, not hidden. Full numbers in
  `results/results.json`; test predictions in
  `results/test_predictions.csv`; curve in `results/qini_curve.png`.
- Honest limits: Hillstrom is a 64k-row email-marketing stand-in, not
  Criteo; conversion (~0.9%) is too sparse to rank on, so visit is the
  primary outcome; no hyperparameter tuning was done (val set unused
  except as a held-out spare — tuning is a Project 2 / follow-up step).

Resume bullet (draft, paste after she confirms the repo is public):

> Built an uplift-modeling pipeline on 64k-row email-marketing data
> (Hillstrom stand-in after Criteo's official download was confirmed
> unavailable); two-model logistic regression achieved Qini 0.036 and
> AUUC 0.022 on a held-out test set (n=12,800), with top-decile
> targeting lifting visit uplift 4.1 pp above the average treatment
> effect; Random Forest baseline scored below random and was reported
> as such.

Still hers to do in the 14:00 block: `git init` + create the public
GitHub repo + first push (the folder is not a repo yet), then confirm
the bullet above.

## Plan to Nov 17

- Sat 11-14 night: light launch (this file) — repo + data loads + shape.
- Sun 11-15: settle the Criteo file (ladder above) or commit to Hillstrom;
  train/validation/test split + a first two-model uplift baseline.
- Mon 11-16: Qini + AUUC evaluation, README + resume bullet draft.
- Tue 11-17: ship — public repo, README with real numbers, no claims
  about Criteo if the run used Hillstrom (say so explicitly).

## Layout

- `scripts/01_load_data.py` — load Criteo if `data/criteo.csv.gz` exists,
  else Hillstrom stand-in; prints shape + treatment/outcome rates.
- `data/` — gitignored; `hillstrom_no_indices.csv.gz` already here.
- `scikit-uplift` fork (`~/workspace/projects/scikit-uplift`) already has
  the models (`sklift.models`) and metrics (`sklift.metrics`, Qini/AUUC)
  this project needs — reuse them, do not rewrite.
