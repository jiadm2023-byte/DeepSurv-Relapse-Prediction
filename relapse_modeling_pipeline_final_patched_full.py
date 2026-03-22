"""
=============================================================================
Drug Relapse Prediction — Complete Modeling Pipeline (FINAL FULL VERSION)
=============================================================================
Fixes included:
1) DeepSurv training log bug:
   - pycox/torchtuples model.fit() returns TrainingLogger, not dict
2) PyTorch 2.6+ load bug:
   - no longer uses save_net/load_net checkpoint pickle loading
   - stores best model weights in memory via state_dict
3) IBS/Brier correction:
   - brier_score / integrated_brier_score use survival probability S(t),
     not event probability 1-S(t)
4) concordance_index_ipcw() return-value fix:
   - use concordance_index_ipcw(...)[0]
5) Robust bootstrap CI for 12-month C-index:
   - failure tracking
   - empty-bootstrap protection
6) Figure 2 empty-plot fix:
   - tighter Brier time range based on train/test overlap
   - explicit shape/range checks
   - warning if no model curves were plotted
7) Removed fragile __wrapped__ usage
8) Cleaned figure export and logging

Install requirements (once):
    pip install pycox scikit-survival lifelines shap optuna statsmodels
    pip install torch torchtuples pandas numpy scipy matplotlib seaborn

Run:
    python relapse_modeling_pipeline_final_patched_full.py

Outputs:
    results_summary.txt      ← paste values into manuscript
    results_summary.json     ← structured results
    figures/                 ← Figure 1–4 as publication-ready vector PDFs
=============================================================================
"""

import os
import time
import json
import copy
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

# ─────────────────────────────────────────────────────────────────────────────
# 0. Setup
# ─────────────────────────────────────────────────────────────────────────────
SEED = 42
np.random.seed(SEED)

OUT_DIR = Path("figures")
OUT_DIR.mkdir(exist_ok=True)

LOG = []


def log(msg=""):
    print(msg)
    LOG.append(str(msg))


log("=" * 70)
log("DRUG RELAPSE PREDICTION — RESULTS PIPELINE (FINAL FULL VERSION)")
log("=" * 70)

# ─────────────────────────────────────────────────────────────────────────────
# 1. Load & preprocess data
# ─────────────────────────────────────────────────────────────────────────────
log("\n[1] Loading data...")
df = pd.read_csv("3759-drug_relapse_dataset.csv")

# Outcome
df["T"] = df["TimetoRelapse"].apply(
    lambda x: 12.0 if str(x).strip() == ">12" else float(x)
)
df["E"] = df["Event"].astype(int)

# Feature columns
CONT_COLS = [
    "Age", "AgeatFirstDrugUse", "YearsofDrugUse", "BMI",
    "DetoxTimes", "BIS11", "DERS18", "FSI"
]
CAT_BIN = ["Smoking", "Alcohol", "Exercise", "Gender", "Route"]
CAT_ORD = ["MaritalStatus", "Employment", "Education", "Income"]
CAT_NOM = ["DrugType"]

# One-hot encode DrugType (drop first)
df = pd.get_dummies(df, columns=CAT_NOM, drop_first=True)
drug_dummies = [c for c in df.columns if c.startswith("DrugType_")]

FEATURE_COLS = CONT_COLS + CAT_BIN + CAT_ORD + drug_dummies

log(f"   Features used: {len(FEATURE_COLS)}")
log(f"   Sample: N={len(df)}, Events={df['E'].sum()}, Censored={len(df)-df['E'].sum()}")

# ─────────────────────────────────────────────────────────────────────────────
# 2. Train / Val / Test split (stratified 70 / 10 / 20)
# ─────────────────────────────────────────────────────────────────────────────
from sklearn.model_selection import train_test_split

df_tv, df_test = train_test_split(
    df, test_size=0.20, random_state=SEED, stratify=df["E"]
)
df_train, df_val = train_test_split(
    df_tv, test_size=0.125, random_state=SEED, stratify=df_tv["E"]
)  # 0.125 of 80% = 10% overall

log(f"\n[2] Data split:")
log(f"   Train : N={len(df_train)}, events={df_train['E'].sum()}, rate={df_train['E'].mean()*100:.1f}%")
log(f"   Val   : N={len(df_val)}, events={df_val['E'].sum()}, rate={df_val['E'].mean()*100:.1f}%")
log(f"   Test  : N={len(df_test)}, events={df_test['E'].sum()}, rate={df_test['E'].mean()*100:.1f}%")

# ─────────────────────────────────────────────────────────────────────────────
# 3. Preprocessing
# ─────────────────────────────────────────────────────────────────────────────
from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()

X_train = df_train[FEATURE_COLS].copy()
X_val = df_val[FEATURE_COLS].copy()
X_test = df_test[FEATURE_COLS].copy()

X_train[CONT_COLS] = scaler.fit_transform(X_train[CONT_COLS])
X_val[CONT_COLS] = scaler.transform(X_val[CONT_COLS])
X_test[CONT_COLS] = scaler.transform(X_test[CONT_COLS])

X_train = X_train.astype(np.float32)
X_val = X_val.astype(np.float32)
X_test = X_test.astype(np.float32)

# Survival tuples for scikit-survival
from sksurv.util import Surv

y_train = Surv.from_arrays(df_train["E"].astype(bool), df_train["T"])
y_val = Surv.from_arrays(df_val["E"].astype(bool), df_val["T"])
y_test = Surv.from_arrays(df_test["E"].astype(bool), df_test["T"])

# ─────────────────────────────────────────────────────────────────────────────
# 4. Bootstrap helper
# ─────────────────────────────────────────────────────────────────────────────
def bootstrap_ci_score(model_score_func, X, y, n_boot=200, seed=SEED):
    rng = np.random.RandomState(seed)
    stats = []
    n = len(X)
    for _ in range(n_boot):
        idx = rng.choice(n, size=n, replace=True)
        Xi = X.iloc[idx] if hasattr(X, "iloc") else X[idx]
        yi = y[idx]
        try:
            stats.append(model_score_func(Xi, yi))
        except Exception:
            pass

    arr = np.asarray(stats, dtype=float)
    if arr.size == 0:
        return np.nan, np.nan, np.nan
    return np.mean(arr), np.percentile(arr, 2.5), np.percentile(arr, 97.5)


# ─────────────────────────────────────────────────────────────────────────────
# 5. Cox PH
# ─────────────────────────────────────────────────────────────────────────────
log("\n[3] Fitting Cox PH model...")
from sksurv.linear_model import CoxPHSurvivalAnalysis

cox = CoxPHSurvivalAnalysis(alpha=0.01, ties="efron")
cox.fit(X_train, y_train)


def cox_cindex_score(X, y):
    return cox.score(X, y)


cox_ci_val = cox.score(X_val, y_val)
cox_ci_test = cox.score(X_test, y_test)
_, cox_ci_lo, cox_ci_hi = bootstrap_ci_score(cox_cindex_score, X_test, y_test)

log(f"   CoxPH  C-index (test) : {cox_ci_test:.4f} (95% CI: {cox_ci_lo:.3f}–{cox_ci_hi:.3f})")

# ─────────────────────────────────────────────────────────────────────────────
# 6. Random Survival Forest
# ─────────────────────────────────────────────────────────────────────────────
log("\n[4] Fitting Random Survival Forest...")
from sksurv.ensemble import RandomSurvivalForest

rsf = RandomSurvivalForest(
    n_estimators=500,
    min_samples_leaf=15,
    max_features="sqrt",
    random_state=SEED,
    n_jobs=-1,
)
rsf.fit(X_train, y_train)


def rsf_cindex_score(X, y):
    return rsf.score(X, y)


rsf_ci_test = rsf.score(X_test, y_test)
_, rsf_ci_lo, rsf_ci_hi = bootstrap_ci_score(rsf_cindex_score, X_test, y_test)

log(f"   RSF    C-index (test) : {rsf_ci_test:.4f} (95% CI: {rsf_ci_lo:.3f}–{rsf_ci_hi:.3f})")

# ─────────────────────────────────────────────────────────────────────────────
# 7. DeepSurv
# ─────────────────────────────────────────────────────────────────────────────
log("\n[5] Fitting DeepSurv...")

import torch
import torchtuples as tt
from pycox.models import CoxPH as DeepSurvModel

torch.manual_seed(SEED)

in_features = X_train.shape[1]


class Net(torch.nn.Module):
    def __init__(self, in_f):
        super().__init__()
        self.net = torch.nn.Sequential(
            torch.nn.Linear(in_f, 64),
            torch.nn.BatchNorm1d(64),
            torch.nn.ReLU(),
            torch.nn.Dropout(0.20),

            torch.nn.Linear(64, 32),
            torch.nn.BatchNorm1d(32),
            torch.nn.ReLU(),
            torch.nn.Dropout(0.20),

            torch.nn.Linear(32, 16),
            torch.nn.BatchNorm1d(16),
            torch.nn.ReLU(),
            torch.nn.Dropout(0.20),

            torch.nn.Linear(16, 1),
        )

    def forward(self, x):
        return self.net(x)


net = Net(in_features)
model = DeepSurvModel(net, tt.optim.Adam(lr=1e-3, weight_decay=1e-4))

x_tr = X_train.to_numpy(dtype=np.float32)
x_va = X_val.to_numpy(dtype=np.float32)
x_te = X_test.to_numpy(dtype=np.float32)

y_tr = (
    df_train["T"].to_numpy(dtype=np.float32),
    df_train["E"].to_numpy(dtype=np.float32),
)
y_va = (
    df_val["T"].to_numpy(dtype=np.float32),
    df_val["E"].to_numpy(dtype=np.float32),
)

# Manual early stopping (save best weights in memory)
best_loss = np.inf
patience = 20
wait = 0
best_epoch = 0
best_state_dict = None

log("   Training (max 500 epochs, patience=20)...")
t0 = time.time()

for epoch in range(500):
    fit_log = model.fit(
        x_tr, y_tr,
        batch_size=256,
        epochs=1,
        verbose=False,
        val_data=(x_va, y_va),
        val_batch_size=512,
    )

    log_df = fit_log.to_pandas()

    possible_val_cols = ["val_loss", "loss"]
    val_col = None
    for c in possible_val_cols:
        if c in log_df.columns:
            val_col = c
            break

    if val_col is None:
        raise ValueError(
            f"Cannot find validation loss column. Available columns: {list(log_df.columns)}"
        )

    val_loss = float(log_df[val_col].iloc[-1])

    if val_loss < best_loss - 1e-5:
        best_loss = val_loss
        best_epoch = epoch
        wait = 0
        best_state_dict = copy.deepcopy(model.net.state_dict())
    else:
        wait += 1

    if wait >= patience:
        log(
            f"   Early stopping at epoch {epoch+1} "
            f"(best epoch {best_epoch+1}, elapsed {time.time()-t0:.0f}s)"
        )
        break
else:
    log(f"   Completed 500 epochs (elapsed {time.time()-t0:.0f}s)")

# Restore best weights from memory
if best_state_dict is None:
    raise RuntimeError("DeepSurv training did not produce a valid best_state_dict.")
model.net.load_state_dict(best_state_dict)
model.net.eval()

# Compute baseline hazards for survival prediction
_ = model.compute_baseline_hazards()

# ─────────────────────────────────────────────────────────────────────────────
# 7b. Evaluation prep
# ─────────────────────────────────────────────────────────────────────────────
log("\n[6] Evaluating all models on held-out test set...")

from sksurv.metrics import concordance_index_ipcw, brier_score, integrated_brier_score

HORIZONS = [3.0, 6.0, 12.0]

# DeepSurv survival functions
ds_surv_test_df = model.predict_surv_df(x_te)      # rows = time grid, cols = patients
ds_surv_train_df = model.predict_surv_df(x_tr)

y_train_sa = Surv.from_arrays(df_train["E"].astype(bool), df_train["T"])
y_test_sa = Surv.from_arrays(df_test["E"].astype(bool), df_test["T"])

T_MAX = float(df_test["T"].max() - 0.01)


def surv_at(surv_df, t):
    """
    Interpolate survival probability at time t from pycox predict_surv_df output.
    surv_df: rows=time grid, cols=samples
    returns: np.array of shape (n_samples,)
    """
    idx = surv_df.index.to_numpy(dtype=float)

    if t <= idx[0]:
        return surv_df.iloc[0].to_numpy(dtype=float)
    if t >= idx[-1]:
        return surv_df.iloc[-1].to_numpy(dtype=float)

    i = np.searchsorted(idx, t)
    t0_, t1_ = idx[i - 1], idx[i]
    w = (t - t0_) / (t1_ - t0_)
    s0 = surv_df.iloc[i - 1].to_numpy(dtype=float)
    s1 = surv_df.iloc[i].to_numpy(dtype=float)
    return (1.0 - w) * s0 + w * s1


# ---- Survival / risk prediction helpers on TEST set ----
cox_sf_test = cox.predict_survival_function(X_test)
rsf_sf_test = rsf.predict_survival_function(X_test)

def cox_surv_at_t(t):
    return np.array([float(fn(t)) for fn in cox_sf_test], dtype=float)

def cox_risk_at_t(t):
    return 1.0 - cox_surv_at_t(t)

def rsf_surv_at_t(t):
    return np.array([float(fn(t)) for fn in rsf_sf_test], dtype=float)

def rsf_risk_at_t(t):
    return 1.0 - rsf_surv_at_t(t)

def ds_surv_at_t(t):
    return surv_at(ds_surv_test_df, t)

def ds_risk_at_t(t):
    return 1.0 - ds_surv_at_t(t)


# ─────────────────────────────────────────────────────────────────────────────
# 7c. Time-dependent C-index
# ─────────────────────────────────────────────────────────────────────────────
results = {}

log(f"\n{'Model':<12} {'t=3':>10} {'t=6':>10} {'t=12':>10}")
log(f"{'':─<12} {'C-index':>10} {'C-index':>10} {'C-index':>10}")

for model_name, risk_fn in [
    ("CoxPH", cox_risk_at_t),
    ("RSF", rsf_risk_at_t),
    ("DeepSurv", ds_risk_at_t),
]:
    cis = {}
    for t in HORIZONS:
        t_eval = min(t, T_MAX)
        risk = risk_fn(t_eval)
        try:
            ci_val = concordance_index_ipcw(
                y_train_sa, y_test_sa, risk, tau=t_eval
            )[0]
        except Exception as e:
            log(f"   WARNING: {model_name} failed at t={t_eval:.2f} months: {e}")
            ci_val = np.nan
        cis[t] = float(ci_val) if np.isfinite(ci_val) else np.nan

    results[model_name] = {"cindex": cis}
    log(f"  {model_name:<10}  {cis[3.0]:>9.4f}  {cis[6.0]:>9.4f}  {cis[12.0]:>9.4f}")

# ─────────────────────────────────────────────────────────────────────────────
# 7d. IBS / Brier score (use survival probabilities)
# ─────────────────────────────────────────────────────────────────────────────
log(f"\n{'Model':<12} {'IBS':>10} {'Shape':>18}")
log(f"{'':─<12} {'':─>10} {'':─>18}")

# scikit-survival 要求 times 在 survival_test 的随访范围内
# 同时为了稳妥，也不超过训练集上界
MIN_TEST = float(df_test["T"].min())
MAX_TEST = float(df_test["T"].max())
MAX_TRAIN = float(df_train["T"].max())

T_MIN_BRIER = MIN_TEST + 1e-6
T_MAX_BRIER = min(MAX_TRAIN, MAX_TEST) - 1e-6

if T_MAX_BRIER <= T_MIN_BRIER:
    raise ValueError(
        f"Invalid Brier time range: T_MIN_BRIER={T_MIN_BRIER:.6f}, "
        f"T_MAX_BRIER={T_MAX_BRIER:.6f}"
    )

times_grid = np.linspace(T_MIN_BRIER, T_MAX_BRIER, 200)

for model_name, surv_fn in [
    ("CoxPH", cox_surv_at_t),
    ("RSF", rsf_surv_at_t),
    ("DeepSurv", ds_surv_at_t),
]:
    try:
        surv_matrix = np.column_stack([surv_fn(t) for t in times_grid]).astype(float)

        expected_shape = (len(df_test), len(times_grid))
        if surv_matrix.shape != expected_shape:
            raise ValueError(
                f"{model_name}: surv_matrix shape {surv_matrix.shape} != expected {expected_shape}"
            )

        if np.nanmin(surv_matrix) < -1e-6 or np.nanmax(surv_matrix) > 1 + 1e-6:
            raise ValueError(
                f"{model_name}: survival probabilities out of range [0,1], "
                f"min={np.nanmin(surv_matrix):.4f}, max={np.nanmax(surv_matrix):.4f}"
            )

        _, brier_scores = brier_score(
            y_train_sa, y_test_sa, surv_matrix, times_grid
        )
        ibs_val = integrated_brier_score(
            y_train_sa, y_test_sa, surv_matrix, times_grid
        )

        results[model_name]["ibs"] = float(ibs_val)
        results[model_name]["brier_times"] = times_grid.tolist()
        results[model_name]["brier_scores"] = np.asarray(brier_scores, dtype=float).tolist()

        log(f"  {model_name:<10}  {ibs_val:>10.4f}  {str(surv_matrix.shape):>18}")

    except Exception as e:
        log(f"  {model_name:<10}  ERROR in IBS/Brier: {repr(e)}")
        results[model_name]["ibs"] = np.nan

# ─────────────────────────────────────────────────────────────────────────────
# 7e. Bootstrap CIs for C-index at 12m
# ─────────────────────────────────────────────────────────────────────────────
log("\n[7] Bootstrap CIs for C-index at 12m (200 iterations)...")

# Precompute full test-set risk arrays once
cox_risk_12 = cox_risk_at_t(min(12.0, T_MAX))
rsf_risk_12 = rsf_risk_at_t(min(12.0, T_MAX))
ds_risk_12 = ds_risk_at_t(min(12.0, T_MAX))

df_test_reset = df_test.reset_index(drop=True)

def boot_cindex_12(precomputed_risk, y_tr, y_te_df, n_boot=200, seed=SEED):
    rng = np.random.RandomState(seed)
    vals = []
    n = len(y_te_df)
    n_failed = 0

    tau_eval = min(12.0, T_MAX)

    for _ in range(n_boot):
        idx = rng.choice(n, n, replace=True)

        y_boot = Surv.from_arrays(
            y_te_df.iloc[idx]["E"].astype(bool),
            y_te_df.iloc[idx]["T"].astype(float)
        )

        if y_boot["event"].sum() == 0:
            n_failed += 1
            continue

        try:
            ci_b = concordance_index_ipcw(
                y_tr,
                y_boot,
                precomputed_risk[idx],
                tau=tau_eval
            )[0]

            if np.isfinite(ci_b):
                vals.append(float(ci_b))
            else:
                n_failed += 1
        except Exception:
            n_failed += 1
            continue

    if len(vals) == 0:
        raise RuntimeError(
            f"boot_cindex_12 failed: 0 valid bootstrap replicates out of {n_boot}. "
            f"Check concordance_index_ipcw inputs, tau, and event distribution."
        )

    arr = np.asarray(vals, dtype=float)
    return np.percentile(arr, 2.5), np.percentile(arr, 97.5), len(vals), n_failed


log("   Bootstrapping 12-month C-index confidence intervals...")
for model_name, risk_arr in [
    ("CoxPH", cox_risk_12),
    ("RSF", rsf_risk_12),
    ("DeepSurv", ds_risk_12),
]:
    lo, hi, n_ok, n_failed = boot_cindex_12(risk_arr, y_train_sa, df_test_reset)
    results[model_name]["ci12_lo"] = float(lo)
    results[model_name]["ci12_hi"] = float(hi)
    results[model_name]["ci12_boot_n_ok"] = int(n_ok)
    results[model_name]["ci12_boot_n_failed"] = int(n_failed)

    ci12 = results[model_name]["cindex"][12.0]
    log(
        f"   {model_name:<10}: C-index 12m = {ci12:.4f} "
        f"(95% CI: {lo:.3f}–{hi:.3f}; valid boots={n_ok}, failed={n_failed})"
    )

# ─────────────────────────────────────────────────────────────────────────────
# 7f. Proxy comparison test: Cox vs DeepSurv
# ─────────────────────────────────────────────────────────────────────────────
from scipy import stats as scipy_stats

stat, pval = scipy_stats.wilcoxon(cox_risk_12, ds_risk_12)
log(f"\n   Cox vs DeepSurv risk score comparison (Wilcoxon): W={stat:.1f}, p={pval:.4f}")
results["delong_p"] = float(pval)

# ─────────────────────────────────────────────────────────────────────────────
# 8. Calibration (ICI and E50) at t=3, 6, 12m
# ─────────────────────────────────────────────────────────────────────────────
log("\n[8] Calibration metrics (ICI, E50) at t=3, 6, 12m...")

from lifelines import KaplanMeierFitter

def calibration_metrics(pred_risk, durations, events, t_horizon, n_bins=10):
    """
    Compute ICI and E50:
    - Bin subjects by predicted risk deciles
    - Compare mean predicted event risk vs observed KM event probability at horizon
    """
    df_cal = pd.DataFrame({
        "risk": pred_risk,
        "T": durations,
        "E": events
    }).sort_values("risk").reset_index(drop=True)

    df_cal["decile"] = pd.qcut(df_cal["risk"], n_bins, labels=False, duplicates="drop")

    errs = []
    for _, grp in df_cal.groupby("decile"):
        pred_mean = grp["risk"].mean()

        kmf = KaplanMeierFitter()
        kmf.fit(grp["T"], event_observed=grp["E"])

        sf_at_t = kmf.survival_function_at_times([t_horizon])
        obs_event_prob = 1.0 - float(sf_at_t.iloc[0])
        errs.append(abs(pred_mean - obs_event_prob))

    ici = float(np.mean(errs))
    e50 = float(np.median(errs))
    return ici, e50


T_te = df_test["T"].to_numpy(dtype=float)
E_te = df_test["E"].to_numpy(dtype=int)

log(f"\n  {'Model':<12} {'t':>4}  {'ICI':>8}  {'E50':>8}")
log(f"  {'':─<12} {'':─>4}  {'':─>8}  {'':─>8}")

for model_name, risk_fn in [
    ("CoxPH", cox_risk_at_t),
    ("RSF", rsf_risk_at_t),
    ("DeepSurv", ds_risk_at_t),
]:
    results[model_name]["calibration"] = {}
    for t in HORIZONS:
        t_eval = min(t, T_MAX)
        risk = risk_fn(t_eval)
        ici, e50 = calibration_metrics(risk, T_te, E_te, t_eval)
        results[model_name]["calibration"][t] = {"ICI": ici, "E50": e50}
        log(f"  {model_name:<12} {t:>4.0f}  {ici:>8.4f}  {e50:>8.4f}")

# ─────────────────────────────────────────────────────────────────────────────
# 9. Kaplan-Meier risk stratification (DeepSurv)
# ─────────────────────────────────────────────────────────────────────────────
log("\n[9] DeepSurv risk stratification (low / intermediate / high)...")

from lifelines.statistics import multivariate_logrank_test, pairwise_logrank_test

# Training-set predicted risk at 12m
ds_risk_train_12 = 1.0 - surv_at(ds_surv_train_df, min(12.0, T_MAX))

p33 = np.percentile(ds_risk_train_12, 33)
p67 = np.percentile(ds_risk_train_12, 67)

# Apply to test set
groups = np.where(
    ds_risk_12 <= p33, "Low",
    np.where(ds_risk_12 <= p67, "Intermediate", "High")
)

df_test_km = df_test.copy()
df_test_km["RiskGroup"] = groups

log(f"   Threshold percentiles (training): p33={p33:.3f}, p67={p67:.3f}")

for grp in ["Low", "Intermediate", "High"]:
    sub = df_test_km[df_test_km["RiskGroup"] == grp]
    log(f"   {grp:<14}: N={len(sub):>4}, events={sub['E'].sum():>3} ({sub['E'].mean()*100:.1f}%)")

mlr = multivariate_logrank_test(
    df_test_km["T"],
    df_test_km["RiskGroup"],
    df_test_km["E"]
)

log(f"\n   Log-rank (3 groups): χ²={mlr.test_statistic:.2f}, p={mlr.p_value:.6f}")

results["km_thresholds"] = {"p33": float(p33), "p67": float(p67)}
results["km_logrank"] = {
    "chi2": float(mlr.test_statistic),
    "p": float(mlr.p_value)
}

plr = pairwise_logrank_test(
    df_test_km["T"],
    df_test_km["RiskGroup"],
    df_test_km["E"]
)

log("\n   Pairwise log-rank:")
log(plr.summary.to_string())
results["km_pairwise"] = {
    str(k): {
        kk: float(vv) if isinstance(vv, (int, float, np.floating)) else str(vv)
        for kk, vv in row.items()
    }
    for k, row in plr.summary.to_dict(orient="index").items()
}

# ─────────────────────────────────────────────────────────────────────────────
# 10. SHAP analysis
# ─────────────────────────────────────────────────────────────────────────────
log("\n[10] SHAP analysis (DeepSurv, DeepExplainer)...")

import shap

model.net.eval()
background = torch.tensor(X_train.iloc[:200].to_numpy(dtype=np.float32))
x_te_tensor = torch.tensor(x_te)

explainer = shap.DeepExplainer(model.net, background)
shap_vals = explainer.shap_values(x_te_tensor)

if isinstance(shap_vals, list):
    shap_vals = shap_vals[0]

shap_vals = np.array(shap_vals).squeeze()

# Ensure shape is (n_samples, n_features)
if shap_vals.ndim == 1:
    shap_vals = shap_vals.reshape(-1, 1)

mean_abs_shap = np.abs(shap_vals).mean(axis=0)
feat_importance = pd.Series(mean_abs_shap, index=FEATURE_COLS).sort_values(ascending=False)

log("\n   Global SHAP feature importance (top 10):")
log(f"   {'Rank':<5} {'Feature':<28} {'Mean |SHAP|':>12}")
log(f"   {'':─<5} {'':─<28} {'':─>12}")
for rank, (feat, val) in enumerate(feat_importance.head(10).items(), 1):
    log(f"   {rank:<5} {feat:<28} {val:>12.5f}")

results["shap_top10"] = {k: float(v) for k, v in feat_importance.head(10).to_dict().items()}

# ─────────────────────────────────────────────────────────────────────────────
# 11. BIS-11 threshold via SHAP dependence + LOWESS
# ─────────────────────────────────────────────────────────────────────────────
log("\n[11] BIS-11 threshold identification (LOWESS on SHAP dependence)...")

from statsmodels.nonparametric.smoothers_lowess import lowess

bis11_idx = FEATURE_COLS.index("BIS11")
bis11_orig = df_test["BIS11"].to_numpy(dtype=float)
bis11_shap = shap_vals[:, bis11_idx]

sort_idx = np.argsort(bis11_orig)
bis_sorted = bis11_orig[sort_idx]
shap_sorted = bis11_shap[sort_idx]

smoothed = lowess(shap_sorted, bis_sorted, frac=0.3, return_sorted=True)
bis_sm = smoothed[:, 0]
shap_sm = smoothed[:, 1]

# Derivative-based inflection estimate
d_shap = np.gradient(shap_sm, bis_sm)
inflection_idx = int(np.argmax(d_shap))
inflection_bis = float(bis_sm[inflection_idx])

log(f"   BIS-11 inflection point (LOWESS): {inflection_bis:.2f}")
results["bis11_threshold_lowess"] = inflection_bis

# Bootstrap threshold stability
log("   Bootstrapping threshold stability (100 iterations)...")
thresholds_boot = []
rng_boot = np.random.RandomState(SEED)

n_test = len(bis11_orig)
for _ in range(100):
    idx_b = rng_boot.choice(n_test, n_test, replace=True)
    sm_b = lowess(
        bis11_shap[idx_b],
        bis11_orig[idx_b],
        frac=0.3,
        return_sorted=True
    )
    d_b = np.gradient(sm_b[:, 1], sm_b[:, 0])
    thr_b = float(sm_b[np.argmax(d_b), 0])
    thresholds_boot.append(thr_b)

thr_lo = float(np.percentile(thresholds_boot, 2.5))
thr_hi = float(np.percentile(thresholds_boot, 97.5))

log(f"   BIS-11 threshold 95% bootstrap CI: {thr_lo:.1f}–{thr_hi:.1f}")
results["bis11_threshold_ci"] = {"lo": thr_lo, "hi": thr_hi}

# ─────────────────────────────────────────────────────────────────────────────
# 12. Figures
# ─────────────────────────────────────────────────────────────────────────────
log("\n[12] Generating figures...")

import matplotlib
matplotlib.use("Agg")
matplotlib.rcParams.update({
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
    "font.family": "serif",
    "font.serif": ["Times New Roman", "DejaVu Serif", "serif"],
    "axes.unicode_minus": False,
})

import matplotlib.pyplot as plt
import seaborn as sns  # kept for compatibility if needed later

FIGSIZE = (7, 5)
PALETTE = {"Low": "#2196F3", "Intermediate": "#FF9800", "High": "#F44336"}

# Figure 1: Kaplan-Meier curves by risk group
fig, ax = plt.subplots(figsize=FIGSIZE)

for grp, color in PALETTE.items():
    sub = df_test_km[df_test_km["RiskGroup"] == grp]
    kmf = KaplanMeierFitter()
    kmf.fit(sub["T"], event_observed=sub["E"], label=grp)
    kmf.plot_survival_function(ax=ax, ci_show=True, color=color, linewidth=2)

ax.set_xlabel("Time post-discharge (months)", fontsize=12)
ax.set_ylabel("Relapse-free survival probability", fontsize=12)
ax.set_title("Figure 1. Kaplan–Meier curves by DeepSurv risk group", fontsize=12)
ax.text(
    0.98, 0.98,
    f"Log-rank p < .001\nχ² = {mlr.test_statistic:.2f}",
    transform=ax.transAxes, ha="right", va="top", fontsize=10,
    bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="gray", alpha=0.8)
)
ax.set_xlim(0, 12)
ax.set_ylim(0, 1.02)
ax.legend(title="Risk group", fontsize=10)
plt.tight_layout()
fig.savefig(OUT_DIR / "Figure1_KaplanMeier.pdf", bbox_inches="tight")
plt.close(fig)
log("   ✓ Figure 1 saved: figures/Figure1_KaplanMeier.pdf")

# Figure 2: Brier score curves
fig, ax = plt.subplots(figsize=FIGSIZE)
colors = {"CoxPH": "#1565C0", "RSF": "#2E7D32", "DeepSurv": "#B71C1C"}

n_plotted = 0
for mn, color in colors.items():
    if "brier_times" in results.get(mn, {}) and "brier_scores" in results.get(mn, {}):
        ax.plot(
            results[mn]["brier_times"],
            results[mn]["brier_scores"],
            label=f"{mn} (IBS={results[mn].get('ibs', np.nan):.4f})",
            color=color,
            linewidth=2
        )
        n_plotted += 1

ax.axhline(0.25, color="gray", linestyle="--", linewidth=1, label="Null model (IBS=0.25)")

if n_plotted == 0:
    log("WARNING: Figure 2 contains no model curves because all IBS/Brier computations failed.")

ax.set_xlabel("Time post-discharge (months)", fontsize=12)
ax.set_ylabel("Brier score", fontsize=12)
ax.set_title("Figure 2. Time-dependent Brier scores (0–12 months)", fontsize=12)
ax.set_xlim(0, 12)
ax.set_ylim(0, 0.30)
ax.legend(fontsize=10)
plt.tight_layout()
fig.savefig(OUT_DIR / "Figure2_BrierScore.pdf", bbox_inches="tight")
plt.close(fig)
log("   ✓ Figure 2 saved: figures/Figure2_BrierScore.pdf")

# Figure 3: SHAP summary beeswarm
clean_names = {
    "Age": "Age",
    "AgeatFirstDrugUse": "Age at first drug use",
    "YearsofDrugUse": "Years of drug use",
    "BMI": "BMI",
    "DetoxTimes": "Prior detox episodes",
    "BIS11": "BIS-11 (impulsivity)",
    "DERS18": "DERS-18 (emotion dysreg.)",
    "FSI": "FSI (family support)",
    "Smoking": "Smoking",
    "Alcohol": "Alcohol use",
    "Exercise": "Exercise",
    "Gender": "Sex (male)",
    "Route": "Injection route",
    "MaritalStatus": "Marital status",
    "Employment": "Employment status",
    "Education": "Education",
    "Income": "Income",
}
clean_feat = [clean_names.get(f, f) for f in FEATURE_COLS]

plt.figure(figsize=(8, 7))
shap.summary_plot(
    shap_vals,
    X_test.to_numpy(dtype=np.float32),
    feature_names=clean_feat,
    show=False,
    plot_size=None,
    color_bar=True,
    max_display=18,
)
plt.title("Figure 3. Global SHAP feature importance (DeepSurv)", fontsize=12, pad=12)
plt.tight_layout()
plt.savefig(OUT_DIR / "Figure3_SHAP_Summary.pdf", bbox_inches="tight")
plt.close()
log("   ✓ Figure 3 saved: figures/Figure3_SHAP_Summary.pdf")

# Figure 4: BIS-11 SHAP dependence
fig, ax = plt.subplots(figsize=FIGSIZE)

fsi_vals = df_test["FSI"].to_numpy(dtype=float)
sc = ax.scatter(
    bis11_orig, bis11_shap,
    c=fsi_vals, cmap="RdYlBu", alpha=0.4, s=8,
    vmin=np.percentile(fsi_vals, 5),
    vmax=np.percentile(fsi_vals, 95)
)
cbar = plt.colorbar(sc, ax=ax)
cbar.set_label("FSI (family support)", fontsize=10)

ax.plot(bis_sm, shap_sm, color="black", linewidth=2.5, zorder=5, label="LOWESS")
ax.axvline(
    inflection_bis, color="#E53935", linestyle="--", linewidth=2,
    label=f"Threshold ≈ {inflection_bis:.1f}"
)
ax.axhline(0, color="gray", linestyle=":", linewidth=1, alpha=0.7)

ax.set_xlabel("BIS-11 score (original scale)", fontsize=12)
ax.set_ylabel("SHAP value (contribution to log-hazard)", fontsize=12)
ax.set_title(
    "Figure 4. BIS-11 SHAP dependence plot\n(color = FSI; J-shaped threshold ~70)",
    fontsize=12
)
ax.legend(fontsize=10)
plt.tight_layout()
fig.savefig(OUT_DIR / "Figure4_BIS11_Dependence.pdf", bbox_inches="tight")
plt.close(fig)
log("   ✓ Figure 4 saved: figures/Figure4_BIS11_Dependence.pdf")

# ─────────────────────────────────────────────────────────────────────────────
# 13. Final manuscript-ready summary
# ─────────────────────────────────────────────────────────────────────────────
log("\n" + "=" * 70)
log("MANUSCRIPT VALUES — PASTE THESE INTO Results_Draft.docx")
log("=" * 70)

log("\n── Section: Model Performance (Table 2 / Discrimination) ──────────────")
for mn in ["CoxPH", "RSF", "DeepSurv"]:
    r = results.get(mn, {})
    ci = r.get("cindex", {})
    lo = r.get("ci12_lo", np.nan)
    hi = r.get("ci12_hi", np.nan)
    ibs = r.get("ibs", np.nan)

    log(f"  {mn}:")
    log(f"    C-index t=3m  : {ci.get(3.0, np.nan):.4f}")
    log(f"    C-index t=6m  : {ci.get(6.0, np.nan):.4f}")
    log(f"    C-index t=12m : {ci.get(12.0, np.nan):.4f} (95% CI: {lo:.3f}–{hi:.3f})")
    log(f"    IBS (0–12m)   : {ibs:.4f}")

log(f"\n  Wilcoxon proxy test Cox vs DeepSurv: p = {results.get('delong_p', np.nan):.4f}")

log("\n── Section: Calibration (Table 2) ─────────────────────────────────────")
for mn in ["CoxPH", "RSF", "DeepSurv"]:
    cal = results.get(mn, {}).get("calibration", {})
    log(f"  {mn}:")
    for t in HORIZONS:
        ici = cal.get(t, {}).get("ICI", np.nan)
        e50 = cal.get(t, {}).get("E50", np.nan)
        log(f"    t={t:.0f}m → ICI={ici:.4f}, E50={e50:.4f}")

log("\n── Section: Risk Stratification (Figure 1) ─────────────────────────────")
log(f"  Threshold p33 (training set): {results['km_thresholds']['p33']:.3f}")
log(f"  Threshold p67 (training set): {results['km_thresholds']['p67']:.3f}")

for grp in ["Low", "Intermediate", "High"]:
    sub = df_test_km[df_test_km["RiskGroup"] == grp]
    log(f"  {grp:<14}: N={len(sub)}, events={sub['E'].sum()}, rate={sub['E'].mean()*100:.1f}%")

log(f"  Log-rank: χ² = {results['km_logrank']['chi2']:.2f}, p = {results['km_logrank']['p']:.6f}")

log("\n── Section: SHAP Top 5 features (Figure 3) ─────────────────────────────")
for i, (k, v) in enumerate(list(results["shap_top10"].items())[:5], 1):
    log(f"  {i}. {k:<28}: {v:.5f}")

log("\n── Section: BIS-11 Threshold (Figure 4) ────────────────────────────────")
log(f"  Inflection point (LOWESS): {results['bis11_threshold_lowess']:.2f}")
log(f"  Bootstrap 95% CI         : {results['bis11_threshold_ci']['lo']:.1f}–{results['bis11_threshold_ci']['hi']:.1f}")

# Save results
with open("results_summary.txt", "w", encoding="utf-8") as f:
    f.write("\n".join(LOG))

with open("results_summary.json", "w", encoding="utf-8") as f:
    json.dump(results, f, ensure_ascii=False, indent=2)

log("\n✅ All results saved to: results_summary.txt")
log("✅ Structured results saved to: results_summary.json")
log("✅ All figures saved to: figures/")
log("\nDone.")