#!/usr/bin/env python3
"""
Layer 2, steps 3 + 4: attrition risk model, SHAP explanations, fairness audit.

  1. Compare a logistic-regression baseline with gradient boosting (5-fold CV).
  2. Sanity checks: does dropping protected attributes or tenure-capped
     variables change accuracy?  (leakage / fairness cost check)
  3. Calibration and top-decile lift from out-of-fold predictions.
  4. SHAP: global drivers + three "reason codes" per employee.
  5. Fairness audit by gender and age band.
  6. Write mart.employee_attrition_risk to Postgres (feeds Layer 3 + Power BI).

Data: staging.hr_employee (IBM HR - SYNTHETIC dataset).
Run:  python layer2_risk_model.py     (needs DATABASE_URL; pip install scikit-learn shap)
Outputs go to reports/ and data/processed/.
"""
import os
import warnings
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.inspection import permutation_importance
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, brier_score_loss, roc_auc_score
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sqlalchemy import create_engine, text

warnings.filterwarnings("ignore")
ROOT = Path(__file__).resolve().parent
REPORTS = ROOT / "reports"
PROCESSED = ROOT / "data" / "processed"
REPORTS.mkdir(exist_ok=True)
PROCESSED.mkdir(parents=True, exist_ok=True)

url = os.environ.get("DATABASE_URL")
if not url:
    raise SystemExit("Set DATABASE_URL first, e.g. postgresql+psycopg2://postgres:PASSWORD@localhost:5432/hr_attrition")
engine = create_engine(url)
df = pd.read_sql("SELECT * FROM staging.hr_employee", engine)
y = df["attrition_flag"].to_numpy()
print(f"{len(df)} employees, attrition rate {y.mean():.1%}")

# ------------------------------------------------------------------ features
DERIVED = ["employee_id", "attrition_flag", "tenure_band", "age_band", "income_quartile"]
PROTECTED = ["age", "gender", "marital_status"]  # NOT used by the deployed model
TENURE_CAPPED = ["years_in_current_role", "years_with_curr_manager", "years_since_last_promotion"]


def build(drop):
    raw = df.drop(columns=DERIVED + drop).copy()
    raw["is_overtime"] = raw["is_overtime"].astype(int)
    cats = raw.select_dtypes(exclude=[np.number]).columns.tolist()
    return raw, cats, pd.get_dummies(raw, columns=cats).astype(float)


raw, cat_cols, X = build(PROTECTED)

cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
logit = make_pipeline(StandardScaler(), LogisticRegression(max_iter=3000, C=0.3))
gbm = GradientBoostingClassifier(n_estimators=200, learning_rate=0.05, max_depth=3,
                                 subsample=0.8, random_state=42)


def oof(model, Xm):
    return cross_val_predict(model, Xm, y, cv=cv, method="predict_proba")[:, 1]


# ------------------------------------------------------- 1. model comparison
rows = []
scores = {}
for name, model in [("Logistic regression", logit), ("Gradient boosting", gbm)]:
    p = oof(model, X)
    scores[name] = p
    rows.append({"model": name, "roc_auc": roc_auc_score(y, p),
                 "avg_precision": average_precision_score(y, p), "brier": brier_score_loss(y, p)})
metrics = pd.DataFrame(rows).round(3)
metrics.to_csv(REPORTS / "risk_model_metrics.csv", index=False)
print("\n1) Out-of-fold performance (5-fold CV; random guessing: AUC 0.50, avg precision "
      f"{y.mean():.2f})")
print(metrics.to_string(index=False))

# deployed model = gradient boosting (explainable with SHAP); the baseline is the benchmark
p_final = scores["Gradient boosting"]

# --------------------------------------------------------- 2. sanity checks
print("\n2) Sanity checks (gradient boosting, same CV)")
for label, drop in [("deployed (no protected attributes)", PROTECTED),
                    ("with age/gender/marital status added", []),
                    ("without tenure-capped variables too", PROTECTED + TENURE_CAPPED)]:
    _, _, Xc = build(drop)
    print(f"   {label:42s} AUC {roc_auc_score(y, oof(gbm, Xc)):.3f}")

# ------------------------------------------------- 3. calibration and lift
cal = pd.DataFrame({"p": p_final, "y": y})
cal["decile"] = pd.qcut(cal["p"].rank(method="first"), 10, labels=False) + 1
cal_t = cal.groupby("decile").agg(mean_predicted=("p", "mean"), actual_rate=("y", "mean"),
                                  employees=("y", "size")).round(3)
cal_t.to_csv(REPORTS / "risk_calibration.csv")
print("\n3) Calibration by risk decile (1 = lowest risk, 10 = highest)")
print(cal_t.to_string())

order = np.argsort(-p_final)
print("\n   Lift: share of all leavers found in the top-scored employees")
for frac in (0.10, 0.20, 0.30):
    k = int(len(df) * frac)
    cap = y[order[:k]].sum() / y.sum()
    print(f"   top {frac:.0%} of scores -> {cap:.0%} of leavers  ({cap / frac:.1f}x random)")

fig, ax = plt.subplots(figsize=(5.5, 5))
ax.plot([0, 0.5], [0, 0.5], "--", color="grey")
ax.plot(cal_t["mean_predicted"], cal_t["actual_rate"], "o-")
ax.set_xlabel("Predicted attrition probability (decile mean)")
ax.set_ylabel("Observed attrition rate")
ax.set_title("Calibration (out-of-fold)", weight="bold")
ax.grid(alpha=0.3)
fig.tight_layout()
fig.savefig(REPORTS / "risk_calibration.png", dpi=150)
plt.close(fig)

# ------------------------------------------------------------ 4. explanations
gbm.fit(X, y)
parent = {}
for col in X.columns:
    owner = next((c for c in cat_cols if col.startswith(c + "_")), col)
    parent[col] = owner

reasons = [""] * len(df)
try:
    import shap
    sv = shap.TreeExplainer(gbm).shap_values(X)
    if isinstance(sv, list):
        sv = sv[1]
    sv = np.asarray(sv)
    if sv.ndim == 3:
        sv = sv[:,:, 1]
    sv_df = pd.DataFrame(sv, columns=X.columns)
    by_feature = sv_df.T.groupby(parent).sum().T  # sum dummy columns back to their parent feature
    imp = by_feature.abs().mean().sort_values(ascending=False)
    shap.summary_plot(sv, X, show=False, max_display=15)
    plt.tight_layout()
    plt.savefig(REPORTS / "shap_summary.png", dpi=150, bbox_inches="tight")
    plt.close()
    # reason codes: the three features pushing each person's risk up the most
    for i in range(len(df)):
        top = by_feature.iloc[i].sort_values(ascending=False).head(3)
        reasons[i] = "; ".join(
            f"{f}={'Yes' if raw[f].iloc[i] == 1 and f == 'is_overtime' else raw[f].iloc[i]}"
            for f, v in top.items() if v > 0)
    expl = "SHAP"
except ImportError:
    print("\n(shap not installed - using permutation importance instead; no reason codes)")
    pi = permutation_importance(gbm, X, y, n_repeats=5, random_state=42, scoring="roc_auc")
    imp = pd.Series(pi.importances_mean, index=X.columns).groupby(parent).sum().sort_values(ascending=False)
    expl = "permutation importance"

imp.round(4).to_csv(REPORTS / "risk_feature_importance.csv", header=["mean_abs_effect"])
print(f"\n4) Top drivers of predicted risk ({expl}, grouped by original feature)")
print(imp.head(12).round(3).to_string())

# --------------------------------------------------------- 5. fairness audit
# Flag the top 20% of scores for a retention conversation, then compare groups.
flag = np.zeros(len(df), dtype=bool)
flag[order[: int(len(df) * 0.20)]] = True
audit = []
for dim in ("gender", "age_band"):
    for grp, idx in df.groupby(dim).groups.items():
        m = df.index.get_indexer(idx)
        yy, ff, pp = y[m], flag[m], p_final[m]
        pos, neg = yy == 1, yy == 0
        audit.append({
            "dimension": dim, "group": grp, "employees": len(m),
            "actual_attrition_pct": round(100 * yy.mean(), 1),
            "flagged_pct": round(100 * ff.mean(), 1),
            "recall_pct": round(100 * ff[pos].mean(), 1) if pos.sum() else np.nan,  # leavers caught
            "false_alarm_pct": round(100 * ff[neg].mean(), 1) if neg.sum() else np.nan,  # stayers flagged
            "precision_pct": round(100 * yy[ff].mean(), 1) if ff.sum() else np.nan,
            "auc": round(roc_auc_score(yy, pp), 3) if 0 < pos.sum() < len(yy) else np.nan,
        })
audit = pd.DataFrame(audit)
audit["flag_rate_vs_highest"] = audit.groupby("dimension")["flagged_pct"].transform(lambda s: (s / s.max()).round(2))
audit.to_csv(REPORTS / "fairness_by_group.csv", index=False)
print("\n5) Fairness audit - top 20% flagged (age and gender are NOT model inputs)")
print(audit.to_string(index=False))
print("   Read: similar recall / false-alarm rates across groups = equal treatment of equal risk;\n"
      "   different flag rates mostly reflect different real attrition rates (column 4).")

# ------------------------------------------------------------ 6. save scores
pct_rank = pd.Series(p_final).rank(pct=True, method="first").to_numpy()
tier = np.where(pct_rank > 0.90, "High", np.where(pct_rank > 0.70, "Medium", "Low"))
out = pd.DataFrame({
    "employee_id": df["employee_id"], "department": df["department"], "job_role": df["job_role"],
    "risk_probability": p_final.round(4), "risk_percentile": (100 * pct_rank).round(1),
    "risk_tier": tier, "has_left": y, "top_reasons": reasons,
})
out.to_csv(PROCESSED / "employee_attrition_risk.csv", index=False)
with engine.begin() as conn:
    conn.execute(text("CREATE SCHEMA IF NOT EXISTS mart"))
    if conn.execute(text("SELECT to_regclass('mart.employee_attrition_risk')")).scalar():
        conn.execute(text("TRUNCATE mart.employee_attrition_risk"))
out.to_sql("employee_attrition_risk", engine, schema="mart", if_exists="append", index=False)
tiers = out.groupby("risk_tier").agg(employees=("has_left", "size"), actual_attrition_pct=("has_left", lambda s: round(100 * s.mean(), 1)))
print("\n6) Saved mart.employee_attrition_risk. Observed attrition by tier:")
print(tiers.reindex(["High", "Medium", "Low"]).to_string())
print(f"\nCharts and tables are in {REPORTS}")
