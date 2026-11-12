#!/usr/bin/env python3
"""
Layer 3: what does attrition cost in rupees, and which retention action pays back?

  1. Calibrate the Layer-2 risk scores to REAL IT-services attrition (Layer 1).
  2. Price each current employee's exit  (Gallup/SHRM replacement-cost range).
  3. Expected cost  =  12-month exit probability  x  replacement cost.
  4. What-if: overtime cap / pay correction / stock options / less travel -
     re-score employees with the model, haircut the effect, net it against the
     programme cost, and rank by return.
  5. Write mart.employee_attrition_cost + mart.retention_scenarios and an
     executive memo (reports/retention_recommendation.md).

Data: staging.hr_employee, mart.employee_attrition_risk (IBM HR - SYNTHETIC),
      mart.attrition_comparable (REAL company data from Layer 1).
Run:  python layer3_cost_and_scenarios.py        (needs DATABASE_URL)
EDIT THE ASSUMPTIONS BELOW - they are assumptions, not facts.
"""
import os
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier
from sqlalchemy import create_engine, text

warnings.filterwarnings("ignore")
ROOT = Path(__file__).resolve().parent
REPORTS = ROOT / "reports"
REPORTS.mkdir(exist_ok=True)

# =========================================================== ASSUMPTIONS ====
USD_INR = 95.0  # the dataset's pay is treated as USD; ~95-96 INR/USD in mid-2026 (update!)
SALARY_SCALE = 1.0  # e.g. 0.25 to approximate Indian IT pay levels instead of US-level pay
FALLBACK_ATTRITION = 13.0  # % per year, used only if Layer-1 views are missing
REALIZATION = 0.5  # share of the model-implied effect we assume an action really delivers
# Replacement cost as a multiple of annual salary, by job level (Gallup/SHRM: 0.5x - 2x)
REPLACEMENT_MULTIPLE = {1: 0.5, 2: 0.75, 3: 1.0, 4: 1.5, 5: 2.0}
COST_SPLIT = {"hiring": 0.25, "onboarding/training": 0.25, "productivity ramp-up": 0.50}
# programme cost per treated person, as % of that person's annual salary
LEVERS = {
    "Cap overtime": {"cost_pct": 0.03},  # backfill capacity
    "Pay correction +10%": {"cost_pct": 0.10},  # the raise itself
    "Stock-option grant": {"cost_pct": 0.04},
    "Cut frequent travel": {"cost_pct": 0.01},
}
TARGET_TIERS = ("High", "Medium")
# ==========================================================================

url = os.environ.get("DATABASE_URL")
if not url:
    raise SystemExit("Set DATABASE_URL first, e.g. postgresql+psycopg2://postgres:PASSWORD@localhost:5432/hr_attrition")
engine = create_engine(url)

emp = pd.read_sql("SELECT * FROM staging.hr_employee", engine)
risk = pd.read_sql("SELECT employee_id, risk_probability, risk_tier FROM mart.employee_attrition_risk", engine)
emp = emp.merge(risk, on="employee_id", how="inner")
assert len(emp) == len(risk), "employee ids do not match between staging and risk table"

# ------------------------------------------- 1. calibrate to real attrition
try:
    bench = pd.read_sql(
        "SELECT DISTINCT ON (company) company, attrition_pct, fiscal_year, fiscal_quarter "
        "FROM mart.attrition_comparable WHERE company IN ('TCS','Infosys') "
        "ORDER BY company, period_end DESC", engine)
    if bench.empty:
        raise ValueError("no TCS/Infosys rows in mart.attrition_comparable")
    target = float(bench["attrition_pct"].astype(float).mean())
    bench_txt = ", ".join(f"{r.company} {float(r.attrition_pct):.1f}% ({r.fiscal_year} Q{int(r.fiscal_quarter)})"
                          for r in bench.itertuples())
except Exception:
    target, bench_txt = FALLBACK_ATTRITION, "fallback assumption (Layer-1 views not found)"
print(f"Target annual attrition: {target:.1f}%  [{bench_txt}]")


def logit(p):
    p = np.clip(p, 1e-4, 1 - 1e-4)
    return np.log(p / (1 - p))


def sigmoid(z):
    return 1 / (1 + np.exp(-z))


cur = emp[emp["attrition_flag"] == 0].reset_index(drop=True)  # people we can still retain
z0 = logit(cur["risk_probability"].to_numpy())
lo, hi = -5.0, 5.0
for _ in range(60):  # bisection for the intercept shift
    mid = (lo + hi) / 2
    if sigmoid(z0 + mid).mean() < target / 100:
        lo = mid
    else:
        hi = mid
shift = (lo + hi) / 2
cur["p12"] = sigmoid(z0 + shift)
print(f"{len(cur)} current employees; mean Layer-2 score {cur['risk_probability'].mean():.1%} "
      f"-> calibrated 12-month risk {cur['p12'].mean():.1%} (intercept shift {shift:+.2f})")

# ------------------------------------------------------ 2. replacement cost
cur["annual_salary_inr"] = cur["monthly_income"] * 12 * USD_INR * SALARY_SCALE
cur["replacement_multiple"] = cur["job_level"].map(REPLACEMENT_MULTIPLE)
cur["replacement_cost_inr"] = cur["annual_salary_inr"] * cur["replacement_multiple"]
cur["expected_cost_inr"] = cur["p12"] * cur["replacement_cost_inr"]

crore = 1e7
tot_exits = cur["p12"].sum()
tot_cost = cur["expected_cost_inr"].sum()


def inr(x):
    return f"Rs {x / crore:,.1f} crore" if abs(x) >= crore else f"Rs {x / 1e5:,.1f} lakh"


print(f"\nExpected exits next 12 months: {tot_exits:,.0f}   expected replacement cost: {inr(tot_cost)}")
print(f"Average cost per exit: {inr(tot_cost / tot_exits)}")
by_tier = cur.groupby("risk_tier").agg(employees=("p12", "size"), expected_exits=("p12", "sum"),
                                       expected_cost_inr=("expected_cost_inr", "sum")).reindex(["High", "Medium", "Low"])
by_tier["share_of_cost_pct"] = (100 * by_tier["expected_cost_inr"] / tot_cost).round(1)
by_tier["expected_cost"] = by_tier["expected_cost_inr"].map(inr)
print("\nBy risk tier:")
print(by_tier[["employees", "expected_exits", "expected_cost", "share_of_cost_pct"]].round(1).to_string())
by_role = (cur.groupby("job_role").agg(employees=("p12", "size"), expected_exits=("p12", "sum"),
                                       expected_cost_inr=("expected_cost_inr", "sum"))
           .sort_values("expected_cost_inr", ascending=False))
by_role["expected_cost"] = by_role["expected_cost_inr"].map(inr)
print("\nTop job roles by expected cost:")
print(by_role[["employees", "expected_exits", "expected_cost"]].head(5).round(1).to_string())

# ------------------------------------------------------- 3. what-if levers
PROTECTED = ["age", "gender", "marital_status"]  # same exclusions as Layer 2
DERIVED = ["employee_id", "attrition_flag", "tenure_band", "age_band", "income_quartile",
           "risk_probability", "risk_tier"]


def design(frame):
    raw = frame.drop(columns=[c for c in DERIVED + PROTECTED if c in frame.columns]).copy()
    raw["is_overtime"] = raw["is_overtime"].astype(int)
    cats = raw.select_dtypes(exclude=[np.number]).columns.tolist()
    return pd.get_dummies(raw, columns=cats).astype(float)


X_all = design(emp)
gbm = GradientBoostingClassifier(n_estimators=200, learning_rate=0.05, max_depth=3,
                                 subsample=0.8, random_state=42).fit(X_all, emp["attrition_flag"])
cols = X_all.columns


def model_logit(frame):
    return logit(gbm.predict_proba(design(frame).reindex(columns=cols, fill_value=0.0))[:, 1])


eligible_tier = cur["risk_tier"].isin(TARGET_TIERS)
masks = {
    "Cap overtime": eligible_tier & (cur["is_overtime"]),
    "Pay correction +10%": eligible_tier & (cur["monthly_income"] < cur["monthly_income"].quantile(0.5)),
    "Stock-option grant": eligible_tier & (cur["stock_option_level"] == 0),
    "Cut frequent travel": eligible_tier & (cur["business_travel"] == "Travel_Frequently"),
}


def apply(name, frame, mask):
    f = frame.copy()
    if name == "Cap overtime":
        f.loc[mask, "is_overtime"] = False
    elif name == "Pay correction +10%":
        f.loc[mask, "monthly_income"] = f.loc[mask, "monthly_income"] * 1.10
    elif name == "Stock-option grant":
        f.loc[mask, "stock_option_level"] = 1
    elif name == "Cut frequent travel":
        f.loc[mask, "business_travel"] = "Travel_Rarely"
    return f


base_logit = model_logit(cur)
delta, cost = {}, {}
for name, mask in masks.items():
    delta[name] = model_logit(apply(name, cur, mask)) - base_logit  # model-implied change, per person
    cost[name] = np.where(mask, LEVERS[name]["cost_pct"] * cur["annual_salary_inr"], 0.0)
COMBINED = "Combined (positive-ROI levers)"


def evaluate(name, r):
    p_after = sigmoid(logit(cur["p12"].to_numpy()) + r * delta[name])
    saved = (cur["p12"].to_numpy() - p_after)
    benefit = saved * cur["replacement_cost_inr"].to_numpy()
    net_i = np.where(masks[name], benefit - cost[name], 0.0)
    return saved, benefit, net_i


# combined package = only the levers that pay back for someone (effects are not additive,
# so the package is re-scored jointly rather than summed)
keep = [n for n in LEVERS if evaluate(n, REALIZATION)[2].clip(min=0).sum() > 0]
combo = cur.copy()
for name in keep:
    combo = apply(name, combo, masks[name])
delta[COMBINED] = model_logit(combo) - base_logit
cost[COMBINED] = sum(cost[n] for n in keep) if keep else np.zeros(len(cur))
masks[COMBINED] = (pd.concat([masks[n] for n in keep], axis=1).any(axis=1)
                   if keep else pd.Series(False, index=cur.index))

rows = []
for name in list(LEVERS) + [COMBINED]:
    saved, benefit, net_i = evaluate(name, REALIZATION)
    m = masks[name].to_numpy()
    spend = cost[name].sum()
    best = net_i > 0
    rows.append({
        "intervention": name, "employees_targeted": int(m.sum()),
        "exits_avoided": saved[m].sum(), "savings_inr": benefit[m].sum(), "program_cost_inr": spend,
        "net_benefit_inr": benefit[m].sum() - spend,
        "roi": (benefit[m].sum() - spend) / spend if spend else np.nan,
        "employees_roi_positive": int(best.sum()),
        "net_benefit_if_targeted_inr": net_i[best].sum(),
        "cost_if_targeted_inr": cost[name][best].sum(),
    })
scen = pd.DataFrame(rows)
show = scen.copy()
for c in ("savings_inr", "program_cost_inr", "net_benefit_inr", "net_benefit_if_targeted_inr"):
    show[c] = show[c].map(inr)
show["exits_avoided"] = show["exits_avoided"].round(1)
show["roi"] = (show["roi"] * 100).round(0).astype("Int64").astype(str) + "%"
print(f"\nWhat-if scenarios (effect haircut: {REALIZATION:.0%} of the model-implied change is assumed real)")
print(show[["intervention", "employees_targeted", "exits_avoided", "program_cost_inr", "net_benefit_inr",
            "roi", "employees_roi_positive", "net_benefit_if_targeted_inr"]].to_string(index=False))

sens = []
for r in (0.25, 0.5, 0.75, 1.0):
    for name in list(LEVERS) + [COMBINED]:
        _, _, net_i = evaluate(name, r)
        sens.append({"realization": r, "intervention": name, "net_benefit_if_targeted_inr": net_i[net_i > 0].sum()})
sens = pd.DataFrame(sens)
piv = sens.pivot(index="intervention", columns="realization", values="net_benefit_if_targeted_inr")
print("\nSensitivity: net benefit (ROI-targeted) by how much of the effect is real")
print(piv.map(inr).to_string() if hasattr(piv, "map") else piv.applymap(inr).to_string())

# ------------------------------------------------------------- 4. save
cost_cols = ["employee_id", "department", "job_role", "job_level", "risk_tier", "risk_probability", "p12",
             "annual_salary_inr", "replacement_multiple", "replacement_cost_inr", "expected_cost_inr"]
out_cost = cur[cost_cols].rename(columns={"p12": "risk_12m_calibrated"}).round(4)
out_scen = scen.merge(sens[sens.realization == 1.0][["intervention"]].drop_duplicates(), on="intervention")
out_scen["realization_assumed"] = REALIZATION
out_scen["target_attrition_pct"] = round(target, 2)
out_scen["usd_inr"] = USD_INR
out_cost.to_csv(REPORTS / "employee_attrition_cost.csv", index=False)
out_scen.to_csv(REPORTS / "retention_scenarios.csv", index=False)
sens.to_csv(REPORTS / "retention_sensitivity.csv", index=False)
with engine.begin() as conn:
    conn.execute(text("CREATE SCHEMA IF NOT EXISTS mart"))
    for tbl in ("employee_attrition_cost", "retention_scenarios"):
        if conn.execute(text(f"SELECT to_regclass('mart.{tbl}')")).scalar():
            conn.execute(text(f"TRUNCATE mart.{tbl}"))
out_cost.to_sql("employee_attrition_cost", engine, schema="mart", if_exists="append", index=False)
out_scen.to_sql("retention_scenarios", engine, schema="mart", if_exists="append", index=False)

# ---------------------------------------------------------- 5. exec memo
best_row = scen[scen.intervention != COMBINED].sort_values("net_benefit_if_targeted_inr", ascending=False).iloc[0]
combo_row = scen[scen.intervention == COMBINED].iloc[0]
hm_share = by_tier.loc[["High", "Medium"], "expected_cost_inr"].sum() / tot_cost * 100
hm_n = int(by_tier.loc[["High", "Medium"], "employees"].sum())
never = scen[(scen.intervention != COMBINED) & (scen.employees_roi_positive == 0)]["intervention"].tolist()
broad_neg = scen[(scen.intervention != COMBINED) & (scen.net_benefit_inr < 0)
                 & (scen.employees_roi_positive > 0)]["intervention"].tolist()
memo = f"""# Attrition risk and retention recommendation

*Prepared from a synthetic IBM HR dataset (method demonstration) calibrated to real IT-services attrition.*

## Headline
- Workforce in scope: **{len(cur):,} current employees**. Annual attrition is set to **{target:.1f}%**, the latest reported level for large Indian IT services firms ({bench_txt}).
- Expected exits in the next 12 months: **~{tot_exits:,.0f}**, with an expected replacement cost of **{inr(tot_cost)}** (about {inr(tot_cost / tot_exits)} per exit).
- High and Medium tiers ({hm_n} people, {hm_n / len(cur):.0%} of staff) carry **{hm_share:.0f}%** of the expected cost; the rest is spread thinly across the large Low tier, so a high-risk-only programme would miss about half of it.

## What to do
- Most valuable single action: **{best_row.intervention}**, applied only where it pays back for the individual
  ({int(best_row.employees_roi_positive)} employees): net benefit **{inr(best_row.net_benefit_if_targeted_inr)}** per year
  after programme cost, assuming {REALIZATION:.0%} of the model-implied effect is real.
- Combined package of the levers that pay back ({', '.join(keep) if keep else 'none'}), targeted person by person: **{inr(combo_row.net_benefit_if_targeted_inr)}** net per year (re-scored jointly, so smaller than the sum of the parts).
{('- Do not fund (no employee pays back at these assumptions): ' + ', '.join(never) + '.') if never else ''}
{('- Only for selected individuals, not as a broad programme: ' + ', '.join(broad_neg) + '.') if broad_neg else ''}
- Start with a **controlled pilot** (treated vs comparable untreated group) before scaling: the model shows association, not proof of cause, and picking the employees with the best estimated payoff overstates the benefit (selection effect).

## Key assumptions (edit in the script)
- Replacement cost = {REPLACEMENT_MULTIPLE[1]:.2f}x to {REPLACEMENT_MULTIPLE[5]:.1f}x annual salary by job level (Gallup and SHRM cite 0.5x-2x).
- Pay in the dataset is treated as USD at Rs {USD_INR:.0f}/USD (pay scale factor {SALARY_SCALE}).
- Programme costs: {', '.join(f"{k} {v['cost_pct']:.0%} of salary" for k, v in LEVERS.items())}.
- Effect realization: {REALIZATION:.0%} of the model-implied change. Sensitivity is in `retention_sensitivity.csv`.

## Limits
- Synthetic data: use for method, not for company-specific claims.
- The risk model measures association; the what-if effects are model estimates, hence the haircut and the pilot recommendation.
- Age and gender are not model inputs, but age-related differences in detection exist (see the Layer 2 fairness audit); use scores as decision support, not for automated decisions.
"""
(REPORTS / "retention_recommendation.md").write_text(memo, encoding="utf-8")
print(f"\nSaved mart.employee_attrition_cost, mart.retention_scenarios and {REPORTS / 'retention_recommendation.md'}")
