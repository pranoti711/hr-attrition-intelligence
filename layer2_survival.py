#!/usr/bin/env python3
"""
Layer 2, step 2: survival analysis of employee attrition.

  * Kaplan-Meier curves: how long do employees stay, by overtime / income / role group?
  * Cox proportional-hazards model: which factors raise or lower the risk of
    leaving, holding the others fixed (hazard ratios with 95% CI).

Data: staging.hr_employee (IBM HR - SYNTHETIC dataset, see README).
Run:  python layer2_survival.py        (needs DATABASE_URL, pip install lifelines)
Outputs go to reports/.
"""
import os
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sqlalchemy import create_engine

try:
    from lifelines import CoxPHFitter, KaplanMeierFitter  # type: ignore[import-not-found]
    from lifelines.statistics import logrank_test, proportional_hazard_test  # type: ignore[import-not-found]
except ImportError as exc:
    raise SystemExit(
        "Missing dependency: install lifelines with 'pip install lifelines' before running this script."
    ) from exc

ROOT = Path(__file__).resolve().parent
OUT = ROOT / "reports"
OUT.mkdir(exist_ok=True)

url = os.environ.get("DATABASE_URL")
if not url:
    raise SystemExit("Set DATABASE_URL first, e.g. postgresql+psycopg2://postgres:PASSWORD@localhost:5432/hr_attrition")
df = pd.read_sql("SELECT * FROM staging.hr_employee", create_engine(url))

# years_at_company is in whole years, so centre each person inside their year
# (+0.5). This also avoids zero-length durations.
df["duration"] = df["years_at_company"] + 0.5
df["event"] = df["attrition_flag"]
print(f"{len(df)} employees, {df['event'].sum()} leavers, "
      f"median tenure of everyone {df['duration'].median():.1f} yrs")


# ---------------------------------------------------------------- Kaplan-Meier
def km_plot(groups: dict, title: str, fname: str) -> None:
    fig, ax = plt.subplots(figsize=(8, 5))
    for label, g in groups.items():
        km = KaplanMeierFitter()
        km.fit(g["duration"], g["event"], label=f"{label} (n={len(g)})")
        km.plot_survival_function(ax=ax, ci_show=True)
    ax.set_title(title, weight="bold")
    ax.set_xlabel("Years at company")
    ax.set_ylabel("Share still employed")
    ax.set_ylim(0, 1.02)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(OUT / fname, dpi=150)
    plt.close(fig)


def retention_at(g: pd.DataFrame, years: float) -> float:
    km = KaplanMeierFitter().fit(g["duration"], g["event"])
    return float(km.predict(years))


ot = {"Overtime": df[df.is_overtime], "No overtime": df[~df.is_overtime]}
km_plot(ot, "Retention by overtime", "km_overtime.png")
lr = logrank_test(ot["Overtime"]["duration"], ot["No overtime"]["duration"],ot["Overtime"]["event"], ot["No overtime"]["event"])
print(f"\nOvertime vs no overtime: log-rank p = {lr.p_value:.2e}")
for name, g in ot.items():
    print(f"  {name:12s} retained at 2y: {retention_at(g, 2):.0%}, 5y: {retention_at(g, 5):.0%}, 10y: {retention_at(g, 10):.0%}")

inc = {f"Income Q{q}": df[df.income_quartile == q] for q in (1, 2, 3, 4)}
km_plot(inc, "Retention by income quartile", "km_income.png")

df["role_group"] = np.where(df.job_role == "Sales Representative", "Sales Representative", "All other roles")
km_plot({k: g for k, g in df.groupby("role_group")}, "Retention: Sales Representatives vs rest", "km_role.png")

# ---------------------------------------------------------------------- Cox
X = pd.DataFrame({
    "duration": df["duration"],
    "event": df["event"],
    "overtime": df["is_overtime"].astype(int),
    "age": df["age"],
    "monthly_income_k": df["monthly_income"] / 1000.0,  # per 1,000 units of income
    "single": (df["marital_status"] == "Single").astype(int),
    "travel_frequently": (df["business_travel"] == "Travel_Frequently").astype(int),
    "sales_rep": (df["job_role"] == "Sales Representative").astype(int),
    "job_satisfaction": df["job_satisfaction"],  # 1 low .. 4 high
    "environment_satisfaction": df["environment_satisfaction"],
    "work_life_balance": df["work_life_balance"],
    "job_involvement": df["job_involvement"],
    "distance_from_home": df["distance_from_home"],
    "num_companies_worked": df["num_companies_worked"],
    "stock_option_level": df["stock_option_level"],
})
# NOTE: job_level, total_working_years, years_in_current_role and
# years_since_last_promotion are left out on purpose: they are capped by tenure
# itself (nobody can go 8 years without a promotion in 3 years at the company),
# so they would mechanically look protective and inflate the model's accuracy.

cph = CoxPHFitter(penalizer=0.01)
cph.fit(X, duration_col="duration", event_col="event")
s = cph.summary
res = pd.DataFrame({
    "hazard_ratio": np.exp(s["coef"]),
    "ci_low": s["exp(coef) lower 95%"],
    "ci_high": s["exp(coef) upper 95%"],
    "p_value": s["p"],
}).sort_values("hazard_ratio", ascending=False).round(3)
res.to_csv(OUT / "cox_hazard_ratios.csv")
print(f"\nCox model concordance index: {cph.concordance_index_:.3f}")
print(res.to_string())

# forest plot
r = res.sort_values("hazard_ratio")
fig, ax = plt.subplots(figsize=(8, 6))
ax.errorbar(r["hazard_ratio"], range(len(r)),
            xerr=[r["hazard_ratio"] - r["ci_low"], r["ci_high"] - r["hazard_ratio"]],
            fmt="o", capsize=3, color="C0")
ax.axvline(1, color="grey", ls="--")
ax.set_yticks(range(len(r)))
ax.set_yticklabels(r.index)
ax.set_xscale("log")
ax.set_xlabel("Hazard ratio of leaving (log scale; >1 = higher risk)")
ax.set_title("Cox model: what raises or lowers the risk of leaving", weight="bold")
ax.grid(alpha=0.3)
fig.tight_layout()
fig.savefig(OUT / "cox_forest.png", dpi=150)
plt.close(fig)

# does the proportional-hazards assumption hold?  (p < 0.05 = possible violation)
ph = proportional_hazard_test(cph, X, time_transform="rank").summary
bad = ph[ph["p"] < 0.05].index.tolist()
print("\nProportional-hazards check - variables with p < 0.05:", bad if bad else "none")
ph.round(3).to_csv(OUT / "cox_ph_test.csv")
print(f"\nSaved charts and tables to {OUT}")
