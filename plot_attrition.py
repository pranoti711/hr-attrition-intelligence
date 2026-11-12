#!/usr/bin/env python3
"""First chart: quarterly attrition trend by company (from company_attrition.csv).

Run from the project folder:
    python plot_attrition.py
Output: reports/attrition_trend.png
"""
from pathlib import Path

import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import pandas as pd

ROOT = Path(__file__).resolve().parent
CSV = ROOT / "data" / "processed" / "company_attrition.csv"
OUT = ROOT / "reports" / "attrition_trend.png"
START = "2019-04-01"  # plot FY20 onwards (older Infosys data is a separate era)

df = pd.read_csv(CSV, parse_dates=["period_end"])
n_review = int((df["status"] == "needs_review").sum())
df = df[(df["status"] != "needs_review") & df["attrition_pct"].notna() & (df["period_end"] >= START)]

fig, ax = plt.subplots(figsize=(11, 6))
# Keep LTM-style and annualized measures on separate lines: they are not comparable.
df["basis"] = df["definition"].fillna("").where(df["definition"].fillna("").str.contains("annualized"), "LTM")
colors = {c: f"C{i}" for i, c in enumerate(sorted(df["company"].unique()))}
for (company, basis), g in df.groupby(["company", "basis"]):
    g = g.sort_values("period_end")
    # break the line wherever quarters are missing instead of drawing across the gap
    xs, ys, prev = [], [], None
    for d, v in zip(g["period_end"], g["attrition_pct"]):
        if prev is not None and (d - prev).days > 120:
            xs.append(d - pd.Timedelta(days=45))
            ys.append(float("nan"))
        xs.append(d)
        ys.append(v)
        prev = d
    ax.plot(xs, ys, marker="o", lw=2, color=colors[company],
            ls={"LTM": "-", "annualized_consolidated": "--"}.get(basis, ":"),
            label=f"{company} ({basis})")

# annotate the TCS peak
tcs = df[df["company"] == "TCS"]
if not tcs.empty:
    pk = tcs.loc[tcs["attrition_pct"].idxmax()]
    ax.annotate(
        f"TCS peak {pk['attrition_pct']:.1f}%\n({pk['fiscal_year']} Q{int(pk['quarter'])})",
        xy=(pk["period_end"], pk["attrition_pct"]),
        xytext=(pk["period_end"] + pd.Timedelta(days=150), pk["attrition_pct"] + 1.5),
        arrowprops=dict(arrowstyle="->"),
        fontsize=10,
    )

# TCS switched from "LTM" to "voluntary LTM" reporting here
switch = pd.Timestamp("2025-09-30")
ax.axvline(switch, ls="--", color="grey", lw=1)
ax.text(switch, ax.get_ylim()[0] + 0.3, " TCS: voluntary LTM\n basis from here",
        fontsize=8, color="grey", va="bottom")

ax.set_title("IT services attrition (LTM %) by quarter", fontsize=14, weight="bold")
ax.set_ylabel("Attrition, last twelve months (%)")
ax.set_xlabel("Quarter end")
ax.xaxis.set_major_formatter(mdates.DateFormatter("%b %Y"))
ax.grid(alpha=0.3)
ax.legend(frameon=False)
fig.text(0.01, 0.01, "Source: company quarterly fact sheets. Definitions differ across companies/periods.",
         fontsize=8, color="grey")
fig.tight_layout(rect=(0, 0.03, 1, 1))

OUT.parent.mkdir(parents=True, exist_ok=True)
fig.savefig(OUT, dpi=160)
print(f"saved {OUT}  (excluded {n_review} needs_review rows)")
