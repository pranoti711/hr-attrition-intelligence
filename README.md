# HR Attrition Intelligence

**From real IT-services attrition to a rupee-level retention decision.**

![Python](https://img.shields.io/badge/Python-3.10-3776AB?logo=python&logoColor=white)
![PostgreSQL](https://img.shields.io/badge/PostgreSQL-warehouse-4169E1?logo=postgresql&logoColor=white)
![Power BI](https://img.shields.io/badge/Power%20BI-dashboard-F2C811?logo=powerbi&logoColor=black)
![scikit-learn](https://img.shields.io/badge/scikit--learn-SHAP%20%7C%20survival-F7931E?logo=scikitlearn&logoColor=white)
![HTML](https://img.shields.io/badge/HTML%2FJS-what--if%20simulator-E34F26?logo=html5&logoColor=white)

![Home](docs/images/dashboard_home.png)

An end-to-end people-analytics project built around one question:
**where does employee attrition cost the most, why does it happen, and which retention action pays back?**

It combines **real company disclosures** (TCS and Infosys quarterly fact sheets, 2020 to 2026) with a
**synthetic employee-level dataset** (IBM HR Analytics) to show a complete method: market context, risk modelling,
fairness checks, and a cost-based decision in rupees. The results are delivered three ways: a Power BI report,
an interactive what-if simulator, and a one-page recommendation memo.

> **Data note:** the market layer uses real published figures. The employee layer uses a *synthetic* public
> dataset created by IBM to demonstrate method, so employee-level results are illustrative, not claims about any real company.

## Contents
[At a glance](#at-a-glance) | [Key results](#key-results) | [Dashboard](#power-bi-dashboard) | [Simulator](#interactive-what-if-simulator) |
[Architecture](#architecture) | [Tech stack](#tech-stack) | [Data sources](#data-sources) | [Methods](#methods) |
[Repository structure](#repository-structure) | [Reproduce it](#reproduce-it) |
[Data definitions and caveats](#data-definitions-and-caveats) | [Limitations](#limitations) | [Roadmap](#roadmap)

## At a glance

| Layer | Question | Data | Main output |
|---|---|---|---|
| 1. Market | How has attrition moved at TCS and Infosys, and are the two comparable? | Real quarterly fact sheets, 2020 to 2026 | Harmonised attrition series, peer-gap and data-quality views |
| 2. Risk | Who is likely to leave, and why? Is the model fair? | Synthetic IBM HR data (1,470 employees) | Kaplan-Meier and Cox survival models, risk model with SHAP, fairness audit |
| 3. Decisions | What does attrition cost in rupees, and which action pays back? | Layer 2 scores calibrated to Layer 1 | Cost model, lever scenarios, simulator, recommendation memo |

## Key results

### Layer 1: the market (real data)
![Attrition trend](reports/attrition_trend.png)

- **TCS** attrition (last twelve months) fell to **7.2%** in FY21 Q4, peaked at **21.5%** in FY23 Q2, and was **13.3%** in FY27 Q2.
- **Infosys** (voluntary, IT services) peaked higher and earlier: **28.4%** in FY23 Q1, and was **12.7%** in FY27 Q2.
- Infosys led the wave by about one quarter. The gap peaked at about **+10 percentage points** (FY22 Q4); from FY24 the two stay within about one point of each other.
- Only a few quarters are truly like-for-like, because the two companies report attrition on different bases (see [caveats](#data-definitions-and-caveats)).

### Layer 2: who leaves, and why (synthetic employee data, 1,470 employees, 16.1% attrition)
| Question | Result |
|---|---|
| Which factors raise the risk of leaving, holding others fixed? | Cox model hazard ratios: **Sales Representative 3.3**, **overtime 3.1**, frequent travel 1.6, single 1.6; protective: job involvement 0.73, work-life balance 0.78, job satisfaction 0.80 |
| How well can we rank who is at risk? | 5-fold cross-validated **ROC-AUC 0.83**; the top 10% of scores contain **42%** of all leavers; well calibrated (top decile: 65% predicted vs 68% observed) |
| Does a complex model beat a simple one? | No: logistic regression ties gradient boosting (AUC 0.830 vs 0.832) |
| Is it fair? | Age, gender and marital status are **not** model inputs. Gender is balanced (flag rates 20.1% female vs 20.0% male; recall 69% vs 58%). **Age shows a gap:** recall is 76.9% under 30 but 41.2% for 40-49 and 34.8% for 50+ |

Overtime is the clearest single signal: employees who work overtime are retained far less over time than those who do not (see the Kaplan-Meier chart on the dashboard and in `reports/km_overtime.png`).

### Layer 3: what it costs and what pays back
Risk scores were rescaled so the workforce's expected attrition matches the real IT-services level from Layer 1 (13.0%, the average of TCS 13.3% and Infosys 12.7%).

- **1,233 current employees**, about **160 expected exits** in 12 months, about **Rs 101.5 crore** expected replacement cost (about Rs 63 lakh per exit under the pay assumptions below).
- The High and Medium risk tiers are 22% of staff but carry about **47%** of the expected cost.
- Manager and Sales Executive roles carry the most expected cost (about Rs 25 crore each). Sales Representatives are the highest-risk role but cheap to replace; Managers and Research Directors are the reverse.
- Retention levers, net benefit per year when applied only where it pays back for the individual (assuming 50% of the model-implied effect is real):

| Lever | Net benefit |
|---|---|
| Cap overtime | Rs 4.5 crore |
| Stock-option grant | Rs 3.0 crore |
| Cut frequent travel | Rs 1.1 crore |
| Pay correction (+10%) | **never pays back** (about Rs -5.7 crore if applied to everyone eligible) |
| Combined package of the three that pay back | **Rs 7.0 crore** (Rs 1.8 crore if only 25% of the effect is real; Rs 15.7 crore at 100%) |

The one-page memo is in [`reports/retention_recommendation.md`](reports/retention_recommendation.md).

## Power BI dashboard
A Power BI report (home page plus eight pages) reads the PostgreSQL `mart` schema. Each page answers one step of the story:
*real market trend > who is at risk > what to do about it, in rupees.*
The file is at [`dashboard/dashboard.pbix`](dashboard/dashboard.pbix).

| # | Page | What it shows |
|---|---|---|
| 0 | **Home** | Three-part storyline (Market, Risk, Decisions) and navigation to every page |
| 1 | **Executive summary** | Headline KPIs (current employees, expected exits, expected cost, cost per exit, best package net benefit), expected cost by risk tier and by job role |
| 2 | **Market trend (real data)** | Attrition % for Infosys, TCS and Wipro over time, plus peak, trough and recovery timing for each company |
| 3 | **Peer gap and data quality** | Infosys minus TCS gap, like-for-like vs scope-differs quarters, missing LTM quarters, and big quarterly jumps flagged for review |
| 4 | **Who leaves** | Overall attrition, retention curves with and without overtime, and segment attrition ranked by lift |
| 5 | **Risk drivers and model** | Cox hazard ratios, model vs baseline AUC, observed attrition by risk tier, predicted vs actual rate by decile |
| 6 | **Fairness** | Recall, flag rate, false-alarm rate and AUC by gender |
| 7 | **Cost and what-if** | Net benefit of each retention lever (treat all eligible vs treat only where it pays), employees targeted, exits avoided, sensitivity to the share of effect that is real |
| 8 | **Action list** | Employee-level list with 12-month risk and expected cost, filterable by department, role and risk tier |


![home page ](docs/images/home.png)

![Executive summary](docs/images/dashboard_executive.png)
![Market trend](docs/images/dashboard_market.png)
![Peer gap and data quality](docs/images/dashboard_peer_gap.png)
![Who leaves](docs/images/dashboard_who_leaves.png)
![Risk drivers and model](docs/images/dashboard_risk.png)
![Fairness](docs/images/dashboard_fairness.png)
![Cost and what-if](docs/images/dashboard_cost.png)
![Action list](docs/images/dashboard_action_list.png)


## Interactive what-if simulator
A single self-contained HTML file (no server needed) lets you change the assumptions and see every number update live:
the annual attrition target, the share of the effect that is really achievable, replacement-cost scale, pay scale, USD to INR rate,
which risk tiers to treat, and which levers to include and at what cost.

Open [`reports/what_if_simulator.html`](reports/what_if_simulator.html) in a browser.
Live demo: *add your GitHub Pages link here after publishing.*

![What-if simulator](docs/images/what_if_simulator.png)

At the default settings it shows 160 expected exits, Rs 101.5 crore expected replacement cost, 29.2 exits avoided by the package,
Rs 5.4 crore programme cost, Rs 7.5 crore net benefit and a 139% return on spend. A second panel re-runs the package at 25%, 50%, 75%
and 100% of the modelled effect, and a risk-by-cost matrix shows which job roles to fix first.

## Architecture
```mermaid
flowchart LR
    A["TCS and Infosys fact sheets (PDF)"] --> B["ingest_attrition.py: download and extract"]
    B --> C[("PostgreSQL: raw, staging, mart")]
    D["IBM HR dataset (synthetic)"] --> E["layer2_load_ibm.py"]
    E --> C
    C --> F["Survival analysis: Kaplan-Meier and Cox"]
    C --> G["Risk model: SHAP and fairness audit"]
    G --> H["Cost model and what-if scenarios"]
    C --> I["Power BI dashboard"]
    H --> I
    H --> J["Interactive HTML simulator"]
    H --> K["Retention recommendation memo"]
```

## Tech stack

| Area | Tools |
|---|---|
| Data engineering | Python (requests, regex and PDF text extraction), PostgreSQL with `raw`, `staging` and `mart` schemas |
| Analysis and modelling | pandas, scikit-learn, lifelines (Kaplan-Meier, Cox), SHAP, matplotlib |
| SQL | Window functions, conditional aggregation, percentiles, life tables, data-quality checks |
| BI and delivery | Power BI (DAX, page navigation, slicers), self-contained HTML/JavaScript simulator |
| Tooling | VS Code, Git and GitHub, PowerShell |

## Data sources
| Source | Used for | Notes |
|---|---|---|
| TCS quarterly fact sheets (investor relations) | Layer 1 | FY20 to FY27 Q2; reports LTM attrition, switching to a voluntary basis from FY26 Q2 |
| Infosys quarterly fact sheets (investor relations) | Layer 1 | FY04 to FY27; reporting basis changes over time (see caveats) |
| Wipro | Layer 1 | Only 3 quarters, entered by hand from press reports and flagged `manual_verify` |
| IBM HR Analytics Employee Attrition and Performance | Layers 2 and 3 | Synthetic, 1,470 rows, downloaded by `layer2_load_ibm.py` |
| Replacement-cost range | Layer 3 assumption | Commonly cited 0.5x to 2x annual salary (Gallup, SHRM), applied by job level |

Company PDFs are downloaded by the scripts and are **not** stored in this repository.

## Methods
**Layer 1: data engineering and market analysis**
- Python downloads quarterly fact sheets and extracts attrition and headcount with layout-specific regex patterns and a fallback PDF reader.
- Rows that fail validation are flagged `needs_review` and excluded. Every row keeps its source URL and its reporting definition.
- A PostgreSQL `raw, staging, mart` warehouse builds a harmonised LTM series, peer-gap and data-quality views.

**Layer 2: employee-level analysis**
- Descriptive segment analysis in SQL (window functions, conditional aggregation, percentiles) and a yearly life table that reconciles with the Python survival curves.
- Kaplan-Meier curves, log-rank test, and a Cox proportional-hazards model with a proportional-hazards check.
- Gradient-boosting risk model vs a logistic baseline with 5-fold cross-validation, calibration, lift, SHAP explanations and per-employee reason codes.
- Sensitivity checks for leakage (dropping tenure-capped variables) and for the cost of excluding protected attributes, plus a fairness audit by gender and age band.

**Layer 3: decision analytics**
- Calibrates risk to real attrition with an intercept (log-odds) shift, prices each exit as annual salary times a job-level multiple, and computes expected cost.
- Re-scores employees under four levers (overtime cap, pay correction, stock options, less travel), applies an effect haircut, nets against programme cost, and targets only people for whom the estimated saving exceeds the cost.
- Sensitivity analysis on how much of the effect is real; a risk-by-cost matrix by role; an interactive simulator.

## Repository structure
```
hr-attrition-intelligence/
├── README.md
├── requirements.txt
├── .gitignore
├── ingest_attrition.py              # Layer 1: download fact sheets, extract metrics, load to Postgres
├── plot_attrition.py                # Layer 1: trend chart
├── layer2_load_ibm.py               # Layer 2: load the IBM HR dataset
├── layer2_survival.py               # Layer 2: Kaplan-Meier and Cox model
├── layer2_risk_model.py             # Layer 2: risk model, SHAP, fairness audit
├── layer3_cost_and_scenarios.py     # Layer 3: cost in rupees and retention scenarios
├── build_simulator.py               # Layer 3: builds the interactive HTML simulator
├── patch_infosys.py                 # one-off patch applied to the Infosys data
├── patch2.py                        # one-off patch applied after ingestion
├── sql/
│   └── staging/
│       ├── 01_layer1_mart.sql
│       ├── 02_layer2_staging.sql
│       ├── 03_layer3_risk_cost_matrix.sql
│       ├── 04_powerbi_views.sql
│       └── 05_layer2_analysis_queries.sql
├── data/
│   ├── manual/
│   │   └── manual_attrition.csv     # hand-entered figures (Wipro)
│   ├── processed/
│   │   ├── company_attrition.csv    # extracted TCS and Infosys series
│   │   └── employee_attrition_risk.csv
│   └── raw/                         # downloaded files (not committed)
│       ├── ibm_hr/
│       ├── infosys/
│       └── tcs/
├── notebooks/                       # exploratory work
├── reports/
│   ├── attrition_trend.png          # Layer 1 chart
│   ├── km_overtime.png              # Kaplan-Meier curves
│   ├── km_income.png
│   ├── km_role.png
│   ├── cox_forest.png               # Cox hazard ratios
│   ├── cox_hazard_ratios.csv
│   ├── cox_ph_test.csv              # proportional-hazards check
│   ├── shap_summary.png             # risk drivers
│   ├── risk_feature_importance.csv
│   ├── risk_model_metrics.csv
│   ├── risk_calibration.png
│   ├── risk_calibration.csv
│   ├── fairness_by_group.csv
│   ├── employee_attrition_cost.csv  # per-employee expected cost
│   ├── retention_scenarios.csv
│   ├── retention_sensitivity.csv
│   ├── retention_recommendation.md  # one-page memo
│   └── what_if_simulator.html       # interactive simulator
├── dashboard/
│   └── dashboard.pbix               # Power BI report
└── docs/images/                     # screenshots used in this README
```

## Reproduce it
**Requirements:** Python 3.10+, PostgreSQL 13+, Power BI Desktop (only for the dashboard).

```powershell
git clone https://github.com/pranoti711/hr-attrition-intelligence.git
cd hr-attrition-intelligence
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt

# create an empty database called hr_attrition (pgAdmin or createdb), then:
$env:DATABASE_URL = "postgresql+psycopg2://USER:PASSWORD@localhost:5432/hr_attrition"
```
Never commit credentials; the scripts read the connection string from the environment.
On macOS or Linux use `export DATABASE_URL="..."` and `source .venv/bin/activate`.

Run in this order (SQL files run in pgAdmin or `psql -d hr_attrition -f file.sql`):

| Step | Command or file | Creates |
|---|---|---|
| 1 | `python ingest_attrition.py all` | downloads PDFs, extracts `data/processed/company_attrition.csv` |
| 2 | `python ingest_attrition.py load` | `raw.company_attrition` |
| 3 | `sql/staging/01_layer1_mart.sql` | Layer 1 dimensions, staging and mart views |
| 4 | `python layer2_load_ibm.py` | `raw.ibm_hr_employee` |
| 5 | `sql/staging/02_layer2_staging.sql` | `staging.hr_employee`, segment view |
| 6 | `python layer2_survival.py` | Kaplan-Meier charts, Cox hazard ratios |
| 7 | `python layer2_risk_model.py` | `mart.employee_attrition_risk`, SHAP, fairness audit |
| 8 | `python layer3_cost_and_scenarios.py` | `mart.employee_attrition_cost`, `mart.retention_scenarios`, memo |
| 9 | `sql/staging/03`, `04`, `05` | risk-by-cost matrix, Power BI views, SQL analysis pack |
| 10 | `python build_simulator.py` and `python plot_attrition.py` | simulator HTML and trend chart |
| 11 | Open `dashboard/dashboard.pbix` | Power BI report; point its data source at your `hr_attrition` database |

Models use fixed random seeds, but results can differ slightly across library versions.

## Data definitions and caveats
Companies report attrition on different bases, and the basis changes over time. The pipeline stores the definition with every row and never mixes bases silently.

| Company | Period | Basis stored as |
|---|---|---|
| TCS | FY20 to FY26 Q1 | `ltm_unspecified` (LTM, IT services; whether it includes involuntary exits is not yet confirmed from the footnotes) |
| TCS | FY26 Q2 onward | `voluntary_ltm` |
| Infosys | FY04 to FY15 | `total_ltm` |
| Infosys | FY16 to FY20 | `annualized_consolidated` (one quarter annualized, **not** comparable with LTM) |
| Infosys | FY21 | `annualized_voluntary_it` |
| Infosys | FY22 onward | `voluntary_ltm` |

The harmonised series in `mart.attrition_comparable` uses LTM-type measures only, so **Infosys has no comparable value for FY16 to FY21**,
which is where its COVID-era low sits. The peer-gap view carries a `same_scope` flag, and the surge-era gap should not be read as like-for-like.
The "Peer gap and data quality" dashboard page makes this visible: it counts the like-for-like quarters, the missing LTM quarters and the large quarter-to-quarter jumps.

**Data quality:** seven SQL checks (row counts, uniqueness, ranges, referential integrity) pass, and a SQL life table reproduces the Python
survival estimates (overtime vs no overtime retained after four years: 80.5% vs 93.0% in SQL against 81% vs 93% at five years in Python, the small difference being a whole-year convention).

## Limitations
- **The employee data is synthetic**, and it is a one-time snapshot, so tenure is measured at the snapshot or at exit rather than from a proper follow-up study. Treat Layer 2 and 3 results as a demonstration of method.
- **Associations, not causes.** The what-if effects are model estimates; they are haircut by 50% by default and the memo recommends a controlled pilot. Choosing the people with the best estimated payoff also overstates the benefit.
- **Costs rest on assumptions** (replacement-cost multiples by job level, programme costs, an exchange rate of Rs 95 per USD, and US-level pay from the dataset converted to rupees). All are set at the top of `layer3_cost_and_scenarios.py` and adjustable in the simulator.
- **The simulator adds lever effects on the log-odds scale**, which slightly overstates the joint effect compared with the exact re-scoring in the script (Rs 7.5 crore vs Rs 7.0 crore for the default package).
- **Fairness:** age is not a model input, yet detection is much weaker for older employees through correlated features. The scores are intended as decision support, not automated decisions.
- **Coverage:** only TCS and Infosys are complete. Wipro has three hand-entered quarters; HCLTech and Tech Mahindra are not included.
- **Extraction** is regex-based on PDF text. Rows that fail validation are flagged and excluded, but a spot check against the source PDFs is advisable when a new quarter is added.

## Roadmap
- Add Wipro, HCLTech and Tech Mahindra from their fact sheets and confirm TCS's pre-FY26 attrition definition.
- Replace the synthetic employee layer with real HRIS data where available, and add time-varying covariates to the survival model.
- Design the retention pilot (treated vs comparable untreated group) and track realised effects.
- Schedule the Layer 1 refresh each quarter and add automated tests for the extraction patterns.

## Author
**Pranoti Munjankar** | [LinkedIn](www.linkedin.com/in/pranoti-munjankar-1760092b9) | 

[GitHub](https://github.com/pranoti711)

