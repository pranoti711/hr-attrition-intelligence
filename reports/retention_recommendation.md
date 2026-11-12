# Attrition risk and retention recommendation

*Prepared from a synthetic IBM HR dataset (method demonstration) calibrated to real IT-services attrition.*

## Headline
- Workforce in scope: **1,233 current employees**. Annual attrition is set to **13.0%**, the latest reported level for large Indian IT services firms (Infosys 12.7% (FY27 Q2), TCS 13.3% (FY27 Q2)).
- Expected exits in the next 12 months: **~160**, with an expected replacement cost of **Rs 101.5 crore** (about Rs 63.3 lakh per exit).
- High and Medium tiers (265 people, 21% of staff) carry **47%** of the expected cost; the rest is spread thinly across the large Low tier, so a high-risk-only programme would miss about half of it.

## What to do
- Most valuable single action: **Cap overtime**, applied only where it pays back for the individual
  (108 employees): net benefit **Rs 4.5 crore** per year
  after programme cost, assuming 50% of the model-implied effect is real.
- Combined package of the levers that pay back (Cap overtime, Stock-option grant, Cut frequent travel), targeted person by person: **Rs 7.0 crore** net per year (re-scored jointly, so smaller than the sum of the parts).
- Do not fund (no employee pays back at these assumptions): Pay correction +10%.

- Start with a **controlled pilot** (treated vs comparable untreated group) before scaling: the model shows association, not proof of cause, and picking the employees with the best estimated payoff overstates the benefit (selection effect).

## Key assumptions (edit in the script)
- Replacement cost = 0.50x to 2.0x annual salary by job level (Gallup and SHRM cite 0.5x-2x).
- Pay in the dataset is treated as USD at Rs 95/USD (pay scale factor 1.0).
- Programme costs: Cap overtime 3% of salary, Pay correction +10% 10% of salary, Stock-option grant 4% of salary, Cut frequent travel 1% of salary.
- Effect realization: 50% of the model-implied change. Sensitivity is in `retention_sensitivity.csv`.

## Limits
- Synthetic data: use for method, not for company-specific claims.
- The risk model measures association; the what-if effects are model estimates, hence the haircut and the pilot recommendation.
- Age and gender are not model inputs, but age-related differences in detection exist (see the Layer 2 fairness audit); use scores as decision support, not for automated decisions.
