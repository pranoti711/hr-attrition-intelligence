-- Two helper views for Power BI. Run in pgAdmin on hr_attrition. Safe to re-run.

-- Retention curve by overtime (SQL life table), ready for a line chart
CREATE OR REPLACE VIEW mart.retention_curve AS
WITH tenure AS (
    SELECT is_overtime, years_at_company AS t,
           COUNT(*) AS people, SUM(attrition_flag) AS leavers
    FROM staging.hr_employee
    GROUP BY is_overtime, years_at_company
),
atrisk AS (
    SELECT is_overtime, t, leavers,
           SUM(people) OVER (PARTITION BY is_overtime ORDER BY t DESC) AS at_risk
    FROM tenure
)
SELECT CASE WHEN is_overtime THEN 'Overtime' ELSE 'No overtime' END AS grp,
       t AS tenure_year, at_risk::int AS at_risk, leavers::int AS leavers,
       ROUND(leavers::numeric / at_risk, 3) AS hazard,
       ROUND(EXP(SUM(LN(GREATEST(1 - leavers::numeric / at_risk, 1e-9)))
                 OVER (PARTITION BY is_overtime ORDER BY t)), 3) AS retained
FROM atrisk
WHERE at_risk >= 30;                       -- hide the noisy tail where fewer than 30 people remain

-- One row per current employee for the action-list page
CREATE OR REPLACE VIEW mart.employee_action_list AS
SELECT c.employee_id, c.department, c.job_role, c.job_level, c.risk_tier,
       ROUND((100 * c.risk_12m_calibrated)::numeric, 1)   AS risk_12m_pct,
       ROUND((c.replacement_cost_inr / 1e5)::numeric, 1)  AS replacement_cost_lakh,
       ROUND((c.expected_cost_inr / 1e5)::numeric, 1)     AS expected_cost_lakh,
       r.top_reasons
FROM mart.employee_attrition_cost c
JOIN mart.employee_attrition_risk r USING (employee_id);
