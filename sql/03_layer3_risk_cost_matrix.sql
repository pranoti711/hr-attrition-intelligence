-- Layer 3: segment risk x cost matrix
-- Run in pgAdmin on hr_attrition AFTER python layer3_cost_and_scenarios.py. Safe to re-run.
CREATE OR REPLACE VIEW mart.risk_cost_matrix AS
WITH seg AS (
    SELECT 'job_role' AS dimension, job_role AS segment,
           COUNT(*) AS employees,
           AVG(risk_12m_calibrated)   AS avg_risk,
           AVG(replacement_cost_inr)  AS avg_replacement_cost_inr,
           SUM(risk_12m_calibrated)   AS expected_exits,
           SUM(expected_cost_inr)     AS expected_cost_inr
    FROM mart.employee_attrition_cost GROUP BY job_role
    UNION ALL
    SELECT 'department', department,
           COUNT(*), AVG(risk_12m_calibrated), AVG(replacement_cost_inr),
           SUM(risk_12m_calibrated), SUM(expected_cost_inr)
    FROM mart.employee_attrition_cost GROUP BY department
),
overall AS (
    SELECT AVG(risk_12m_calibrated) AS r, AVG(replacement_cost_inr) AS c
    FROM mart.employee_attrition_cost
)
SELECT
    s.dimension, s.segment, s.employees,
    ROUND((100 * s.avg_risk)::numeric, 1)                          AS avg_risk_pct,
    ROUND((s.avg_replacement_cost_inr / 1e5)::numeric, 1)          AS avg_replacement_cost_lakh,
    ROUND(s.expected_exits::numeric, 1)                            AS expected_exits,
    ROUND((s.expected_cost_inr / 1e7)::numeric, 2)                 AS expected_cost_crore,
    ROUND((100.0 * s.expected_cost_inr
           / SUM(s.expected_cost_inr) OVER (PARTITION BY s.dimension))::numeric, 1) AS share_of_cost_pct,
    CASE WHEN s.avg_risk >= o.r AND s.avg_replacement_cost_inr >= o.c THEN 'Fix first'
         WHEN s.avg_risk >= o.r                                       THEN 'Manage efficiently'
         WHEN s.avg_replacement_cost_inr >= o.c                       THEN 'Protect and monitor'
         ELSE 'Maintain' END                                          AS quadrant
FROM seg s CROSS JOIN overall o;
