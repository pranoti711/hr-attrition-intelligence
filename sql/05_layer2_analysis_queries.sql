-- =====================================================================
-- Layer 2b: SQL analysis pack on the employee data
-- Run each query on its own in pgAdmin (select it, press F5).
-- Needs: staging.hr_employee, mart.employee_attrition_risk, mart.employee_attrition_cost
-- =====================================================================

-- Q1. Retention curve in pure SQL (a yearly life table = discrete Kaplan-Meier)
--     at_risk = people who reached tenure year t; hazard = leavers / at_risk;
--     survival = running product of (1 - hazard).
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
       t AS tenure_year, at_risk, leavers,
       ROUND(leavers::numeric / at_risk, 3) AS hazard,
       ROUND(EXP(SUM(LN(GREATEST(1 - leavers::numeric / at_risk, 1e-9)))
                 OVER (PARTITION BY is_overtime ORDER BY t)), 3) AS retained
FROM atrisk
ORDER BY grp, t;

-- Q2. Which job role is the attrition hotspot inside each department? (window functions)
WITH overall AS (SELECT AVG(attrition_flag) AS r FROM staging.hr_employee),
roles AS (
    SELECT department, job_role, COUNT(*) AS employees,
           SUM(attrition_flag) AS leavers, AVG(attrition_flag) AS rate
    FROM staging.hr_employee GROUP BY department, job_role
)
SELECT department, job_role, employees, leavers,
       ROUND(100 * rate, 1)                                   AS attrition_pct,
       ROUND(100 * (rate - o.r), 1)                           AS pp_vs_company,
       DENSE_RANK() OVER (PARTITION BY department ORDER BY rate DESC) AS rank_in_department
FROM roles CROSS JOIN overall o
ORDER BY department, rank_in_department;

-- Q3. Overtime x tenure: does overtime hurt at every stage? (conditional aggregation)
SELECT tenure_band,
       COUNT(*) FILTER (WHERE is_overtime)                                   AS overtime_n,
       ROUND(100 * AVG(attrition_flag) FILTER (WHERE is_overtime), 1)        AS overtime_attrition_pct,
       COUNT(*) FILTER (WHERE NOT is_overtime)                               AS no_overtime_n,
       ROUND(100 * AVG(attrition_flag) FILTER (WHERE NOT is_overtime), 1)    AS no_overtime_attrition_pct
FROM staging.hr_employee
GROUP BY tenure_band
ORDER BY MIN(years_at_company);

-- Q4. Pay of leavers vs stayers (quartiles, not just averages)
SELECT CASE WHEN attrition_flag = 1 THEN 'Left' ELSE 'Stayed' END AS status,
       COUNT(*) AS employees,
       PERCENTILE_CONT(0.25) WITHIN GROUP (ORDER BY monthly_income) AS p25,
       PERCENTILE_CONT(0.50) WITHIN GROUP (ORDER BY monthly_income) AS median,
       PERCENTILE_CONT(0.75) WITHIN GROUP (ORDER BY monthly_income) AS p75
FROM staging.hr_employee
GROUP BY attrition_flag
ORDER BY attrition_flag DESC;

-- Q5. Pay position vs peers (compa-ratio = pay / median pay of the same job level)
WITH med AS (
    SELECT job_level, PERCENTILE_CONT(0.5) WITHIN GROUP (ORDER BY monthly_income) AS median_pay
    FROM staging.hr_employee GROUP BY job_level
),
ratio AS (
    SELECT e.attrition_flag, e.monthly_income / m.median_pay AS compa
    FROM staging.hr_employee e JOIN med m USING (job_level)
)
SELECT CASE WHEN compa < 0.8 THEN '1) below 80% of peers'
            WHEN compa < 1.0 THEN '2) 80-100%'
            WHEN compa < 1.2 THEN '3) 100-120%'
            ELSE '4) above 120%' END AS pay_position,
       COUNT(*) AS employees,
       ROUND(100 * AVG(attrition_flag), 1) AS attrition_pct
FROM ratio
GROUP BY 1
ORDER BY 1;

-- Q6. Career stagnation and manager change
SELECT CASE WHEN years_since_last_promotion = 0 THEN '0 (promoted this year)'
            WHEN years_since_last_promotion <= 2 THEN '1-2'
            WHEN years_since_last_promotion <= 5 THEN '3-5'
            ELSE '6+' END                            AS years_since_promotion,
       COUNT(*)                                      AS employees,
       ROUND(100 * AVG(attrition_flag), 1)           AS attrition_pct
FROM staging.hr_employee
GROUP BY 1
ORDER BY MIN(years_since_last_promotion);

SELECT CASE WHEN years_with_curr_manager = 0 THEN 'new manager (0)'
            WHEN years_with_curr_manager <= 2 THEN '1-2 years'
            ELSE '3+ years' END                      AS time_with_manager,
       COUNT(*)                                      AS employees,
       ROUND(100 * AVG(attrition_flag), 1)           AS attrition_pct
FROM staging.hr_employee
GROUP BY 1
ORDER BY MIN(years_with_curr_manager);

-- Q7. Top 3 retention priorities in every department (current staff, by expected cost)
WITH ranked AS (
    SELECT c.department, c.job_role, c.employee_id, c.risk_tier,
           c.risk_12m_calibrated, c.expected_cost_inr, r.top_reasons,
           ROW_NUMBER() OVER (PARTITION BY c.department ORDER BY c.expected_cost_inr DESC) AS rn
    FROM mart.employee_attrition_cost c
    JOIN mart.employee_attrition_risk r USING (employee_id)
)
SELECT department, rn AS priority, employee_id, job_role, risk_tier,
       ROUND((100 * risk_12m_calibrated)::numeric, 1)  AS risk_pct,
       ROUND((expected_cost_inr / 1e5)::numeric, 1)    AS expected_cost_lakh,
       top_reasons
FROM ranked
WHERE rn <= 3
ORDER BY department, rn;

-- Q8. Data-quality checks (every row should say PASS)
SELECT 'employee rows = 1470' AS check_name,
       CASE WHEN COUNT(*) = 1470 THEN 'PASS' ELSE 'FAIL' END AS result, COUNT(*)::text AS value
FROM staging.hr_employee
UNION ALL
SELECT 'employee ids unique',
       CASE WHEN COUNT(*) = COUNT(DISTINCT employee_id) THEN 'PASS' ELSE 'FAIL' END, COUNT(DISTINCT employee_id)::text
FROM staging.hr_employee
UNION ALL
SELECT 'leavers = 237 (matches Python)',
       CASE WHEN SUM(attrition_flag) = 237 THEN 'PASS' ELSE 'FAIL' END, SUM(attrition_flag)::text
FROM staging.hr_employee
UNION ALL
SELECT 'no negative tenure or income',
       CASE WHEN COUNT(*) = 0 THEN 'PASS' ELSE 'FAIL' END, COUNT(*)::text
FROM staging.hr_employee WHERE years_at_company < 0 OR monthly_income <= 0
UNION ALL
SELECT 'years at company <= total working years',
       CASE WHEN COUNT(*) = 0 THEN 'PASS' ELSE 'FAIL' END, COUNT(*)::text
FROM staging.hr_employee WHERE years_at_company > total_working_years
UNION ALL
SELECT 'every risk score has an employee',
       CASE WHEN COUNT(*) = 0 THEN 'PASS' ELSE 'FAIL' END, COUNT(*)::text
FROM mart.employee_attrition_risk r
LEFT JOIN staging.hr_employee e USING (employee_id) WHERE e.employee_id IS NULL
UNION ALL
SELECT 'risk probabilities between 0 and 1',
       CASE WHEN COUNT(*) = 0 THEN 'PASS' ELSE 'FAIL' END, COUNT(*)::text
FROM mart.employee_attrition_risk WHERE risk_probability < 0 OR risk_probability > 1;
