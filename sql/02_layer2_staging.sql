-- Layer 2 / step 1: staging + descriptive marts for the IBM HR dataset
-- Run in pgAdmin on hr_attrition AFTER python layer2_load_ibm.py. Safe to re-run.
CREATE SCHEMA IF NOT EXISTS staging;
CREATE SCHEMA IF NOT EXISTS mart;

CREATE UNIQUE INDEX IF NOT EXISTS ux_raw_ibm_hr_employee
    ON raw.ibm_hr_employee (employee_number);

CREATE OR REPLACE VIEW staging.hr_employee AS
SELECT
    employee_number                          AS employee_id,
    attrition_flag,                                           -- 1 = left, 0 = still employed
    age, gender, marital_status, education, education_field,
    department, job_role, job_level, business_travel,
    (over_time = 'Yes')                      AS is_overtime,
    distance_from_home, monthly_income, percent_salary_hike, stock_option_level,
    num_companies_worked, total_working_years, training_times_last_year,
    years_at_company, years_in_current_role, years_since_last_promotion, years_with_curr_manager,
    environment_satisfaction, job_satisfaction, job_involvement,
    relationship_satisfaction, work_life_balance, performance_rating,
    CASE WHEN years_at_company < 1  THEN '0-1y'
         WHEN years_at_company < 3  THEN '1-3y'
         WHEN years_at_company < 5  THEN '3-5y'
         WHEN years_at_company < 10 THEN '5-10y'
         ELSE '10y+' END                     AS tenure_band,
    CASE WHEN age < 30 THEN '<30'
         WHEN age < 40 THEN '30-39'
         WHEN age < 50 THEN '40-49'
         ELSE '50+' END                      AS age_band,
    NTILE(4) OVER (ORDER BY monthly_income)  AS income_quartile
FROM raw.ibm_hr_employee;

-- attrition rate by every segment we care about, with lift vs the overall rate
CREATE OR REPLACE VIEW mart.hr_attrition_by_segment AS
WITH seg AS (
    SELECT 'department' AS dimension, department AS segment, COUNT(*) AS employees, SUM(attrition_flag) AS leavers FROM staging.hr_employee GROUP BY department
    UNION ALL SELECT 'job_role', job_role, COUNT(*), SUM(attrition_flag) FROM staging.hr_employee GROUP BY job_role
    UNION ALL SELECT 'overtime', CASE WHEN is_overtime THEN 'Yes' ELSE 'No' END, COUNT(*), SUM(attrition_flag) FROM staging.hr_employee GROUP BY is_overtime
    UNION ALL SELECT 'tenure_band', tenure_band, COUNT(*), SUM(attrition_flag) FROM staging.hr_employee GROUP BY tenure_band
    UNION ALL SELECT 'age_band', age_band, COUNT(*), SUM(attrition_flag) FROM staging.hr_employee GROUP BY age_band
    UNION ALL SELECT 'income_quartile', 'Q' || income_quartile, COUNT(*), SUM(attrition_flag) FROM staging.hr_employee GROUP BY income_quartile
    UNION ALL SELECT 'business_travel', business_travel, COUNT(*), SUM(attrition_flag) FROM staging.hr_employee GROUP BY business_travel
    UNION ALL SELECT 'marital_status', marital_status, COUNT(*), SUM(attrition_flag) FROM staging.hr_employee GROUP BY marital_status
    UNION ALL SELECT 'job_satisfaction', job_satisfaction::text, COUNT(*), SUM(attrition_flag) FROM staging.hr_employee GROUP BY job_satisfaction
    UNION ALL SELECT 'work_life_balance', work_life_balance::text, COUNT(*), SUM(attrition_flag) FROM staging.hr_employee GROUP BY work_life_balance
    UNION ALL SELECT 'gender', gender, COUNT(*), SUM(attrition_flag) FROM staging.hr_employee GROUP BY gender
),
overall AS (SELECT AVG(attrition_flag) AS r FROM staging.hr_employee)
SELECT
    s.dimension, s.segment, s.employees, s.leavers,
    ROUND(100.0 * s.leavers / s.employees, 1)                  AS attrition_pct,
    ROUND((100.0 * s.leavers / s.employees) / (o.r * 100), 2)  AS lift_vs_overall
FROM seg s CROSS JOIN overall o;
