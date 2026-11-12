-- =====================================================================
-- HR Attrition Intelligence - SQL layer
-- Run in pgAdmin Query Tool on database hr_attrition, AFTER:
--     python ingest_attrition.py load        (creates raw.company_attrition)
-- Safe to re-run: everything uses CREATE OR REPLACE / IF NOT EXISTS / upsert.
-- =====================================================================

-- ---------------------------------------------------------------------
-- 0. Schemas, clean-up of the earlier trial view, raw-table guard
-- ---------------------------------------------------------------------
CREATE SCHEMA IF NOT EXISTS raw;
CREATE SCHEMA IF NOT EXISTS staging;
CREATE SCHEMA IF NOT EXISTS mart;

DROP VIEW IF EXISTS mart.attrition_quarterly;   -- replaced by the views below

-- one row per company + quarter + definition (the Python script already dedupes)
CREATE UNIQUE INDEX IF NOT EXISTS ux_raw_company_attrition
    ON raw.company_attrition (company, period_end, definition);

-- ---------------------------------------------------------------------
-- 1. Dimensions
-- ---------------------------------------------------------------------
CREATE TABLE IF NOT EXISTS mart.dim_company (
    company  TEXT PRIMARY KEY,
    ticker   TEXT,
    country  TEXT,
    segment  TEXT
);
INSERT INTO mart.dim_company (company, ticker, country, segment) VALUES
    ('TCS',     'TCS',   'India', 'IT services'),
    ('Infosys', 'INFY',  'India', 'IT services'),
    ('Wipro',   'WIPRO', 'India', 'IT services')
ON CONFLICT (company) DO NOTHING;

-- How each reported measure should be interpreted and which one to prefer
CREATE TABLE IF NOT EXISTS mart.dim_definition (
    definition          TEXT PRIMARY KEY,
    measure_type        TEXT NOT NULL,   -- 'ltm' (last twelve months) or 'annualized' (one quarter x4)
    scope               TEXT NOT NULL,   -- 'voluntary' / 'total' / 'unspecified'
    comparability_rank  INT  NOT NULL,   -- lower = preferred in the harmonised series
    description         TEXT
);
INSERT INTO mart.dim_definition VALUES
    ('voluntary_ltm',           'ltm',        'voluntary',   1, 'Voluntary attrition, last twelve months (IT services where stated)'),
    ('ltm_unspecified',         'ltm',        'unspecified', 2, 'LTM attrition as reported without a voluntary label (TCS before FY26 Q2); scope to be confirmed from footnotes'),
    ('total_ltm',               'ltm',        'total',       3, 'LTM attrition, older Infosys sheets'),
    ('annualized_consolidated', 'annualized', 'total',       9, 'Quarterly attrition annualized, consolidated (Infosys FY16-FY20)'),
    ('annualized_standalone',   'annualized', 'total',       9, 'Quarterly attrition annualized, standalone (Infosys FY16-FY20)'),
    ('annualized_voluntary_it', 'annualized', 'voluntary',   9, 'Quarterly voluntary attrition annualized, IT services (Infosys FY21)')
ON CONFLICT (definition) DO UPDATE SET
    measure_type = EXCLUDED.measure_type,
    scope = EXCLUDED.scope,
    comparability_rank = EXCLUDED.comparability_rank,
    description = EXCLUDED.description;

-- Indian fiscal quarters (Q1 = Apr-Jun). Use as the date table in Power BI.
CREATE OR REPLACE VIEW mart.dim_quarter AS
WITH q AS (
    SELECT (qs + INTERVAL '3 months' - INTERVAL '1 day')::date AS period_end
    FROM generate_series(TIMESTAMP '2003-04-01', TIMESTAMP '2027-04-01', INTERVAL '3 months') AS qs
)
SELECT
    period_end,
    'FY' || to_char((EXTRACT(YEAR FROM period_end)::int
                     + CASE WHEN EXTRACT(MONTH FROM period_end) = 3 THEN 0 ELSE 1 END) % 100, 'FM00') AS fiscal_year,
    CASE EXTRACT(MONTH FROM period_end)::int WHEN 6 THEN 1 WHEN 9 THEN 2 WHEN 12 THEN 3 ELSE 4 END      AS fiscal_quarter,
    EXTRACT(YEAR FROM period_end)::int                                                                 AS calendar_year,
    to_char(period_end, 'Mon YYYY')                                                                    AS label
FROM q;

-- ---------------------------------------------------------------------
-- 2. Staging: typed, trusted rows only
-- ---------------------------------------------------------------------
CREATE OR REPLACE VIEW staging.attrition_clean AS
SELECT
    company,
    fiscal_year,
    quarter::smallint               AS fiscal_quarter,
    period_end::date                AS period_end,
    attrition_pct::numeric(5,2)     AS attrition_pct,
    definition,
    headcount::bigint               AS headcount,
    gross_adds::bigint              AS gross_adds,
    net_adds::bigint                AS net_adds,
    attrition_count::bigint         AS attrition_count,
    status,
    source_url,
    source_file
FROM raw.company_attrition
WHERE attrition_pct IS NOT NULL
  AND status IN ('ok', 'manual_verify');

-- ---------------------------------------------------------------------
-- 3. Marts
-- ---------------------------------------------------------------------

-- 3a. Everything, every definition, with changes computed WITHIN a definition
--     (so a change of reporting basis never shows up as a fake jump)
CREATE OR REPLACE VIEW mart.fact_attrition_quarterly AS
WITH s AS (
    SELECT a.company, a.fiscal_year, a.fiscal_quarter, a.period_end, a.attrition_pct,
           a.definition, d.measure_type, d.scope, a.headcount, a.status, a.source_url
    FROM staging.attrition_clean a
    JOIN mart.dim_definition d ON d.definition = a.definition
)
SELECT
    cur.*,
    ROUND(cur.attrition_pct - pq.attrition_pct, 2) AS qoq_change_pp,
    ROUND(cur.attrition_pct - py.attrition_pct, 2) AS yoy_change_pp
FROM s cur
LEFT JOIN s pq
       ON pq.company = cur.company AND pq.definition = cur.definition
      AND pq.period_end = (date_trunc('month', cur.period_end) - INTERVAL '2 months' - INTERVAL '1 day')::date
LEFT JOIN s py
       ON py.company = cur.company AND py.definition = cur.definition
      AND py.period_end = (cur.period_end - INTERVAL '1 year')::date;

-- 3b. Harmonised LTM series: one value per company per quarter, preferring
--     voluntary > unspecified > total. Annualized measures are excluded.
--     basis_changed_vs_prev_q flags quarters where the definition switched.
CREATE OR REPLACE VIEW mart.attrition_comparable AS
WITH pick AS (
    SELECT DISTINCT ON (a.company, a.period_end)
        a.company, a.fiscal_year, a.fiscal_quarter, a.period_end,
        a.attrition_pct, a.definition, d.scope, a.headcount
    FROM staging.attrition_clean a
    JOIN mart.dim_definition d ON d.definition = a.definition
    WHERE d.measure_type = 'ltm'
    ORDER BY a.company, a.period_end, d.comparability_rank
)
SELECT
    cur.*,
    ROUND(cur.attrition_pct - pq.attrition_pct, 2) AS qoq_change_pp,
    ROUND(cur.attrition_pct - py.attrition_pct, 2) AS yoy_change_pp,
    (pq.definition IS NOT NULL AND pq.definition <> cur.definition) AS basis_changed_vs_prev_q,
    CASE WHEN py.attrition_pct IS NULL THEN 'n/a'
         WHEN cur.attrition_pct - py.attrition_pct >=  3 THEN 'surge'
         WHEN cur.attrition_pct - py.attrition_pct <= -3 THEN 'cool-down'
         ELSE 'stable' END AS phase
FROM pick cur
LEFT JOIN pick pq
       ON pq.company = cur.company
      AND pq.period_end = (date_trunc('month', cur.period_end) - INTERVAL '2 months' - INTERVAL '1 day')::date
LEFT JOIN pick py
       ON py.company = cur.company
      AND py.period_end = (cur.period_end - INTERVAL '1 year')::date;

-- 3c. Attrition cycle per company (since FY20): trough -> peak -> latest.
--     CAUTION: Infosys' LTM series only starts in FY22 (FY16-FY21 are annualized),
--     so its LTM "trough" is mid-cycle. series_start shows where each series begins.
CREATE OR REPLACE VIEW mart.attrition_cycle_summary AS
WITH s AS (
    SELECT * FROM mart.attrition_comparable WHERE period_end >= DATE '2019-04-01'
),
n AS (
    SELECT company, COUNT(*) AS n_quarters, MIN(period_end) AS series_start
    FROM s GROUP BY company
),
peak AS (
    SELECT DISTINCT ON (company) company, period_end AS peak_date, attrition_pct AS peak_pct
    FROM s ORDER BY company, attrition_pct DESC, period_end
),
trough AS (
    SELECT DISTINCT ON (s.company) s.company, s.period_end AS trough_date, s.attrition_pct AS trough_pct
    FROM s JOIN peak p ON p.company = s.company AND s.period_end <= p.peak_date
    ORDER BY s.company, s.attrition_pct, s.period_end
),
latest AS (
    SELECT DISTINCT ON (company) company, period_end AS latest_date, attrition_pct AS latest_pct
    FROM s ORDER BY company, period_end DESC
)
SELECT
    p.company,
    n.series_start,
    n.n_quarters,
    t.trough_date, t.trough_pct,
    p.peak_date,   p.peak_pct,
    ROUND(p.peak_pct - t.trough_pct, 1)                      AS rise_pp,
    ROUND(p.peak_pct / NULLIF(t.trough_pct, 0), 2)           AS peak_to_trough_x,
    (EXTRACT(YEAR  FROM age(p.peak_date, t.trough_date)) * 12
   + EXTRACT(MONTH FROM age(p.peak_date, t.trough_date)))::int AS months_trough_to_peak,
    l.latest_date, l.latest_pct,
    ROUND(p.peak_pct - l.latest_pct, 1)                      AS decline_from_peak_pp
FROM peak p
JOIN n      USING (company)
JOIN trough t USING (company)
JOIN latest l USING (company)
WHERE n.n_quarters >= 8;       -- skip series too short to describe a cycle (e.g. Wipro for now)

-- 3d. TCS vs Infosys, quarter by quarter, with a flag for like-for-like basis
CREATE OR REPLACE VIEW mart.peer_gap_tcs_infosys AS
SELECT
    t.period_end, t.fiscal_year, t.fiscal_quarter,
    t.attrition_pct AS tcs_pct,     t.definition AS tcs_definition,
    i.attrition_pct AS infosys_pct, i.definition AS infosys_definition,
    ROUND(i.attrition_pct - t.attrition_pct, 1) AS infosys_minus_tcs_pp,
    (t.scope = i.scope)                          AS same_scope
FROM mart.attrition_comparable t
JOIN mart.attrition_comparable i ON i.period_end = t.period_end
WHERE t.company = 'TCS' AND i.company = 'Infosys';

-- 3e. Rough exits implied by the LTM rate (APPROXIMATION: LTM rate x closing headcount)
CREATE OR REPLACE VIEW mart.estimated_exits AS
SELECT
    company, fiscal_year, fiscal_quarter, period_end,
    attrition_pct, headcount,
    ROUND(headcount * attrition_pct / 100.0) AS est_exits_ltm
FROM mart.attrition_comparable
WHERE headcount IS NOT NULL;

-- ---------------------------------------------------------------------
-- 4. Data-quality views
-- ---------------------------------------------------------------------
-- quarters with no LTM value between a company's first and last LTM quarter
CREATE OR REPLACE VIEW mart.dq_missing_quarters AS
WITH bounds AS (
    SELECT company, MIN(period_end) AS lo, MAX(period_end) AS hi
    FROM mart.attrition_comparable GROUP BY company
)
SELECT b.company, q.fiscal_year, q.fiscal_quarter, q.period_end
FROM bounds b
JOIN mart.dim_quarter q ON q.period_end BETWEEN b.lo AND b.hi
LEFT JOIN mart.attrition_comparable a ON a.company = b.company AND a.period_end = q.period_end
WHERE a.period_end IS NULL;

-- quarter-on-quarter moves of 3+ points inside one definition: check against the PDF
CREATE OR REPLACE VIEW mart.dq_big_jumps AS
SELECT company, fiscal_year, fiscal_quarter, period_end, definition, attrition_pct, qoq_change_pp
FROM mart.fact_attrition_quarterly
WHERE ABS(qoq_change_pp) >= 3;
