DROP VIEW IF EXISTS employee_score CASCADE;
DROP TABLE IF EXISTS tv_meta CASCADE;
DROP TABLE IF EXISTS talent_benchmarks CASCADE;

CREATE TABLE talent_benchmarks (
  job_vacancy_id serial PRIMARY KEY,
  role_name text,
  job_level text,
  role_purpose text,
  weights_config jsonb
);
INSERT INTO talent_benchmarks VALUES (
  DEFAULT,
  'Data Analyst',
  'V',
  'Turn raw data into actionable business insights to drive decision-making',
  '{
    "Motivation & Drive": 0.205389,
    "Social Orientation & Collaboration": 0.186995,
    "Creativity & Innovation Orientation": 0.166815,
    "Cognitive Complexity & Problem-Solving": 0.160294,
    "Cultural & Values Urgency": 0.093856,
    "Leadership & Influence": 0.039483
  }'::jsonb
);

-- Feature dictionary from Step 1
CREATE TABLE tv_meta (
  tgv_name text,
  tv_name text,
  feature_weight numeric
);

INSERT INTO tv_meta (tgv_name, tv_name, feature_weight) VALUES
  ('Motivation & Drive', 'competencies_qdd', 0.168140),
  ('Motivation & Drive', 'papi_a', 0.037249),
  ('Social Orientation & Collaboration', 'competencies_sea', 0.149106),
  ('Social Orientation & Collaboration', 'papi_s', 0.037889),
  ('Creativity & Innovation Orientation', 'competencies_ids', 0.132537),
  ('Creativity & Innovation Orientation', 'strength_ideation', 0.034278),
  ('Cognitive Complexity & Problem-Solving', 'competencies_sto', 0.125195),
  ('Cognitive Complexity & Problem-Solving', 'iq', 0.035099),
  ('Cultural & Values Urgency', 'competencies_vcu', 0.093856),
  ('Leadership & Influence', 'disc_d', 0.039483);

-- Unpivot the table so computing later will be easier
CREATE OR REPLACE VIEW employee_score AS
SELECT employee_id, 'competencies_qdd' AS tv_name, competencies_qdd::numeric AS score FROM master_dataset
UNION ALL SELECT employee_id, 'papi_a', papi_a::numeric FROM master_dataset
UNION ALL SELECT employee_id, 'competencies_sea', competencies_sea::numeric FROM master_dataset
UNION ALL SELECT employee_id, 'papi_s', papi_s::numeric FROM master_dataset
UNION ALL SELECT employee_id, 'competencies_ids', competencies_ids::numeric FROM master_dataset
UNION ALL SELECT employee_id, 'strength_ideation', strength_ideation::int FROM master_dataset
UNION ALL SELECT employee_id, 'competencies_sto', competencies_sto::numeric FROM master_dataset
UNION ALL SELECT employee_id, 'iq', iq::numeric FROM master_dataset
UNION ALL SELECT employee_id, 'competencies_vcu', competencies_vcu::numeric FROM master_dataset
UNION ALL SELECT employee_id, 'disc_d', disc_d::int FROM master_dataset;

WITH role_tb AS ( 
  SELECT * FROM talent_benchmarks WHERE job_vacancy_id = 1
), selected_benchmarks AS (
  SELECT employee_id
  FROM master_dataset
  WHERE performance_rating = 5
),

-- Algorithm
benchmark_scores AS (
  SELECT
    m.tgv_name,
    es.tv_name,
    percentile_cont(0.5) WITHIN GROUP (ORDER BY es.score) AS baseline_score
  FROM employee_score es
  JOIN selected_benchmarks sb ON sb.employee_id = es.employee_id
  JOIN tv_meta m ON es.tv_name = m.tv_name
  GROUP BY m.tgv_name, es.tv_name
),

tv_match AS (
  SELECT
    e.employee_id,
    e.fullname,
    e.directorate_name,
    e.position_name,
    e.grade_name,
    m.tgv_name,
    es.tv_name,
    m.feature_weight,
    b.baseline_score,
    es.score AS user_score,
    CASE
      WHEN b.baseline_score IS NULL THEN 0
      -- Boolean OHE categoricals
      WHEN es.tv_name LIKE 'disc_%' THEN
        CASE WHEN es.score = b.baseline_score THEN 100 ELSE 0 END
      -- Variables with inverse scale (lower is better), or rank-based TVs
      WHEN es.tv_name LIKE 'papi_z%' OR es.tv_name LIKE 'papi_k%' or es.tv_name LIKE 'strength_%' THEN
        ROUND(((2 * b.baseline_score - es.score) / NULLIF(b.baseline_score, 0) * 100)::numeric, 2)
      -- Numeric TVs
      WHEN es.tv_name NOT LIKE 'papi_z%' AND es.tv_name NOT LIKE 'papi_k%' AND es.tv_name NOT LIKE 'strength_%' AND es.tv_name NOT LIKE 'disc_%' THEN
        ROUND((es.score / NULLIF(b.baseline_score, 0) * 100)::numeric, 2)
      ELSE NULL
    END AS tv_match_rate,
    COALESCE(m.feature_weight, 1.0) AS tv_weight
  FROM employee_score es
  JOIN master_dataset e ON e.employee_id = es.employee_id
  JOIN tv_meta m ON es.tv_name = m.tv_name
  LEFT JOIN benchmark_scores b ON b.tv_name = es.tv_name
),

tgv_match AS (
  SELECT
    tm.employee_id,
    tm.tgv_name,
    ROUND(SUM(tm.tv_match_rate * tm.tv_weight) / NULLIF(SUM(tm.tv_weight),0), 2) AS tgv_match_rate,
    COALESCE((tb.weights_config ->> tm.tgv_name)::numeric, 1.0) AS tgv_weight_for_final
  FROM tv_match tm
  CROSS JOIN role_tb tb
  GROUP BY tm.employee_id, tm.tgv_name, tb.weights_config
),

final_match AS (
  SELECT
    employee_id,
    ROUND(SUM(tgv_match_rate * tgv_weight_for_final) / NULLIF(SUM(tgv_weight_for_final),0), 2) AS final_match_rate
  FROM tgv_match
  GROUP BY employee_id
)

-- Output
SELECT
  tm.employee_id,
  tm.directorate_name,
  tm.position_name AS role_name,
  tm.grade_name,
  tm.tgv_name,
  tm.tv_name,
  tm.baseline_score,
  tm.user_score,
  tm.tv_match_rate,
  tg.tgv_match_rate,
  fm.final_match_rate
FROM tv_match tm
JOIN tgv_match tg
  ON tm.employee_id = tg.employee_id AND tm.tgv_name = tg.tgv_name
JOIN final_match fm
  ON tm.employee_id = fm.employee_id
ORDER BY fm.final_match_rate DESC, tm.employee_id, tm.tgv_name, tm.tv_name;