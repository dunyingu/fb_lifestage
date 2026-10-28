SELECT
max(SYSTEMDATE) AS last_score_date,
datediff(day, last_score_date, CURRENT_DATE()) AS DAYS_IN_BETWEEN
FROM ?score_output_tbl