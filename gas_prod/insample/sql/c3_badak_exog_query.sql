SELECT prod_date AS date, realization AS lpg_c3
FROM
lng_lpg_c3_daily
WHERE prod_date BETWEEN '{}' AND '{}'
ORDER BY prod_date 