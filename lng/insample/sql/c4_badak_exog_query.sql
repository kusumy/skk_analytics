SELECT prod_date AS date, realization AS lpg_c4
FROM lng_lpg_c4_daily
WHERE prod_date BETWEEN '{}' AND '{}'
ORDER BY prod_date 