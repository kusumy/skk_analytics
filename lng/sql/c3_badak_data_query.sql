SELECT prod_date AS date, realization AS lpg_c3
FROM lng_lpg_c3_daily
WHERE prod_date BETWEEN '{}' AND '{}'
AND prod_date <= (SELECT max(prod_date) FROM lng_lpg_c3_daily WHERE realization is not null)
ORDER BY prod_date