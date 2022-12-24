SELECT prod_date AS date, realization AS lpg_c3
FROM
lng_lpg_c3_daily
WHERE prod_date BETWEEN '2022-11-11' AND '2023-04-30'
ORDER BY prod_date 