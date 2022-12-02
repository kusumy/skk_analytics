SELECT prod_date AS date, realization AS lpg_c3
FROM
lng_lpg_c3_daily
WHERE prod_date BETWEEN '2019-03-28' AND '2022-11-10'
ORDER BY prod_date 