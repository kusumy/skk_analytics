SELECT prod_date AS date, realization AS lpg_c4
FROM
lng_lpg_c4_daily
WHERE prod_date BETWEEN '2018-02-01' AND '2022-11-10'
ORDER BY prod_date 