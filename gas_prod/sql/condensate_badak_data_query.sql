SELECT prod_date AS date, realization AS condensate
FROM
lng_condensate_daily
WHERE lng_plant = 'PT Badak'
AND prod_date < '2022-11-11'
ORDER BY prod_date 