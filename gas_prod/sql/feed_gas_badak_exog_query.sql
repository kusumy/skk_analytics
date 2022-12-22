SELECT prod_date AS date, realization AS feed_gas
FROM
lng_feed_gas_daily
WHERE lng_plant = 'PT Badak'
AND prod_date BETWEEN '2022-11-11' AND '2023-04-30'
ORDER BY prod_date 