SELECT prod_date AS date, realization AS feed_gas
FROM
lng_feed_gas_daily
WHERE lng_plant = 'PT Badak'
AND prod_date < '2022-11-11'
ORDER BY prod_date 