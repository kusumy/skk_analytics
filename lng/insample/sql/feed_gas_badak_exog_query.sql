SELECT prod_date AS date, realization AS feed_gas
FROM lng_feed_gas_daily WHERE lng_plant = 'PT Badak'
AND prod_date BETWEEN '{}' AND '{}'
ORDER BY prod_date 