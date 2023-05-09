SELECT prod_date AS date, realization AS feed_gas
FROM lng_feed_gas_daily
WHERE prod_date BETWEEN '{}' AND '{}'
AND prod_date <= (SELECT max(prod_date) FROM lng_feed_gas_daily WHERE realization is not null)
AND lng_plant = 'PT Badak'
ORDER BY prod_date
