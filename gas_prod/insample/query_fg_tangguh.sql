SELECT lng_plant, prod_date AS date, realization AS feed_gas
FROM
lng_feed_gas_daily
WHERE lng_plant = 'BP Tangguh'