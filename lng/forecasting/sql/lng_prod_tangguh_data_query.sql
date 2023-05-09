SELECT prod_date AS date, realization_prod AS lng_production, is_planned_shutdown AS planned_shutdown, is_unplanned_shutdown AS unplanned_shutdown
FROM f_lng_prod_bptangguh('BP Tangguh', '{}', '{}')
WHERE prod_date <= (SELECT max(prod_date) FROM lng_production_daily WHERE realization is not null and lng_plant = 'BP Tangguh')