SELECT prod_date AS date, target_wpnb AS wpnb_gas, is_planned_shutdown AS planned_shutdown
FROM f_lng_feed_gas_bptangguh('BP Tangguh', '{}', '{}')