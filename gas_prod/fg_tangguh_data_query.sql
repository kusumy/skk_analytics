SELECT prod_date AS date, realization_prod AS feed_gas, target_wpnb AS wpnb_gas, is_unplanned_shutdown AS unplanned_shutdown, is_planned_shutdown AS planned_shutdown
FROM public.f_lng_feed_gas_bptangguh('BP Tangguh', '20200101', '20221120')