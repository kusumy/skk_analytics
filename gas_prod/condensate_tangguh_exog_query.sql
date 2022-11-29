SELECT prod_date AS date, target_wpnb AS wpnb_minyak, is_planned_shutdown AS planned_shutdown
FROM f_lng_condensate_bptangguh('BP Tangguh', '20220914', '20221231')